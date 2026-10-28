"""Train fusion heads on FashionIQ train only, with an image-disjoint holdout.

No model downloads or official validation annotations are used. Run-specific
checkpoints, traces and full Top-50 rankings stay in artifacts/fusion-runs/.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.nn import functional as F

from cir_baseline.adaptive import confidence_bands, has_identity_intent, quantile_thresholds, select_weight_from_ranks
from cir_baseline.contracts import IndexManifest
from cir_baseline.datasets import load_fashioniq
from cir_baseline.fusion import l2_normalize_rows
from cir_baseline.index import FaissFlatIPIndex
from cir_baseline.learned_fusion import LearnedFusion, image_disjoint_split, multi_positive_contrastive_loss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(ranks, categories) -> dict:
    by_category = {
        category: {f"recall@{k}": float(np.mean(ranks[categories == category] <= k)) for k in (10, 50)}
        for category in sorted(set(categories.tolist()))
    }
    return {
        "query_count": len(ranks), "by_category": by_category,
        **{f"category_mean_recall@{k}": float(np.mean([row[f"recall@{k}"] for row in by_category.values()])) for k in (10, 50)},
    }


def objective(result) -> tuple:
    return result["category_mean_recall@10"], result["category_mean_recall@50"]


class ExperimentData:
    def __init__(self, storage: Path, config: dict) -> None:
        self.bundle = load_fashioniq(storage / "datasets" / "fashioniq", split="train")
        self.queries = self.bundle.queries
        self.categories = np.asarray([q.category for q in self.queries])
        self.references = [q.reference_image_id for q in self.queries]
        self.targets = [q.target_image_ids[0] for q in self.queries]
        self.partition, self.groups = image_disjoint_split(self.references, self.targets, config["split_seed"])
        self.positions = {key: np.flatnonzero(self.partition == key) for key in ("fit", "dev", "holdout")}
        self.split_audit = {}
        supervised_ids = {}
        for split, positions in self.positions.items():
            if not len(positions):
                raise ValueError(f"empty split: {split}")
            supervised_ids[split] = {item for i in positions for item in (self.references[i], self.targets[i])}
            self.split_audit[split] = {"query_count": len(positions), "component_count": len(set(self.groups[positions])),
                                     "supervised_image_count": len(supervised_ids[split]),
                                     "categories": dict(Counter(self.categories[positions]))}
            if set(self.categories[positions]) != {"dress", "shirt", "toptee"}:
                raise ValueError("a split is missing a category")
        overlap = {f"{a}:{b}": len(supervised_ids[a] & supervised_ids[b])
                   for a, b in (("fit", "dev"), ("fit", "holdout"), ("dev", "holdout"))}
        if any(overlap.values()):
            raise ValueError("supervised images cross splits")
        self.split_audit["supervised_image_overlap"] = overlap
        cache_root = storage / "derived" / "embeddings"
        paths = {}
        for name, pattern in (("gallery", "fashioniq_train_official_*.npz"), ("text", "fashioniq_train_text-ablation-v1_*.npz")):
            matches = sorted(cache_root.glob(pattern))
            if len(matches) != 1:
                raise ValueError(f"expected one {name} cache, found {len(matches)}")
            paths[name] = matches[0]
        with np.load(paths["gallery"], allow_pickle=False) as data:
            self.ids = data["image_ids"].astype(str).tolist()
            self.gallery = l2_normalize_rows(data["embeddings"])
            self.encoder_version = str(data["encoder_version"].item())
            if self.ids != [item.image_id for item in self.bundle.gallery]:
                raise ValueError("gallery cache ordering mismatch")
            if [Path(p).resolve() for p in data["image_paths"].astype(str)] != [item.path.resolve() for item in self.bundle.gallery]:
                raise ValueError("gallery cache image paths mismatch")
        with np.load(paths["text"], allow_pickle=False) as data:
            if data["query_ids"].astype(str).tolist() != [q.query_id for q in self.queries]:
                raise ValueError("text cache query IDs mismatch")
            if str(data["encoder_version"].item()) != self.encoder_version:
                raise ValueError("text/gallery encoder mismatch")
            digest = hashlib.sha256()
            for query in self.queries:
                digest.update(query.query_id.encode())
                for caption in query.modification_texts:
                    digest.update(b"\0")
                    digest.update(caption.encode())
            if str(data["text_digest"].item()) != digest.hexdigest():
                raise ValueError("text cache annotations mismatch")
            self.texts = l2_normalize_rows(data[config["text_strategy"]])
        if self.texts.shape != (len(self.queries), self.gallery.shape[1]):
            raise ValueError("incompatible cached embedding dimensions")
        self.cache = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}
        id_to_position = {key: i for i, key in enumerate(self.ids)}
        self.ref_positions = np.asarray([id_to_position[key] for key in self.references])
        self.target_positions = np.asarray([id_to_position[key] for key in self.targets])
        self.ref_vectors = self.gallery[self.ref_positions]
        self.indices = {}
        for category in sorted(set(self.categories)):
            positions = [i for i, item in enumerate(self.bundle.gallery) if category in item.categories]
            self.indices[category] = FaissFlatIPIndex(
                [self.ids[i] for i in positions], self.gallery[positions],
                IndexManifest(f"train-{category}-{self.cache['gallery']['sha256'][:12]}", self.encoder_version, self.gallery.shape[1]),
            )

    def search(self, positions, vectors, keep_rankings=False):
        ranks = np.full(len(positions), 51, dtype=np.int16)
        rankings = [None] * len(positions) if keep_rankings else None
        confidence = np.zeros(len(positions), dtype=np.float32)
        for category, index in self.indices.items():
            local = np.flatnonzero(self.categories[positions] == category)
            if not len(local):
                continue
            hits = index.search_many(vectors[local], 51)
            for offset, row in zip(local, hits, strict=True):
                query = self.queries[positions[offset]]
                filtered = [hit for hit in row if hit.image_id != query.reference_image_id][:50]
                ids = [hit.image_id for hit in filtered]
                if query.target_image_ids[0] in ids:
                    ranks[offset] = ids.index(query.target_image_ids[0]) + 1
                confidence[offset] = filtered[0].score - np.mean([hit.score for hit in filtered[1:10]])
                if rankings is not None:
                    rankings[offset] = {"ranked_image_ids": ids, "scores": [hit.score for hit in filtered]}
        return ranks, rankings, confidence

    def weighted(self, positions, weights):
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim:
            w = w[:, None]
        return l2_normalize_rows(w * self.ref_vectors[positions] + (1 - w) * self.texts[positions])


def fit_baselines(data, config):
    fit = data.positions["fit"]
    grids, confidence = [], None
    weights = config["weight_grid"]
    for weight in weights:
        ranks, _, conf = data.search(fit, data.weighted(fit, weight))
        grids.append(ranks)
        if weight == 0:
            confidence = conf
    grid = np.stack(grids)
    global_weight = weights[max(range(len(weights)), key=lambda i: (*objective(metrics(grid[i], data.categories[fit])), -abs(weights[i] - .3)))]
    identity = np.asarray([has_identity_intent(data.queries[i].modification_text) for i in fit])
    thresholds, gates, category_weights = {}, {}, {}
    for category in sorted(set(data.categories)):
        positions = np.flatnonzero(data.categories[fit] == category)
        category_weight, _ = select_weight_from_ranks(weights, grid, positions)
        category_weights[category] = category_weight
        ordinary = positions[~identity[positions]]
        edges = quantile_thresholds(confidence[ordinary])
        thresholds[category] = edges
        bands = confidence_bands(confidence[ordinary], edges)
        groups = {"identity": positions[identity[positions]],
                  **{f"confidence_{band}": ordinary[bands == band] for band in range(3)}}
        for name, indices in groups.items():
            gate = category_weight
            if len(indices) >= config["min_gate_group_size"]:
                gate, _ = select_weight_from_ranks(weights, grid, indices)
            gates[f"{category}:{name}"] = gate
    return {"global_image_weight": global_weight, "category_weights": category_weights,
            "confidence_thresholds": thresholds, "gate_weights": gates, "selection_split": "fit only"}


def baseline_vectors(data, positions, parameters):
    _, _, confidence = data.search(positions, data.texts[positions])
    dynamic = []
    for i, conf in zip(positions, confidence, strict=True):
        query = data.queries[i]
        if has_identity_intent(query.modification_text):
            suffix = "identity"
        else:
            band = int(confidence_bands([conf], parameters["confidence_thresholds"][query.category])[0])
            suffix = f"confidence_{band}"
        dynamic.append(parameters["gate_weights"][f"{query.category}:{suffix}"])
    return {
        "B0_image_only": data.weighted(positions, 1.),
        "B1_text_only": data.weighted(positions, 0.),
        "B2_equal": data.weighted(positions, .5),
        "A0_refit_global": data.weighted(positions, parameters["global_image_weight"]),
        "A2_refit_dynamic": data.weighted(positions, dynamic),
    }


def infer(model, data, positions, device):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(positions), 512):
            batch = positions[offset:offset + 512]
            vectors = model(torch.from_numpy(data.ref_vectors[batch]).to(device), torch.from_numpy(data.texts[batch]).to(device))
            chunks.append(vectors.cpu().numpy())
    return np.concatenate(chunks)


def train_one(data, config, variant, seed, run_dir, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = LearnedFusion(dimension=data.gallery.shape[1], hidden_dim=config["hidden_dim"],
                          image_weight=config["initial_image_weight"], residual=variant == "residual_fusion",
                          dropout=config["dropout"], residual_scale=config["residual_scale"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    fit, dev = data.positions["fit"], data.positions["dev"]
    refs = torch.from_numpy(data.ref_vectors).to(device)
    texts = torch.from_numpy(data.texts).to(device)
    gallery = torch.from_numpy(data.gallery).to(device)
    target_ids = torch.from_numpy(data.target_positions).to(device)
    best, best_epoch, trace = None, 0, []
    checkpoint = run_dir / f"{variant}-seed{seed}.pt"
    for epoch in range(config["epochs"] + 1):
        total_loss, samples = 0., 0
        if epoch:
            model.train()
            batches = []
            for category in sorted(set(data.categories)):
                selected = rng.permutation(fit[data.categories[fit] == category])
                batches.extend(selected[i:i + config["batch_size"]] for i in range(0, len(selected), config["batch_size"]))
            rng.shuffle(batches)
            for positions in batches:
                if len(positions) < 2:
                    continue
                indices = torch.as_tensor(positions, device=device)
                output = model(refs[indices], texts[indices])
                ids = target_ids[indices]
                loss = multi_positive_contrastive_loss(output, gallery[ids], ids, ids, config["temperature"])
                prior = F.normalize(config["initial_image_weight"] * refs[indices] + (1 - config["initial_image_weight"]) * texts[indices], dim=-1)
                loss = loss + config["anchor_penalty"] * (1 - (output * prior).sum(-1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += float(loss.detach()) * len(positions)
                samples += len(positions)
        if epoch % config["evaluate_every"] == 0 or epoch == config["epochs"]:
            ranks, _, _ = data.search(dev, infer(model, data, dev, device))
            observed = metrics(ranks, data.categories[dev])
            trace.append({"epoch": epoch, "loss": total_loss / samples if samples else None, "dev": observed})
            if best is None or objective(observed) > objective(best):
                best, best_epoch = observed, epoch
                torch.save({"model_config": model.config, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "encoder_version": data.encoder_version, "seed": seed, "epoch": epoch,
                            "variant": variant, "text_strategy": config["text_strategy"]}, checkpoint)
            print(f"variant={variant} seed={seed} epoch={epoch} dev_R10={observed['category_mean_recall@10']:.5f} dev_R50={observed['category_mean_recall@50']:.5f}", flush=True)
    write_json(run_dir / f"{variant}-seed{seed}-training.json", trace)
    return {"variant": variant, "seed": seed, "best_epoch": best_epoch, "dev": best,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint)}


def cluster_bootstrap(data, positions, baseline_ranks, candidate_ranks, repeats=2000):
    groups, inverse = np.unique(data.groups[positions], return_inverse=True)
    categories = sorted(set(data.categories))
    sizes = np.zeros((len(groups), len(categories)))
    changes = np.zeros_like(sizes)
    difference = np.mean(np.stack(candidate_ranks) <= 10, axis=0) - (baseline_ranks <= 10)
    for c, category in enumerate(categories):
        selected = data.categories[positions] == category
        np.add.at(sizes[:, c], inverse[selected], 1)
        np.add.at(changes[:, c], inverse[selected], difference[selected])
    rng = np.random.default_rng(20261028)
    deltas = []
    for _ in range(repeats):
        sample = rng.integers(0, len(groups), len(groups))
        counts = sizes[sample].sum(0)
        if np.all(counts > 0):
            deltas.append(float(np.mean(changes[sample].sum(0) / counts)))
    return {"unit": "reference-target connected component", "repeats": len(deltas),
            "ci95_low": float(np.quantile(deltas, .025)), "ci95_high": float(np.quantile(deltas, .975)),
            "scope": "sampling uncertainty for seed-averaged hit rate; seed SD reported separately"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "learned_fusion.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "fusion-runs" / "fashioniq-train-v1")
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts" / "metrics" / "learned_fusion_train_v1.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["dataset_split"] != "train" or config["text_strategy"] != "symmetric_join_mean":
        raise ValueError("this preregistered run only permits train and frozen caption strategy")
    if args.output_dir.exists():
        raise ValueError("output directory exists; choose a fresh run directory to preserve evidence")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    data = ExperimentData(args.storage_root, config)
    print(json.dumps(data.split_audit, ensure_ascii=False), flush=True)
    write_json(args.output_dir / "config.json", config)
    split_rows = [{"query_id": q.query_id, "partition": str(data.partition[i]), "component": str(data.groups[i])} for i, q in enumerate(data.queries)]
    write_json(args.output_dir / "split_manifest.json", split_rows)
    parameters = fit_baselines(data, config)
    write_json(args.output_dir / "baseline_parameters.json", parameters)
    dev = data.positions["dev"]
    dev_baselines = {name: metrics(data.search(dev, vectors)[0], data.categories[dev]) for name, vectors in baseline_vectors(data, dev, parameters).items()}
    runs = [train_one(data, config, variant, seed, args.output_dir, args.device)
            for variant in config["variants"] for seed in config["seeds"]]
    dev_means = {variant: np.mean([objective(run["dev"]) for run in runs if run["variant"] == variant], axis=0).tolist() for variant in config["variants"]}
    selected = max(config["variants"], key=lambda variant: tuple(dev_means[variant]))
    selection = {"selected_variant": selected, "selection_source": "dev only", "dev_variant_means_R10_R50": dev_means,
                 "runs": runs, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / "selection_frozen_before_holdout.json", selection)
    print(f"selection_frozen={selected}; evaluating holdout once", flush=True)
    holdout = data.positions["holdout"]
    all_metrics, all_ranks = {}, {}
    ranking_file = args.output_dir / "holdout_rankings.jsonl.gz"
    with gzip.open(ranking_file, "wt", encoding="utf-8") as handle:
        def report(name, vectors):
            ranks, rankings, _ = data.search(holdout, vectors, keep_rankings=True)
            all_metrics[name] = metrics(ranks, data.categories[holdout])
            all_ranks[name] = ranks
            for offset, position in enumerate(holdout):
                q = data.queries[position]
                handle.write(json.dumps({"model": name, "query_id": q.query_id, "category": q.category,
                                         "reference_image_id": q.reference_image_id, "target_image_ids": q.target_image_ids,
                                         "target_rank": int(ranks[offset]), **rankings[offset]}, ensure_ascii=False) + "\n")

        for name, vectors in baseline_vectors(data, holdout, parameters).items():
            report(name, vectors)
        for run in runs:
            if run["variant"] != selected:
                continue
            checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            model = LearnedFusion(**checkpoint["model_config"]).to(args.device)
            model.load_state_dict(checkpoint["state_dict"])
            report(f"{selected}-seed{run['seed']}", infer(model, data, holdout, args.device))
    names = [f"{selected}-seed{seed}" for seed in config["seeds"]]
    values = np.asarray([objective(all_metrics[name]) for name in names])
    mean, sd = values.mean(0), values.std(0, ddof=1)
    baseline = all_metrics["A2_refit_dynamic"]
    interval = cluster_bootstrap(data, holdout, all_ranks["A2_refit_dynamic"], [all_ranks[name] for name in names])
    gates = {"mean_R10_improves": bool(mean[0] > objective(baseline)[0]),
             "all_seeds_R10_improve": bool(np.all(values[:, 0] > objective(baseline)[0])),
             "mean_R50_not_worse": bool(mean[1] >= objective(baseline)[1]),
             "cluster_bootstrap_lower_positive": bool(interval["ci95_low"] > 0)}
    summary = {
        "status": "pass", "scope": "FashionIQ train-only image-disjoint internal holdout; not official val/test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "cache": data.cache, "encoder_version": data.encoder_version,
        "split_audit": data.split_audit, "split_manifest_sha256": sha256(args.output_dir / "split_manifest.json"),
        "baseline_parameters": parameters, "dev_baselines": dev_baselines, "selection": selection,
        "selection_manifest_sha256": sha256(args.output_dir / "selection_frozen_before_holdout.json"),
        "holdout": all_metrics,
        "three_seed_summary": {"category_mean_R10": float(mean[0]), "R10_sample_sd": float(sd[0]),
                               "category_mean_R50": float(mean[1]), "R50_sample_sd": float(sd[1]),
                               "delta_R10_vs_refit_A2": float(mean[0] - objective(baseline)[0]),
                               "delta_R50_vs_refit_A2": float(mean[1] - objective(baseline)[1]),
                               "group_bootstrap_R10_delta": interval},
        "promotion_checks": gates, "decision": "research_candidate_passes_internal_gate" if all(gates.values()) else "retain_A2_no_promotion",
        "deployed_model_changed": False,
        "artifacts": {"run_dir": str(args.output_dir.resolve()), "rankings": str(ranking_file.resolve()), "rankings_sha256": sha256(ranking_file)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": args.device, "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(args.summary, summary)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"selected": selected, "summary": summary["three_seed_summary"], "decision": summary["decision"], "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
