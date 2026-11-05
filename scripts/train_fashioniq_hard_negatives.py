"""Paired fit/dev-only negative sampling experiment; never score holdout/val."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from train_fashioniq_fusion import (
    ROOT, ExperimentData, baseline_vectors, infer, metrics, objective, sha256, write_json,
)
import numpy as np
import torch
from torch.nn import functional as F

from cir_baseline.hard_negatives import batch_candidate_masks, fit_image_pool, positive_key
from cir_baseline.learned_fusion import LearnedFusion, masked_multi_positive_loss


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate(model, data, split, device, rankings=False):
    if split not in ("fit", "dev"):
        raise ValueError("this experiment may evaluate fit/dev only")
    positions = data.positions[split]
    ranks, hits, _ = data.search(positions, infer(model, data, positions, device), rankings)
    return metrics(ranks, data.categories[positions]), ranks, hits


def prepare_banks(data, config, output, device):
    fit = data.positions["fit"]
    positive_map = defaultdict(set)
    for i in fit:
        positive_map[positive_key(data.ref_positions[i], data.queries[i].modification_texts)].add(int(data.target_positions[i]))
    positives = [positive_map[positive_key(data.ref_positions[i], data.queries[i].modification_texts)] for i in fit]
    width = config["bank_width"]
    hard, random = np.empty((len(fit), width), dtype=np.int64), np.empty((len(fit), width), dtype=np.int64)
    rng = np.random.default_rng(config["negative_seed"])
    gallery = torch.from_numpy(data.gallery).to(device)
    pools = {}
    for category in sorted(set(data.categories)):
        pool = fit_image_pool(data.partition, data.ref_positions, data.target_positions, data.categories, category)
        if len(pool) < width:
            raise ValueError("candidate pool smaller than bank")
        pool_set = set(pool.tolist())
        local = np.flatnonzero(data.categories[fit] == category)
        if any(not positives[i] <= pool_set for i in local):
            raise ValueError("positive outside category fit pool")
        pool_vectors = gallery[torch.as_tensor(pool, device=device)]
        for start in range(0, len(local), 256):
            batch = local[start:start + 256]
            vectors = torch.from_numpy(data.weighted(fit[batch], config["initial_image_weight"])).to(device)
            # Stable ties respect sorted gallery IDs; frozen prior is the miner.
            order = torch.argsort(vectors @ pool_vectors.T, dim=1, descending=True, stable=True)[:, :width]
            hard[batch] = pool[order.cpu().numpy()]
        for i in local:
            random[i] = rng.choice(pool, width, replace=False)
        pools[category] = {"image_count": len(pool), "query_count": len(local),
                           "image_ids_sha256": hashlib.sha256("\n".join(data.ids[i] for i in pool).encode()).hexdigest()}
    np.savez_compressed(output / "negative_banks.npz", fit_positions=fit, hard=hard, random=random)
    audit = {"source": "same-category fit reference/target images only", "pools": pools,
             "multi_positive_query_count": sum(len(row) > 1 for row in positives),
             "maximum_positive_count": max(map(len, positives)), "bank_width": width,
             "banks_sha256": sha256(output / "negative_banks.npz"),
             "fit_dev_holdout_supervised_overlap": data.split_audit["supervised_image_overlap"]}
    write_json(output / "negative_source_audit.json", audit)
    return positives, {"hard_16": hard, "random_16": random}, audit


def prior_diagnosis(data, output, device):
    previous = ROOT / "artifacts" / "fusion-runs" / "fashioniq-train-v1"
    old_config = json.loads((previous / "config.json").read_text(encoding="utf-8"))
    if old_config["split_seed"] != 20261028:
        raise ValueError("unexpected prior split")
    old_split = json.loads((previous / "split_manifest.json").read_text(encoding="utf-8"))
    if [(r["query_id"], r["partition"], r["component"]) for r in old_split] != [
        (q.query_id, str(data.partition[i]), str(data.groups[i])) for i, q in enumerate(data.queries)
    ]:
        raise ValueError("prior split manifest mismatch")
    parameters = json.loads((previous / "baseline_parameters.json").read_text(encoding="utf-8"))
    if parameters["selection_split"] != "fit only":
        raise ValueError("baseline was not fit-only")
    diagnosis = {"scope": "prior residual checkpoints evaluated on fit/dev only", "baselines": {}, "runs": []}
    npz = {}
    baseline_ranks = {}
    for split in ("fit", "dev"):
        positions = data.positions[split]
        vectors = baseline_vectors(data, positions, parameters)["A2_refit_dynamic"]
        ranks, _, _ = data.search(positions, vectors)
        baseline_ranks[split] = ranks
        diagnosis["baselines"][split] = metrics(ranks, data.categories[positions])
        npz[f"A2_{split}"] = ranks
    cases = []
    for seed in (17, 29, 43):
        path = previous / f"residual_fusion-seed{seed}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["encoder_version"] != data.encoder_version:
            raise ValueError("checkpoint encoder mismatch")
        model = LearnedFusion(**saved["model_config"]).to(device)
        model.load_state_dict(saved["state_dict"])
        row = {"seed": seed, "best_epoch": saved["epoch"], "checkpoint_sha256": sha256(path),
               "trace": json.loads((previous / f"residual_fusion-seed{seed}-training.json").read_text(encoding="utf-8"))}
        for split in ("fit", "dev"):
            observed, ranks, hits = evaluate(model, data, split, device, rankings=split == "dev" and seed == 17)
            row[split] = observed
            npz[f"residual{seed}_{split}"] = ranks
            base = baseline_ranks[split]
            row[f"{split}_transitions_at_10"] = {
                "both_hit": int(((base <= 10) & (ranks <= 10)).sum()),
                "rescued": int(((base > 10) & (ranks <= 10)).sum()),
                "regressed": int(((base <= 10) & (ranks > 10)).sum()),
                "both_miss": int(((base > 10) & (ranks > 10)).sum()),
                "outside_top50": int((ranks > 50).sum()),
            }
            if hits is not None:
                for category in sorted(set(data.categories)):
                    for label, condition in (
                        ("rescued", (base > 10) & (ranks <= 10)),
                        ("regressed", (base <= 10) & (ranks > 10)),
                        ("outside_top50", ranks > 50),
                    ):
                        choices = [j for j, i in enumerate(data.positions["dev"]) if condition[j] and data.categories[i] == category]
                        choices.sort(key=lambda j: data.queries[data.positions["dev"][j]].query_id)
                        for j in choices[:2]:
                            q = data.queries[data.positions["dev"][j]]
                            cases.append({"query_id": q.query_id, "category": category, "failure_group": label,
                                          "reference": q.reference_image_id, "target": q.target_image_ids[0],
                                          "text": list(q.modification_texts), "A2_rank": int(base[j]),
                                          "residual_rank": int(ranks[j]), "top5": hits[j]["ranked_image_ids"][:5]})
        diagnosis["runs"].append(row)
    diagnosis["case_selection"] = "seed17 fixed in advance; first two query IDs in each category/transition; dev only"
    diagnosis["cases"] = cases
    write_json(output / "prior_fit_dev_diagnosis.json", diagnosis)
    np.savez_compressed(output / "prior_fit_dev_ranks.npz", **npz)
    return diagnosis


def train_one(data, config, arm, seed, positives, banks, output, device, resume):
    prefix = output / f"{arm}-seed{seed}"
    result_path = Path(str(prefix) + "-result.json")
    if resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if sha256(Path(result["checkpoint"])) != result["checkpoint_sha256"]:
            raise ValueError("completed checkpoint changed")
        print(f"resume: completed {arm} seed={seed}", flush=True)
        return result
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = LearnedFusion(dimension=data.gallery.shape[1], hidden_dim=config["hidden_dim"],
                          image_weight=config["initial_image_weight"], residual=True,
                          dropout=config["dropout"], residual_scale=config["residual_scale"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    fit = data.positions["fit"]
    refs, texts, gallery = [torch.from_numpy(x).to(device) for x in (data.ref_vectors[fit], data.texts[fit], data.gallery)]
    best, best_epoch, trace, first_epoch, orders = None, 0, [], 0, []
    stats = {"samples": 0, "extra_count": 0, "extra_cosine_sum": 0., "valid_candidate_sum": 0}
    checkpoint, last_path = Path(str(prefix) + ".pt"), Path(str(prefix) + "-resume.pt")
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["arm"] != arm or saved["seed"] != seed:
            raise ValueError("resume protocol mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best, best_epoch, trace, stats, orders = [saved[key] for key in ("best", "best_epoch", "trace", "stats", "batch_orders")]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first_epoch = saved["epoch"] + 1
        print(f"resume: {arm} seed={seed} epoch={first_epoch}", flush=True)
    extra_count = 0 if arm == "in_batch" else config["extra_negatives"]
    for epoch in range(first_epoch, config["epochs"] + 1):
        loss_sum, count = 0., 0
        if epoch:
            model.train()
            batches = []
            for category in sorted(set(data.categories)):
                selected = rng.permutation(np.flatnonzero(data.categories[fit] == category))
                batches.extend(selected[i:i + config["batch_size"]] for i in range(0, len(selected), config["batch_size"]))
            rng.shuffle(batches)
            digest = hashlib.sha256()
            for local in batches:
                if len(local) < 2:
                    continue
                digest.update(np.asarray(local, dtype="<i8").tobytes())
                digest.update(b";")
                positions = fit[local]
                ids, positive, valid, extra = batch_candidate_masks(
                    data.ref_positions[positions], [positives[i] for i in local], data.target_positions[positions],
                    banks[arm][local] if extra_count else None, extra_count)
                idx = torch.as_tensor(local, device=device)
                output_vectors = model(refs[idx], texts[idx])
                prior = F.normalize(config["initial_image_weight"] * refs[idx] + (1 - config["initial_image_weight"]) * texts[idx], dim=-1)
                loss = masked_multi_positive_loss(output_vectors, gallery[torch.as_tensor(ids, device=device)],
                                                 torch.as_tensor(positive, device=device), torch.as_tensor(valid, device=device), config["temperature"])
                loss = loss + config["anchor_penalty"] * (1 - (output_vectors * prior).sum(-1)).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(local)
                count += len(local)
                stats["samples"] += len(local)
                stats["valid_candidate_sum"] += int(valid.sum())
                if extra_count:
                    stats["extra_count"] += extra.size
                    stats["extra_cosine_sum"] += float((gallery[torch.as_tensor(extra, device=device)] * prior.detach()[:, None, :]).sum(-1).sum())
            orders.append(digest.hexdigest())
        if epoch % config["evaluate_every"] == 0 or epoch == config["epochs"]:
            observed, _, _ = evaluate(model, data, "dev", device)
            fit_observed, _, _ = evaluate(model, data, "fit", device)
            trace.append({"epoch": epoch, "loss": loss_sum / count if count else None, "fit": fit_observed, "dev": observed})
            if best is None or objective(observed) > objective(best):
                best, best_epoch = observed, epoch
                atomic_torch_save({"model_config": model.config,
                                   "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                   "encoder_version": data.encoder_version, "seed": seed, "epoch": epoch,
                                   "variant": arm, "text_strategy": config["text_strategy"]}, checkpoint)
            print(f"arm={arm} seed={seed} epoch={epoch} fit_R10={objective(fit_observed)[0]:.5f} dev_R10={objective(observed)[0]:.5f} dev_R50={objective(observed)[1]:.5f}", flush=True)
            write_json(Path(str(prefix) + "-training.json"), trace)
        atomic_torch_save({"config": config, "arm": arm, "seed": seed, "epoch": epoch,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "best": best, "best_epoch": best_epoch, "trace": trace, "stats": stats,
                           "batch_orders": orders, "numpy_rng": rng.bit_generator.state,
                           "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}, last_path)
    result = {"arm": arm, "seed": seed, "best_epoch": best_epoch, "dev": best,
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
              "batch_order_sha256_by_epoch": orders, "negative_audit": stats}
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fashioniq_hard_negatives.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "fusion-runs" / "fashioniq-hard-negatives-dev-v1")
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts" / "metrics" / "fashioniq_hard_negatives_dev_v1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["dataset_split"] != "train" or config["split_seed"] != 20261028 or config["text_strategy"] != "symmetric_join_mean":
        raise ValueError("only frozen train fit/dev protocol is allowed")
    if config["arms"] != ["in_batch", "random_16", "hard_16"] or config["extra_negatives"] != 16:
        raise ValueError("unexpected experiment arms")
    if args.output_dir.exists() and not args.resume:
        raise ValueError("run exists; use --resume or a fresh directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise ValueError("run already complete; preserve its evidence")
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    data = ExperimentData(args.storage_root, config)
    source_files = [Path(__file__), ROOT / "src/cir_baseline/hard_negatives.py", ROOT / "src/cir_baseline/learned_fusion.py", ROOT / "scripts/train_fashioniq_fusion.py"]
    protocol = {"config": config, "cache": data.cache, "split_audit": data.split_audit,
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
                "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": args.device,
                                "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None}}
    protocol_file = args.output_dir / "protocol.json"
    if args.resume and protocol_file.exists():
        previous = json.loads(protocol_file.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "registered_at_utc"} != protocol:
            raise ValueError("data/code/environment/config changed; cannot resume")
    else:
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        for path in source_files:
            snapshot = args.output_dir / "sources" / path.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
        write_json(protocol_file, protocol)
    diagnosis_file = args.output_dir / "prior_fit_dev_diagnosis.json"
    diagnosis = json.loads(diagnosis_file.read_text(encoding="utf-8")) if args.resume and diagnosis_file.exists() else prior_diagnosis(data, args.output_dir, args.device)
    positives, banks, audit = prepare_banks(data, config, args.output_dir, args.device)
    runs = [train_one(data, config, arm, seed, positives, banks, args.output_dir, args.device, args.resume)
            for arm in config["arms"] for seed in config["seeds"]]
    comparisons = {}
    for seed in config["seeds"]:
        paired = [row for row in runs if row["seed"] == seed]
        if len({tuple(row["batch_order_sha256_by_epoch"]) for row in paired}) != 1:
            raise ValueError("batch orders differ between paired arms")
        sampled = [row["negative_audit"] for row in paired if row["arm"] != "in_batch"]
        if len({(s["samples"], s["extra_count"], s["valid_candidate_sum"]) for s in sampled}) != 1:
            raise ValueError("random/hard candidate counts differ")
    for arm in config["arms"]:
        values = np.asarray([objective(row["dev"]) for row in runs if row["arm"] == arm])
        comparisons[arm] = {"mean_R10": float(values[:, 0].mean()), "sd_R10": float(values[:, 0].std(ddof=1)),
                            "mean_R50": float(values[:, 1].mean()), "sd_R50": float(values[:, 1].std(ddof=1))}
    summary = {"status": "pass", "scope": "development evidence only; no new holdout/official val evaluation",
               "protocol_sha256": sha256(protocol_file), "config": config, "split_audit": data.split_audit,
               "negative_source_audit": audit, "runs": runs, "dev_three_seed_summary": comparisons,
               "A2_fit_only_baseline": diagnosis["baselines"],
               "decision": "retain_deployed_A2_pending_independent_confirmation", "deployed_model_changed": False,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds_this_invocation": round(time.perf_counter() - started, 3)}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.summary, summary)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
