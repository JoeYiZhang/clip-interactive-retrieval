"""Cached-embedding LaSCo trainer; full validation and epoch-exact continuation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .embedding_cache import atomic_json, digest_file, load_cache, read_json
from .lasco_training import candidate_masks, query_key
from .learned_fusion import LearnedFusion, masked_multi_positive_loss


def save_torch(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cache_inputs(cache):
    plan, matrices, identity = load_cache(cache)
    if plan["dataset"] != "lasco":
        raise ValueError("LaSCo cache required")
    ids = [row["id"] for row in plan["images"]]
    lookup = {key: i for i, key in enumerate(ids)}
    queries = plan["queries"]
    references = np.asarray([lookup[row["reference_id"]] for row in queries])
    targets = np.asarray([lookup[row["target_id"]] for row in queries])
    return {"plan": plan, "matrices": matrices, "identity": identity, "ids": ids,
            "references": references, "targets": targets}


def validate_training_inputs(train, dev, config):
    if config["mode"] not in ("full", "engineering") or train["plan"]["split"] != "train" or train["plan"]["mode"] != config["mode"]:
        raise ValueError("training cache scope/split mismatch")
    if config["epochs"] < 1 or config["batch_size"] < 2 or len(train["plan"]["queries"]) < 2:
        raise ValueError("insufficient training budget/data")
    if config["mode"] == "full" and dev is None:
        raise ValueError("full training requires a complete independent val cache")
    if config["mode"] == "engineering" and dev is not None:
        raise ValueError("engineering run must not select a model using val")
    if dev is not None:
        if dev["plan"]["split"] != "val" or dev["plan"]["mode"] != "full":
            raise ValueError("complete val cache required")
        if set(train["ids"]) & set(dev["ids"]):
            raise ValueError("train/val supervised image overlap")
        if any(train["plan"][key] != dev["plan"][key] for key in ("encoder_version", "text_policy", "dimension")):
            raise ValueError("train/val cache encoder contract mismatch")


def evaluate_dev(model, data, device):
    from .contracts import IndexManifest
    from .index import FaissFlatIPIndex
    if "index" not in data:
        data["index"] = FaissFlatIPIndex(data["ids"], data["matrices"]["images"],
            IndexManifest("lasco-val-" + data["identity"][:12], data["plan"]["encoder_version"], data["plan"]["dimension"]))
    model.eval()
    ranks = []
    with torch.inference_mode():
        for start in range(0, len(data["plan"]["queries"]), 512):
            queries = data["plan"]["queries"][start:start + 512]
            refs = data["matrices"]["images"][data["references"][start:start + 512]]
            texts = data["matrices"]["texts"][start:start + 512]
            vectors = model(torch.from_numpy(refs).to(device), torch.from_numpy(texts).to(device)).cpu().numpy()
            for row, hits in zip(queries, data["index"].search_many(vectors, 51), strict=True):
                ids = [hit.image_id for hit in hits if hit.image_id != row["reference_id"]][:50]
                ranks.append(ids.index(row["target_id"]) + 1 if row["target_id"] in ids else 51)
    return {"query_count": len(ranks), "target_R10": float(np.mean(np.asarray(ranks) <= 10)),
            "target_R50": float(np.mean(np.asarray(ranks) <= 50)), "gallery_count": len(data["ids"])}


def train_cached(train, dev, config, seed, output, device="cuda", resume=False, stop_after_epoch=None):
    from filelock import FileLock
    validate_training_inputs(train, dev, config)
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError("training output exists; choose a new directory or resume")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".writer.lock"), timeout=0):
        source_files = [Path(__file__), Path(__file__).with_name("learned_fusion.py"),
                        Path(__file__).with_name("lasco_training.py"), Path(__file__).with_name("embedding_cache.py")]
        contract = {"config": config, "seed": seed, "train_cache_sha256": train["identity"],
                    "dev_cache_sha256": dev["identity"] if dev else None, "device": device,
                    "torch": str(torch.__version__), "source_sha256": {path.name: digest_file(path) for path in source_files}}
        if (output / "contract.json").exists():
            if read_json(output / "contract.json") != contract:
                raise ValueError("training resume contract changed")
        elif resume and (output / "resume.pt").exists():
            raise ValueError("resume missing provenance contract")
        atomic_json(output / "contract.json", contract)
        if (output / "result.json").exists():
            result = read_json(output / "result.json")
            if digest_file(output / result["checkpoint_file"]) != result["checkpoint_sha256"]:
                raise ValueError("completed checkpoint changed")
            return result
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = LearnedFusion(dimension=train["plan"]["dimension"], hidden_dim=config["hidden_dim"],
                              image_weight=config["initial_image_weight"], dropout=config["dropout"],
                              residual=True, residual_scale=config["residual_scale"]).to(device)
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        history, first_epoch, best, best_epoch = [], 1, None, None
        if resume and (output / "resume.pt").exists():
            saved = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
            if saved["contract"] != contract:
                raise ValueError("checkpoint resume contract mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            history, best, best_epoch = saved["history"], saved["best"], saved["best_epoch"]
            first_epoch = saved["epoch"] + 1
            rng.bit_generator.state = saved["numpy_rng"]
            torch.set_rng_state(saved["torch_rng"])
            if device.startswith("cuda"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        gallery = torch.from_numpy(train["matrices"]["images"]).to(device)
        image_ids = np.asarray(train["ids"])
        queries = train["plan"]["queries"]
        positives = {query_key(row["reference_id"], row["text"]): set(row["positive_ids"]) for row in queries}
        def weight_payload(epoch):
            return {"model_config": model.config, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "encoder_version": train["plan"]["encoder_version"], "text_strategy": "single_text",
                    "seed": seed, "epoch": epoch, "variant": "lasco_cached_multi_positive", "mode": config["mode"]}
        for epoch in range(first_epoch, config["epochs"] + 1):
            model.train()
            order = rng.permutation(len(queries))
            total_loss, sample_count = 0., 0
            for start in range(0, len(order), config["batch_size"]):
                positions = order[start:start + config["batch_size"]]
                if len(positions) < 2:
                    continue
                rows = [queries[i] for i in positions]
                candidates = np.unique(train["targets"][positions])
                positive, valid = candidate_masks(rows, image_ids[candidates], positives)
                refs = gallery[torch.as_tensor(train["references"][positions], device=device)]
                texts = torch.from_numpy(train["matrices"]["texts"][positions]).to(device)
                output_vectors = model(refs, texts)
                loss = masked_multi_positive_loss(output_vectors, gallery[torch.as_tensor(candidates, device=device)],
                    torch.as_tensor(positive, device=device), torch.as_tensor(valid, device=device), config["temperature"])
                prior = F.normalize(config["initial_image_weight"] * refs + (1 - config["initial_image_weight"]) * texts, dim=-1)
                loss = loss + config["anchor_penalty"] * (1 - (output_vectors * prior).sum(-1)).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite LaSCo loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
                sample_count += len(rows)
            observed = evaluate_dev(model, dev, device) if dev is not None else None
            if observed is not None and (best is None or (observed["target_R10"], observed["target_R50"]) > (best["target_R10"], best["target_R50"])):
                best, best_epoch = observed, epoch
                save_torch(output / "best.pt", weight_payload(epoch))
            history.append({"epoch": epoch, "loss": total_loss / sample_count, "training_samples": sample_count, "dev": observed})
            save_torch(output / "resume.pt", {"contract": contract, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "history": history, "best": best, "best_epoch": best_epoch, "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []})
            atomic_json(output / "history.json", history)
            print(f"LaSCo seed={seed} epoch={epoch} loss={history[-1]['loss']:.6f} samples={sample_count}", flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < config["epochs"]:
                return {"status": "interrupted_at_epoch_boundary", "epoch": epoch, "mode": config["mode"]}
        save_torch(output / "final.pt", weight_payload(config["epochs"]))
        changed = any(not torch.equal(value.detach().cpu(), initial[key]) for key, value in model.state_dict().items())
        if not changed:
            raise ValueError("training did not update fusion parameters")
        checkpoint = "best.pt" if dev is not None else "final.pt"
        result = {"status": "pass", "mode": config["mode"], "seed": seed, "epochs": config["epochs"],
                  "train_queries": len(queries), "train_gallery": len(train["ids"]), "parameters_updated": changed,
                  "best_dev": best, "best_epoch": best_epoch, "checkpoint_file": checkpoint,
                  "checkpoint_sha256": digest_file(output / checkpoint), "history": history,
                  "scope": "engineering optimizer verification only" if dev is None else "LaSCo project validation protocol; no external confirmation or deployment"}
        atomic_json(output / "result.json", result)
        return result
