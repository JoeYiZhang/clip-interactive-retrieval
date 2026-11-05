"""Frozen CIRCO evaluation with official gallery gates and separate smoke scope."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from cir_baseline.circo_evaluation import evaluate_circo
from cir_baseline.contracts import IndexManifest
from cir_baseline.dataset_plans import make_dataset_plan
from cir_baseline.embedding_cache import atomic_json, digest_file, load_cache, read_json
from cir_baseline.fusion import l2_normalize_rows
from cir_baseline.index import FaissFlatIPIndex
from cir_baseline.learned_fusion import LearnedFusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/circo_evaluation.json")
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--mode", choices=("full", "engineering"), default="full")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("evaluation output exists; preserve evidence and choose a fresh directory")
    torch.set_num_threads(4)
    import faiss
    faiss.omp_set_num_threads(4)
    plan, matrices, cache_identity = load_cache(args.cache_dir)
    config = read_json(args.config)
    if plan["dataset"] != "circo" or plan["mode"] != args.mode or plan["split"] not in ("val", "test") or plan["text_policy"] != config["text_policy"]:
        raise ValueError("CIRCO cache scope/text mismatch")
    current = make_dataset_plan(args.storage_root, "circo", plan["split"], args.mode, plan["encoder_version"], len(plan["queries"]))
    if any(plan[key] != current[key] for key in ("sources", "images", "queries", "text_policy")):
        raise ValueError("current source annotations/gallery differ from cache")
    if args.mode == "full":
        if len(plan["queries"]) != config["full_query_counts"][plan["split"]] or plan["sources"]["official_inventory_sha256"] != config["official_inventory_sha256"]:
            raise ValueError("official full-query/inventory contract mismatch")
    manifest = read_json(ROOT / config["candidate_manifest"])
    checkpoint = Path(manifest["checkpoint"])
    if digest_file(checkpoint) != config["candidate_checkpoint_sha256"] or manifest["checkpoint_sha256"] != config["candidate_checkpoint_sha256"]:
        raise ValueError("frozen candidate checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["encoder_version"] != plan["encoder_version"] or saved["text_strategy"] != "symmetric_join_mean":
        raise ValueError("candidate/cache encoder mismatch")
    args.output_dir.mkdir(parents=True)
    protocol = {"config": config, "mode": args.mode, "split": plan["split"], "cache_sha256": cache_identity,
                "source_annotation_sha256": plan["sources"]["annotation_sha256"], "candidate_sha256": digest_file(checkpoint),
                "gallery_count": len(plan["images"]), "query_count": len(plan["queries"]),
                "official_metric_code": "https://github.com/miccunifi/CIRCO/blob/main/src/evaluation.py",
                "implementation_sha256": {name: digest_file(ROOT / name) for name in ("scripts/evaluate_circo_cached.py", "src/cir_baseline/circo_evaluation.py", "src/cir_baseline/dataset_plans.py")}}
    atomic_json(args.output_dir / "protocol.json", protocol)
    ids = [row["id"] for row in plan["images"]]
    lookup = {key: i for i, key in enumerate(ids)}
    refs = matrices["images"][[lookup[row["reference_id"]] for row in plan["queries"]]]
    texts = matrices["texts"]
    model = LearnedFusion(**saved["model_config"]).to(args.device).eval()
    model.load_state_dict(saved["state_dict"])
    with torch.inference_mode():
        learned = model(torch.from_numpy(refs).to(args.device), torch.from_numpy(texts).to(args.device)).cpu().numpy()
    weight = config["image_weight"]
    vectors = {"image_only": refs, "text_only": texts, "fixed_0.3": l2_normalize_rows(weight * refs + (1-weight) * texts),
               "fashioniq_low_lr_seed17": learned}
    if set(vectors) != set(config["models"]):
        raise ValueError("frozen evaluation model list changed")
    index = FaissFlatIPIndex(ids, matrices["images"], IndexManifest("circo-" + cache_identity[:12], plan["encoder_version"], plan["dimension"]))
    results = {}
    ks = tuple(config["ks"]) if args.mode == "full" else (5, 10)
    for name in config["models"]:
        predictions, scores = {}, {}
        for row, hits in zip(plan["queries"], index.search_many(vectors[name], min(51, len(ids))), strict=True):
            hits = [hit for hit in hits if hit.image_id != row["reference_id"]][:50]
            predictions[row["query_id"]] = [hit.image_id for hit in hits]
            scores[row["query_id"]] = [hit.score for hit in hits]
        if args.mode == "full" and any(len(row) != 50 or len(set(row)) != 50 for row in predictions.values()):
            raise ValueError("full CIRCO requires fifty unique results per query")
        atomic_json(args.output_dir / f"{name}-predictions.json", {key: list(map(int, values)) for key, values in predictions.items()})
        atomic_json(args.output_dir / f"{name}-scores.json", scores)
        results[name] = evaluate_circo(plan["queries"], predictions, ids, ks) if plan["split"] == "val" else {"status": "local_test_submission_only", "query_count": len(predictions)}
    summary = {"status": "pass", "mode": args.mode, "split": plan["split"], "gallery_count": len(ids), "query_count": len(plan["queries"]),
               "scope": "engineering partial-gallery check only; NOT benchmark performance" if args.mode == "engineering" else "frozen full-gallery evaluation or local test submission; no tuning or deployment",
               "results": results, "protocol_sha256": digest_file(args.output_dir / "protocol.json")}
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
