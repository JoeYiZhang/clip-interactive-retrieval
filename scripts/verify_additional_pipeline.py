"""Recheck saved LaSCo/CIRCO engineering evidence without downloading or tuning."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from cir_baseline.circo_evaluation import evaluate_circo
from cir_baseline.dataset_plans import make_dataset_plan
from cir_baseline.embedding_cache import atomic_json, digest_file, load_cache, read_json


def verify(args):
    records = []

    def check(name, condition):
        if not condition:
            raise ValueError(name)
        records.append({"name": name, "status": "pass"})

    lasco, _, lasco_identity = load_cache(args.lasco_cache)
    circo, _, circo_identity = load_cache(args.circo_cache)
    check("explicit engineering scopes", lasco["mode"] == circo["mode"] == "engineering")
    check("fixed sample inventory", (len(lasco["images"]), len(lasco["queries"]), len(circo["images"]), len(circo["queries"])) == (14, 8, 42, 8))
    shard = args.lasco_cache / "images-000000000.npz"
    check("committed shard retained after continuation", digest_file(shard) == args.committed_shard_sha256.lower())
    a = torch.load(args.resumed / "seed17/resume.pt", map_location="cpu", weights_only=True)
    b = torch.load(args.uninterrupted / "seed17/resume.pt", map_location="cpu", weights_only=True)
    check("exact resumed training history", a["history"] == b["history"] and len(a["history"]) == 3)
    check("exact resumed model tensors", set(a["model"]) == set(b["model"]) and all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"]))
    check("identical saved weight files", digest_file(args.resumed / "seed17/final.pt") == digest_file(args.uninterrupted / "seed17/final.pt"))
    trained = read_json(args.resumed / "seed17/result.json")
    check("training updated parameters without dev selection", trained["parameters_updated"] and trained["best_dev"] is None and trained["mode"] == "engineering")
    summary = read_json(args.circo_run / "summary.json")
    check("CIRCO small-gallery scope", summary["status"] == "pass" and summary["mode"] == "engineering" and summary["gallery_count"] == 42)
    artifacts = {}
    for model, result in summary["results"].items():
        path = args.circo_run / f"{model}-predictions.json"
        predictions = {key: list(map(str, values)) for key, values in read_json(path).items()}
        scores = read_json(args.circo_run / f"{model}-scores.json")
        check(f"{model}: all 32 query/model rankings cover 41 non-reference images", len(predictions) == 8 and all(len(values) == 41 for values in predictions.values()))
        check(f"{model}: descending score/rank alignment", set(scores) == set(predictions) and all(len(scores[key]) == len(values) and
              all(first >= second for first, second in zip(scores[key], scores[key][1:])) for key, values in predictions.items()))
        recomputed = evaluate_circo(circo["queries"], predictions, [row["id"] for row in circo["images"]], (5, 10))
        check(f"{model}: recomputed exported ranking metrics", recomputed == result)
        artifacts[model] = {"predictions_sha256": digest_file(path), "query_count": len(predictions), "ranks_per_query": 41}
    parity = read_json(ROOT / "artifacts/metrics/circo_official_metric_parity.json")
    check("official metric parity certificate", parity["status"] == "pass" and len(parity["checks"]) == 17 and
          parity["project_metric_source_sha256"] == digest_file(ROOT / "src/cir_baseline/circo_evaluation.py") and
          all(row["absolute_error"] <= 1e-12 for row in parity["checks"]))
    suite = ET.parse(ROOT / "artifacts/metrics/additional_pipeline_tests.xml").getroot().find("testsuite")
    check("regression suite", int(suite.attrib["tests"]) >= 67 and all(suite.attrib[key] == "0" for key in ("errors", "failures", "skipped")))
    gates = {}
    for dataset, split in (("lasco", "train"), ("lasco", "val"), ("circo", "val")):
        try:
            full = make_dataset_plan(args.storage_root, dataset, split, "full", lasco["encoder_version"])
            gates[f"{dataset}/{split}"] = {"status": "inventory_ready", "images": len(full["images"])}
        except FileNotFoundError as error:
            gates[f"{dataset}/{split}"] = {"status": "blocked_waiting_verified_archive", "reason": str(error)}
    progress_path = args.storage_root / "downloads/coco/unlabeled2017.zip.progress.json"
    progress = read_json(progress_path)
    completed = sum(min(progress["chunk_size"], progress["size"] - i * progress["chunk_size"]) for i in set(progress["done"]))
    return {"status": "pass", "checked_at_local": datetime.now().astimezone().isoformat(),
            "scope": "engineering verification only; full training and full-gallery model evaluation NOT executed",
            "checks": records, "check_count": len(records), "unit_tests": int(suite.attrib["tests"]),
            "lasco": {"images": 14, "queries": 8, "cache_sha256": lasco_identity, "seed": 17, "epochs": 3,
                      "first_shard_sha256": digest_file(shard), "history": a["history"], "checkpoint_sha256": trained["checkpoint_sha256"]},
            "circo": {"images": 42, "queries": 8, "cache_sha256": circo_identity, "models": artifacts,
                      "official_metric_checks": 17, "official_metric_max_error": max(row["absolute_error"] for row in parity["checks"]),
                      "official_inventory_count": parity["gallery_inventory_count"]},
            "full_data_gates": gates, "download_snapshot": {"archive": "unlabeled2017", "verified_completed_chunk_bytes": completed,
                         "total_bytes": progress["size"], "percent": 100 * completed / progress["size"]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--lasco-cache", type=Path, default=ROOT / "artifacts/cache/lasco-train-engineering-v1")
    parser.add_argument("--circo-cache", type=Path, default=ROOT / "artifacts/cache/circo-val-engineering-v1")
    parser.add_argument("--resumed", type=Path, default=ROOT / "artifacts/fusion-runs/lasco-engineering-resume-v1")
    parser.add_argument("--uninterrupted", type=Path, default=ROOT / "artifacts/fusion-runs/lasco-engineering-uninterrupted-v1")
    parser.add_argument("--circo-run", type=Path, default=ROOT / "artifacts/fusion-runs/circo-engineering-v1")
    parser.add_argument("--committed-shard-sha256", required=True, help="Hash recorded before continuing the interrupted cache")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/metrics/additional_pipeline_verification.json")
    args = parser.parse_args()
    result = verify(args)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
