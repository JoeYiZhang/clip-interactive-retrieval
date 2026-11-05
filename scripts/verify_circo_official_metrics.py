"""Compare our metric equations with unmodified official code using its fixture.

The official example predictions are external test fixtures, NOT our model output.
Only the dataset loader is replaced with an annotation reader; metric code is intact.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cir_baseline.circo_evaluation import evaluate_circo
from cir_baseline.embedding_cache import atomic_json, digest_file, digest_json, read_json


def verify(storage):
    repo = storage / "downloads/circo-official"
    annotation = repo / "annotations/val.json"
    source = repo / "src/evaluation.py"
    fixture = repo / "submission_examples/submission_val.json"
    rows, predictions = read_json(annotation), read_json(fixture)
    if [row["id"] for row in rows] != list(range(len(rows))):
        raise ValueError("official dataset uses annotation position as query ID")
    info_path = storage / "downloads/coco/image_info_unlabeled2017.json"
    gallery = [str(row["id"]) for row in read_json(info_path)["images"]]
    # The official example includes its reference for query 189. Adapt only that
    # input contract, then feed exactly the same adjusted fixture to both scorers.
    adjustments = []
    for row in rows:
        key, ref = str(row["id"]), row["reference_img_id"]
        if ref in predictions[key]:
            values = [value for value in predictions[key] if value != ref]
            replacement = next(int(value) for value in gallery if int(value) not in set(predictions[key]) | set(row["gt_img_ids"]))
            predictions[key] = values + [replacement]
            adjustments.append({"query_id": key, "removed_reference_id": ref, "appended_nonpositive_id": replacement})

    class AnnotationOnlyDataset:
        def __init__(self, data_path, split, mode, preprocess):
            if Path(data_path) != repo or split != "val" or mode != "relative":
                raise ValueError("unexpected official loader arguments")

        def get_target_img_ids(self, index):
            return {key: rows[index][key] for key in ("gt_img_ids", "target_img_id")}

        def get_semantic_aspects(self, index):
            return rows[index].get("semantic_aspects", [])

    module = types.ModuleType("dataset")
    module.CIRCODataset = AnnotationOnlyDataset
    spec = importlib.util.spec_from_file_location("official_circo_metrics", source)
    official = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"dataset": module}):
        spec.loader.exec_module(official)
    ks = (5, 10, 25, 50)
    maps, recalls, semantic = official.compute_metrics(repo, predictions, list(ks))
    queries = [{"query_id": str(row["id"]), "reference_id": str(row["reference_img_id"]), "target_id": str(row["target_img_id"]),
                "positive_ids": list(map(str, row["gt_img_ids"])), "semantic_aspects": row.get("semantic_aspects", [])} for row in rows]
    ours = evaluate_circo(queries, {key: list(map(str, values)) for key, values in predictions.items()}, gallery, ks)
    comparisons = [{"metric": f"mAP@{k}", "absolute_error": abs(ours[f"mAP@{k}"] - maps[k])} for k in ks]
    comparisons += [{"metric": f"target_recall@{k}", "absolute_error": abs(ours[f"target_recall@{k}"] - recalls[k])} for k in ks]
    if set(semantic) != set(ours["semantic_mAP@10"]):
        raise ValueError("semantic metric coverage mismatch")
    comparisons += [{"metric": f"semantic_mAP@10/{key}", "absolute_error": abs(value - ours["semantic_mAP@10"][key])} for key, value in semantic.items()]
    if not all(row["absolute_error"] <= 1e-12 for row in comparisons):
        raise ValueError("official metric parity failed")
    return {"status": "pass", "scope": "official external prediction fixture; NOT project model performance", "query_count": len(rows),
            "gallery_inventory_count": len(gallery), "checks": comparisons,
            "official_repository_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "official_metric_source_sha256": digest_file(source), "official_annotations_sha256": digest_file(annotation),
            "official_predictions_sha256": digest_file(fixture), "official_inventory_sha256": digest_file(info_path),
            "fixture_adjustments": adjustments, "compared_prediction_content_sha256": digest_json(predictions),
            "project_metric_source_sha256": digest_file(ROOT / "src/cir_baseline/circo_evaluation.py"),
            "official_source": "https://github.com/miccunifi/CIRCO/blob/main/src/evaluation.py"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/metrics/circo_official_metric_parity.json")
    args = parser.parse_args()
    result = verify(args.storage_root)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
