"""Audit all LaSCo supervision before training; never load FashionIQ holdout."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from test_additional_datasets import ROOT, audit

def describe(rows):
    return {
        "image_ids": {int(row[key][0]) for row in rows for key in ("query-image", "target-image")},
        "pairs": {(int(row["query-image"][0]), int(row["target-image"][0])) for row in rows},
    }

def main():
    storage = Path(r"D:\image_data\downloads")
    reports, sets = {}, {}
    for split in ("train", "val"):
        path = storage / f"lasco_{split}.json"
        rows = json.loads(path.read_text("utf-8"))
        inspection = audit(rows, "lasco", allow_repeated_qids=True)
        excluded = {row["row_index"] for row in inspection["invalid"]}
        valid = [row for i, row in enumerate(rows) if i not in excluded]
        sets[split] = describe(valid)
        reports[split] = {
            "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "audit": inspection, "unique_usable_images": len(sets[split]["image_ids"]),
            "unique_usable_ordered_pairs": len(sets[split]["pairs"]),
            "extra_queries_sharing_an_ordered_pair": len(valid)-len(sets[split]["pairs"]),
            "image_folders": dict(Counter(Path(row[key][1]).parts[0] for row in valid for key in ("query-image", "target-image"))),
        }
    overlap = sets["train"]["image_ids"] & sets["val"]["image_ids"]
    circo_ids = set()
    for split in ("val", "test"):
        path = storage / "circo-official/annotations" / f"{split}.json"
        for row in json.loads(path.read_text("utf-8")):
            circo_ids.update([row["reference_img_id"], *row.get("gt_img_ids", [])])
    output = {
        "scope": "annotation audit only; images and full CIRCO gallery still require verification",
        "splits": reports,
        "train_val_image_overlap": len(overlap), "overlap_examples": sorted(overlap)[:20],
        "train_val_ordered_pair_overlap": len(sets["train"]["pairs"] & sets["val"]["pairs"]),
        "circo_public_query_image_id_overlap": {split: len(value["image_ids"] & circo_ids) for split, value in sets.items()},
        "circo_overlap_scope": "public references/validation positives only, not the full gallery or decoded-image duplicate audit",
        "training_notes": ["Quarantine invalid rows with query IDs; preserve originals.",
                           "Source qid is not a unique example key; use split plus row index and retain original qid.",
                           "Shared image pairs may have multiple valid texts; do not count them as independent image pairs.",
                           "Deduplicate candidate image IDs and mask known valid positives in hard-negative mining.",
                           "Use official train for gradient updates; val for preregistered selection only; reserve CIRCO for frozen external evaluation."],
        "status": "annotation_gate_pass_with_quarantine" if not overlap else "split_overlap_requires_resolution",
    }
    path = ROOT / "artifacts/metrics/lasco_training_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "splits": {k: {"queries": v["audit"]["queries"], "usable": v["audit"]["usable_queries"], "invalid": len(v["audit"]["invalid"]), "images": v["unique_usable_images"]} for k,v in reports.items()}, "train_val_image_overlap":len(overlap), "circo_public_query_image_id_overlap":output["circo_public_query_image_id_overlap"]}, indent=2))

if __name__ == "__main__":
    main()
