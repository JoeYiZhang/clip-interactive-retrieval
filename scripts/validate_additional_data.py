"""Final coverage gate after full official COCO archives have been verified."""
import json
from pathlib import Path
from test_additional_datasets import ROOT, audit

def main():
    root = Path(r"D:\image_data")
    result = {"scope": "full image coverage and annotation audit; no model quality claim", "datasets": {}}
    for kind in ("circo", "lasco"):
        archives = ("unlabeled2017",) if kind == "circo" else ("train2014", "val2014")
        certificates = [root / "downloads/coco" / f"{name}.zip.verified.json" for name in archives]
        certified = all(path.is_file() and json.loads(path.read_text())["status"] == "complete" for path in certificates)
        splits = ("val", "test") if kind == "circo" else ("val", "train")
        reports = {}
        for split in splits:
            path = root / (f"downloads/circo-official/annotations/{split}.json" if kind == "circo" else f"downloads/lasco_{split}.json")
            if not path.exists():
                reports[split] = {"status": "annotations_pending"}
                continue
            rows = json.loads(path.read_text("utf-8"))
            if kind == "circo" and split == "test":
                ids = [str(row["id"]) for row in rows]
                inspection = {"queries": len(rows), "query_ids_unique": len(ids) == len(set(ids)), "invalid": []}
            else:
                inspection = audit(rows, kind)
            needed = set()
            for row in rows:
                if kind == "circo":
                    for image_id in [row["reference_img_id"], *row.get("gt_img_ids", [])]:
                        needed.add(root / "datasets/coco/unlabeled2017" / f"{image_id:012}.jpg")
                else:
                    for key in ("query-image", "target-image"):
                        needed.add(root / "datasets/coco" / row[key][1])
            missing = [str(path) for path in sorted(needed) if not path.is_file()]
            reports[split] = {"audit": inspection, "required_images": len(needed), "missing_images": len(missing), "missing_examples": missing[:10]}
        ready = certified and all("audit" in row and row["missing_images"] == 0 for row in reports.values())
        result["datasets"][kind] = {"ready_for_full_evaluation": ready, "archives_crc_verified": certified, "splits": reports}
    result["status"] = "pass" if all(row["ready_for_full_evaluation"] for row in result["datasets"].values()) else "pending_download"
    (ROOT / "artifacts/metrics/additional_data_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2

if __name__ == "__main__":
    raise SystemExit(main())
