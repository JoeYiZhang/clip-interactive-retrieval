"""Explicit complete-gallery versus fixed engineering-subset data contracts."""
from pathlib import Path
import zipfile

from .embedding_cache import digest_file, read_json, validate_plan
from .lasco_training import prepare_examples, query_key


def certified_gallery(storage, name):
    storage = Path(storage)
    archive = storage / "downloads/coco" / f"{name}.zip"
    certificate = archive.with_suffix(".zip.verified.json")
    if not certificate.is_file():
        raise FileNotFoundError(f"full gallery blocked: verified {name} archive not available")
    record = read_json(certificate)
    if record.get("status") != "complete" or record.get("zip_crc") != "pass" or not archive.is_file() or archive.stat().st_size != record["size"]:
        raise ValueError("archive certificate/size mismatch")
    with zipfile.ZipFile(archive) as source:
        members = [item.filename for item in source.infolist() if not item.is_dir()]
    images = []
    for member in members:
        path = Path(member)
        if len(path.parts) != 2 or path.parts[0] != name or path.suffix.lower() != ".jpg":
            raise ValueError("unexpected official archive member")
        raw_id = path.stem if name == "unlabeled2017" else path.stem.removeprefix(f"COCO_{name}_")
        image_id = str(int(raw_id))
        expected = f"{int(image_id):012}.jpg" if name == "unlabeled2017" else f"COCO_{name}_{int(image_id):012}.jpg"
        if path.name != expected:
            raise ValueError("image ID/name mismatch")
        images.append({"id": image_id, "path": path.as_posix()})
    if len(images) != record["image_count"] or len({row["id"] for row in images}) != len(images):
        raise ValueError("archive image inventory mismatch")
    return sorted(images, key=lambda row: int(row["id"])), {"certificate_sha256": digest_file(certificate), "archive_sha256_from_crc_certificate": record["sha256"]}


def make_dataset_plan(storage, dataset, split, mode, encoder_version, sample_count=8):
    storage = Path(storage)
    if mode not in ("full", "engineering") or sample_count < 1:
        raise ValueError("invalid explicit dataset scope")
    sources = {}
    if dataset == "lasco":
        if split not in ("train", "val"):
            raise ValueError("LaSCo split must be train or val")
        path = storage / "downloads" / f"lasco_{split}.json"
        examples, positives, quarantine = prepare_examples(read_json(path), split)
        selected = examples[:sample_count] if mode == "engineering" else examples
        queries = [{"query_id": row["example_id"], "source_qid": row["source_qid"], "source_row_index": row["source_row_index"],
                    "reference_id": str(row["reference_id"]), "target_id": str(row["target_id"]), "text": row["text"],
                    "positive_ids": sorted(map(str, positives[query_key(row["reference_id"], row["text"])]), key=int)} for row in selected]
        if mode == "full":
            images, certificate = certified_gallery(storage, "train2014" if split == "train" else "val2014")
            sources.update(certificate)
        else:
            paths = {str(row[key]): row[relative] for row in selected for key, relative in (("reference_id", "reference_path"), ("target_id", "target_path"))}
            images = [{"id": key, "path": paths[key]} for key in sorted(paths, key=int)]
        sources.update(annotation_sha256=digest_file(path), valid_annotation_count=len(examples), quarantined_rows=len(quarantine))
        text_policy = "query-text-verbatim-stripped-v1"
    elif dataset == "circo":
        if split not in ("val", "test"):
            raise ValueError("CIRCO split must be val or test")
        path = storage / "downloads/circo-official/annotations" / f"{split}.json"
        rows = read_json(path)
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError("duplicate CIRCO query IDs")
        selected = rows[:sample_count] if mode == "engineering" else rows
        queries = []
        for row in selected:
            ref = str(row["reference_img_id"])
            positive_ids = list(map(str, row.get("gt_img_ids", [])))
            if split == "val" and (not positive_ids or len(set(positive_ids)) != len(positive_ids) or ref in positive_ids or str(row["target_img_id"]) not in positive_ids):
                raise ValueError("invalid CIRCO positive labels")
            if not row["shared_concept"].strip() or not row["relative_caption"].strip():
                raise ValueError("empty CIRCO text")
            queries.append({"query_id": str(row["id"]), "reference_id": ref,
                            "target_id": str(row["target_img_id"]) if split == "val" else None,
                            "positive_ids": positive_ids, "text": f"{row['shared_concept']} {row['relative_caption']}",
                            "semantic_aspects": row.get("semantic_aspects", [])})
        if mode == "full":
            images, certificate = certified_gallery(storage, "unlabeled2017")
            info_path = storage / "downloads/coco/image_info_unlabeled2017.json"
            official = read_json(info_path)["images"]
            expected = {str(row["id"]): "unlabeled2017/" + row["file_name"] for row in official}
            if len(expected) != len(official) or expected != {row["id"]: row["path"] for row in images}:
                raise ValueError("CIRCO gallery differs from official COCO inventory")
            sources.update(certificate, official_inventory_sha256=digest_file(info_path))
        else:
            needed = {key for row in queries for key in [row["reference_id"], *row["positive_ids"]]}
            images = [{"id": key, "path": f"unlabeled2017/{int(key):012}.jpg"} for key in sorted(needed, key=int)]
        sources.update(annotation_sha256=digest_file(path), annotation_count=len(rows))
        text_policy = "shared-concept-space-relative-caption-v1"
    else:
        raise ValueError("unsupported dataset")
    plan = {"schema": 1, "dataset": dataset, "split": split, "mode": mode, "dimension": 512,
            "image_root": str((storage / "datasets/coco").resolve()), "encoder_version": encoder_version,
            "text_policy": text_policy, "sources": sources, "images": images, "queries": queries}
    validate_plan(plan)
    return plan
