import tempfile
import unittest
import zipfile
from pathlib import Path

from cir_baseline.dataset_plans import certified_gallery, make_dataset_plan
from cir_baseline.embedding_cache import atomic_json, digest_file


class GalleryGateTests(unittest.TestCase):
    def test_partial_archive_never_satisfies_full_gallery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "downloads/coco/unlabeled2017.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"unfinished archive")
            with self.assertRaisesRegex(FileNotFoundError, "verified"):
                certified_gallery(root, "unlabeled2017")
            atomic_json(archive.with_suffix(".zip.verified.json"), {"status": "building", "zip_crc": "pending"})
            with self.assertRaises(ValueError):
                certified_gallery(root, "unlabeled2017")

    def test_official_circo_inventory_must_match_certified_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "downloads/coco/unlabeled2017.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as handle:
                for image_id in (1, 2, 3):
                    handle.writestr(f"unlabeled2017/{image_id:012}.jpg", b"synthetic")
            atomic_json(archive.with_suffix(".zip.verified.json"), {"status": "complete", "zip_crc": "pass", "size": archive.stat().st_size,
                        "image_count": 3, "sha256": digest_file(archive)})
            atomic_json(root / "downloads/circo-official/annotations/val.json", [{"id": 0, "reference_img_id": 1, "target_img_id": 2,
                        "gt_img_ids": [2], "shared_concept": "cat", "relative_caption": "sleeping"}])
            info = root / "downloads/coco/image_info_unlabeled2017.json"
            rows = [{"id": i, "file_name": f"{i:012}.jpg"} for i in (1, 2, 3)]
            atomic_json(info, {"images": rows})
            plan = make_dataset_plan(root, "circo", "val", "full", "tiny")
            self.assertEqual(len(plan["images"]), 3)  # Includes the unlabelled distractor.
            atomic_json(info, {"images": rows[:2]})
            with self.assertRaisesRegex(ValueError, "official COCO inventory"):
                make_dataset_plan(root, "circo", "val", "full", "tiny")
