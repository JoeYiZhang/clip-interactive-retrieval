from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cir_baseline.datasets import load_cirr, load_fashioniq, validate_bundle


class DatasetAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=artifact_root)
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_cirr_adapter_and_validation(self) -> None:
        root = self.root / "cirr"
        self._write_json(
            root / "image_splits" / "split.rc2.val.json",
            {"ref": "./dev/ref.png", "target": "./dev/target.png"},
        )
        self._write_json(
            root / "captions" / "cap.rc2.val.json",
            [
                {
                    "pairid": 7,
                    "reference": "ref",
                    "target_hard": "target",
                    "caption": "make it blue",
                    "img_set": {"members": ["ref", "target"]},
                }
            ],
        )
        for name in ("ref", "target"):
            path = root / "img_raw" / "dev" / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        bundle = load_cirr(root)
        report = validate_bundle(bundle)
        self.assertEqual(bundle.queries[0].query_id, "7")
        self.assertEqual(bundle.queries[0].target_image_ids, ("target",))
        self.assertEqual(bundle.gallery[0].path, root / "img_raw" / "dev" / "ref.png")
        self.assertTrue(report.ready_for_experiment)

    def test_cirr_missing_images_are_separate_from_structure(self) -> None:
        root = self.root / "cirr"
        self._write_json(
            root / "image_splits" / "split.rc2.val.json",
            {"ref": "./dev/ref.png", "target": "./dev/target.png"},
        )
        self._write_json(
            root / "captions" / "cap.rc2.val.json",
            [
                {
                    "pairid": 1,
                    "reference": "ref",
                    "target_hard": "target",
                    "caption": "change it",
                    "img_set": {"members": ["ref", "target"]},
                }
            ],
        )
        report = validate_bundle(load_cirr(root))
        self.assertTrue(report.structural_ok)
        self.assertFalse(report.images_ok)
        self.assertFalse(report.ready_for_experiment)

    def test_fashioniq_caption_join_and_category(self) -> None:
        root = self.root / "fashioniq"
        self._write_json(
            root / "image_splits" / "split.dress.val.json", ["a", "b"]
        )
        self._write_json(
            root / "captions" / "cap.dress.val.json",
            [
                {
                    "candidate": "a",
                    "target": "b",
                    "captions": ["shorter sleeves", "bright red"],
                }
            ],
        )
        for name in ("a", "b"):
            path = root / "images" / f"{name}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        bundle = load_fashioniq(root, categories=("dress",))
        report = validate_bundle(bundle)
        self.assertEqual(
            bundle.queries[0].modification_text, "shorter sleeves and bright red"
        )
        self.assertEqual(bundle.queries[0].category, "dress")
        self.assertTrue(report.ready_for_experiment)

    def test_test_split_without_targets_is_not_experiment_ready(self) -> None:
        root = self.root / "fashioniq"
        self._write_json(
            root / "image_splits" / "split.dress.test.json", ["a"]
        )
        self._write_json(
            root / "captions" / "cap.dress.test.json",
            [{"candidate": "a", "captions": ["red", "short"]}],
        )
        image = root / "images" / "a.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()

        report = validate_bundle(
            load_fashioniq(root, split="test", categories=("dress",))
        )
        self.assertTrue(report.structural_ok)
        self.assertEqual(report.queries_with_targets, 0)
        self.assertFalse(report.ready_for_experiment)

    def test_fashioniq_shared_image_belongs_to_both_categories(self) -> None:
        root = self.root / "fashioniq"
        for category in ("shirt", "toptee"):
            self._write_json(
                root / "image_splits" / f"split.{category}.val.json",
                ["shared", f"{category}-target"],
            )
            self._write_json(
                root / "captions" / f"cap.{category}.val.json",
                [
                    {
                        "candidate": "shared",
                        "target": f"{category}-target",
                        "captions": ["different", "style"],
                    }
                ],
            )
        bundle = load_fashioniq(root, categories=("shirt", "toptee"))
        shared = next(item for item in bundle.gallery if item.image_id == "shared")
        self.assertEqual(shared.categories, ("shirt", "toptee"))
        self.assertEqual(len(bundle.gallery), 3)


if __name__ == "__main__":
    unittest.main()
