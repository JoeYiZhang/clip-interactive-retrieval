from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cir_baseline.index_registry import IndexRegistry


class IndexRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        self.temp = tempfile.TemporaryDirectory(dir=artifact_root)
        self.root = Path(self.temp.name) / "fashioniq"
        (self.root / "captions").mkdir(parents=True)
        (self.root / "image_splits").mkdir()
        (self.root / "images").mkdir()
        category_ids = {
            "dress": ["d-ref", "d-item", "shared"],
            "shirt": ["s-ref", "s-item", "shared"],
        }
        captions = {
            "dress": [
                {"candidate": "d-ref", "target": "d-item", "captions": ["blue"]}
            ],
            "shirt": [
                {"candidate": "s-ref", "target": "s-item", "captions": ["plain"]}
            ],
        }
        for category, ids in category_ids.items():
            (self.root / "image_splits" / f"split.{category}.val.json").write_text(
                json.dumps(ids), encoding="utf-8"
            )
            (self.root / "captions" / f"cap.{category}.val.json").write_text(
                json.dumps(captions[category]), encoding="utf-8"
            )
        ordered_ids = ["d-ref", "d-item", "shared", "s-ref", "s-item"]
        paths = []
        for image_id in ordered_ids:
            path = self.root / "images" / f"{image_id}.png"
            path.write_bytes(b"test-image")
            paths.append(str(path.resolve()))
        self.cache = Path(self.temp.name) / "cache.npz"
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.4, 0.4, 0.2],
                [0.0, 1.0, 0.0],
                [0.1, 0.8, 0.1],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            self.cache,
            image_ids=np.asarray(ordered_ids),
            image_paths=np.asarray(paths),
            embeddings=vectors,
            encoder_version=np.asarray("registry-encoder-v1"),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_loads_category_indexes_and_paths(self) -> None:
        registry = IndexRegistry.from_fashioniq_cache(
            data_root=self.root,
            cache_path=self.cache,
            categories=("dress", "shirt"),
        )
        self.assertEqual(registry.keys, ("fashioniq:val:dress", "fashioniq:val:shirt"))
        dress = registry.get("fashioniq:val:dress")
        shirt = registry.get("fashioniq:val:shirt")
        self.assertEqual(dress.index.manifest.item_count, 3)
        self.assertEqual(shirt.index.manifest.item_count, 3)
        self.assertIn("shared", dress.image_path_by_id)
        self.assertIn("shared", shirt.image_path_by_id)
        self.assertTrue(dress.image_path("d-ref").is_file())
        self.assertIs(
            registry.by_version(dress.index.manifest.index_version), dress
        )
        self.assertEqual(dress.index.search(np.asarray([1, 0, 0]), 1)[0].image_id, "d-ref")
        self.assertEqual(len(registry.summaries()), 2)

    def test_rejects_cache_dataset_mismatch(self) -> None:
        with np.load(self.cache, allow_pickle=False) as payload:
            paths = payload["image_paths"]
            vectors = payload["embeddings"]
        bad = Path(self.temp.name) / "bad.npz"
        np.savez_compressed(
            bad,
            image_ids=np.asarray(["wrong"] * 5),
            image_paths=paths,
            embeddings=vectors,
            encoder_version=np.asarray("registry-encoder-v1"),
        )
        with self.assertRaisesRegex(ValueError, "image IDs"):
            IndexRegistry.from_fashioniq_cache(
                data_root=self.root,
                cache_path=bad,
                categories=("dress", "shirt"),
            )


if __name__ == "__main__":
    unittest.main()
