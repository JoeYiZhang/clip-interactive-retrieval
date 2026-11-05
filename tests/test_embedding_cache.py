import tempfile
import unittest
from pathlib import Path

import numpy as np

from cir_baseline.embedding_cache import build_cache, load_cache


class TinyEncoder:
    version = "test-v1"
    def encode_images(self, values):
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(values), 1))
    def encode_texts(self, values):
        return np.tile(np.array([[0., 1.]], dtype=np.float32), (len(values), 1))


class CacheTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / "a.jpg").write_bytes(b"image-a")
        (root / "b.jpg").write_bytes(b"image-b")
        return {"mode": "engineering", "dimension": 2, "image_root": str(root), "encoder_version": "test-v1",
                "images": [{"id": "a", "path": "a.jpg"}, {"id": "b", "path": "b.jpg"}],
                "queries": [{"query_id": "q1", "reference_id": "a", "target_id": "b", "text": "blue", "positive_ids": ["b"]}]}

    def test_resume_keeps_committed_shards_and_complete_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.fixture(folder)
            output = Path(folder) / "cache"
            state = build_cache(plan, output, TinyEncoder, 1, 1, stop_after_shards=1)
            self.assertEqual(state["status"], "building")
            first = (output / "images-000000000.npz").read_bytes()
            with self.assertRaises(ValueError):
                load_cache(output)
            state = build_cache(plan, output, TinyEncoder, 1, 1)
            self.assertEqual(state["status"], "complete")
            self.assertEqual((output / "images-000000000.npz").read_bytes(), first)
            _, matrices, _ = load_cache(output)
            self.assertEqual(matrices["images"].shape, (2, 2))
            def no_encoding():
                raise AssertionError("completed cache must not re-encode")
            build_cache(plan, output, no_encoding, 1, 1)

    def test_missing_or_changed_images_and_cache_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.fixture(folder)
            output = Path(folder) / "cache"
            build_cache(plan, output, TinyEncoder, 1, 1)
            (Path(folder) / "a.jpg").write_bytes(b"new-image")
            with self.assertRaisesRegex(ValueError, "content changed"):
                build_cache(plan, output, TinyEncoder, 1, 1)
            (output / "texts-000000000.npz").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                load_cache(output)
            (Path(folder) / "b.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                build_cache(plan, Path(folder) / "new", TinyEncoder)

    def test_changed_text_or_encoder_cannot_reuse_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.fixture(folder)
            output = Path(folder) / "cache"
            build_cache(plan, output, TinyEncoder, 1, 1)
            plan["queries"][0]["text"] = "red"
            with self.assertRaises(ValueError):
                build_cache(plan, output, TinyEncoder, 1, 1)
            plan["encoder_version"] = "different"
            with self.assertRaises(ValueError):
                build_cache(plan, Path(folder) / "other", TinyEncoder)
