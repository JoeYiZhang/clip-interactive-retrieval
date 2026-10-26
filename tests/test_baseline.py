from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cir_baseline.contracts import (
    IndexManifest,
    QuerySpec,
    ensure_compatible_versions,
)
from cir_baseline.feedback import FeedbackWeights, update_query
from cir_baseline.fusion import fuse_weighted, l2_normalize
from cir_baseline.index import NumpyFlatIPIndex
from cir_baseline.metrics import ann_recall_at_k, recall_at_k


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = IndexManifest(
            index_version="test-index-v1",
            encoder_version="test-encoder-v1",
            embedding_dim=3,
        )
        self.ids = ["x", "y", "z"]
        self.embeddings = np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )

    def test_normalization_and_fusion(self) -> None:
        fused = fuse_weighted(
            np.asarray([2, 0, 0], dtype=np.float32),
            np.asarray([0, 4, 0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(np.linalg.norm(fused)), 1.0, places=6)
        self.assertTrue(np.allclose(fused, [2**-0.5, 2**-0.5, 0], atol=1e-6))
        with self.assertRaises(ValueError):
            l2_normalize(np.zeros(3, dtype=np.float32))

    def test_exact_search_and_persistence(self) -> None:
        index = NumpyFlatIPIndex(self.ids, self.embeddings, self.manifest)
        results = index.search(np.asarray([0.9, 0.1, 0], dtype=np.float32), 2)
        self.assertEqual([result.image_id for result in results], ["x", "y"])
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            target = Path(directory) / "index.npz"
            index.save(target)
            restored = NumpyFlatIPIndex.load(target)
            restored_results = restored.search(
                np.asarray([0.9, 0.1, 0], dtype=np.float32), 2
            )
        self.assertEqual(results, restored_results)
        self.assertEqual(restored.manifest.item_count, 3)

    def test_metrics(self) -> None:
        ranked = [["a", "b"], ["c", "d"]]
        relevant = [{"a"}, {"d"}]
        self.assertEqual(recall_at_k(ranked, relevant, 1), 0.5)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 1.0)
        self.assertEqual(ann_recall_at_k(ranked, ranked, 2), 1.0)

    def test_version_guard(self) -> None:
        valid = QuerySpec("ref", "text", "test-encoder-v1", "test-index-v1")
        ensure_compatible_versions(valid, self.manifest)
        invalid = QuerySpec("ref", "text", "other", "test-index-v1")
        with self.assertRaises(ValueError):
            ensure_compatible_versions(invalid, self.manifest)

    def test_feedback_drift_guard(self) -> None:
        q0 = np.asarray([1, 0, 0], dtype=np.float32)
        accepted = update_query(
            q0,
            q0,
            positive_embeddings=[np.asarray([0.9, 0.1, 0], dtype=np.float32)],
        )
        self.assertTrue(accepted.applied)
        blocked = update_query(
            q0,
            q0,
            feedback_text_embedding=np.asarray([-1, 0, 0], dtype=np.float32),
            weights=FeedbackWeights(
                anchor_weight=0,
                previous_weight=0,
                positive_weight=0,
                negative_weight=0,
                text_weight=1,
                update_strength=1,
                max_cosine_drift=0.1,
            ),
        )
        self.assertFalse(blocked.applied)
        self.assertTrue(np.allclose(blocked.vector, q0))


if __name__ == "__main__":
    unittest.main()

