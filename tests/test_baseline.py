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
from cir_baseline.adaptive import (
    confidence_bands,
    has_identity_intent,
    quantile_thresholds,
    select_weight_from_ranks,
)
from cir_baseline.feedback import FeedbackWeights, update_query
from cir_baseline.fusion import fuse_weighted, l2_normalize
from cir_baseline.index import FaissFlatIPIndex, NumpyFlatIPIndex
from cir_baseline.interactive import InteractiveSession
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

    def test_faiss_batch_search_matches_single_search(self) -> None:
        index = FaissFlatIPIndex(self.ids, self.embeddings, self.manifest)
        queries = np.asarray([[0.9, 0.1, 0], [0, 0.2, 0.8]], dtype=np.float32)
        batched = index.search_many(queries, 2)
        singles = [index.search(query, 2) for query in queries]
        self.assertEqual(batched, singles)

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

    def test_identity_intent_is_specific(self) -> None:
        self.assertTrue(has_identity_intent("is exactly the same product"))
        self.assertTrue(has_identity_intent("looks identical"))
        self.assertFalse(has_identity_intent("has the same color but shorter sleeves"))

    def test_confidence_bands_use_train_thresholds(self) -> None:
        thresholds = quantile_thresholds([0.0, 1.0, 2.0, 3.0])
        bands = confidence_bands([0.0, 1.5, 3.0], thresholds)
        self.assertEqual(bands.tolist(), [0, 1, 2])

    def test_weight_selection_uses_recall_then_prior(self) -> None:
        weights = [0.0, 0.2, 0.4]
        ranks = np.asarray(
            [
                [1, 60, 60, 60],
                [1, 2, 60, 60],
                [1, 2, 60, 60],
            ]
        )
        selected, metrics = select_weight_from_ranks(
            weights, ranks, np.arange(4), prior_weight=0.2
        )
        self.assertEqual(selected, 0.2)
        self.assertEqual(metrics["recall@10"], 0.5)

    def _interactive_session(
        self,
        *,
        max_rounds: int = 3,
        weights: FeedbackWeights = FeedbackWeights(),
    ) -> InteractiveSession:
        index = NumpyFlatIPIndex(self.ids, self.embeddings, self.manifest)
        embedding_by_id = dict(zip(self.ids, self.embeddings, strict=True))
        return InteractiveSession(
            session_id="anon-session-1",
            reference_image_id="x",
            initial_query=np.asarray([0.8, 0.6, 0], dtype=np.float32),
            encoder_version="test-encoder-v1",
            index=index,
            embedding_by_id=embedding_by_id,
            max_rounds=max_rounds,
            weights=weights,
        )

    def test_interactive_round_and_explicit_feedback(self) -> None:
        session = self._interactive_session()
        shown = session.retrieve(top_k=2)
        shown_ids = tuple(item.image_id for item in shown)
        self.assertNotIn("x", shown_ids)
        transition = session.submit_feedback(
            event_id="event-1",
            shown_image_ids=shown_ids,
            accepted_ids=("y",),
            rejected_ids=("z",),
        )
        self.assertEqual(transition.round_after, 1)
        self.assertEqual(transition.cumulative_positive_ids, ("y",))
        self.assertEqual(transition.cumulative_negative_ids, ("z",))
        self.assertEqual(session.snapshot().status, "active")

    def test_interactive_requires_recorded_exposure(self) -> None:
        session = self._interactive_session()
        with self.assertRaises(RuntimeError):
            session.submit_feedback(event_id="event-1", shown_image_ids=("y",))
        shown_ids = tuple(item.image_id for item in session.retrieve(top_k=2))
        with self.assertRaises(ValueError):
            session.submit_feedback(
                event_id="event-2",
                shown_image_ids=shown_ids,
                accepted_ids=("not-shown",),
            )
        with self.assertRaises(ValueError):
            session.submit_feedback(
                event_id="event-3",
                shown_image_ids=shown_ids,
                accepted_ids=(shown_ids[0],),
                rejected_ids=(shown_ids[0],),
            )

    def test_interactive_event_id_is_idempotent(self) -> None:
        session = self._interactive_session()
        shown_ids = tuple(item.image_id for item in session.retrieve(top_k=2))
        first = session.submit_feedback(
            event_id="event-1",
            shown_image_ids=shown_ids,
            accepted_ids=(shown_ids[0],),
        )
        repeated = session.submit_feedback(
            event_id="event-1",
            shown_image_ids=shown_ids,
            accepted_ids=(shown_ids[0],),
        )
        self.assertEqual(first, repeated)
        self.assertEqual(session.snapshot().round_id, 1)
        with self.assertRaises(ValueError):
            session.submit_feedback(
                event_id="event-1",
                shown_image_ids=shown_ids,
                rejected_ids=(shown_ids[0],),
            )

    def test_interactive_max_rounds_and_close(self) -> None:
        session = self._interactive_session(max_rounds=1)
        shown_ids = tuple(item.image_id for item in session.retrieve(top_k=2))
        transition = session.submit_feedback(
            event_id="event-1",
            shown_image_ids=shown_ids,
            accepted_ids=(shown_ids[0],),
        )
        self.assertEqual(transition.status, "completed")
        with self.assertRaises(RuntimeError):
            session.submit_feedback(
                event_id="event-2",
                shown_image_ids=shown_ids,
                accepted_ids=(shown_ids[0],),
            )
        snapshot = session.close()
        self.assertEqual(snapshot.status, "completed")

    def test_interactive_drift_guard_rolls_back(self) -> None:
        session = self._interactive_session(
            weights=FeedbackWeights(
                anchor_weight=0,
                previous_weight=0,
                positive_weight=0,
                negative_weight=0,
                text_weight=1,
                update_strength=1,
                max_cosine_drift=0.01,
            )
        )
        before = session.current_query
        shown_ids = tuple(item.image_id for item in session.retrieve(top_k=2))
        transition = session.submit_feedback(
            event_id="event-1",
            shown_image_ids=shown_ids,
            feedback_text_embedding=-before,
        )
        self.assertFalse(transition.update_applied)
        self.assertTrue(np.allclose(session.current_query, before))


if __name__ == "__main__":
    unittest.main()

