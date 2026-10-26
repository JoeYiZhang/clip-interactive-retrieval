from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cir_baseline.api_models import (
    CloseSessionRequest,
    CreateSessionRequest,
    FeedbackRequest,
    RetrieveRequest,
)
from cir_baseline.contracts import IndexManifest
from cir_baseline.index import NumpyFlatIPIndex
from cir_baseline.session_service import (
    SessionAlreadyExists,
    SessionService,
    SessionVersionConflict,
)
from cir_baseline.session_store import SQLiteSessionStore


class SessionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = ["reference", "a", "b", "c"]
        self.embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.2, 0.9, 0.3],
                [0.0, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        self.manifest = IndexManifest(
            index_version="service-index-v1",
            encoder_version="service-encoder-v1",
            embedding_dim=3,
        )
        self.index = NumpyFlatIPIndex(self.ids, self.embeddings, self.manifest)
        self.embedding_by_id = dict(zip(self.ids, self.embeddings, strict=True))

    def _service(self, db_path: Path, max_rounds: int = 3) -> SessionService:
        return SessionService(
            index=self.index,
            embedding_by_id=self.embedding_by_id,
            store=SQLiteSessionStore(db_path),
            max_rounds=max_rounds,
        )

    def _create_request(self, session_id: str = "anon-service-1") -> CreateSessionRequest:
        return CreateSessionRequest(
            session_id=session_id,
            reference_image_id="reference",
            initial_query=(0.7, 0.7, 0.0),
            expected_encoder_version=self.manifest.encoder_version,
            expected_index_version=self.manifest.index_version,
        )

    def test_request_models_reject_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            RetrieveRequest.from_dict({"top_k": 0})
        with self.assertRaises(ValueError):
            FeedbackRequest.from_dict(
                {"event_id": "event", "shown_image_ids": "not-a-list"}
            )
        request = CreateSessionRequest.from_dict(
            {
                "session_id": "anon",
                "reference_image_id": "reference",
                "initial_query": [1, 0, 0],
                "expected_encoder_version": self.manifest.encoder_version,
                "expected_index_version": self.manifest.index_version,
            }
        )
        self.assertEqual(request.initial_query, (1.0, 0.0, 0.0))

    def test_sqlite_restart_restores_exposure_and_feedback(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            service = self._service(db_path)
            created = service.create_session(self._create_request())
            self.assertEqual(created["round_id"], 0)
            retrieved = service.retrieve("anon-service-1", RetrieveRequest(top_k=3))
            shown = tuple(item["image_id"] for item in retrieved.results)

            restarted = self._service(db_path)
            restored = restarted.get_session("anon-service-1")
            self.assertEqual(tuple(restored["last_shown_image_ids"]), shown)
            response = restarted.submit_feedback(
                "anon-service-1",
                FeedbackRequest(
                    event_id="event-1",
                    shown_image_ids=shown,
                    accepted_ids=("b",),
                    rejected_ids=("c",),
                ),
            )
            self.assertEqual(response.session["round_id"], 1)
            self.assertEqual(restarted.store.event_count("anon-service-1"), 1)

            restarted_again = self._service(db_path)
            snapshot = restarted_again.get_session("anon-service-1")
            self.assertEqual(tuple(snapshot["positive_image_ids"]), ("b",))
            self.assertEqual(tuple(snapshot["negative_image_ids"]), ("c",))

    def test_duplicate_event_is_idempotent_across_restart(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            service = self._service(db_path)
            service.create_session(self._create_request())
            retrieved = service.retrieve("anon-service-1", RetrieveRequest(top_k=3))
            shown = tuple(item["image_id"] for item in retrieved.results)
            request = FeedbackRequest(
                event_id="event-1",
                shown_image_ids=shown,
                accepted_ids=("a",),
            )
            first = service.submit_feedback("anon-service-1", request)
            restarted = self._service(db_path)
            repeated = restarted.submit_feedback("anon-service-1", request)
            self.assertEqual(first, repeated)
            self.assertEqual(restarted.store.event_count("anon-service-1"), 1)

    def test_version_conflict_duplicate_session_and_close(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            service = self._service(Path(directory) / "sessions.sqlite3")
            invalid = CreateSessionRequest(
                session_id="bad-version",
                reference_image_id="reference",
                initial_query=(1, 0, 0),
                expected_encoder_version="wrong",
                expected_index_version=self.manifest.index_version,
            )
            with self.assertRaises(SessionVersionConflict):
                service.create_session(invalid)
            service.create_session(self._create_request())
            with self.assertRaises(SessionAlreadyExists):
                service.create_session(self._create_request())
            closed = service.close_session(
                "anon-service-1", CloseSessionRequest("test_complete")
            )
            self.assertEqual(closed["status"], "closed")
            restarted = self._service(Path(directory) / "sessions.sqlite3")
            self.assertEqual(
                restarted.get_session("anon-service-1")["close_reason"],
                "test_complete",
            )


if __name__ == "__main__":
    unittest.main()
