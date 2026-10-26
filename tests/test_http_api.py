from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from cir_baseline.contracts import IndexManifest
from cir_baseline.http_api import create_app
from cir_baseline.index import NumpyFlatIPIndex
from cir_baseline.session_service import SessionService
from cir_baseline.session_store import SQLiteSessionStore


class HttpApiTests(unittest.TestCase):
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
            index_version="api-index-v1",
            encoder_version="api-encoder-v1",
            embedding_dim=3,
        )
        self.index = NumpyFlatIPIndex(self.ids, self.embeddings, self.manifest)
        self.embedding_by_id = dict(zip(self.ids, self.embeddings, strict=True))

    def _client(self, path: Path) -> TestClient:
        service = SessionService(
            index=self.index,
            embedding_by_id=self.embedding_by_id,
            store=SQLiteSessionStore(path),
            max_rounds=3,
        )
        return TestClient(create_app(service))

    def _create_payload(self) -> dict:
        return {
            "session_id": "anon-api-1",
            "reference_image_id": "reference",
            "initial_query": [0.7, 0.7, 0.0],
            "expected_encoder_version": self.manifest.encoder_version,
            "expected_index_version": self.manifest.index_version,
        }

    def test_http_lifecycle_restart_and_idempotency(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            db_path = Path(directory) / "api.sqlite3"
            with self._client(db_path) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["index_version"], "api-index-v1")

                created = client.post("/v1/sessions", json=self._create_payload())
                self.assertEqual(created.status_code, 201)
                self.assertEqual(created.json()["round_id"], 0)
                duplicate = client.post("/v1/sessions", json=self._create_payload())
                self.assertEqual(duplicate.status_code, 409)

                retrieved = client.post(
                    "/v1/sessions/anon-api-1/retrieve", json={"top_k": 3}
                )
                self.assertEqual(retrieved.status_code, 200)
                shown = [item["image_id"] for item in retrieved.json()["results"]]
                self.assertNotIn("reference", shown)
                feedback_payload = {
                    "event_id": "event-1",
                    "shown_image_ids": shown,
                    "accepted_ids": ["b"],
                    "rejected_ids": ["c"],
                }
                first = client.post(
                    "/v1/sessions/anon-api-1/feedback", json=feedback_payload
                )
                self.assertEqual(first.status_code, 200)
                repeated = client.post(
                    "/v1/sessions/anon-api-1/feedback", json=feedback_payload
                )
                self.assertEqual(repeated.status_code, 200)
                self.assertEqual(first.json(), repeated.json())

            with self._client(db_path) as restarted:
                restored = restarted.get("/v1/sessions/anon-api-1")
                self.assertEqual(restored.status_code, 200)
                self.assertEqual(restored.json()["round_id"], 1)
                self.assertEqual(restored.json()["positive_image_ids"], ["b"])
                retrieved = restarted.post(
                    "/v1/sessions/anon-api-1/retrieve", json={"top_k": 2}
                )
                shown = [item["image_id"] for item in retrieved.json()["results"]]
                invalid = restarted.post(
                    "/v1/sessions/anon-api-1/feedback",
                    json={
                        "event_id": "event-2",
                        "shown_image_ids": shown,
                        "accepted_ids": ["not-shown"],
                    },
                )
                self.assertEqual(invalid.status_code, 422)
                closed = restarted.post(
                    "/v1/sessions/anon-api-1/close",
                    json={"reason": "integration_complete"},
                )
                self.assertEqual(closed.status_code, 200)
                self.assertEqual(closed.json()["status"], "closed")

    def test_http_validation_versions_and_openapi(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            with self._client(Path(directory) / "api.sqlite3") as client:
                malformed = client.post(
                    "/v1/sessions", json={"session_id": "missing-fields"}
                )
                self.assertEqual(malformed.status_code, 422)
                wrong = self._create_payload()
                wrong["session_id"] = "wrong-version"
                wrong["expected_index_version"] = "unavailable"
                conflict = client.post("/v1/sessions", json=wrong)
                self.assertEqual(conflict.status_code, 409)
                missing = client.get("/v1/sessions/does-not-exist")
                self.assertEqual(missing.status_code, 404)
                openapi = client.get("/openapi.json")
                self.assertEqual(openapi.status_code, 200)
                self.assertIn("/v1/sessions/{session_id}/feedback", openapi.json()["paths"])


if __name__ == "__main__":
    unittest.main()
