from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from cir_baseline.contracts import IndexManifest
from cir_baseline.demo_app import create_demo_app
from cir_baseline.demo_service import (
    FashionIQQueryComposer,
    MultiIndexSessionService,
)
from cir_baseline.index import NumpyFlatIPIndex
from cir_baseline.index_registry import IndexRegistry, RegisteredIndex
from cir_baseline.session_store import SQLiteSessionStore


class FakeTextEncoder:
    version = "demo-encoder-v1"

    def encode_texts(self, texts):
        return np.stack(
            [np.asarray([0.1, 0.9, 0.2], dtype=np.float32) for _ in texts]
        )


class DemoAppTests(unittest.TestCase):
    def setUp(self) -> None:
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
        self.temp = tempfile.TemporaryDirectory(dir=artifact_root)
        root = Path(self.temp.name)
        self.ids = ["reference", "a", "b", "c"]
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.2, 0.9, 0.3],
                [0.0, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        manifest = IndexManifest(
            index_version="demo-dress-index-v1",
            encoder_version=FakeTextEncoder.version,
            embedding_dim=3,
        )
        index = NumpyFlatIPIndex(self.ids, vectors, manifest)
        paths = {}
        for image_id in self.ids:
            path = root / f"{image_id}.png"
            path.write_bytes(f"image:{image_id}".encode())
            paths[image_id] = path
        entry = RegisteredIndex(
            key="fashioniq:val:dress",
            dataset="fashioniq",
            split="val",
            category="dress",
            index=index,
            embedding_by_id=dict(zip(self.ids, index.embeddings, strict=True)),
            image_path_by_id=paths,
            cache_path=root / "synthetic.npz",
            cache_sha256="0" * 64,
        )
        self.registry = IndexRegistry([entry])
        self.database = root / "demo.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _client(self) -> TestClient:
        service = MultiIndexSessionService(
            registry=self.registry,
            store=SQLiteSessionStore(self.database),
            composer=FashionIQQueryComposer(FakeTextEncoder()),
            max_rounds=3,
        )
        return TestClient(create_demo_app(service))

    def test_real_surface_lifecycle_restart_and_conflict(self) -> None:
        with self._client() as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/assets/demo.js").status_code, 200)
            health = client.get("/v1/demo/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["indexes"][0]["item_count"], 4)
            references = client.get(
                "/v1/demo/indexes/fashioniq:val:dress/references?limit=2"
            )
            self.assertEqual(len(references.json()["items"]), 2)
            created = client.post(
                "/v1/demo/sessions",
                json={
                    "index_key": "fashioniq:val:dress",
                    "reference_image_id": "reference",
                    "modification_text": "make it blue",
                },
            )
            self.assertEqual(created.status_code, 201)
            self.assertEqual(
                created.json()["composition"]["policy_version"],
                "fashioniq-a2-train-frozen-v1",
            )
            session_id = created.json()["session"]["session_id"]
            retrieved = client.post(
                f"/v1/demo/sessions/{session_id}/retrieve", json={"top_k": 3}
            )
            self.assertEqual(retrieved.status_code, 200)
            results = retrieved.json()["results"]
            shown = [row["image_id"] for row in results]
            self.assertNotIn("reference", shown)
            image = client.get(results[0]["image_url"])
            self.assertEqual(image.status_code, 200)
            payload = {
                "event_id": "event-1",
                "expected_round_id": 0,
                "shown_image_ids": shown,
                "accepted_ids": [shown[0]],
                "rejected_ids": [shown[-1]],
                "feedback_text": "",
            }
            feedback = client.post(
                f"/v1/demo/sessions/{session_id}/feedback", json=payload
            )
            self.assertEqual(feedback.status_code, 200)
            repeated = client.post(
                f"/v1/demo/sessions/{session_id}/feedback", json=payload
            )
            self.assertEqual(repeated.status_code, 200)
            self.assertEqual(feedback.json(), repeated.json())
            stale = dict(payload)
            stale["event_id"] = "event-stale"
            self.assertEqual(
                client.post(
                    f"/v1/demo/sessions/{session_id}/feedback", json=stale
                ).status_code,
                409,
            )

        with self._client() as restarted:
            restored = restarted.get(f"/v1/demo/sessions/{session_id}")
            self.assertEqual(restored.status_code, 200)
            self.assertEqual(restored.json()["round_id"], 1)
            self.assertEqual(restored.json()["index_key"], "fashioniq:val:dress")
            next_page = restarted.post(
                f"/v1/demo/sessions/{session_id}/retrieve", json={"top_k": 2}
            )
            self.assertEqual(next_page.status_code, 200)
            metrics = restarted.get("/v1/demo/metrics")
            self.assertTrue(metrics.json()["routes"])
            closed = restarted.post(
                f"/v1/demo/sessions/{session_id}/close",
                json={"reason": "test_complete"},
            )
            self.assertEqual(closed.json()["status"], "closed")

    def test_validation_and_image_whitelist(self) -> None:
        with self._client() as client:
            self.assertEqual(
                client.get(
                    "/v1/demo/images/fashioniq:val:dress/not-registered"
                ).status_code,
                404,
            )
            invalid = client.post(
                "/v1/demo/sessions",
                json={
                    "index_key": "unknown",
                    "reference_image_id": "reference",
                    "modification_text": "blue",
                },
            )
            self.assertEqual(invalid.status_code, 404)


if __name__ == "__main__":
    unittest.main()
