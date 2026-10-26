"""Exercise the local FastAPI + SQLite session lifecycle end to end."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.contracts import IndexManifest  # noqa: E402
from cir_baseline.http_api import create_app  # noqa: E402
from cir_baseline.index import NumpyFlatIPIndex  # noqa: E402
from cir_baseline.session_service import SessionService  # noqa: E402
from cir_baseline.session_store import SQLiteSessionStore  # noqa: E402


def _service(db_path: Path) -> SessionService:
    ids = ["reference", "candidate-a", "candidate-b", "candidate-c"]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.2, 0.9, 0.3],
            [0.0, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    manifest = IndexManifest(
        index_version="api-smoke-index-v1",
        encoder_version="api-smoke-encoder-v1",
        embedding_dim=3,
    )
    index = NumpyFlatIPIndex(ids, embeddings, manifest)
    return SessionService(
        index=index,
        embedding_by_id=dict(zip(ids, embeddings, strict=True)),
        store=SQLiteSessionStore(db_path),
        max_rounds=3,
    )


def main() -> int:
    artifact_dir = PROJECT_ROOT / "artifacts"
    checks: dict[str, bool] = {}
    evidence: dict = {}
    with tempfile.TemporaryDirectory(dir=artifact_dir) as directory:
        db_path = Path(directory) / "api-smoke.sqlite3"
        service = _service(db_path)
        app = create_app(service)
        openapi_path = artifact_dir / "api" / "openapi.json"
        openapi_path.parent.mkdir(parents=True, exist_ok=True)
        openapi_path.write_text(
            json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with TestClient(app) as client:
            health = client.get("/health")
            checks["health"] = health.status_code == 200
            created = client.post(
                "/v1/sessions",
                json={
                    "session_id": "anon-api-smoke",
                    "reference_image_id": "reference",
                    "initial_query": [0.7, 0.7, 0.0],
                    "expected_encoder_version": "api-smoke-encoder-v1",
                    "expected_index_version": "api-smoke-index-v1",
                },
            )
            checks["create_session"] = created.status_code == 201
            retrieved = client.post(
                "/v1/sessions/anon-api-smoke/retrieve", json={"top_k": 3}
            )
            shown = [row["image_id"] for row in retrieved.json()["results"]]
            checks["retrieve_and_reference_filter"] = (
                retrieved.status_code == 200 and "reference" not in shown
            )
            payload = {
                "event_id": "api-smoke-event-1",
                "shown_image_ids": shown,
                "accepted_ids": ["candidate-b"],
                "rejected_ids": ["candidate-c"],
            }
            feedback = client.post(
                "/v1/sessions/anon-api-smoke/feedback", json=payload
            )
            repeated = client.post(
                "/v1/sessions/anon-api-smoke/feedback", json=payload
            )
            checks["feedback"] = feedback.status_code == 200
            checks["http_idempotency"] = (
                repeated.status_code == 200 and repeated.json() == feedback.json()
            )
            evidence["first_transition"] = feedback.json()["transition"]

        restarted_app = create_app(_service(db_path))
        with TestClient(restarted_app) as restarted:
            restored = restarted.get("/v1/sessions/anon-api-smoke")
            checks["restart_recovery"] = (
                restored.status_code == 200
                and restored.json()["round_id"] == 1
                and restored.json()["positive_image_ids"] == ["candidate-b"]
                and restored.json()["negative_image_ids"] == ["candidate-c"]
            )
            closed = restarted.post(
                "/v1/sessions/anon-api-smoke/close",
                json={"reason": "smoke_complete"},
            )
            checks["close"] = (
                closed.status_code == 200 and closed.json()["status"] == "closed"
            )
            invalid = restarted.post(
                "/v1/sessions",
                json={
                    "session_id": "bad-version",
                    "reference_image_id": "reference",
                    "initial_query": [1.0, 0.0, 0.0],
                    "expected_encoder_version": "api-smoke-encoder-v1",
                    "expected_index_version": "wrong-index",
                },
            )
            checks["version_conflict_http_409"] = invalid.status_code == 409
            checks["openapi_route_present"] = (
                "/v1/sessions/{session_id}/feedback"
                in restarted_app.openapi()["paths"]
            )

    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "local FastAPI + SQLite integration smoke; not a research result",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "evidence": evidence,
        "openapi": "artifacts/api/openapi.json",
    }
    target = artifact_dir / "metrics" / "stage3_api_smoke.json"
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
