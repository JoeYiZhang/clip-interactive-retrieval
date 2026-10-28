"""Exercise the real FashionIQ registry, model, HTTP UI API and SQLite recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from cir_baseline.datasets import load_fashioniq  # noqa: E402
from cir_baseline.demo_app import create_demo_app  # noqa: E402
from cir_baseline.demo_service import (  # noqa: E402
    FashionIQQueryComposer,
    LazyTextEncoder,
    MultiIndexSessionService,
)
from cir_baseline.encoders import OpenCLIPEncoder  # noqa: E402
from cir_baseline.index_registry import IndexRegistry  # noqa: E402
from cir_baseline.session_store import SQLiteSessionStore  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage-root", type=Path, default=Path(r"D:\image_data")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "metrics"
        / "stage3_fashioniq_demo_smoke.json",
    )
    parser.add_argument(
        "--openapi-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "api" / "fashioniq-demo-openapi.json",
    )
    args = parser.parse_args()
    cache_paths = sorted(
        (args.storage_root / "derived" / "embeddings").glob(
            "fashioniq_val_official_*.npz"
        )
    )
    if len(cache_paths) != 1:
        raise SystemExit("expected one FashionIQ validation embedding cache")

    started = time.perf_counter()
    registry = IndexRegistry.from_fashioniq_cache(
        data_root=args.storage_root / "datasets" / "fashioniq",
        cache_path=cache_paths[0],
    )
    registry_seconds = time.perf_counter() - started
    bundle = load_fashioniq(
        args.storage_root / "datasets" / "fashioniq",
        split="val",
        categories=("dress",),
    )
    query = bundle.queries[0]
    expected_version = registry.get("fashioniq:val:dress").index.manifest.encoder_version
    lazy_encoder = LazyTextEncoder(
        expected_version=expected_version,
        factory=lambda: OpenCLIPEncoder(
            checkpoint_path=args.checkpoint,
            device=args.device,
            force_quick_gelu=True,
            weights_only=False,
        ),
    )

    checks: dict[str, bool] = {}
    latency: dict[str, float] = {"registry_load_ms": round(registry_seconds * 1000, 3)}
    artifact_root = PROJECT_ROOT / "artifacts"
    with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
        database = Path(directory) / "demo.sqlite3"

        def client() -> TestClient:
            service = MultiIndexSessionService(
                registry=registry,
                store=SQLiteSessionStore(database),
                composer=FashionIQQueryComposer(lazy_encoder),
                max_rounds=3,
            )
            return TestClient(create_demo_app(service, text_encoder=lazy_encoder))

        with client() as first:
            checks["page"] = first.get("/").status_code == 200
            openapi = first.get("/openapi.json")
            checks["openapi"] = (
                openapi.status_code == 200
                and "/v1/demo/sessions" in openapi.json()["paths"]
            )
            health = first.get("/v1/demo/health")
            checks["three_real_indexes"] = (
                health.status_code == 200 and len(health.json()["indexes"]) == 3
            )
            create_started = time.perf_counter()
            created = first.post(
                "/v1/demo/sessions",
                json={
                    "index_key": "fashioniq:val:dress",
                    "reference_image_id": query.reference_image_id,
                    "modification_text": query.modification_text,
                },
            )
            latency["first_text_session_ms"] = round(
                (time.perf_counter() - create_started) * 1000, 3
            )
            checks["create"] = created.status_code == 201
            session_id = created.json()["session"]["session_id"]
            warm_started = time.perf_counter()
            warm = first.post(
                "/v1/demo/sessions",
                json={
                    "index_key": "fashioniq:val:dress",
                    "reference_image_id": query.reference_image_id,
                    "modification_text": query.modification_text,
                },
            )
            latency["warm_text_session_ms"] = round(
                (time.perf_counter() - warm_started) * 1000, 3
            )
            checks["warm_create"] = warm.status_code == 201
            warm_session_id = warm.json()["session"]["session_id"]
            checks["warm_close"] = (
                first.post(
                    f"/v1/demo/sessions/{warm_session_id}/close",
                    json={"reason": "warm_latency_observed"},
                ).status_code
                == 200
            )
            retrieve_started = time.perf_counter()
            retrieved = first.post(
                f"/v1/demo/sessions/{session_id}/retrieve", json={"top_k": 12}
            )
            latency["retrieve_ms"] = round(
                (time.perf_counter() - retrieve_started) * 1000, 3
            )
            results = retrieved.json()["results"]
            shown = [row["image_id"] for row in results]
            checks["retrieve"] = retrieved.status_code == 200 and len(shown) == 12
            checks["reference_filtered"] = query.reference_image_id not in shown
            image = first.get(results[0]["image_url"])
            checks["image_delivery"] = image.status_code == 200 and len(image.content) > 0
            feedback_payload = {
                "event_id": "real-event-1",
                "expected_round_id": 0,
                "shown_image_ids": shown,
                "accepted_ids": [shown[0]],
                "rejected_ids": [shown[-1]],
                "feedback_text": "",
            }
            feedback = first.post(
                f"/v1/demo/sessions/{session_id}/feedback", json=feedback_payload
            )
            checks["feedback"] = (
                feedback.status_code == 200
                and feedback.json()["session"]["round_id"] == 1
            )
            repeated = first.post(
                f"/v1/demo/sessions/{session_id}/feedback", json=feedback_payload
            )
            checks["idempotent_retry"] = (
                repeated.status_code == 200 and repeated.json() == feedback.json()
            )
            stale = dict(feedback_payload)
            stale["event_id"] = "stale-event"
            checks["stale_round_rejected"] = (
                first.post(
                    f"/v1/demo/sessions/{session_id}/feedback", json=stale
                ).status_code
                == 409
            )

        with client() as restarted:
            restored = restarted.get(f"/v1/demo/sessions/{session_id}")
            checks["restart_restore"] = (
                restored.status_code == 200
                and restored.json()["round_id"] == 1
                and restored.json()["index_key"] == "fashioniq:val:dress"
            )
            second = restarted.post(
                f"/v1/demo/sessions/{session_id}/retrieve", json={"top_k": 5}
            )
            checks["retrieve_after_restart"] = (
                second.status_code == 200 and len(second.json()["results"]) == 5
            )
            metrics = restarted.get("/v1/demo/metrics")
            checks["latency_metrics"] = (
                metrics.status_code == 200 and bool(metrics.json()["routes"])
            )
            closed = restarted.post(
                f"/v1/demo/sessions/{session_id}/close",
                json={"reason": "real_demo_smoke_complete"},
            )
            checks["close"] = (
                closed.status_code == 200 and closed.json()["status"] == "closed"
            )

    args.openapi_output.parent.mkdir(parents=True, exist_ok=True)
    args.openapi_output.write_text(
        json.dumps(openapi.json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    passed = all(checks.values())
    output = {
        "status": "pass" if passed else "fail",
        "scope": (
            "real FashionIQ index and local HTTP engineering acceptance; "
            "not a benchmark metric"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "registry": registry.summaries(),
        "encoder": {
            "version": expected_version,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "loaded": lazy_encoder.loaded,
        },
        "query": {
            "query_id": query.query_id,
            "category": query.category,
            "reference_image_id": query.reference_image_id,
        },
        "checks": checks,
        "latency": latency,
        "openapi": {
            "path": str(args.openapi_output.resolve()),
            "sha256": _sha256(args.openapi_output),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
