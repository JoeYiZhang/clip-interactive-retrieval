from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .api_models import CloseSessionRequest, FeedbackRequest, RetrieveRequest
from .demo_service import LazyTextEncoder, MultiIndexSessionService
from .session_service import (
    SessionAlreadyExists,
    SessionNotFound,
    SessionVersionConflict,
)
from .session_store import SessionStoreError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DemoCreateBody(StrictModel):
    index_key: str = Field(min_length=1, max_length=128)
    reference_image_id: str = Field(min_length=1, max_length=256)
    modification_text: str = Field(default="", max_length=1000)


class DemoRetrieveBody(StrictModel):
    top_k: int = Field(default=12, ge=1, le=50)


class DemoFeedbackBody(StrictModel):
    event_id: str = Field(min_length=1, max_length=128)
    expected_round_id: int = Field(ge=0)
    shown_image_ids: list[str] = Field(min_length=1)
    accepted_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    feedback_text: str = Field(default="", max_length=1000)


class DemoCloseBody(StrictModel):
    reason: str = Field(default="user_closed", min_length=1, max_length=256)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


class LatencyMetrics:
    def __init__(self, history_size: int = 2000) -> None:
        self._values: dict[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._lock = RLock()

    def record(self, key: str, milliseconds: float, status_code: int) -> None:
        with self._lock:
            self._values[key].append((float(milliseconds), int(status_code)))

    def snapshot(self) -> dict:
        with self._lock:
            output = {}
            for key, rows in sorted(self._values.items()):
                durations = [row[0] for row in rows]
                output[key] = {
                    "count": len(rows),
                    "error_count": sum(row[1] >= 400 for row in rows),
                    "p50_ms": round(_percentile(durations, 0.50), 3),
                    "p95_ms": round(_percentile(durations, 0.95), 3),
                    "max_ms": round(max(durations, default=0.0), 3),
                }
            return output


def _invoke(function: Callable[[], Any]) -> Any:
    try:
        return function()
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (SessionAlreadyExists, SessionVersionConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_demo_app(
    service: MultiIndexSessionService,
    *,
    text_encoder: LazyTextEncoder | None = None,
) -> FastAPI:
    app = FastAPI(
        title="FashionIQ Interactive Retrieval Demo",
        version="0.2.0",
        description=(
            "Local-only real-index demo. It is an engineering acceptance surface, "
            "not a new FashionIQ or CIRR benchmark result."
        ),
    )
    metrics = LatencyMetrics()
    static_root = Path(__file__).with_name("static")

    @app.middleware("http")
    async def observe_latency(request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            route = request.scope.get("route")
            template = getattr(route, "path", request.url.path)
            metrics.record(f"{request.method} {template}", elapsed_ms, status_code)
            if "response" in locals():
                response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(static_root / "demo.html", media_type="text/html")

    @app.get("/assets/demo.css", include_in_schema=False)
    def stylesheet():
        return FileResponse(static_root / "demo.css", media_type="text/css")

    @app.get("/assets/demo.js", include_in_schema=False)
    def javascript():
        return FileResponse(static_root / "demo.js", media_type="text/javascript")

    @app.get("/v1/demo/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mode": "local-engineering-demo",
            "text_encoder_loaded": text_encoder.loaded if text_encoder else None,
            "indexes": service.registry.summaries(),
        }

    @app.get("/v1/demo/catalog")
    def catalog() -> dict:
        return {
            "scope": "FashionIQ validation gallery; engineering demo only",
            "indexes": service.registry.summaries(),
        }

    @app.get("/v1/demo/indexes/{index_key}/references")
    def references(
        index_key: str,
        offset: int = 0,
        limit: int = 24,
    ) -> dict:
        return _invoke(
            lambda: service.reference_page(index_key, offset=offset, limit=limit)
        )

    @app.get("/v1/demo/images/{index_key}/{image_id}")
    def image(index_key: str, image_id: str):
        path = _invoke(lambda: service.image_path(index_key, image_id))
        return FileResponse(path)

    @app.post(
        "/v1/demo/sessions",
        status_code=status.HTTP_201_CREATED,
    )
    def create_session(body: DemoCreateBody) -> dict:
        return _invoke(
            lambda: service.create_composed_session(
                session_id=f"demo-{uuid4().hex}",
                index_key=body.index_key,
                reference_image_id=body.reference_image_id,
                modification_text=body.modification_text,
            )
        )

    @app.get("/v1/demo/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        return _invoke(lambda: service.get_session(session_id))

    @app.post("/v1/demo/sessions/{session_id}/retrieve")
    def retrieve(session_id: str, body: DemoRetrieveBody) -> dict:
        response = _invoke(
            lambda: service.retrieve(session_id, RetrieveRequest(top_k=body.top_k))
        )
        index_key = response.session["index_key"]
        return {
            "session": response.session,
            "results": [
                {
                    **row,
                    "image_url": (
                        f"/v1/demo/images/{index_key}/{row['image_id']}"
                    ),
                }
                for row in response.results
            ],
        }

    @app.post("/v1/demo/sessions/{session_id}/feedback")
    def feedback(session_id: str, body: DemoFeedbackBody) -> dict:
        request = FeedbackRequest(
            event_id=body.event_id,
            shown_image_ids=tuple(body.shown_image_ids),
            accepted_ids=tuple(body.accepted_ids),
            rejected_ids=tuple(body.rejected_ids),
            expected_round_id=body.expected_round_id,
        )
        return _invoke(
            lambda: service.submit_feedback(
                session_id,
                request,
                feedback_text=body.feedback_text,
            ).to_dict()
        )

    @app.post("/v1/demo/sessions/{session_id}/close")
    def close(session_id: str, body: DemoCloseBody) -> dict:
        return _invoke(
            lambda: service.close_session(
                session_id, CloseSessionRequest(reason=body.reason)
            )
        )

    @app.get("/v1/demo/metrics")
    def latency_metrics() -> dict:
        return {
            "scope": "process-local rolling window; not a capacity benchmark",
            "routes": metrics.snapshot(),
        }

    return app
