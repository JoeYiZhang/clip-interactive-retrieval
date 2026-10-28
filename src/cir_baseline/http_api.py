from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from .api_models import (
    CloseSessionRequest,
    CreateSessionRequest,
    FeedbackRequest,
    RetrieveRequest,
)
from .session_service import (
    SessionAlreadyExists,
    SessionNotFound,
    SessionService,
    SessionVersionConflict,
)
from .session_store import SessionStoreError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionBody(StrictModel):
    session_id: str = Field(min_length=1, max_length=128)
    reference_image_id: str = Field(min_length=1)
    initial_query: list[float] = Field(min_length=1)
    expected_encoder_version: str = Field(min_length=1)
    expected_index_version: str = Field(min_length=1)


class RetrieveBody(StrictModel):
    top_k: int = Field(default=10, ge=1, le=100)


class FeedbackBody(StrictModel):
    event_id: str = Field(min_length=1, max_length=128)
    shown_image_ids: list[str] = Field(min_length=1)
    accepted_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    feedback_text_embedding: list[float] | None = None
    expected_round_id: int | None = Field(default=None, ge=0)


class CloseBody(StrictModel):
    reason: str = Field(default="user_closed", min_length=1, max_length=256)


class SessionModel(StrictModel):
    session_id: str
    round_id: int
    reference_image_id: str
    encoder_version: str
    index_version: str
    reranker_version: str
    prompt_version: str
    positive_image_ids: list[str] | tuple[str, ...]
    negative_image_ids: list[str] | tuple[str, ...]
    last_shown_image_ids: list[str] | tuple[str, ...]
    processed_event_count: int
    max_rounds: int
    status: str
    close_reason: str | None


class SearchResultModel(StrictModel):
    image_id: str
    score: float
    rank: int


class RetrievalResponseModel(StrictModel):
    session: SessionModel
    results: list[SearchResultModel]


class TransitionModel(StrictModel):
    event_id: str
    round_before: int
    round_after: int
    update_applied: bool
    proposed_cosine_drift: float
    reason: str
    accepted_ids: list[str] | tuple[str, ...]
    rejected_ids: list[str] | tuple[str, ...]
    cumulative_positive_ids: list[str] | tuple[str, ...]
    cumulative_negative_ids: list[str] | tuple[str, ...]
    status: str


class FeedbackResponseModel(StrictModel):
    session: SessionModel
    transition: TransitionModel


class HealthModel(StrictModel):
    status: str
    encoder_version: str
    index_version: str
    item_count: int


def _invoke(function: Callable[[], Any]) -> Any:
    try:
        return function()
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    except (SessionAlreadyExists, SessionVersionConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_app(service: SessionService) -> FastAPI:
    app = FastAPI(
        title="CLIP Interactive Retrieval API",
        version="0.1.0",
        description="Versioned multi-turn retrieval session API.",
    )

    @app.get("/health", response_model=HealthModel)
    def health() -> dict:
        return {
            "status": "ok",
            "encoder_version": service.index.manifest.encoder_version,
            "index_version": service.index.manifest.index_version,
            "item_count": service.index.manifest.item_count,
        }

    @app.post(
        "/v1/sessions",
        response_model=SessionModel,
        status_code=status.HTTP_201_CREATED,
    )
    def create_session(body: CreateSessionBody) -> dict:
        request = CreateSessionRequest.from_dict(body.model_dump())
        return _invoke(lambda: service.create_session(request))

    @app.get("/v1/sessions/{session_id}", response_model=SessionModel)
    def get_session(session_id: str) -> dict:
        return _invoke(lambda: service.get_session(session_id))

    @app.post(
        "/v1/sessions/{session_id}/retrieve",
        response_model=RetrievalResponseModel,
    )
    def retrieve(session_id: str, body: RetrieveBody) -> dict:
        request = RetrieveRequest.from_dict(body.model_dump())
        return _invoke(lambda: service.retrieve(session_id, request).to_dict())

    @app.post(
        "/v1/sessions/{session_id}/feedback",
        response_model=FeedbackResponseModel,
    )
    def feedback(session_id: str, body: FeedbackBody) -> dict:
        request = FeedbackRequest.from_dict(body.model_dump())
        return _invoke(lambda: service.submit_feedback(session_id, request).to_dict())

    @app.post("/v1/sessions/{session_id}/close", response_model=SessionModel)
    def close(session_id: str, body: CloseBody) -> dict:
        request = CloseSessionRequest.from_dict(body.model_dump())
        return _invoke(lambda: service.close_session(session_id, request))

    return app
