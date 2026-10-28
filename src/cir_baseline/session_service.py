from __future__ import annotations

from dataclasses import asdict
from threading import RLock
from typing import Mapping

import numpy as np

from .api_models import (
    CloseSessionRequest,
    CreateSessionRequest,
    FeedbackRequest,
    FeedbackResponse,
    RetrievalResponse,
    RetrieveRequest,
)
from .feedback import FeedbackWeights
from .interactive import InteractiveSession, SearchIndex
from .session_store import (
    SQLiteSessionStore,
    StoredSessionAlreadyExists,
    StoredSessionNotFound,
)


class SessionNotFound(KeyError):
    pass


class SessionAlreadyExists(ValueError):
    pass


class SessionVersionConflict(ValueError):
    pass


class SessionService:
    def __init__(
        self,
        *,
        index: SearchIndex,
        embedding_by_id: Mapping[str, np.ndarray],
        store: SQLiteSessionStore,
        weights: FeedbackWeights = FeedbackWeights(),
        max_rounds: int = 6,
    ) -> None:
        self.index = index
        self.embedding_by_id = embedding_by_id
        self.store = store
        self.weights = weights
        self.max_rounds = max_rounds
        self._lock = RLock()

    def _load(self, session_id: str) -> InteractiveSession:
        try:
            return self.store.load_session(
                session_id,
                index=self.index,
                embedding_by_id=self.embedding_by_id,
            )
        except StoredSessionNotFound as exc:
            raise SessionNotFound(session_id) from exc

    def create_session(self, request: CreateSessionRequest) -> dict:
        if request.expected_encoder_version != self.index.manifest.encoder_version:
            raise SessionVersionConflict("encoder version does not match loaded index")
        if request.expected_index_version != self.index.manifest.index_version:
            raise SessionVersionConflict("index version does not match loaded index")
        session = InteractiveSession(
            session_id=request.session_id,
            reference_image_id=request.reference_image_id,
            initial_query=np.asarray(request.initial_query, dtype=np.float32),
            encoder_version=request.expected_encoder_version,
            index=self.index,
            embedding_by_id=self.embedding_by_id,
            weights=self.weights,
            max_rounds=self.max_rounds,
        )
        try:
            self.store.create_session(session)
        except StoredSessionAlreadyExists as exc:
            raise SessionAlreadyExists(request.session_id) from exc
        return session.snapshot().to_dict()

    def get_session(self, session_id: str) -> dict:
        return self._load(session_id).snapshot().to_dict()

    def retrieve(
        self, session_id: str, request: RetrieveRequest
    ) -> RetrievalResponse:
        with self._lock:
            session = self._load(session_id)
            results = session.retrieve(request.top_k)
            self.store.record_exposure(session)
        return RetrievalResponse(
            session=session.snapshot().to_dict(),
            results=tuple(asdict(result) for result in results),
        )

    def submit_feedback(
        self, session_id: str, request: FeedbackRequest
    ) -> FeedbackResponse:
        with self._lock:
            session = self._load(session_id)
            if (
                request.expected_round_id is not None
                and request.expected_round_id != session.snapshot().round_id
                and not session.has_event(request.event_id)
            ):
                raise SessionVersionConflict(
                    "session round changed; retrieve the latest state before feedback"
                )
            feedback_vector = (
                None
                if request.feedback_text_embedding is None
                else np.asarray(request.feedback_text_embedding, dtype=np.float32)
            )
            transition = session.submit_feedback(
                event_id=request.event_id,
                shown_image_ids=request.shown_image_ids,
                accepted_ids=request.accepted_ids,
                rejected_ids=request.rejected_ids,
                feedback_text_embedding=feedback_vector,
            )
            self.store.record_feedback(
                session,
                shown_image_ids=request.shown_image_ids,
                accepted_ids=request.accepted_ids,
                rejected_ids=request.rejected_ids,
                feedback_text_embedding=feedback_vector,
                transition=transition,
            )
        return FeedbackResponse(
            session=session.snapshot().to_dict(),
            transition=transition.to_dict(),
        )

    def close_session(self, session_id: str, request: CloseSessionRequest) -> dict:
        with self._lock:
            session = self._load(session_id)
            snapshot = session.close(request.reason)
            self.store.record_close(session)
        return snapshot.to_dict()
