from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Callable, Protocol, Sequence

import numpy as np

from .adaptive import confidence_bands, has_identity_intent
from .api_models import (
    CloseSessionRequest,
    CreateSessionRequest,
    FeedbackRequest,
    FeedbackResponse,
    RetrievalResponse,
    RetrieveRequest,
)
from .feedback import FeedbackWeights
from .fusion import l2_normalize
from .index_registry import IndexRegistry, RegisteredIndex
from .session_service import SessionNotFound, SessionService, SessionVersionConflict
from .session_store import SQLiteSessionStore, StoredSessionNotFound


class TextEncoder(Protocol):
    version: str

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class LazyTextEncoder:
    """Load the heavyweight text encoder only when free-form text is submitted."""

    def __init__(
        self,
        *,
        expected_version: str,
        factory: Callable[[], TextEncoder],
    ) -> None:
        self.version = str(expected_version)
        self._factory = factory
        self._encoder: TextEncoder | None = None
        self._lock = RLock()

    @property
    def loaded(self) -> bool:
        return self._encoder is not None

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            if self._encoder is None:
                encoder = self._factory()
                if encoder.version != self.version:
                    raise ValueError(
                        "loaded text encoder version does not match the index registry"
                    )
                self._encoder = encoder
            return self._encoder.encode_texts(texts)


@dataclass(frozen=True)
class CompositionResult:
    vector: np.ndarray
    index_key: str
    reference_image_id: str
    policy_version: str
    gate_group: str
    image_weight: float
    text_confidence: float | None
    identity_intent: bool

    def metadata(self) -> dict:
        payload = asdict(self)
        payload.pop("vector")
        return payload


class FashionIQQueryComposer:
    """Deploy the FashionIQ A2 gate frozen from train-only selection.

    The online demo accepts one free-form text string. The scientific FashionIQ
    evaluation averages forward/reverse two-caption joins; therefore demo use is
    an engineering validation and must not be reported as a benchmark score.
    """

    POLICY_VERSION = "fashioniq-a2-train-frozen-v1"
    CONFIDENCE_THRESHOLDS = {
        "dress": (0.005648404359817505, 0.009234100580215454),
        "shirt": (0.005581816037495931, 0.009576062361399329),
        "toptee": (0.006196260452270508, 0.01031455397605896),
    }
    IMAGE_WEIGHTS = {
        "dress:identity": 0.5,
        "dress:confidence_0": 0.3,
        "dress:confidence_1": 0.3,
        "dress:confidence_2": 0.3,
        "shirt:identity": 0.9,
        "shirt:confidence_0": 0.4,
        "shirt:confidence_1": 0.3,
        "shirt:confidence_2": 0.2,
        "toptee:identity": 0.7,
        "toptee:confidence_0": 0.3,
        "toptee:confidence_1": 0.3,
        "toptee:confidence_2": 0.2,
    }

    def __init__(self, text_encoder: TextEncoder) -> None:
        self.text_encoder = text_encoder

    def _encode_text(self, text: str, dimension: int) -> np.ndarray:
        matrix = np.asarray(self.text_encoder.encode_texts([text]), dtype=np.float32)
        if matrix.shape != (1, dimension):
            raise ValueError("text encoder returned an incompatible embedding shape")
        return l2_normalize(matrix[0])

    @staticmethod
    def _confidence(
        entry: RegisteredIndex,
        text_vector: np.ndarray,
        reference_image_id: str,
    ) -> float:
        hits = [
            hit
            for hit in entry.index.search(text_vector, min(11, len(entry.index.image_ids)))
            if hit.image_id != reference_image_id
        ][:10]
        if len(hits) < 2:
            return 0.0
        return float(hits[0].score - np.mean([hit.score for hit in hits[1:]]))

    def compose(
        self,
        entry: RegisteredIndex,
        *,
        reference_image_id: str,
        modification_text: str,
    ) -> CompositionResult:
        reference = entry.embedding_by_id.get(str(reference_image_id))
        if reference is None:
            raise ValueError("reference image is not present in the selected index")
        text = str(modification_text).strip()
        if not text:
            return CompositionResult(
                vector=l2_normalize(reference),
                index_key=entry.key,
                reference_image_id=str(reference_image_id),
                policy_version=self.POLICY_VERSION,
                gate_group=f"{entry.category}:image_only",
                image_weight=1.0,
                text_confidence=None,
                identity_intent=False,
            )
        if self.text_encoder.version != entry.index.manifest.encoder_version:
            raise ValueError("text encoder version does not match selected index")
        text_vector = self._encode_text(text, entry.index.manifest.embedding_dim)
        identity = has_identity_intent(text)
        confidence = self._confidence(entry, text_vector, str(reference_image_id))
        if identity:
            group = f"{entry.category}:identity"
        else:
            thresholds = self.CONFIDENCE_THRESHOLDS.get(entry.category)
            if thresholds is None:
                raise ValueError(f"no frozen gate for category: {entry.category}")
            band = int(confidence_bands([confidence], thresholds)[0])
            group = f"{entry.category}:confidence_{band}"
        image_weight = self.IMAGE_WEIGHTS[group]
        vector = l2_normalize(
            image_weight * l2_normalize(reference)
            + (1.0 - image_weight) * text_vector
        )
        return CompositionResult(
            vector=vector,
            index_key=entry.key,
            reference_image_id=str(reference_image_id),
            policy_version=self.POLICY_VERSION,
            gate_group=group,
            image_weight=image_weight,
            text_confidence=confidence,
            identity_intent=identity,
        )


class MultiIndexSessionService:
    """Resolve persisted sessions to immutable registered index versions."""

    def __init__(
        self,
        *,
        registry: IndexRegistry,
        store: SQLiteSessionStore,
        composer: FashionIQQueryComposer,
        weights: FeedbackWeights = FeedbackWeights(),
        max_rounds: int = 6,
    ) -> None:
        self.registry = registry
        self.store = store
        self.composer = composer
        self.weights = weights
        self.max_rounds = max_rounds
        self._lock = RLock()

    def _service(self, entry: RegisteredIndex) -> SessionService:
        return SessionService(
            index=entry.index,
            embedding_by_id=entry.embedding_by_id,
            store=self.store,
            weights=self.weights,
            max_rounds=self.max_rounds,
        )

    def _entry_for_session(self, session_id: str) -> RegisteredIndex:
        try:
            version = self.store.session_index_version(session_id)
        except StoredSessionNotFound as exc:
            raise SessionNotFound(session_id) from exc
        try:
            return self.registry.by_version(version)
        except KeyError as exc:
            raise SessionVersionConflict(str(exc)) from exc

    def create_composed_session(
        self,
        *,
        session_id: str,
        index_key: str,
        reference_image_id: str,
        modification_text: str,
    ) -> dict:
        with self._lock:
            entry = self.registry.get(index_key)
            composition = self.composer.compose(
                entry,
                reference_image_id=reference_image_id,
                modification_text=modification_text,
            )
            session = self._service(entry).create_session(
                CreateSessionRequest(
                    session_id=str(session_id),
                    reference_image_id=str(reference_image_id),
                    initial_query=tuple(float(value) for value in composition.vector),
                    expected_encoder_version=entry.index.manifest.encoder_version,
                    expected_index_version=entry.index.manifest.index_version,
                )
            )
        return {
            "session": {**session, "index_key": entry.key},
            "composition": composition.metadata(),
        }

    def get_session(self, session_id: str) -> dict:
        with self._lock:
            entry = self._entry_for_session(session_id)
            snapshot = self._service(entry).get_session(session_id)
        return {**snapshot, "index_key": entry.key}

    def retrieve(self, session_id: str, request: RetrieveRequest) -> RetrievalResponse:
        with self._lock:
            entry = self._entry_for_session(session_id)
            response = self._service(entry).retrieve(session_id, request)
        return RetrievalResponse(
            session={**response.session, "index_key": entry.key},
            results=response.results,
        )

    def submit_feedback(
        self,
        session_id: str,
        request: FeedbackRequest,
        *,
        feedback_text: str = "",
    ) -> FeedbackResponse:
        with self._lock:
            entry = self._entry_for_session(session_id)
            text = str(feedback_text).strip()
            if text and request.feedback_text_embedding is not None:
                raise ValueError("submit feedback_text or an embedding, not both")
            if text:
                vector = self.composer._encode_text(
                    text, entry.index.manifest.embedding_dim
                )
                request = replace(
                    request,
                    feedback_text_embedding=tuple(float(value) for value in vector),
                )
            response = self._service(entry).submit_feedback(session_id, request)
        return FeedbackResponse(
            session={**response.session, "index_key": entry.key},
            transition=response.transition,
        )

    def close_session(self, session_id: str, request: CloseSessionRequest) -> dict:
        with self._lock:
            entry = self._entry_for_session(session_id)
            snapshot = self._service(entry).close_session(session_id, request)
        return {**snapshot, "index_key": entry.key}

    def image_path(self, index_key: str, image_id: str):
        return self.registry.get(index_key).image_path(image_id)

    def reference_page(self, index_key: str, *, offset: int, limit: int) -> dict:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit <= 0 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        entry = self.registry.get(index_key)
        selected = entry.index.image_ids[offset : offset + limit]
        return {
            "index_key": entry.key,
            "offset": offset,
            "limit": limit,
            "total": len(entry.index.image_ids),
            "items": [
                {
                    "image_id": image_id,
                    "image_url": f"/v1/demo/images/{entry.key}/{image_id}",
                }
                for image_id in selected
            ],
        }
