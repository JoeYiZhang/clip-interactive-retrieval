from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .contracts import IndexManifest, QuerySpec, SearchResult, ensure_compatible_versions
from .feedback import FeedbackWeights, update_query
from .fusion import l2_normalize


class SearchIndex(Protocol):
    image_ids: list[str]
    manifest: IndexManifest

    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]: ...


@dataclass(frozen=True)
class SessionTransition:
    event_id: str
    round_before: int
    round_after: int
    update_applied: bool
    proposed_cosine_drift: float
    reason: str
    accepted_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    cumulative_positive_ids: tuple[str, ...]
    cumulative_negative_ids: tuple[str, ...]
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    round_id: int
    reference_image_id: str
    encoder_version: str
    index_version: str
    reranker_version: str
    prompt_version: str
    positive_image_ids: tuple[str, ...]
    negative_image_ids: tuple[str, ...]
    last_shown_image_ids: tuple[str, ...]
    processed_event_count: int
    max_rounds: int
    status: str
    close_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class InteractiveSession:
    """In-memory multi-turn query state machine.

    The caller owns persistence and authentication. This core enforces exposure
    consistency, explicit-only negatives, idempotent event IDs, model/index
    compatibility, maximum rounds and query-drift fallback.
    """

    def __init__(
        self,
        *,
        session_id: str,
        reference_image_id: str,
        initial_query: np.ndarray,
        encoder_version: str,
        index: SearchIndex,
        embedding_by_id: Mapping[str, np.ndarray],
        weights: FeedbackWeights = FeedbackWeights(),
        max_rounds: int = 6,
        reranker_version: str = "none",
        prompt_version: str = "none",
    ) -> None:
        if not str(session_id).strip():
            raise ValueError("session_id must be non-empty and de-identified")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if reference_image_id not in index.image_ids:
            raise ValueError("reference image is not present in the loaded index")
        query = QuerySpec(
            reference_image_id=reference_image_id,
            modification_text="",
            encoder_version=encoder_version,
            index_version=index.manifest.index_version,
        )
        ensure_compatible_versions(query, index.manifest)
        initial = l2_normalize(initial_query)
        if initial.shape[0] != index.manifest.embedding_dim:
            raise ValueError("initial query dimension does not match index")

        self.session_id = str(session_id)
        self.reference_image_id = str(reference_image_id)
        self.encoder_version = str(encoder_version)
        self.index = index
        self.embedding_by_id = embedding_by_id
        self.weights = weights
        self.max_rounds = int(max_rounds)
        self.reranker_version = str(reranker_version)
        self.prompt_version = str(prompt_version)
        self._initial_query = initial
        self._current_query = initial.copy()
        self._round_id = 0
        self._positive_ids: set[str] = set()
        self._negative_ids: set[str] = set()
        self._last_shown: tuple[str, ...] = ()
        self._last_shown_round: int | None = None
        self._events: dict[str, tuple[str, SessionTransition]] = {}
        self._status = "active"
        self._close_reason: str | None = None

    @property
    def current_query(self) -> np.ndarray:
        return self._current_query.copy()

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self.session_id,
            round_id=self._round_id,
            reference_image_id=self.reference_image_id,
            encoder_version=self.encoder_version,
            index_version=self.index.manifest.index_version,
            reranker_version=self.reranker_version,
            prompt_version=self.prompt_version,
            positive_image_ids=tuple(sorted(self._positive_ids)),
            negative_image_ids=tuple(sorted(self._negative_ids)),
            last_shown_image_ids=self._last_shown,
            processed_event_count=len(self._events),
            max_rounds=self.max_rounds,
            status=self._status,
            close_reason=self._close_reason,
        )

    def retrieve(self, top_k: int = 10) -> tuple[SearchResult, ...]:
        if self._status == "closed":
            raise RuntimeError("session is closed")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        requested = min(top_k + 1, self.index.manifest.item_count)
        hits = self.index.search(self._current_query, requested)
        filtered = tuple(
            hit for hit in hits if hit.image_id != self.reference_image_id
        )[:top_k]
        self._last_shown = tuple(hit.image_id for hit in filtered)
        self._last_shown_round = self._round_id
        return filtered

    def _embedding(self, image_id: str) -> np.ndarray:
        vector = self.embedding_by_id.get(image_id)
        if vector is None:
            raise ValueError(f"feedback image has no registered embedding: {image_id}")
        normalized = l2_normalize(vector)
        if normalized.shape[0] != self.index.manifest.embedding_dim:
            raise ValueError(f"feedback embedding dimension mismatch: {image_id}")
        return normalized

    @staticmethod
    def _event_digest(
        shown_image_ids: Sequence[str],
        accepted_ids: Sequence[str],
        rejected_ids: Sequence[str],
        feedback_text_embedding: np.ndarray | None,
    ) -> str:
        payload = {
            "shown": list(shown_image_ids),
            "accepted": sorted(str(value) for value in accepted_ids),
            "rejected": sorted(str(value) for value in rejected_ids),
            "text_embedding_sha256": None,
        }
        if feedback_text_embedding is not None:
            values = np.ascontiguousarray(
                np.asarray(feedback_text_embedding, dtype=np.float32)
            )
            payload["text_embedding_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def submit_feedback(
        self,
        *,
        event_id: str,
        shown_image_ids: Sequence[str],
        accepted_ids: Sequence[str] = (),
        rejected_ids: Sequence[str] = (),
        feedback_text_embedding: np.ndarray | None = None,
    ) -> SessionTransition:
        if not str(event_id).strip():
            raise ValueError("event_id must be non-empty")
        shown = tuple(str(value) for value in shown_image_ids)
        accepted = tuple(sorted(set(str(value) for value in accepted_ids)))
        rejected = tuple(sorted(set(str(value) for value in rejected_ids)))
        digest = self._event_digest(
            shown, accepted, rejected, feedback_text_embedding
        )
        existing = self._events.get(str(event_id))
        if existing is not None:
            if existing[0] != digest:
                raise ValueError("event_id was already used with different feedback")
            return existing[1]

        if self._status != "active":
            raise RuntimeError(f"session is not active: {self._status}")
        if self._last_shown_round != self._round_id or not self._last_shown:
            raise RuntimeError("retrieve must be called before feedback in each round")
        if shown != self._last_shown:
            raise ValueError("shown_image_ids do not match the recorded exposure")
        overlap = set(accepted) & set(rejected)
        if overlap:
            raise ValueError(f"same image cannot be accepted and rejected: {sorted(overlap)}")
        exposed = set(shown)
        if not set(accepted).issubset(exposed) or not set(rejected).issubset(exposed):
            raise ValueError("accepted/rejected images must be explicitly shown")
        if not accepted and not rejected and feedback_text_embedding is None:
            raise ValueError("feedback event contains no explicit signal")
        if feedback_text_embedding is not None:
            feedback_text_embedding = l2_normalize(feedback_text_embedding)
            if feedback_text_embedding.shape[0] != self.index.manifest.embedding_dim:
                raise ValueError("feedback text embedding dimension does not match index")

        self._negative_ids.difference_update(accepted)
        self._positive_ids.update(accepted)
        self._positive_ids.difference_update(rejected)
        self._negative_ids.update(rejected)
        update = update_query(
            self._initial_query,
            self._current_query,
            positive_embeddings=[
                self._embedding(image_id) for image_id in sorted(self._positive_ids)
            ],
            negative_embeddings=[
                self._embedding(image_id) for image_id in sorted(self._negative_ids)
            ],
            feedback_text_embedding=feedback_text_embedding,
            weights=self.weights,
        )
        round_before = self._round_id
        self._round_id += 1
        self._current_query = update.vector.copy()
        self._last_shown = ()
        self._last_shown_round = None
        if self._round_id >= self.max_rounds:
            self._status = "completed"
            self._close_reason = "max_rounds_reached"
        transition = SessionTransition(
            event_id=str(event_id),
            round_before=round_before,
            round_after=self._round_id,
            update_applied=update.applied,
            proposed_cosine_drift=update.proposed_cosine_drift,
            reason=update.reason,
            accepted_ids=accepted,
            rejected_ids=rejected,
            cumulative_positive_ids=tuple(sorted(self._positive_ids)),
            cumulative_negative_ids=tuple(sorted(self._negative_ids)),
            status=self._status,
        )
        self._events[str(event_id)] = (digest, transition)
        return transition

    def close(self, reason: str = "user_closed") -> SessionSnapshot:
        if self._status == "active":
            self._status = "closed"
            self._close_reason = str(reason) or "user_closed"
            self._last_shown = ()
            self._last_shown_round = None
        return self.snapshot()
