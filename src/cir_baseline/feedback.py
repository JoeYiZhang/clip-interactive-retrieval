from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fusion import l2_normalize, l2_normalize_rows


@dataclass(frozen=True)
class FeedbackWeights:
    anchor_weight: float = 0.35
    previous_weight: float = 1.0
    positive_weight: float = 0.75
    negative_weight: float = 0.25
    text_weight: float = 0.50
    update_strength: float = 0.50
    max_cosine_drift: float = 0.35


@dataclass(frozen=True)
class QueryUpdate:
    vector: np.ndarray
    proposed_cosine_drift: float
    applied: bool
    reason: str


def _mean_or_zero(vectors: Sequence[np.ndarray], dimension: int) -> np.ndarray:
    if not vectors:
        return np.zeros(dimension, dtype=np.float32)
    matrix = l2_normalize_rows(np.stack(vectors))
    return matrix.mean(axis=0)


def update_query(
    initial_query: np.ndarray,
    previous_query: np.ndarray,
    positive_embeddings: Sequence[np.ndarray] = (),
    negative_embeddings: Sequence[np.ndarray] = (),
    feedback_text_embedding: np.ndarray | None = None,
    weights: FeedbackWeights = FeedbackWeights(),
) -> QueryUpdate:
    q0 = l2_normalize(initial_query)
    previous = l2_normalize(previous_query)
    if q0.shape != previous.shape:
        raise ValueError("initial and previous query dimensions differ")
    if not 0 <= weights.update_strength <= 1:
        raise ValueError("update_strength must be between zero and one")
    if not 0 <= weights.max_cosine_drift <= 2:
        raise ValueError("max_cosine_drift must be between zero and two")

    dimension = q0.shape[0]
    positive = _mean_or_zero(positive_embeddings, dimension)
    negative = _mean_or_zero(negative_embeddings, dimension)
    text = (
        np.zeros(dimension, dtype=np.float32)
        if feedback_text_embedding is None
        else l2_normalize(feedback_text_embedding)
    )

    feedback = (
        weights.anchor_weight * q0
        + weights.previous_weight * previous
        + weights.positive_weight * positive
        - weights.negative_weight * negative
        + weights.text_weight * text
    )
    feedback = l2_normalize(feedback)
    candidate = l2_normalize(
        (1 - weights.update_strength) * q0
        + weights.update_strength * feedback
    )
    drift = float(1 - np.clip(np.dot(previous, candidate), -1.0, 1.0))
    if drift > weights.max_cosine_drift:
        return QueryUpdate(
            vector=previous,
            proposed_cosine_drift=drift,
            applied=False,
            reason="query drift exceeded the configured guard; previous query retained",
        )
    return QueryUpdate(
        vector=candidate,
        proposed_cosine_drift=drift,
        applied=True,
        reason="feedback update accepted",
    )

