from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np


_IDENTITY_INTENT = re.compile(
    r"\b(?:"
    r"identical|"
    r"exact(?:ly)?\s+(?:the\s+)?same|"
    r"same\s+(?:exact\s+)?product|"
    r"provided\s+product|"
    r"looks?\s+(?:exactly\s+)?(?:the\s+)?same|"
    r"no\s+change|unchanged"
    r")\b",
    flags=re.IGNORECASE,
)


def has_identity_intent(text: str) -> bool:
    """Return whether a modification explicitly asks for the same item.

    The rule deliberately avoids matching broad phrases such as ``same color``;
    those describe one retained attribute rather than identity of the product.
    """

    return _IDENTITY_INTENT.search(str(text)) is not None


def quantile_thresholds(
    values: Sequence[float] | np.ndarray,
    quantiles: Sequence[float] = (1.0 / 3.0, 2.0 / 3.0),
) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain NaN or infinity")
    requested = np.asarray(quantiles, dtype=np.float64)
    if requested.ndim != 1 or np.any((requested <= 0) | (requested >= 1)):
        raise ValueError("quantiles must be a 1-D sequence strictly inside (0, 1)")
    if np.any(np.diff(requested) <= 0):
        raise ValueError("quantiles must be strictly increasing")
    return tuple(float(value) for value in np.quantile(array, requested))


def confidence_bands(
    values: Sequence[float] | np.ndarray,
    thresholds: Sequence[float],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    edges = np.asarray(thresholds, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be 1-D")
    if edges.ndim != 1 or np.any(np.diff(edges) < 0):
        raise ValueError("thresholds must be a non-decreasing 1-D sequence")
    if not np.all(np.isfinite(array)) or not np.all(np.isfinite(edges)):
        raise ValueError("values and thresholds must be finite")
    return np.searchsorted(edges, array, side="right").astype(np.int64)


def select_weight_from_ranks(
    weights: Sequence[float],
    ranks_by_weight: np.ndarray,
    query_indices: Sequence[int] | np.ndarray,
    prior_weight: float = 0.2,
) -> tuple[float, dict[str, float | int]]:
    """Select a weight by train R@10, then R@50 and a fixed prior.

    Ranks greater than 50 represent a miss. The caller controls which split is
    supplied; production experiments must call this with training ranks only.
    """

    candidates = np.asarray(weights, dtype=np.float64)
    ranks = np.asarray(ranks_by_weight)
    indices = np.asarray(query_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("weights must be a non-empty 1-D sequence")
    if ranks.ndim != 2 or ranks.shape[0] != candidates.size:
        raise ValueError("ranks_by_weight must have shape (weight_count, query_count)")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("query_indices must be a non-empty 1-D sequence")
    if np.any(indices < 0) or np.any(indices >= ranks.shape[1]):
        raise ValueError("query index is out of range")

    best_index = max(
        range(candidates.size),
        key=lambda offset: (
            float(np.mean(ranks[offset, indices] <= 10)),
            float(np.mean(ranks[offset, indices] <= 50)),
            -abs(float(candidates[offset]) - prior_weight),
            -float(candidates[offset]),
        ),
    )
    selected = ranks[best_index, indices]
    return float(candidates[best_index]), {
        "query_count": int(indices.size),
        "recall@10": float(np.mean(selected <= 10)),
        "recall@50": float(np.mean(selected <= 50)),
    }
