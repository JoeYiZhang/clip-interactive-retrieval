from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(
    ranked_ids: Sequence[Sequence[str]],
    relevant_ids: Sequence[set[str]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ranked_ids) != len(relevant_ids):
        raise ValueError("ranked and relevant query counts differ")
    if not ranked_ids:
        raise ValueError("at least one query is required")
    hits = 0
    for ranking, relevant in zip(ranked_ids, relevant_ids, strict=True):
        if not relevant:
            raise ValueError("each query must have at least one relevant item")
        hits += bool(set(ranking[:k]) & relevant)
    return hits / len(ranked_ids)


def ann_recall_at_k(
    approximate_ranked_ids: Sequence[Sequence[str]],
    exact_ranked_ids: Sequence[Sequence[str]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(approximate_ranked_ids) != len(exact_ranked_ids):
        raise ValueError("approximate and exact query counts differ")
    if not exact_ranked_ids:
        raise ValueError("at least one query is required")
    total = 0.0
    for approximate, exact in zip(
        approximate_ranked_ids, exact_ranked_ids, strict=True
    ):
        exact_top = set(exact[:k])
        if not exact_top:
            raise ValueError("exact ranking is empty")
        total += len(set(approximate[:k]) & exact_top) / len(exact_top)
    return total / len(exact_ranked_ids)

