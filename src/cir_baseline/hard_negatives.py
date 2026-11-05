"""Fit-only negative sampling and per-query candidate masking."""
from __future__ import annotations

import numpy as np


def fit_image_pool(partitions, references, targets, categories, category):
    """Reject supervised split leakage before constructing a category pool."""
    if not (len(partitions) == len(references) == len(targets) == len(categories)):
        raise ValueError("unaligned split metadata")
    fit, other, selected = set(), set(), set()
    for part, ref, target, cat in zip(partitions, references, targets, categories, strict=True):
        (fit if part == "fit" else other).update((int(ref), int(target)))
        if part == "fit" and cat == category:
            selected.update((int(ref), int(target)))
    if fit & other:
        raise ValueError("supervised images cross fit/evaluation splits")
    if not selected:
        raise ValueError("empty fit candidate pool")
    return np.asarray(sorted(selected), dtype=np.int64)


def positive_key(reference, captions):
    """Caption order is immaterial for the frozen symmetric text representation."""
    return int(reference), tuple(sorted(" ".join(text.split()) for text in captions))


def batch_candidate_masks(references, positives, batch_targets, extra_orders=None, extra_count=0):
    """Deduplicate candidates; extras are visible only to their own query.

    Including another query's extras in the softmax would confound the number
    of negatives with hardness. Every arm shares batch targets and positives.
    """
    n = len(references)
    if n == 0 or n != len(positives) or n != len(batch_targets) or extra_count < 0:
        raise ValueError("invalid batch metadata")
    if extra_count and (extra_orders is None or len(extra_orders) != n):
        raise ValueError("extra candidate orders required for every query")
    base = set(map(int, batch_targets))
    chosen, valid_rows, positive_rows = [], [], []
    for i, (ref, positive, target) in enumerate(zip(references, positives, batch_targets, strict=True)):
        positive = set(map(int, positive))
        if not positive or int(target) not in positive or int(ref) in positive:
            raise ValueError("invalid positive/reference relationship")
        extras = []
        excluded = base | positive | {int(ref)}
        if extra_count:
            for value in extra_orders[i]:
                value = int(value)
                if value not in excluded:
                    extras.append(value)
                    excluded.add(value)
                    if len(extras) == extra_count:
                        break
            if len(extras) != extra_count:
                raise ValueError("insufficient distinct extra negatives")
        chosen.append(extras)
        positive_rows.append(positive)
        valid_rows.append((base | positive | set(extras)) - {int(ref)})
    candidates = np.asarray(sorted(set.union(*valid_rows)), dtype=np.int64)
    lookup = {int(value): j for j, value in enumerate(candidates)}
    positive_mask = np.zeros((n, len(candidates)), dtype=bool)
    valid_mask = np.zeros_like(positive_mask)
    for i, (positive, valid) in enumerate(zip(positive_rows, valid_rows, strict=True)):
        positive_mask[i, [lookup[value] for value in positive]] = True
        valid_mask[i, [lookup[value] for value in valid]] = True
    return candidates, positive_mask, valid_mask, np.asarray(chosen, dtype=np.int64).reshape(n, extra_count)
