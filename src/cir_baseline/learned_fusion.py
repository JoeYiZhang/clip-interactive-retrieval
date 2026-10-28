"""Small trainable fusion heads over frozen CLIP embeddings (optional torch)."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def image_disjoint_split(
    references: Sequence[str], targets: Sequence[str], seed: int = 20261028
) -> tuple[np.ndarray, np.ndarray]:
    """Hash image-connected components into fit/dev/holdout (60/20/20).

    A reference or target image used in supervision cannot cross partitions.
    The frozen, shared retrieval gallery is not a supervised partition.
    Canonical component IDs make the split independent of annotation order.
    """
    if not references or len(references) != len(targets):
        raise ValueError("references and targets must be nonempty and aligned")
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            old = key
            key = parent[key]
            parent[old] = root
        return root

    for reference, target in zip(references, targets, strict=True):
        left, right = find(str(reference)), find(str(target))
        parent[max(left, right)] = min(left, right)
    groups = np.asarray([find(str(key)) for key in references], dtype=str)
    assignments = {}
    for group in sorted(set(groups)):
        fraction = int.from_bytes(
            hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big"
        ) / 2**64
        assignments[group] = "fit" if fraction < .6 else "dev" if fraction < .8 else "holdout"
    return np.asarray([assignments[group] for group in groups]), groups


class LearnedFusion(nn.Module):
    """Query-only scalar gate with an optional vector residual.

    Zero-initialized output heads start exactly at the weighted CLIP baseline.
    Neither target embeddings nor target IDs are inputs to forward().
    """

    def __init__(
        self, dimension: int = 512, hidden_dim: int = 128,
        image_weight: float = .3, residual: bool = True,
        dropout: float = .1, residual_scale: float = .2,
    ) -> None:
        super().__init__()
        if dimension < 1 or hidden_dim < 1 or not 0 < image_weight < 1:
            raise ValueError("invalid fusion dimensions or prior weight")
        self.config = dict(dimension=dimension, hidden_dim=hidden_dim,
                           image_weight=image_weight, residual=residual,
                           dropout=dropout, residual_scale=residual_scale)
        self.features = nn.Sequential(
            nn.LayerNorm(dimension * 4), nn.Linear(dimension * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(image_weight / (1 - image_weight)))
        self.residual = nn.Linear(hidden_dim, dimension) if residual else None
        if self.residual is not None:
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        if images.ndim != 2 or images.shape != texts.shape:
            raise ValueError("image/text tensors must be aligned matrices")
        images, texts = F.normalize(images, dim=-1), F.normalize(texts, dim=-1)
        features = self.features(torch.cat((images, texts, images * texts, images - texts), -1))
        alpha = self.gate(features).sigmoid()
        query = alpha * images + (1 - alpha) * texts
        if self.residual is not None:
            query = query + self.config["residual_scale"] * self.residual(features)
        return F.normalize(query, dim=-1)


def multi_positive_contrastive_loss(
    queries: torch.Tensor, candidates: torch.Tensor,
    target_ids: torch.Tensor, candidate_ids: torch.Tensor,
    temperature: float = .07,
) -> torch.Tensor:
    """One query may match multiple duplicate target rows in a batch.

    Remove duplicate candidates before softmax so duplicate image IDs do not
    overweight either positive or negative items. The trainer supplies only
    same-category target candidates for harder comparisons.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if queries.ndim != 2 or candidates.ndim != 2 or queries.shape[1] != candidates.shape[1]:
        raise ValueError("incompatible query/candidate matrices")
    if target_ids.shape != (queries.shape[0],) or candidate_ids.shape != (candidates.shape[0],):
        raise ValueError("image ID counts do not match vectors")
    # Stable first occurrence for each ID, implemented on device.
    order = torch.argsort(candidate_ids, stable=True)
    sorted_ids = candidate_ids[order]
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    unique = order[first]
    logits = F.normalize(queries, dim=-1) @ F.normalize(candidates[unique], dim=-1).T
    logits = logits / temperature
    positives = target_ids[:, None] == candidate_ids[unique][None, :]
    if not bool(positives.any(dim=1).all()):
        raise ValueError("each query requires a positive candidate")
    return (torch.logsumexp(logits, 1) - torch.logsumexp(
        logits.masked_fill(~positives, -torch.inf), 1
    )).mean()
