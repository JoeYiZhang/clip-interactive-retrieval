from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import IndexManifest, SearchResult
from .fusion import l2_normalize, l2_normalize_rows


class NumpyFlatIPIndex:
    """Exact inner-product reference index for normalized embeddings."""

    def __init__(
        self,
        image_ids: Sequence[str],
        embeddings: np.ndarray,
        manifest: IndexManifest,
    ) -> None:
        ids = [str(value) for value in image_ids]
        matrix = l2_normalize_rows(embeddings)
        if len(ids) != matrix.shape[0]:
            raise ValueError("image id count does not match embedding row count")
        if len(set(ids)) != len(ids):
            raise ValueError("image ids must be unique")
        if matrix.shape[1] != manifest.embedding_dim:
            raise ValueError("embedding dimension does not match manifest")
        if manifest.item_count not in (0, len(ids)):
            raise ValueError("item count does not match manifest")
        self.image_ids = ids
        self.embeddings = matrix
        self.manifest = IndexManifest(
            index_version=manifest.index_version,
            encoder_version=manifest.encoder_version,
            embedding_dim=manifest.embedding_dim,
            metric=manifest.metric,
            normalized=True,
            item_count=len(ids),
        )

    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = l2_normalize(query)
        if vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query dimension does not match index")
        count = min(top_k, len(self.image_ids))
        scores = self.embeddings @ vector
        order = np.argsort(-scores, kind="stable")[:count]
        return [
            SearchResult(
                image_id=self.image_ids[int(idx)],
                score=float(scores[int(idx)]),
                rank=rank,
            )
            for rank, idx in enumerate(order, start=1)
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            image_ids=np.asarray(self.image_ids, dtype=str),
            embeddings=self.embeddings,
            manifest_json=json.dumps(self.manifest.to_dict(), ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NumpyFlatIPIndex":
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["image_ids"].astype(str).tolist()
            embeddings = payload["embeddings"].astype(np.float32)
            manifest_raw = payload["manifest_json"].item()
        manifest = IndexManifest.from_dict(json.loads(str(manifest_raw)))
        return cls(ids, embeddings, manifest)


class FaissFlatIPIndex:
    """Optional exact FAISS implementation with the same result contract."""

    def __init__(
        self,
        image_ids: Sequence[str],
        embeddings: np.ndarray,
        manifest: IndexManifest,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss is unavailable or its native DLL failed to load. "
                "Install the 'ml' extra and verify the Windows runtime."
            ) from exc
        self._faiss = faiss
        self.image_ids = [str(value) for value in image_ids]
        self.embeddings = l2_normalize_rows(embeddings)
        if len(self.image_ids) != self.embeddings.shape[0]:
            raise ValueError("image id count does not match embedding row count")
        if self.embeddings.shape[1] != manifest.embedding_dim:
            raise ValueError("embedding dimension does not match manifest")
        self.manifest = IndexManifest(
            index_version=manifest.index_version,
            encoder_version=manifest.encoder_version,
            embedding_dim=manifest.embedding_dim,
            metric=manifest.metric,
            normalized=True,
            item_count=len(self.image_ids),
        )
        self.index = faiss.IndexFlatIP(manifest.embedding_dim)
        self.index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))

    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.ascontiguousarray(l2_normalize(query)[None, :], dtype=np.float32)
        count = min(top_k, len(self.image_ids))
        scores, indices = self.index.search(vector, count)
        return [
            SearchResult(
                image_id=self.image_ids[int(idx)],
                score=float(score),
                rank=rank,
            )
            for rank, (score, idx) in enumerate(
                zip(scores[0], indices[0], strict=True), start=1
            )
        ]

    def search_many(
        self, queries: np.ndarray, top_k: int
    ) -> list[list[SearchResult]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        matrix = l2_normalize_rows(queries)
        if matrix.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query dimension does not match index")
        count = min(top_k, len(self.image_ids))
        scores, indices = self.index.search(
            np.ascontiguousarray(matrix, dtype=np.float32), count
        )
        return [
            [
                SearchResult(
                    image_id=self.image_ids[int(item_index)],
                    score=float(score),
                    rank=rank,
                )
                for rank, (score, item_index) in enumerate(
                    zip(row_scores, row_indices, strict=True), start=1
                )
            ]
            for row_scores, row_indices in zip(scores, indices, strict=True)
        ]

