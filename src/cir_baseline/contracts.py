from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QuerySpec:
    reference_image_id: str
    modification_text: str
    encoder_version: str
    index_version: str
    image_weight: float = 0.5
    text_weight: float = 0.5


@dataclass(frozen=True)
class SearchResult:
    image_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class IndexManifest:
    index_version: str
    encoder_version: str
    embedding_dim: int
    metric: str = "inner_product_on_l2_normalized_vectors"
    normalized: bool = True
    item_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IndexManifest":
        return cls(**payload)


def ensure_compatible_versions(query: QuerySpec, manifest: IndexManifest) -> None:
    if query.encoder_version != manifest.encoder_version:
        raise ValueError(
            "encoder/index version mismatch: "
            f"query={query.encoder_version!r}, index={manifest.encoder_version!r}"
        )
    if query.index_version != manifest.index_version:
        raise ValueError(
            "requested index version is not loaded: "
            f"query={query.index_version!r}, loaded={manifest.index_version!r}"
        )

