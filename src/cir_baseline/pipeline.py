from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .contracts import QuerySpec, SearchResult, ensure_compatible_versions
from .fusion import fuse_weighted


class Encoder(Protocol):
    version: str

    def encode_texts(self, texts: list[str]): ...

    def encode_images(self, image_paths: list[str | Path]): ...


class RetrievalPipeline:
    def __init__(self, encoder: Encoder, index) -> None:
        self.encoder = encoder
        self.index = index

    def search(
        self,
        reference_image: str | Path,
        modification_text: str,
        top_k: int,
        image_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> list[SearchResult]:
        query = QuerySpec(
            reference_image_id=str(reference_image),
            modification_text=modification_text,
            encoder_version=self.encoder.version,
            index_version=self.index.manifest.index_version,
            image_weight=image_weight,
            text_weight=text_weight,
        )
        ensure_compatible_versions(query, self.index.manifest)
        image_vector = self.encoder.encode_images([reference_image])[0]
        text_vector = self.encoder.encode_texts([modification_text])[0]
        fused = fuse_weighted(
            image_vector,
            text_vector,
            image_weight=image_weight,
            text_weight=text_weight,
        )
        return self.index.search(fused, top_k=top_k)

