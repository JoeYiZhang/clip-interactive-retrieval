from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import IndexManifest
from .datasets import load_fashioniq
from .index import NumpyFlatIPIndex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RegisteredIndex:
    key: str
    dataset: str
    split: str
    category: str
    index: NumpyFlatIPIndex
    embedding_by_id: Mapping[str, np.ndarray]
    image_path_by_id: Mapping[str, Path]
    cache_path: Path
    cache_sha256: str

    def image_path(self, image_id: str) -> Path:
        path = self.image_path_by_id.get(str(image_id))
        if path is None:
            raise KeyError(f"image is not registered in {self.key}: {image_id}")
        if not path.is_file():
            raise FileNotFoundError(f"registered image is unavailable: {path}")
        return path

    def summary(self) -> dict:
        return {
            "key": self.key,
            "dataset": self.dataset,
            "split": self.split,
            "category": self.category,
            "item_count": self.index.manifest.item_count,
            "embedding_dim": self.index.manifest.embedding_dim,
            "encoder_version": self.index.manifest.encoder_version,
            "index_version": self.index.manifest.index_version,
            "cache_path": str(self.cache_path.resolve()),
            "cache_sha256": self.cache_sha256,
        }


class IndexRegistry:
    """Versioned, immutable collection of deployable retrieval indexes."""

    def __init__(self, entries: Sequence[RegisteredIndex]) -> None:
        if not entries:
            raise ValueError("index registry must contain at least one entry")
        self._by_key: dict[str, RegisteredIndex] = {}
        self._by_version: dict[str, RegisteredIndex] = {}
        for entry in entries:
            if entry.key in self._by_key:
                raise ValueError(f"duplicate registry key: {entry.key}")
            version = entry.index.manifest.index_version
            if version in self._by_version:
                raise ValueError(f"duplicate index version: {version}")
            self._by_key[entry.key] = entry
            self._by_version[version] = entry

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_key))

    def get(self, key: str) -> RegisteredIndex:
        try:
            return self._by_key[str(key)]
        except KeyError as exc:
            raise KeyError(f"unknown index key: {key}") from exc

    def by_version(self, version: str) -> RegisteredIndex:
        try:
            return self._by_version[str(version)]
        except KeyError as exc:
            raise KeyError(f"index version is not loaded: {version}") from exc

    def summaries(self) -> list[dict]:
        return [self._by_key[key].summary() for key in self.keys]

    @classmethod
    def from_fashioniq_cache(
        cls,
        *,
        data_root: str | Path,
        cache_path: str | Path,
        split: str = "val",
        categories: Sequence[str] = ("dress", "shirt", "toptee"),
    ) -> "IndexRegistry":
        root = Path(data_root)
        cache = Path(cache_path)
        selected = tuple(str(value) for value in categories)
        bundle = load_fashioniq(root, split=split, categories=selected)
        with np.load(cache, allow_pickle=False) as payload:
            required = {"image_ids", "image_paths", "embeddings", "encoder_version"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"embedding cache is missing fields: {sorted(missing)}")
            image_ids = payload["image_ids"].astype(str).tolist()
            image_paths = [Path(value) for value in payload["image_paths"].astype(str)]
            embeddings = payload["embeddings"].astype(np.float32)
            encoder_version = str(payload["encoder_version"].item())

        expected_ids = [item.image_id for item in bundle.gallery]
        expected_paths = [item.path.resolve() for item in bundle.gallery]
        actual_paths = [path.resolve() for path in image_paths]
        if image_ids != expected_ids:
            raise ValueError("embedding cache image IDs do not match the dataset bundle")
        if actual_paths != expected_paths:
            raise ValueError("embedding cache image paths do not match the dataset bundle")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(image_ids):
            raise ValueError("embedding cache matrix has an invalid shape")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("embedding cache contains NaN or infinity")

        cache_digest = _sha256(cache)
        position_by_id = {image_id: offset for offset, image_id in enumerate(image_ids)}
        bundle_by_id = {item.image_id: item for item in bundle.gallery}
        entries: list[RegisteredIndex] = []
        for category in selected:
            ids = [
                image_id
                for image_id in image_ids
                if category in bundle_by_id[image_id].categories
            ]
            if not ids:
                raise ValueError(f"FashionIQ category has no gallery items: {category}")
            positions = [position_by_id[image_id] for image_id in ids]
            vectors = embeddings[positions]
            version = f"fashioniq-{split}-{category}-flatip-{cache_digest[:12]}"
            manifest = IndexManifest(
                index_version=version,
                encoder_version=encoder_version,
                embedding_dim=int(vectors.shape[1]),
            )
            index = NumpyFlatIPIndex(ids, vectors, manifest)
            entries.append(
                RegisteredIndex(
                    key=f"fashioniq:{split}:{category}",
                    dataset="fashioniq",
                    split=split,
                    category=category,
                    index=index,
                    embedding_by_id={
                        image_id: index.embeddings[offset]
                        for offset, image_id in enumerate(index.image_ids)
                    },
                    image_path_by_id={
                        image_id: image_paths[position_by_id[image_id]]
                        for image_id in ids
                    },
                    cache_path=cache,
                    cache_sha256=cache_digest,
                )
            )
        return cls(entries)
