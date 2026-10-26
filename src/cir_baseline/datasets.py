from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GalleryItem:
    image_id: str
    path: Path
    dataset: str
    split: str
    category: str | None = None
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComposedQuery:
    query_id: str
    reference_image_id: str
    modification_text: str
    target_image_ids: tuple[str, ...]
    subset_image_ids: tuple[str, ...]
    dataset: str
    split: str
    category: str | None = None


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    split: str
    version: str
    root: Path
    gallery: tuple[GalleryItem, ...]
    queries: tuple[ComposedQuery, ...]


@dataclass(frozen=True)
class DatasetValidation:
    dataset: str
    split: str
    version: str
    gallery_count: int
    query_count: int
    queries_with_targets: int
    duplicate_gallery_ids: tuple[str, ...]
    duplicate_query_ids: tuple[str, ...]
    missing_reference_ids: tuple[str, ...]
    missing_target_ids: tuple[str, ...]
    invalid_subset_query_ids: tuple[str, ...]
    missing_image_count: int
    missing_image_samples: tuple[str, ...]
    structural_ok: bool
    images_ok: bool
    ready_for_experiment: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"annotation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_image(base: Path, image_id: str, relative: str | None = None) -> Path:
    candidates: list[Path] = []
    if relative:
        cleaned = relative.replace("\\", "/").removeprefix("./")
        candidates.extend((base / "img_raw" / cleaned, base / cleaned))
    for image_dir in (base / "images", base / "img_raw", base):
        for suffix in (".jpg", ".jpeg", ".png", ".webp"):
            candidates.append(image_dir / f"{image_id}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_cirr(
    root: str | Path,
    split: str = "val",
    version: str = "rc2",
) -> DatasetBundle:
    base = Path(root)
    caption_path = base / "captions" / f"cap.{version}.{split}.json"
    split_path = base / "image_splits" / f"split.{version}.{split}.json"
    raw_queries = _read_json(caption_path)
    raw_gallery = _read_json(split_path)
    if not isinstance(raw_queries, list) or not isinstance(raw_gallery, dict):
        raise ValueError("CIRR annotations must be a query list and an image-id map")

    gallery = tuple(
        GalleryItem(
            image_id=str(image_id),
            path=_find_image(base, str(image_id), str(relative_path)),
            dataset="cirr",
            split=split,
        )
        for image_id, relative_path in raw_gallery.items()
    )
    queries: list[ComposedQuery] = []
    for row in raw_queries:
        target = row.get("target_hard")
        targets = (str(target),) if target is not None else ()
        image_set = row.get("img_set") or {}
        queries.append(
            ComposedQuery(
                query_id=str(row["pairid"]),
                reference_image_id=str(row["reference"]),
                modification_text=str(row["caption"]).strip(),
                target_image_ids=targets,
                subset_image_ids=tuple(str(item) for item in image_set.get("members", ())),
                dataset="cirr",
                split=split,
            )
        )
    return DatasetBundle(
        dataset="cirr",
        split=split,
        version=version,
        root=base,
        gallery=gallery,
        queries=tuple(queries),
    )


def _join_fashioniq_captions(captions: Sequence[str]) -> str:
    cleaned = [str(value).strip() for value in captions if str(value).strip()]
    if not cleaned:
        raise ValueError("FashionIQ query has no usable captions")
    return " and ".join(cleaned)


def load_fashioniq(
    root: str | Path,
    split: str = "val",
    categories: Iterable[str] = ("dress", "shirt", "toptee"),
) -> DatasetBundle:
    base = Path(root)
    gallery_by_id: dict[str, GalleryItem] = {}
    queries: list[ComposedQuery] = []
    selected_categories = tuple(categories)
    if not selected_categories:
        raise ValueError("at least one FashionIQ category is required")

    for category in selected_categories:
        caption_path = base / "captions" / f"cap.{category}.{split}.json"
        split_path = base / "image_splits" / f"split.{category}.{split}.json"
        raw_queries = _read_json(caption_path)
        raw_gallery = _read_json(split_path)
        if not isinstance(raw_queries, list) or not isinstance(raw_gallery, list):
            raise ValueError("FashionIQ captions and image splits must be lists")
        for image_id in raw_gallery:
            key = str(image_id)
            existing = gallery_by_id.get(key)
            if existing is None:
                gallery_by_id[key] = GalleryItem(
                    image_id=key,
                    path=_find_image(base, key),
                    dataset="fashioniq",
                    split=split,
                    category=category,
                    categories=(category,),
                )
            elif category not in existing.categories:
                gallery_by_id[key] = GalleryItem(
                    image_id=existing.image_id,
                    path=existing.path,
                    dataset=existing.dataset,
                    split=existing.split,
                    category=existing.category,
                    categories=existing.categories + (category,),
                )
        for offset, row in enumerate(raw_queries):
            reference = row.get("candidate")
            if reference is None:
                raise ValueError(f"FashionIQ query lacks candidate: {category}/{offset}")
            target = row.get("target")
            targets = (str(target),) if target is not None else ()
            queries.append(
                ComposedQuery(
                    query_id=f"{category}:{split}:{offset}",
                    reference_image_id=str(reference),
                    modification_text=_join_fashioniq_captions(row.get("captions", ())),
                    target_image_ids=targets,
                    subset_image_ids=(),
                    dataset="fashioniq",
                    split=split,
                    category=category,
                )
            )
    return DatasetBundle(
        dataset="fashioniq",
        split=split,
        version="official",
        root=base,
        gallery=tuple(gallery_by_id.values()),
        queries=tuple(queries),
    )


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return tuple(sorted(duplicate))


def validate_bundle(bundle: DatasetBundle, sample_limit: int = 20) -> DatasetValidation:
    gallery_ids = {item.image_id for item in bundle.gallery}
    duplicate_gallery = _duplicates(item.image_id for item in bundle.gallery)
    duplicate_queries = _duplicates(query.query_id for query in bundle.queries)
    missing_references = tuple(
        sorted(
            {
                query.reference_image_id
                for query in bundle.queries
                if query.reference_image_id not in gallery_ids
            }
        )
    )
    missing_targets = tuple(
        sorted(
            {
                target
                for query in bundle.queries
                for target in query.target_image_ids
                if target not in gallery_ids
            }
        )
    )
    invalid_subset_queries = tuple(
        query.query_id
        for query in bundle.queries
        if query.subset_image_ids
        and (
            query.reference_image_id not in query.subset_image_ids
            or any(target not in query.subset_image_ids for target in query.target_image_ids)
            or any(member not in gallery_ids for member in query.subset_image_ids)
        )
    )
    missing_images = [str(item.path) for item in bundle.gallery if not item.path.is_file()]
    structural_ok = not any(
        (
            duplicate_gallery,
            duplicate_queries,
            missing_references,
            missing_targets,
            invalid_subset_queries,
        )
    )
    images_ok = not missing_images
    queries_with_targets = sum(bool(query.target_image_ids) for query in bundle.queries)
    ready = structural_ok and images_ok and queries_with_targets == len(bundle.queries)
    return DatasetValidation(
        dataset=bundle.dataset,
        split=bundle.split,
        version=bundle.version,
        gallery_count=len(bundle.gallery),
        query_count=len(bundle.queries),
        queries_with_targets=queries_with_targets,
        duplicate_gallery_ids=duplicate_gallery,
        duplicate_query_ids=duplicate_queries,
        missing_reference_ids=missing_references,
        missing_target_ids=missing_targets,
        invalid_subset_query_ids=invalid_subset_queries,
        missing_image_count=len(missing_images),
        missing_image_samples=tuple(missing_images[:sample_limit]),
        structural_ok=structural_ok,
        images_ok=images_ok,
        ready_for_experiment=ready,
    )
