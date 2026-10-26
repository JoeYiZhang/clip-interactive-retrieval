"""Core building blocks for the stage-two retrieval baseline."""

from .contracts import IndexManifest, QuerySpec, SearchResult, ensure_compatible_versions
from .datasets import (
    ComposedQuery,
    DatasetBundle,
    DatasetValidation,
    GalleryItem,
    load_cirr,
    load_fashioniq,
    validate_bundle,
)
from .feedback import FeedbackWeights, QueryUpdate, update_query
from .fusion import fuse_weighted, l2_normalize, l2_normalize_rows
from .index import NumpyFlatIPIndex
from .metrics import ann_recall_at_k, recall_at_k

__all__ = [
    "FeedbackWeights",
    "ComposedQuery",
    "DatasetBundle",
    "DatasetValidation",
    "GalleryItem",
    "IndexManifest",
    "NumpyFlatIPIndex",
    "QuerySpec",
    "QueryUpdate",
    "SearchResult",
    "ann_recall_at_k",
    "ensure_compatible_versions",
    "fuse_weighted",
    "l2_normalize",
    "l2_normalize_rows",
    "load_cirr",
    "load_fashioniq",
    "recall_at_k",
    "update_query",
    "validate_bundle",
]

