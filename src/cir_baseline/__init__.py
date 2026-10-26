"""Core building blocks for reproducible retrieval and multi-turn interaction."""

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
from .interactive import InteractiveSession, SessionSnapshot, SessionTransition
from .metrics import ann_recall_at_k, recall_at_k

__all__ = [
    "FeedbackWeights",
    "ComposedQuery",
    "DatasetBundle",
    "DatasetValidation",
    "GalleryItem",
    "IndexManifest",
    "InteractiveSession",
    "NumpyFlatIPIndex",
    "QuerySpec",
    "QueryUpdate",
    "SearchResult",
    "SessionSnapshot",
    "SessionTransition",
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

