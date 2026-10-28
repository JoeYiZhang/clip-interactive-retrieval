"""Core building blocks for reproducible retrieval and multi-turn interaction."""

from .api_models import (
    CloseSessionRequest,
    CreateSessionRequest,
    FeedbackRequest,
    FeedbackResponse,
    RetrievalResponse,
    RetrieveRequest,
)
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
from .demo_service import (
    FashionIQQueryComposer,
    LazyTextEncoder,
    MultiIndexSessionService,
)
from .feedback import FeedbackWeights, QueryUpdate, update_query
from .fusion import fuse_weighted, l2_normalize, l2_normalize_rows
from .index import NumpyFlatIPIndex
from .index_registry import IndexRegistry, RegisteredIndex
from .interactive import InteractiveSession, SessionSnapshot, SessionTransition
from .metrics import ann_recall_at_k, recall_at_k
from .session_service import SessionService
from .session_store import SQLiteSessionStore

__all__ = [
    "CloseSessionRequest",
    "ComposedQuery",
    "CreateSessionRequest",
    "DatasetBundle",
    "DatasetValidation",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackWeights",
    "FashionIQQueryComposer",
    "GalleryItem",
    "IndexManifest",
    "IndexRegistry",
    "InteractiveSession",
    "LazyTextEncoder",
    "MultiIndexSessionService",
    "NumpyFlatIPIndex",
    "QuerySpec",
    "QueryUpdate",
    "RegisteredIndex",
    "RetrievalResponse",
    "RetrieveRequest",
    "SearchResult",
    "SessionSnapshot",
    "SessionService",
    "SessionTransition",
    "SQLiteSessionStore",
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

