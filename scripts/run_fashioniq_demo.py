"""Run the local FashionIQ real-index interactive retrieval demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.demo_app import create_demo_app  # noqa: E402
from cir_baseline.demo_service import (  # noqa: E402
    FashionIQQueryComposer,
    LazyTextEncoder,
    MultiIndexSessionService,
)
from cir_baseline.encoders import OpenCLIPEncoder  # noqa: E402
from cir_baseline.index_registry import IndexRegistry  # noqa: E402
from cir_baseline.session_store import SQLiteSessionStore  # noqa: E402


def _default_cache(root: Path, split: str) -> Path:
    matches = sorted(
        (root / "derived" / "embeddings").glob(
            f"fashioniq_{split}_official_*.npz"
        )
    )
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one FashionIQ {split} gallery cache, found {len(matches)}"
        )
    return matches[0]


def build_runtime(args: argparse.Namespace):
    cache = args.cache or _default_cache(args.storage_root, args.split)
    registry = IndexRegistry.from_fashioniq_cache(
        data_root=args.storage_root / "datasets" / "fashioniq",
        cache_path=cache,
        split=args.split,
        categories=args.categories,
    )
    versions = {
        registry.get(key).index.manifest.encoder_version for key in registry.keys
    }
    if len(versions) != 1:
        raise SystemExit("registered indexes do not share one encoder version")
    expected_version = versions.pop()
    lazy_encoder = LazyTextEncoder(
        expected_version=expected_version,
        factory=lambda: OpenCLIPEncoder(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            device=args.device,
            force_quick_gelu=True,
            weights_only=False,
        ),
    )
    service = MultiIndexSessionService(
        registry=registry,
        store=SQLiteSessionStore(args.database),
        composer=FashionIQQueryComposer(lazy_encoder),
        max_rounds=args.max_rounds,
    )
    return registry, lazy_encoder, create_demo_app(service, text_encoder=lazy_encoder)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage-root", type=Path, default=Path(r"D:\image_data")
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--categories", nargs="+", default=("dress", "shirt", "toptee")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"),
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runtime" / "fashioniq-demo.sqlite3",
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("this delivery is local-only; host must be a loopback address")
    if not args.checkpoint.is_file():
        raise SystemExit(f"OpenCLIP checkpoint not found: {args.checkpoint}")
    registry, lazy_encoder, app = build_runtime(args)
    summary = {
        "status": "ready",
        "mode": "local-engineering-demo",
        "database": str(args.database.resolve()),
        "text_encoder_loaded": lazy_encoder.loaded,
        "indexes": registry.summaries(),
        "url": f"http://{args.host}:{args.port}/",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return 0

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
