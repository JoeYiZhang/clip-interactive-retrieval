"""Validate official annotations separately from licensed/raw image availability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.datasets import (  # noqa: E402
    load_cirr,
    load_fashioniq,
    validate_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cirr", "fashioniq"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--version", default="rc2", help="CIRR annotation version")
    parser.add_argument(
        "--categories", nargs="+", default=("dress", "shirt", "toptee")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Return success when annotations are valid even if raw images are pending.",
    )
    args = parser.parse_args()

    if args.dataset == "cirr":
        bundle = load_cirr(args.root, split=args.split, version=args.version)
    else:
        bundle = load_fashioniq(
            args.root, split=args.split, categories=args.categories
        )
    report = validate_bundle(bundle)
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    passed = report.structural_ok and (
        report.images_ok or args.allow_missing_images
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
