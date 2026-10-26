"""Create the agreed external data-root layout without downloading licensed images."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\image_data"))
    args = parser.parse_args()
    directories = (
        "datasets/cirr/captions",
        "datasets/cirr/image_splits",
        "datasets/cirr/img_raw",
        "datasets/fashioniq/captions",
        "datasets/fashioniq/image_splits",
        "datasets/fashioniq/images",
        "derived/embeddings",
        "derived/indexes",
        "derived/rankings",
        "downloads",
        "manifests",
    )
    for relative in directories:
        (args.root / relative).mkdir(parents=True, exist_ok=True)
    template = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_template.csv"
    manifest = args.root / "manifests" / "dataset_manifest.csv"
    if template.is_file() and not manifest.exists():
        shutil.copyfile(template, manifest)
    print(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
