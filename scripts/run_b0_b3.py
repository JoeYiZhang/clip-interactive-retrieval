"""Run exact OpenCLIP B0-B3 baselines on CIRR or FashionIQ validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.contracts import IndexManifest  # noqa: E402
from cir_baseline.datasets import (  # noqa: E402
    DatasetBundle,
    load_cirr,
    load_fashioniq,
    validate_bundle,
)
from cir_baseline.encoders import OpenCLIPEncoder  # noqa: E402
from cir_baseline.fusion import fuse_weighted  # noqa: E402
from cir_baseline.index import FaissFlatIPIndex  # noqa: E402
from cir_baseline.metrics import recall_at_k  # noqa: E402


def _batched(values: list, size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _cache_name(bundle: DatasetBundle, encoder_version: str) -> str:
    digest = hashlib.sha256(encoder_version.encode("utf-8")).hexdigest()[:12]
    return f"{bundle.dataset}_{bundle.split}_{bundle.version}_{digest}.npz"


def _gallery_embeddings(
    bundle: DatasetBundle,
    encoder: OpenCLIPEncoder,
    cache_dir: Path,
    batch_size: int,
    refresh: bool,
) -> tuple[list[str], np.ndarray, Path, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_name(bundle, encoder.version)
    expected_ids = [item.image_id for item in bundle.gallery]
    expected_paths = [str(item.path.resolve()) for item in bundle.gallery]
    if cache_path.is_file() and not refresh:
        with np.load(cache_path, allow_pickle=False) as payload:
            ids = payload["image_ids"].astype(str).tolist()
            paths = payload["image_paths"].astype(str).tolist()
            version = str(payload["encoder_version"].item())
            vectors = payload["embeddings"].astype(np.float32)
        if ids == expected_ids and paths == expected_paths and version == encoder.version:
            return ids, vectors, cache_path, True

    encoded: list[np.ndarray] = []
    paths = [item.path for item in bundle.gallery]
    for batch in _batched(paths, batch_size):
        encoded.append(encoder.encode_images(batch))
    vectors = np.concatenate(encoded, axis=0).astype(np.float32)
    np.savez_compressed(
        cache_path,
        image_ids=np.asarray(expected_ids, dtype=str),
        image_paths=np.asarray(expected_paths, dtype=str),
        embeddings=vectors,
        encoder_version=np.asarray(encoder.version),
    )
    return expected_ids, vectors, cache_path, False


def _text_embeddings(
    bundle: DatasetBundle, encoder: OpenCLIPEncoder, batch_size: int
) -> np.ndarray:
    texts = [query.modification_text for query in bundle.queries]
    encoded = [encoder.encode_texts(batch) for batch in _batched(texts, batch_size)]
    return np.concatenate(encoded, axis=0).astype(np.float32)


def _category_indices(bundle: DatasetBundle, image_ids: list[str]) -> dict[str, list[int]]:
    if bundle.dataset == "cirr":
        return {"all": list(range(len(image_ids)))}
    output: dict[str, list[int]] = {}
    for index, item in enumerate(bundle.gallery):
        categories = item.categories or (str(item.category),)
        for category in categories:
            output.setdefault(category, []).append(index)
    return output


def _evaluate(
    bundle: DatasetBundle,
    rankings: list[list[str]],
    subset_rankings: list[list[str]],
) -> dict:
    relevant = [set(query.target_image_ids) for query in bundle.queries]
    if bundle.dataset == "cirr":
        result = {
            f"recall@{k}": recall_at_k(rankings, relevant, k)
            for k in (1, 5, 10, 50)
        }
        result.update(
            {
                f"subset_recall@{k}": recall_at_k(subset_rankings, relevant, k)
                for k in (1, 2, 3)
            }
        )
        return result

    result: dict = {
        f"recall@{k}": recall_at_k(rankings, relevant, k) for k in (10, 50)
    }
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({str(query.category) for query in bundle.queries}):
        indices = [
            index
            for index, query in enumerate(bundle.queries)
            if str(query.category) == category
        ]
        category_rankings = [rankings[index] for index in indices]
        category_relevant = [relevant[index] for index in indices]
        by_category[category] = {
            f"recall@{k}": recall_at_k(category_rankings, category_relevant, k)
            for k in (10, 50)
        }
    result["by_category"] = by_category
    result["category_mean_recall@10"] = float(
        np.mean([value["recall@10"] for value in by_category.values()])
    )
    result["category_mean_recall@50"] = float(
        np.mean([value["recall@50"] for value in by_category.values()])
    )
    return result


def _run_weight(
    bundle: DatasetBundle,
    gallery_ids: list[str],
    gallery_vectors: np.ndarray,
    text_vectors: np.ndarray,
    image_weight: float,
    encoder_version: str,
) -> tuple[dict, list[list[str]], list[list[str]]]:
    id_to_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    category_indices = _category_indices(bundle, gallery_ids)
    indices: dict[str, FaissFlatIPIndex] = {}
    local_ids: dict[str, list[str]] = {}
    for category, positions in category_indices.items():
        ids = [gallery_ids[position] for position in positions]
        vectors = gallery_vectors[positions]
        manifest = IndexManifest(
            index_version=f"{bundle.dataset}-{bundle.split}-{category}-flatip-v1",
            encoder_version=encoder_version,
            embedding_dim=int(vectors.shape[1]),
        )
        indices[category] = FaissFlatIPIndex(ids, vectors, manifest)
        local_ids[category] = ids

    rankings: list[list[str]] = []
    subset_rankings: list[list[str]] = []
    for query_index, query in enumerate(bundle.queries):
        reference_vector = gallery_vectors[id_to_index[query.reference_image_id]]
        vector = fuse_weighted(
            reference_vector,
            text_vectors[query_index],
            image_weight=image_weight,
            text_weight=1.0 - image_weight,
        )
        category = "all" if bundle.dataset == "cirr" else str(query.category)
        top_k = min(51, len(local_ids[category]))
        hits = indices[category].search(vector, top_k=top_k)
        filtered = [
            hit.image_id
            for hit in hits
            if hit.image_id != query.reference_image_id
        ][:50]
        rankings.append(filtered)

        if query.subset_image_ids:
            subset_ids = [
                image_id
                for image_id in query.subset_image_ids
                if image_id != query.reference_image_id
            ]
            subset_vectors = gallery_vectors[
                [id_to_index[image_id] for image_id in subset_ids]
            ]
            scores = subset_vectors @ vector
            order = np.argsort(-scores, kind="stable")
            subset_rankings.append([subset_ids[int(index)] for index in order])
        else:
            subset_rankings.append(filtered)
    return _evaluate(bundle, rankings, subset_rankings), rankings, subset_rankings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cirr", "fashioniq"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--version", default="rc2")
    parser.add_argument(
        "--categories", nargs="+", default=("dress", "shirt", "toptee")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument(
        "--weight-grid",
        type=float,
        nargs="+",
        default=[value / 10 for value in range(11)],
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(r"D:\image_data\derived")
    )
    parser.add_argument("--run-id")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    if args.dataset == "cirr":
        bundle = load_cirr(args.data_root, split=args.split, version=args.version)
    else:
        bundle = load_fashioniq(
            args.data_root, split=args.split, categories=args.categories
        )
    validation = validate_bundle(bundle)
    if not validation.ready_for_experiment:
        print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(
            "dataset is not experiment-ready; run validate_dataset.py for details"
        )
    if not all(0.0 <= value <= 1.0 for value in args.weight_grid):
        raise SystemExit("all image weights must be in [0, 1]")

    encoder = OpenCLIPEncoder(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        force_quick_gelu=True,
        weights_only=False,
    )
    gallery_ids, gallery_vectors, cache_path, cache_hit = _gallery_embeddings(
        bundle,
        encoder,
        args.output_root / "embeddings",
        args.image_batch_size,
        args.refresh_cache,
    )
    text_vectors = _text_embeddings(bundle, encoder, args.text_batch_size)

    named_weights = {"B0_image_only": 1.0, "B1_text_only": 0.0, "B2_equal": 0.5}
    named_results: dict[str, dict] = {}
    saved_rankings: dict[str, tuple[list[list[str]], list[list[str]]]] = {}
    for name, weight in named_weights.items():
        metrics, ranking, subset = _run_weight(
            bundle, gallery_ids, gallery_vectors, text_vectors, weight, encoder.version
        )
        named_results[name] = {"image_weight": weight, "metrics": metrics}
        saved_rankings[name] = (ranking, subset)

    sweep: list[dict] = []
    sweep_rankings: dict[float, tuple[list[list[str]], list[list[str]]]] = {}
    objective = "recall@10"
    for weight in sorted(set(args.weight_grid)):
        metrics, ranking, subset = _run_weight(
            bundle, gallery_ids, gallery_vectors, text_vectors, weight, encoder.version
        )
        sweep.append({"image_weight": weight, "metrics": metrics})
        sweep_rankings[weight] = (ranking, subset)
    best = max(
        sweep,
        key=lambda row: (row["metrics"][objective], -abs(row["image_weight"] - 0.5)),
    )
    best_weight = float(best["image_weight"])
    named_results["B3_validation_tuned"] = {
        "image_weight": best_weight,
        "selection_objective": objective,
        "metrics": best["metrics"],
    }
    saved_rankings["B3_validation_tuned"] = sweep_rankings[best_weight]

    timestamp = datetime.now(timezone.utc)
    run_id = args.run_id or timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / "rankings" / f"{bundle.dataset}_{bundle.split}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "scope": "exact validation baseline; B3 is tuned and reported on the same split",
        "timestamp_utc": timestamp.isoformat(),
        "dataset": {
            "name": bundle.dataset,
            "split": bundle.split,
            "version": bundle.version,
            "root": str(bundle.root.resolve()),
            "gallery_count": len(bundle.gallery),
            "query_count": len(bundle.queries),
            "categories": list(args.categories) if bundle.dataset == "fashioniq" else [],
        },
        "encoder": {
            "version": encoder.version,
            "checkpoint": str(args.checkpoint.resolve()),
            "weights_sha256": encoder.weights_sha256,
            "device": encoder.device,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "cache": {"path": str(cache_path.resolve()), "hit": cache_hit},
        "baselines": named_results,
        "B3_weight_sweep": sweep,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "rankings.jsonl").open("w", encoding="utf-8") as handle:
        for baseline, (ranking, subset) in saved_rankings.items():
            for index, query in enumerate(bundle.queries):
                handle.write(
                    json.dumps(
                        {
                            "baseline": baseline,
                            "query_id": query.query_id,
                            "category": query.category,
                            "reference_image_id": query.reference_image_id,
                            "target_image_ids": query.target_image_ids,
                            "ranked_image_ids": ranking[index],
                            "subset_ranked_image_ids": subset[index]
                            if query.subset_image_ids
                            else [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
