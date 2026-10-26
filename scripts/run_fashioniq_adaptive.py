"""Train FashionIQ fusion choices on train and report once on validation.

This experiment closes the main methodological limitation of the original B3
baseline: caption strategy, global/category weights, confidence thresholds and
gate weights are selected with train targets only. Validation targets are used
only after every choice has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.adaptive import (  # noqa: E402
    confidence_bands,
    has_identity_intent,
    quantile_thresholds,
    select_weight_from_ranks,
)
from cir_baseline.contracts import IndexManifest  # noqa: E402
from cir_baseline.datasets import (  # noqa: E402
    DatasetBundle,
    load_fashioniq,
    validate_bundle,
)
from cir_baseline.encoders import OpenCLIPEncoder  # noqa: E402
from cir_baseline.fusion import l2_normalize_rows  # noqa: E402
from cir_baseline.index import FaissFlatIPIndex  # noqa: E402


TEXT_STRATEGIES = (
    "joined_forward",
    "joined_reverse",
    "separate_mean",
    "symmetric_join_mean",
)


def _batched(values: Sequence, size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _gallery_cache_name(bundle: DatasetBundle, encoder_version: str) -> str:
    return (
        f"{bundle.dataset}_{bundle.split}_{bundle.version}_"
        f"{_digest_text(encoder_version)}.npz"
    )


def _gallery_embeddings(
    bundle: DatasetBundle,
    encoder: OpenCLIPEncoder,
    cache_dir: Path,
    batch_size: int,
    refresh: bool,
) -> tuple[list[str], np.ndarray, Path, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _gallery_cache_name(bundle, encoder.version)
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
    batches = list(_batched(paths, batch_size))
    started = time.perf_counter()
    for batch_index, batch in enumerate(batches, start=1):
        encoded.append(encoder.encode_images(batch))
        if batch_index % 10 == 0 or batch_index == len(batches):
            print(
                f"{bundle.split}_image_encoding={batch_index}/{len(batches)} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    vectors = np.concatenate(encoded, axis=0).astype(np.float32)
    np.savez_compressed(
        cache_path,
        image_ids=np.asarray(expected_ids, dtype=str),
        image_paths=np.asarray(expected_paths, dtype=str),
        embeddings=vectors,
        encoder_version=np.asarray(encoder.version),
    )
    return expected_ids, vectors, cache_path, False


def _query_text_digest(bundle: DatasetBundle) -> str:
    digest = hashlib.sha256()
    for query in bundle.queries:
        digest.update(query.query_id.encode("utf-8"))
        for caption in query.modification_texts:
            digest.update(b"\0")
            digest.update(caption.encode("utf-8"))
    return digest.hexdigest()


def _encode_texts(
    encoder: OpenCLIPEncoder,
    texts: list[str],
    batch_size: int,
    label: str,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    batches = list(_batched(texts, batch_size))
    for batch_index, batch in enumerate(batches, start=1):
        encoded.append(encoder.encode_texts(batch))
        if batch_index % 10 == 0 or batch_index == len(batches):
            print(f"{label}_text_encoding={batch_index}/{len(batches)}", flush=True)
    return np.concatenate(encoded, axis=0).astype(np.float32)


def _text_embeddings(
    bundle: DatasetBundle,
    encoder: OpenCLIPEncoder,
    cache_dir: Path,
    batch_size: int,
    refresh: bool,
) -> tuple[dict[str, np.ndarray], Path, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"fashioniq_{bundle.split}_text-ablation-v1_"
        f"{_digest_text(encoder.version)}.npz"
    )
    query_ids = [query.query_id for query in bundle.queries]
    text_digest = _query_text_digest(bundle)
    if cache_path.is_file() and not refresh:
        with np.load(cache_path, allow_pickle=False) as payload:
            cached_ids = payload["query_ids"].astype(str).tolist()
            cached_version = str(payload["encoder_version"].item())
            cached_digest = str(payload["text_digest"].item())
            if (
                cached_ids == query_ids
                and cached_version == encoder.version
                and cached_digest == text_digest
            ):
                return (
                    {
                        name: payload[name].astype(np.float32)
                        for name in TEXT_STRATEGIES
                    },
                    cache_path,
                    True,
                )

    caption_lists = [query.modification_texts for query in bundle.queries]
    if any(not captions for captions in caption_lists):
        raise ValueError("FashionIQ adaptive experiment requires at least one caption")
    joined = [" and ".join(captions) for captions in caption_lists]
    reversed_joined = [" and ".join(reversed(captions)) for captions in caption_lists]
    individual = [caption for captions in caption_lists for caption in captions]
    joined_vectors = _encode_texts(
        encoder, joined, batch_size, f"{bundle.split}_joined"
    )
    reversed_vectors = _encode_texts(
        encoder, reversed_joined, batch_size, f"{bundle.split}_reversed"
    )
    encoded_individual = _encode_texts(
        encoder, individual, batch_size, f"{bundle.split}_individual"
    )
    individual_means: list[np.ndarray] = []
    cursor = 0
    for captions in caption_lists:
        next_cursor = cursor + len(captions)
        individual_means.append(encoded_individual[cursor:next_cursor].mean(axis=0))
        cursor = next_cursor
    strategies = {
        "joined_forward": joined_vectors,
        "joined_reverse": reversed_vectors,
        "separate_mean": l2_normalize_rows(np.stack(individual_means)),
        "symmetric_join_mean": l2_normalize_rows(joined_vectors + reversed_vectors),
    }
    np.savez_compressed(
        cache_path,
        query_ids=np.asarray(query_ids, dtype=str),
        encoder_version=np.asarray(encoder.version),
        text_digest=np.asarray(text_digest),
        **strategies,
    )
    return strategies, cache_path, False


def _category_gallery_positions(
    bundle: DatasetBundle, gallery_ids: list[str]
) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for index, item in enumerate(bundle.gallery):
        for category in item.categories or (str(item.category),):
            output.setdefault(category, []).append(index)
    if set(output) != {str(query.category) for query in bundle.queries}:
        raise ValueError("query and gallery categories differ")
    if len(gallery_ids) != len(bundle.gallery):
        raise ValueError("gallery id count differs from bundle")
    return output


@dataclass
class ExactFashionIQEvaluator:
    bundle: DatasetBundle
    gallery_ids: list[str]
    gallery_vectors: np.ndarray
    encoder_version: str

    def __post_init__(self) -> None:
        self.id_to_global = {
            image_id: index for index, image_id in enumerate(self.gallery_ids)
        }
        self.reference_vectors = self.gallery_vectors[
            [self.id_to_global[q.reference_image_id] for q in self.bundle.queries]
        ]
        self.indices: dict[str, FaissFlatIPIndex] = {}
        self.local_ids: dict[str, list[str]] = {}
        self.local_id_to_position: dict[str, dict[str, int]] = {}
        for category, positions in _category_gallery_positions(
            self.bundle, self.gallery_ids
        ).items():
            ids = [self.gallery_ids[position] for position in positions]
            vectors = self.gallery_vectors[positions]
            manifest = IndexManifest(
                index_version=f"fashioniq-{self.bundle.split}-{category}-flatip-v2",
                encoder_version=self.encoder_version,
                embedding_dim=int(vectors.shape[1]),
            )
            self.indices[category] = FaissFlatIPIndex(ids, vectors, manifest)
            self.local_ids[category] = ids
            self.local_id_to_position[category] = {
                image_id: index for index, image_id in enumerate(ids)
            }

    def search(
        self,
        text_vectors: np.ndarray,
        image_weights: float | np.ndarray,
        *,
        return_rankings: bool = False,
        return_confidence: bool = False,
    ) -> tuple[np.ndarray, list[list[str]] | None, np.ndarray | None]:
        count = len(self.bundle.queries)
        weights = np.asarray(image_weights, dtype=np.float32)
        if weights.ndim == 0:
            weights = np.full(count, float(weights), dtype=np.float32)
        if weights.shape != (count,) or np.any((weights < 0) | (weights > 1)):
            raise ValueError("image_weights must be a scalar or one value per query")
        query_vectors = l2_normalize_rows(
            weights[:, None] * self.reference_vectors
            + (1.0 - weights[:, None]) * text_vectors
        )
        target_ranks = np.full(count, 51, dtype=np.int16)
        rankings: list[list[str] | None] | None = (
            [None] * count if return_rankings else None
        )
        confidence = (
            np.full(count, np.nan, dtype=np.float32)
            if return_confidence
            else None
        )
        for category, category_index in self.indices.items():
            positions = np.asarray(
                [
                    index
                    for index, query in enumerate(self.bundle.queries)
                    if str(query.category) == category
                ],
                dtype=np.int64,
            )
            top_k = min(51, len(self.local_ids[category]))
            scores, local_indices = category_index.index.search(
                np.ascontiguousarray(query_vectors[positions], dtype=np.float32),
                top_k,
            )
            local_lookup = self.local_id_to_position[category]
            ids = self.local_ids[category]
            for row_offset, query_position in enumerate(positions):
                query = self.bundle.queries[int(query_position)]
                reference_local = local_lookup[query.reference_image_id]
                keep = local_indices[row_offset] != reference_local
                selected_indices = local_indices[row_offset][keep][:50]
                selected_scores = scores[row_offset][keep][:50]
                target_local = local_lookup[query.target_image_ids[0]]
                matches = np.flatnonzero(selected_indices == target_local)
                if matches.size:
                    target_ranks[int(query_position)] = int(matches[0]) + 1
                if rankings is not None:
                    rankings[int(query_position)] = [
                        ids[int(item)] for item in selected_indices
                    ]
                if confidence is not None:
                    tail = selected_scores[1:10]
                    confidence[int(query_position)] = float(
                        selected_scores[0] - np.mean(tail)
                    )
        if np.any(target_ranks < 1):
            raise RuntimeError("invalid target rank")
        final_rankings = None
        if rankings is not None:
            if any(row is None for row in rankings):
                raise RuntimeError("some query rankings were not generated")
            final_rankings = [row or [] for row in rankings]
        if confidence is not None and not np.all(np.isfinite(confidence)):
            raise RuntimeError("some confidence values were not generated")
        return target_ranks, final_rankings, confidence


def _metrics_from_ranks(bundle: DatasetBundle, ranks: np.ndarray) -> dict:
    output: dict = {
        "recall@10": float(np.mean(ranks <= 10)),
        "recall@50": float(np.mean(ranks <= 50)),
    }
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({str(query.category) for query in bundle.queries}):
        indices = np.asarray(
            [
                index
                for index, query in enumerate(bundle.queries)
                if str(query.category) == category
            ],
            dtype=np.int64,
        )
        by_category[category] = {
            "recall@10": float(np.mean(ranks[indices] <= 10)),
            "recall@50": float(np.mean(ranks[indices] <= 50)),
        }
    output["by_category"] = by_category
    output["category_mean_recall@10"] = float(
        np.mean([row["recall@10"] for row in by_category.values()])
    )
    output["category_mean_recall@50"] = float(
        np.mean([row["recall@50"] for row in by_category.values()])
    )
    return output


def _select_global_weight(
    weights: list[float],
    ranks_by_weight: np.ndarray,
    bundle: DatasetBundle,
    prior_weight: float = 0.2,
) -> tuple[float, dict]:
    candidates = [
        _metrics_from_ranks(bundle, ranks_by_weight[index])
        for index in range(len(weights))
    ]
    best_index = max(
        range(len(weights)),
        key=lambda index: (
            candidates[index]["category_mean_recall@10"],
            candidates[index]["category_mean_recall@50"],
            -abs(weights[index] - prior_weight),
            -weights[index],
        ),
    )
    return weights[best_index], candidates[best_index]


def _category_indices(bundle: DatasetBundle, category: str) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, query in enumerate(bundle.queries)
            if str(query.category) == category
        ],
        dtype=np.int64,
    )


def _combine_rankings(
    rankings_by_weight: dict[float, list[list[str]]],
    query_weights: np.ndarray,
) -> list[list[str]]:
    return [
        rankings_by_weight[float(weight)][index]
        for index, weight in enumerate(query_weights)
    ]


def _write_rankings(
    path: Path,
    bundle: DatasetBundle,
    named_rankings: dict[str, list[list[str]]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for baseline, rankings in named_rankings.items():
            for index, query in enumerate(bundle.queries):
                handle.write(
                    json.dumps(
                        {
                            "baseline": baseline,
                            "query_id": query.query_id,
                            "category": query.category,
                            "reference_image_id": query.reference_image_id,
                            "target_image_ids": query.target_image_ids,
                            "ranked_image_ids": rankings[index],
                            "subset_ranked_image_ids": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> int:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\image_data\derived"))
    parser.add_argument("--run-id")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument(
        "--weight-grid",
        type=float,
        nargs="+",
        default=[value / 10 for value in range(11)],
    )
    parser.add_argument("--min-gate-group-size", type=int, default=30)
    parser.add_argument("--refresh-gallery-cache", action="store_true")
    parser.add_argument("--refresh-text-cache", action="store_true")
    args = parser.parse_args()

    weights = sorted(set(float(value) for value in args.weight_grid))
    if not weights or any(value < 0 or value > 1 for value in weights):
        raise SystemExit("weight grid must contain values in [0, 1]")
    for required in (0.0, 0.2):
        if required not in weights:
            raise SystemExit(f"weight grid must include {required}")

    bundles = {
        split: load_fashioniq(args.data_root, split=split)
        for split in ("train", "val")
    }
    for bundle in bundles.values():
        validation = validate_bundle(bundle)
        if not validation.ready_for_experiment:
            raise SystemExit(
                json.dumps(validation.to_dict(), ensure_ascii=False, indent=2)
            )

    encoder_started = time.perf_counter()
    encoder = OpenCLIPEncoder(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        force_quick_gelu=True,
        weights_only=False,
    )
    encoder_seconds = time.perf_counter() - encoder_started
    torch_module = encoder._torch
    if encoder.device.startswith("cuda"):
        torch_module.cuda.reset_peak_memory_stats()

    cache_details: dict[str, dict] = {}
    evaluators: dict[str, ExactFashionIQEvaluator] = {}
    text_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, bundle in bundles.items():
        gallery_ids, gallery_vectors, gallery_cache, gallery_hit = _gallery_embeddings(
            bundle,
            encoder,
            args.output_root / "embeddings",
            args.image_batch_size,
            args.refresh_gallery_cache,
        )
        text_vectors, text_cache, text_hit = _text_embeddings(
            bundle,
            encoder,
            args.output_root / "embeddings",
            args.text_batch_size,
            args.refresh_text_cache,
        )
        evaluators[split] = ExactFashionIQEvaluator(
            bundle, gallery_ids, gallery_vectors, encoder.version
        )
        text_by_split[split] = text_vectors
        cache_details[split] = {
            "gallery": {"path": str(gallery_cache.resolve()), "hit": gallery_hit},
            "text": {"path": str(text_cache.resolve()), "hit": text_hit},
        }

    train = bundles["train"]
    val = bundles["val"]
    train_grids: dict[str, np.ndarray] = {}
    train_confidence: dict[str, np.ndarray] = {}
    caption_ablation: dict[str, dict] = {}
    strategy_selection_started = time.perf_counter()
    for strategy in TEXT_STRATEGIES:
        print(f"strategy={strategy} phase=train_weight_sweep", flush=True)
        rank_rows: list[np.ndarray] = []
        confidence = None
        for weight in weights:
            ranks, _, observed_confidence = evaluators["train"].search(
                text_by_split["train"][strategy],
                weight,
                return_confidence=weight == 0.0,
            )
            rank_rows.append(ranks)
            if observed_confidence is not None:
                confidence = observed_confidence
        if confidence is None:
            raise RuntimeError("text-only confidence was not generated")
        grid = np.stack(rank_rows)
        train_grids[strategy] = grid
        train_confidence[strategy] = confidence
        selected_weight, selected_train_metrics = _select_global_weight(
            weights, grid, train
        )
        val_ranks, _, _ = evaluators["val"].search(
            text_by_split["val"][strategy], selected_weight
        )
        caption_ablation[strategy] = {
            "selected_image_weight_from_train": selected_weight,
            "train_metrics": selected_train_metrics,
            "val_metrics": _metrics_from_ranks(val, val_ranks),
        }

    selected_strategy = max(
        TEXT_STRATEGIES,
        key=lambda name: (
            caption_ablation[name]["train_metrics"]["category_mean_recall@10"],
            caption_ablation[name]["train_metrics"]["category_mean_recall@50"],
            -TEXT_STRATEGIES.index(name),
        ),
    )
    selected_grid = train_grids[selected_strategy]
    global_weight = float(
        caption_ablation[selected_strategy]["selected_image_weight_from_train"]
    )
    categories = sorted({str(query.category) for query in train.queries})
    category_weights: dict[str, float] = {}
    category_selection: dict[str, dict] = {}
    for category in categories:
        category_weight, selection = select_weight_from_ranks(
            weights,
            selected_grid,
            _category_indices(train, category),
        )
        category_weights[category] = category_weight
        category_selection[category] = selection

    identity_train = np.asarray(
        [has_identity_intent(query.modification_text) for query in train.queries],
        dtype=bool,
    )
    confidence_train = train_confidence[selected_strategy]
    confidence_thresholds: dict[str, tuple[float, ...]] = {}
    gate_weights: dict[str, float] = {}
    gate_selection: dict[str, dict] = {}
    for category in categories:
        category_idx = _category_indices(train, category)
        non_identity_idx = category_idx[~identity_train[category_idx]]
        thresholds = quantile_thresholds(confidence_train[non_identity_idx])
        confidence_thresholds[category] = thresholds
        category_bands = confidence_bands(
            confidence_train[non_identity_idx], thresholds
        )
        groups = {
            f"{category}:identity": category_idx[identity_train[category_idx]],
            **{
                f"{category}:confidence_{band}": non_identity_idx[
                    category_bands == band
                ]
                for band in range(len(thresholds) + 1)
            },
        }
        for key, group_indices in groups.items():
            if len(group_indices) < args.min_gate_group_size:
                gate_weights[key] = category_weights[category]
                gate_selection[key] = {
                    "query_count": int(len(group_indices)),
                    "fallback": "category_weight",
                }
                continue
            group_weight, selection = select_weight_from_ranks(
                weights, selected_grid, group_indices
            )
            gate_weights[key] = group_weight
            gate_selection[key] = selection

    print(
        f"selected_strategy={selected_strategy} global_weight={global_weight}",
        flush=True,
    )
    selected_val_vectors = text_by_split["val"][selected_strategy]
    val_rank_rows: list[np.ndarray] = []
    val_rankings_by_weight: dict[float, list[list[str]]] = {}
    val_confidence = None
    for weight in weights:
        ranks, rankings, observed_confidence = evaluators["val"].search(
            selected_val_vectors,
            weight,
            return_rankings=True,
            return_confidence=weight == 0.0,
        )
        if rankings is None:
            raise RuntimeError("validation rankings were not generated")
        val_rank_rows.append(ranks)
        val_rankings_by_weight[weight] = rankings
        if observed_confidence is not None:
            val_confidence = observed_confidence
    if val_confidence is None:
        raise RuntimeError("validation confidence was not generated")
    val_grid = np.stack(val_rank_rows)
    weight_to_row = {weight: index for index, weight in enumerate(weights)}

    category_query_weights = np.asarray(
        [category_weights[str(query.category)] for query in val.queries],
        dtype=np.float64,
    )
    identity_val = np.asarray(
        [has_identity_intent(query.modification_text) for query in val.queries],
        dtype=bool,
    )
    dynamic_query_weights = np.empty(len(val.queries), dtype=np.float64)
    dynamic_group_counts: dict[str, int] = {}
    for category in categories:
        category_idx = _category_indices(val, category)
        category_identity = category_idx[identity_val[category_idx]]
        identity_key = f"{category}:identity"
        dynamic_query_weights[category_identity] = gate_weights[identity_key]
        dynamic_group_counts[identity_key] = int(len(category_identity))
        non_identity_idx = category_idx[~identity_val[category_idx]]
        bands = confidence_bands(
            val_confidence[non_identity_idx], confidence_thresholds[category]
        )
        for band in range(len(confidence_thresholds[category]) + 1):
            group_idx = non_identity_idx[bands == band]
            key = f"{category}:confidence_{band}"
            dynamic_query_weights[group_idx] = gate_weights[key]
            dynamic_group_counts[key] = int(len(group_idx))

    def ranks_for(query_weights: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                val_grid[weight_to_row[float(weight)], index]
                for index, weight in enumerate(query_weights)
            ],
            dtype=np.int16,
        )

    global_query_weights = np.full(len(val.queries), global_weight, dtype=np.float64)
    global_ranks = ranks_for(global_query_weights)
    category_ranks = ranks_for(category_query_weights)
    dynamic_ranks = ranks_for(dynamic_query_weights)

    previous_ranks, previous_rankings, _ = evaluators["val"].search(
        text_by_split["val"]["joined_forward"], 0.2, return_rankings=True
    )
    if previous_rankings is None:
        raise RuntimeError("previous baseline rankings were not generated")

    baselines = {
        "B3_previous_val_tuned": {
            "caption_strategy": "joined_forward",
            "image_weight": 0.2,
            "selection_split": "val (historical; biased comparison anchor)",
            "metrics": _metrics_from_ranks(val, previous_ranks),
        },
        "A0_train_tuned_global": {
            "caption_strategy": selected_strategy,
            "image_weight": global_weight,
            "selection_split": "train",
            "metrics": _metrics_from_ranks(val, global_ranks),
        },
        "A1_train_tuned_category": {
            "caption_strategy": selected_strategy,
            "image_weights": category_weights,
            "selection_split": "train",
            "metrics": _metrics_from_ranks(val, category_ranks),
        },
        "A2_train_tuned_dynamic": {
            "caption_strategy": selected_strategy,
            "gate": "identity intent else per-category text-confidence tertile",
            "selection_split": "train",
            "metrics": _metrics_from_ranks(val, dynamic_ranks),
        },
    }
    named_rankings = {
        "B3_previous_val_tuned": previous_rankings,
        "A0_train_tuned_global": _combine_rankings(
            val_rankings_by_weight, global_query_weights
        ),
        "A1_train_tuned_category": _combine_rankings(
            val_rankings_by_weight, category_query_weights
        ),
        "A2_train_tuned_dynamic": _combine_rankings(
            val_rankings_by_weight, dynamic_query_weights
        ),
    }

    oracle_hits = np.min(val_grid, axis=0)
    timestamp = datetime.now(timezone.utc)
    run_id = args.run_id or timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        args.output_root / "rankings" / f"fashioniq_val_{run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "scope": "FashionIQ choices trained on train and frozen before val reporting",
        "timestamp_utc": timestamp.isoformat(),
        "dataset": {
            "name": "fashioniq",
            "train_gallery_count": len(train.gallery),
            "train_query_count": len(train.queries),
            "val_gallery_count": len(val.gallery),
            "val_query_count": len(val.queries),
            "categories": categories,
            "root": str(args.data_root.resolve()),
        },
        "encoder": {
            "version": encoder.version,
            "checkpoint": str(args.checkpoint.resolve()),
            "weights_sha256": encoder.weights_sha256,
            "device": encoder.device,
        },
        "selection_protocol": {
            "selection_split": "train",
            "report_split": "val",
            "objective": "category_mean_recall@10 for caption/global selection; group recall@10 for gates",
            "weight_grid": weights,
            "confidence_feature": "text-only top1 score minus mean of ranks 2-10 after reference exclusion",
            "confidence_bins": "per-category train tertiles",
            "identity_rule": "fixed lexical rule; no target labels",
            "minimum_gate_group_size": args.min_gate_group_size,
        },
        "caption_ablation": caption_ablation,
        "selected_caption_strategy": selected_strategy,
        "train_parameters": {
            "global_image_weight": global_weight,
            "category_image_weights": category_weights,
            "category_selection": category_selection,
            "confidence_thresholds": {
                key: list(value) for key, value in confidence_thresholds.items()
            },
            "gate_image_weights": gate_weights,
            "gate_selection": gate_selection,
        },
        "val_gate_group_counts": dynamic_group_counts,
        "baselines": baselines,
        "diagnostics": {
            "target_aware_per_query_grid_oracle_not_deployable": _metrics_from_ranks(
                val, oracle_hits
            )
        },
        "cache": cache_details,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "timing": {
            "encoder_load_seconds": round(encoder_seconds, 3),
            "strategy_selection_and_evaluation_seconds": round(
                time.perf_counter() - strategy_selection_started, 3
            ),
            "total_seconds": round(time.perf_counter() - run_started, 3),
        },
        "gpu": {
            "peak_memory_allocated_mib": (
                round(torch_module.cuda.max_memory_allocated() / 1024**2, 1)
                if encoder.device.startswith("cuda")
                else None
            ),
            "peak_memory_reserved_mib": (
                round(torch_module.cuda.max_memory_reserved() / 1024**2, 1)
                if encoder.device.startswith("cuda")
                else None
            ),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_rankings(run_dir / "rankings.jsonl", val, named_rankings)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"run_dir={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
