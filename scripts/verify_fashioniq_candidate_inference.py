"""Check fixed raw-text inference against archived dev rankings, including ties."""
from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path

from train_fashioniq_regularization import ROOT, read_json, sha256, write_json
import numpy as np
import json


def compare_rankings(actual_ids, actual_scores, expected_ids, expected_scores, vectors, tolerance=1e-5):
    if not actual_ids or not (len(actual_ids) == len(actual_scores) == len(expected_ids) == len(expected_scores)):
        raise ValueError("unaligned or empty rankings")
    unique = len(actual_ids) == len(set(actual_ids)) and len(expected_ids) == len(set(expected_ids))
    finite = all(math.isfinite(value) for value in list(actual_scores) + list(expected_scores))
    same_members = set(actual_ids) == set(expected_ids)
    equivalent_vectors = all(np.array_equal(vectors[a], vectors[b]) for a, b in zip(actual_ids, expected_ids, strict=True))
    delta = max(abs(a - b) for a, b in zip(actual_scores, expected_scores, strict=True))
    checks = {"unique_candidates": unique, "same_topk_image_set": same_members,
              "order_difference_only_at_identical_vectors": equivalent_vectors,
              "finite_descending_scores": finite and list(actual_scores) == sorted(actual_scores, reverse=True) and list(expected_scores) == sorted(expected_scores, reverse=True),
              "score_agreement": finite and delta <= tolerance}
    return {"checks": checks, "exact_id_order_equal": actual_ids == expected_ids,
            "reordered_positions": [i + 1 for i, (a, b) in enumerate(zip(actual_ids, expected_ids, strict=True)) if a != b],
            "max_score_absolute_delta": delta, "score_tolerance": tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / "artifacts/metrics/fashioniq_regularization_inference_sample.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/metrics/fashioniq_regularization_candidate_manifest.json")
    parser.add_argument("--rankings", type=Path, default=ROOT / "artifacts/fusion-runs/fashioniq-regularization-dev-v1/dev_rankings.jsonl.gz")
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--query-id", default="dress:train:0")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/metrics/fashioniq_regularization_inference_verification.json")
    args = parser.parse_args()
    sample, manifest = read_json(args.sample), read_json(args.manifest)
    name = f"{manifest['regularization_variant']}-seed{manifest['inference_seed']}"
    with gzip.open(args.rankings, "rt", encoding="utf-8") as handle:
        expected = next((row for row in map(json.loads, handle) if row["model"] == name and row["query_id"] == args.query_id), None)
    if expected is None:
        raise ValueError("fixed query missing from replay archive")
    ids = [row["image_id"] for row in sample["results"]]
    scores = [row["score"] for row in sample["results"]]
    caches = list((args.storage_root / "derived/embeddings").glob("fashioniq_train_official_*.npz"))
    if len(caches) != 1:
        raise ValueError("expected exactly one train cache")
    with np.load(caches[0], allow_pickle=False) as cache:
        lookup = {key: i for i, key in enumerate(cache["image_ids"].astype(str).tolist())}
        gallery = cache["embeddings"]
        vectors = {key: gallery[lookup[key]] for key in set(ids) | set(expected["ranked_image_ids"][:len(ids)])}
    comparison = compare_rankings(ids, scores, expected["ranked_image_ids"][:len(ids)], expected["scores"][:len(ids)], vectors)
    checks = {**comparison["checks"],
              "checkpoint_hash": sha256(Path(sample["checkpoint"])) == manifest["checkpoint_sha256"],
              "seed_epoch_encoder": sample["seed"] == manifest["inference_seed"] and sample["epoch"] == manifest["epoch"] and sample["encoder_version"] == manifest["encoder_version"],
              "fixed_query_reference": sample["reference_image_id"] == expected["reference_image_id"] and sample["category"] == expected["category"],
              "unit_query": abs(sample["query_norm"] - 1) <= 1e-5,
              "ten_nonreference_images": len(ids) == 10 and sample["reference_image_id"] not in ids,
              "image_paths_exist": all(Path(row["image_path"]).is_file() for row in sample["results"])}
    result = {"status": "pass" if all(checks.values()) else "fail", **{key: value for key, value in comparison.items() if key != "checks"},
              "checks": checks, "check_count": len(checks), "query_id": args.query_id,
              "sample_sha256": sha256(args.sample), "gallery_cache_sha256": sha256(caches[0]),
              "verification_source_sha256": sha256(Path(__file__)),
              "scope": "fixed engineering query only; exact Top-10 membership, score agreement and strictly identical-vector tie equivalence; no scientific metric changes"}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise ValueError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    main()
