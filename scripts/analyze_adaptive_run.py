"""Paired analysis for the train-tuned FashionIQ adaptive experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.datasets import load_fashioniq  # noqa: E402


def _rank(ranking: list[str], target: str) -> int:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return 51


def _exact_mcnemar_p(rescued: int, regressed: int) -> float:
    discordant = rescued + regressed
    if discordant == 0:
        return 1.0
    lower = min(rescued, regressed)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def _paired_bootstrap_category_mean(
    old_hits: np.ndarray,
    new_hits: np.ndarray,
    categories: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    names = sorted(set(categories.tolist()))
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        category_deltas = []
        for category in names:
            indices = np.flatnonzero(categories == category)
            sampled = rng.choice(indices, size=len(indices), replace=True)
            category_deltas.append(
                float(np.mean(new_hits[sampled]) - np.mean(old_hits[sampled]))
            )
        differences[repeat] = float(np.mean(category_deltas))
    return {
        "repeats": repeats,
        "seed": seed,
        "mean_delta": float(np.mean(differences)),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()

    bundle = load_fashioniq(args.data_root, split="val")
    rankings: dict[str, dict[str, list[str]]] = {}
    with (args.run_dir / "rankings.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rankings.setdefault(str(row["baseline"]), {})[
                str(row["query_id"])
            ] = [str(value) for value in row["ranked_image_ids"]]

    old_name = "B3_previous_val_tuned"
    global_name = "A0_train_tuned_global"
    new_name = "A2_train_tuned_dynamic"
    required = {old_name, global_name, new_name}
    if not required.issubset(rankings):
        raise SystemExit(f"missing baselines: {sorted(required - set(rankings))}")

    rows: list[dict] = []
    for query in bundle.queries:
        target = query.target_image_ids[0]
        item = {
            "query_id": query.query_id,
            "category": str(query.category),
            "reference": query.reference_image_id,
            "target": target,
            "text": query.modification_text,
        }
        for baseline in (old_name, global_name, new_name):
            ranking = rankings[baseline][query.query_id]
            item[f"{baseline}_rank"] = _rank(ranking, target)
            item[f"{baseline}_top1"] = ranking[0]
        rows.append(item)

    categories = np.asarray([row["category"] for row in rows], dtype=str)

    def comparison(reference_name: str) -> dict:
        old_ranks = np.asarray(
            [row[f"{reference_name}_rank"] for row in rows], dtype=np.int16
        )
        new_ranks = np.asarray(
            [row[f"{new_name}_rank"] for row in rows], dtype=np.int16
        )
        output: dict = {"reference": reference_name, "candidate": new_name}
        for cutoff in (10, 50):
            old_hits = old_ranks <= cutoff
            new_hits = new_ranks <= cutoff
            rescued = int(np.sum(~old_hits & new_hits))
            regressed = int(np.sum(old_hits & ~new_hits))
            output[f"recall@{cutoff}"] = {
                "reference_hits": int(np.sum(old_hits)),
                "candidate_hits": int(np.sum(new_hits)),
                "net_hits": int(np.sum(new_hits) - np.sum(old_hits)),
                "rescued": rescued,
                "regressed": regressed,
                "query_weighted_delta": float(np.mean(new_hits) - np.mean(old_hits)),
                "mcnemar_exact_two_sided_p": _exact_mcnemar_p(rescued, regressed),
                "category_stratified_paired_bootstrap": (
                    _paired_bootstrap_category_mean(
                        old_hits,
                        new_hits,
                        categories,
                        args.bootstrap_repeats,
                        seed=20261026 + cutoff,
                    )
                ),
            }
        output["rank_changes"] = {
            "better": int(np.sum(new_ranks < old_ranks)),
            "worse": int(np.sum(new_ranks > old_ranks)),
            "same": int(np.sum(new_ranks == old_ranks)),
        }
        return output

    old_ranks = np.asarray(
        [row[f"{old_name}_rank"] for row in rows], dtype=np.int16
    )
    new_ranks = np.asarray(
        [row[f"{new_name}_rank"] for row in rows], dtype=np.int16
    )
    rescued_rows = [
        row for row, old, new in zip(rows, old_ranks, new_ranks, strict=True)
        if old > 10 >= new
    ]
    regressed_rows = [
        row for row, old, new in zip(rows, old_ranks, new_ranks, strict=True)
        if old <= 10 < new
    ]

    def sample(values: list[dict], reverse: bool) -> list[dict]:
        return sorted(
            values,
            key=lambda row: (
                row[f"{old_name}_rank"] - row[f"{new_name}_rank"],
                row["query_id"],
            ),
            reverse=reverse,
        )[: args.sample_count]

    report = {
        "status": "pass",
        "query_count": len(rows),
        "comparisons": {
            "dynamic_vs_previous_b3": comparison(old_name),
            "dynamic_vs_train_global": comparison(global_name),
        },
        "dynamic_vs_previous_b3_top10_by_category": {
            category: {
                "rescued": sum(
                    row["category"] == category for row in rescued_rows
                ),
                "regressed": sum(
                    row["category"] == category for row in regressed_rows
                ),
            }
            for category in sorted(set(categories.tolist()))
        },
        "dynamic_top50_misses_by_category": dict(
            sorted(
                Counter(
                    row["category"]
                    for row in rows
                    if row[f"{new_name}_rank"] > 50
                ).items()
            )
        ),
        "rescued_samples": sample(rescued_rows, reverse=True),
        "regressed_samples": sample(regressed_rows, reverse=False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
