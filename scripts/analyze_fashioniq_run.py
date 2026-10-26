"""Summarize FashionIQ successes, regressions and unresolved Top-50 failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.datasets import load_fashioniq  # noqa: E402


def _rank(ranking: list[str], target: str) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    args = parser.parse_args()

    bundle = load_fashioniq(args.data_root, split="val")
    rankings: dict[str, dict[str, list[str]]] = {}
    with (args.run_dir / "rankings.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rankings.setdefault(row["baseline"], {})[row["query_id"]] = row[
                "ranked_image_ids"
            ]

    rows: list[dict] = []
    for query in bundle.queries:
        target = query.target_image_ids[0]
        item = {
            "query_id": query.query_id,
            "category": query.category,
            "reference": query.reference_image_id,
            "target": target,
            "text": query.modification_text,
        }
        for baseline in ("B0_image_only", "B1_text_only", "B2_equal", "B3_validation_tuned"):
            ranking = rankings[baseline][query.query_id]
            item[f"{baseline}_rank"] = _rank(ranking, target)
            item[f"{baseline}_top1"] = ranking[0]
        rows.append(item)

    def numeric(value: int | None) -> int:
        return value if value is not None else 51

    b1_key = "B1_text_only_rank"
    b3_key = "B3_validation_tuned_rank"
    rescued = [row for row in rows if numeric(row[b1_key]) > 10 >= numeric(row[b3_key])]
    regressed = [row for row in rows if numeric(row[b1_key]) <= 10 < numeric(row[b3_key])]
    unresolved = [row for row in rows if row[b3_key] is None]
    improved = [row for row in rows if numeric(row[b3_key]) < numeric(row[b1_key])]
    worsened = [row for row in rows if numeric(row[b3_key]) > numeric(row[b1_key])]
    unchanged = len(rows) - len(improved) - len(worsened)

    def sample(values: list[dict], key) -> list[dict]:
        return sorted(values, key=key)[: args.sample_count]

    report = {
        "status": "pass",
        "query_count": len(rows),
        "comparison": "B3_validation_tuned versus B1_text_only",
        "counts": {
            "b3_rescued_into_top10": len(rescued),
            "b3_regressed_out_of_top10": len(regressed),
            "b3_better_rank": len(improved),
            "b3_worse_rank": len(worsened),
            "b3_same_rank": unchanged,
            "b3_target_outside_top50": len(unresolved),
        },
        "b3_outside_top50_by_category": dict(
            sorted(Counter(str(row["category"]) for row in unresolved).items())
        ),
        "rescued_samples": sample(
            rescued,
            lambda row: numeric(row[b3_key]) - numeric(row[b1_key]),
        ),
        "regressed_samples": sample(
            regressed,
            lambda row: numeric(row[b1_key]) - numeric(row[b3_key]),
        ),
        "unresolved_samples": sample(
            unresolved,
            lambda row: (str(row["category"]), str(row["query_id"])),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
