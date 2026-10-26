"""Independently verify saved rankings and recompute reported recall metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.datasets import load_cirr, load_fashioniq  # noqa: E402
from cir_baseline.metrics import recall_at_k  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cirr", "fashioniq"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--version", default="rc2")
    parser.add_argument(
        "--categories", nargs="+", default=("dress", "shirt", "toptee")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.dataset == "cirr":
        bundle = load_cirr(args.data_root, split=args.split, version=args.version)
    else:
        bundle = load_fashioniq(
            args.data_root, split=args.split, categories=args.categories
        )
    summary = json.loads((args.run_dir / "summary.json").read_text("utf-8"))
    baselines = set(summary["baselines"])
    queries = {query.query_id: query for query in bundle.queries}
    records: dict[tuple[str, str], dict] = {}
    errors: list[str] = []
    with (args.run_dir / "rankings.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            key = (str(row["baseline"]), str(row["query_id"]))
            if key in records:
                errors.append(f"duplicate ranking at line {line_number}: {key}")
                continue
            records[key] = row
            query = queries.get(key[1])
            if key[0] not in baselines or query is None:
                errors.append(f"unknown baseline/query at line {line_number}: {key}")
                continue
            ranking = [str(value) for value in row["ranked_image_ids"]]
            if query.reference_image_id in ranking:
                errors.append(f"reference leaked into ranking: {key}")
            if len(ranking) != len(set(ranking)):
                errors.append(f"duplicate image id in ranking: {key}")
            if len(ranking) < 50:
                errors.append(f"ranking shorter than 50: {key}")

    expected_count = len(baselines) * len(queries)
    if len(records) != expected_count:
        errors.append(f"record count {len(records)} != expected {expected_count}")

    recomputed: dict[str, dict] = {}
    ordered_queries = list(bundle.queries)
    relevant = [set(query.target_image_ids) for query in ordered_queries]
    for baseline in sorted(baselines):
        if any((baseline, query.query_id) not in records for query in ordered_queries):
            continue
        rankings = [
            records[(baseline, query.query_id)]["ranked_image_ids"]
            for query in ordered_queries
        ]
        metrics = {
            f"recall@{k}": recall_at_k(rankings, relevant, k)
            for k in ((1, 5, 10, 50) if args.dataset == "cirr" else (10, 50))
        }
        if args.dataset == "cirr":
            subset = [
                records[(baseline, query.query_id)]["subset_ranked_image_ids"]
                for query in ordered_queries
            ]
            metrics.update(
                {
                    f"subset_recall@{k}": recall_at_k(subset, relevant, k)
                    for k in (1, 2, 3)
                }
            )
        recomputed[baseline] = metrics
        reported = summary["baselines"][baseline]["metrics"]
        for name, value in metrics.items():
            if abs(float(reported[name]) - value) > 1e-12:
                errors.append(
                    f"metric mismatch {baseline}/{name}: {reported[name]} != {value}"
                )

    report = {
        "status": "pass" if not errors else "fail",
        "dataset": args.dataset,
        "split": args.split,
        "baseline_count": len(baselines),
        "query_count": len(queries),
        "ranking_record_count": len(records),
        "expected_ranking_record_count": expected_count,
        "checks": {
            "no_reference_leakage": not any("reference leaked" in item for item in errors),
            "no_duplicate_ranked_ids": not any("duplicate image" in item for item in errors),
            "all_rankings_have_top50": not any("shorter than 50" in item for item in errors),
            "record_count_matches": len(records) == expected_count,
            "reported_metrics_recomputed_exactly": not any(
                "metric mismatch" in item for item in errors
            ),
        },
        "recomputed_metrics": recomputed,
        "errors": errors[:100],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
