"""Independently recompute saved ranking metrics and replay selected checkpoints."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_fashioniq_fusion import ExperimentData, ROOT, baseline_vectors, infer, sha256, write_json
from cir_baseline.learned_fusion import LearnedFusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts/metrics/learned_fusion_train_v1.json")
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/metrics/learned_fusion_verification.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config = summary["config"]
    run_dir = Path(summary["artifacts"]["run_dir"])
    import faiss
    faiss.omp_set_num_threads(4)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = ExperimentData(args.storage_root, config)
    checks = {}
    checks["input_cache_hashes"] = data.cache == summary["cache"]
    split_path = run_dir / "split_manifest.json"
    checks["split_manifest_hash"] = sha256(split_path) == summary["split_manifest_sha256"]
    split_rows = json.loads(split_path.read_text(encoding="utf-8"))
    expected_split = [{"query_id": q.query_id, "partition": str(data.partition[i]), "component": str(data.groups[i])} for i, q in enumerate(data.queries)]
    checks["split_recomputed"] = split_rows == expected_split
    checks["zero_supervised_image_overlap"] = not any(data.split_audit["supervised_image_overlap"].values())
    selection_path = run_dir / "selection_frozen_before_holdout.json"
    checks["selection_hash"] = sha256(selection_path) == summary["selection_manifest_sha256"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checks["selection_matches_summary"] = selection == summary["selection"]
    variant_means = {}
    for variant in config["variants"]:
        runs = [run for run in selection["runs"] if run["variant"] == variant]
        variant_means[variant] = np.mean([
            [run["dev"][f"category_mean_recall@{k}"] for k in (10, 50)] for run in runs
        ], axis=0)
    selected = max(config["variants"], key=lambda name: tuple(variant_means[name]))
    checks["variant_chosen_from_dev"] = selected == selection["selected_variant"]
    checks["checkpoints_and_epoch_selection"] = True
    for run in selection["runs"]:
        trace = json.loads((run_dir / f"{run['variant']}-seed{run['seed']}-training.json").read_text(encoding="utf-8"))
        best = max(trace, key=lambda row: tuple(row["dev"][f"category_mean_recall@{k}"] for k in (10, 50)))
        checks["checkpoints_and_epoch_selection"] &= (
            best["epoch"] == run["best_epoch"] and best["dev"] == run["dev"]
            and sha256(Path(run["checkpoint"])) == run["checkpoint_sha256"]
        )

    holdout = data.positions["holdout"]
    query_by_id = {data.queries[i].query_id: data.queries[i] for i in holdout}
    ids_by_category = {category: set(index.image_ids) for category, index in data.indices.items()}
    rows_by_model = defaultdict(dict)
    ranking_path = Path(summary["artifacts"]["rankings"])
    checks["rankings_hash"] = sha256(ranking_path) == summary["artifacts"]["rankings_sha256"]
    with gzip.open(ranking_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            model, query_id = row["model"], row["query_id"]
            if model not in summary["holdout"] or query_id not in query_by_id or query_id in rows_by_model[model]:
                raise ValueError("unexpected model, query ID or duplicate ranking")
            query = query_by_id[query_id]
            ids, scores = row["ranked_image_ids"], np.asarray(row["scores"])
            if len(ids) != 50 or len(set(ids)) != 50 or scores.shape != (50,):
                raise ValueError("incomplete or duplicate Top-50")
            if not np.all(np.isfinite(scores)) or np.any(np.diff(scores) > 1e-6):
                raise ValueError("invalid or unsorted scores")
            if query.reference_image_id in ids or not set(ids) <= ids_by_category[query.category]:
                raise ValueError("reference leak or cross-category retrieval")
            if row["category"] != query.category or row["reference_image_id"] != query.reference_image_id or row["target_image_ids"] != list(query.target_image_ids):
                raise ValueError("ranking metadata differs from official train annotations")
            target = query.target_image_ids[0]
            rank = ids.index(target) + 1 if target in ids else 51
            if rank != row["target_rank"]:
                raise ValueError("saved target rank cannot be reproduced")
            rows_by_model[model][query_id] = row
    checks["rankings_complete_and_valid"] = (
        set(rows_by_model) == set(summary["holdout"])
        and all(set(rows) == set(query_by_id) for rows in rows_by_model.values())
    )
    checks["metrics_recomputed"] = True
    for name, rows in rows_by_model.items():
        for k in (10, 50):
            category_recalls = []
            for category in sorted(ids_by_category):
                items = [row for row in rows.values() if row["category"] == category]
                recall = sum(row["target_rank"] <= k for row in items) / len(items)
                checks["metrics_recomputed"] &= abs(recall - summary["holdout"][name]["by_category"][category][f"recall@{k}"]) < 1e-12
                category_recalls.append(recall)
            checks["metrics_recomputed"] &= abs(float(np.mean(category_recalls)) - summary["holdout"][name][f"category_mean_recall@{k}"]) < 1e-12
    names = [f"{selected}-seed{seed}" for seed in config["seeds"]]
    checks["three_seed_mean_and_sd"] = True
    for k in (10, 50):
        recalls = [summary["holdout"][name][f"category_mean_recall@{k}"] for name in names]
        checks["three_seed_mean_and_sd"] &= bool(np.isclose(np.mean(recalls), summary["three_seed_summary"][f"category_mean_R{k}"], atol=1e-12))
        checks["three_seed_mean_and_sd"] &= bool(np.isclose(np.std(recalls, ddof=1), summary["three_seed_summary"][f"R{k}_sample_sd"], atol=1e-12))

    replay = {}
    def compare(name, vectors):
        _, rankings, _ = data.search(holdout, vectors, keep_rankings=True)
        matches = 0
        for i, ranking in enumerate(rankings):
            row = rows_by_model[name][data.queries[holdout[i]].query_id]
            if ranking["ranked_image_ids"] == row["ranked_image_ids"] and np.allclose(ranking["scores"], row["scores"], atol=1e-6, rtol=0):
                matches += 1
        replay[name] = {"matching_queries": matches, "expected_queries": len(holdout)}

    for name, vectors in baseline_vectors(data, holdout, summary["baseline_parameters"]).items():
        compare(name, vectors)
    for run in selection["runs"]:
        if run["variant"] != selected:
            continue
        saved = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
        if saved["encoder_version"] != data.encoder_version or saved["text_strategy"] != config["text_strategy"]:
            raise ValueError("checkpoint encoder or caption version mismatch")
        model = LearnedFusion(**saved["model_config"]).to(args.device)
        model.load_state_dict(saved["state_dict"])
        compare(f"{selected}-seed{run['seed']}", infer(model, data, holdout, args.device))
    checks["checkpoint_and_baseline_rankings_replayed"] = all(row["matching_queries"] == row["expected_queries"] for row in replay.values())
    baseline = rows_by_model["A2_refit_dynamic"]
    cases = {"consistent_rescues_at_10": [], "consistent_regressions_at_10": []}
    for query_id in sorted(query_by_id):
        old = baseline[query_id]["target_rank"]
        ranks = [rows_by_model[name][query_id]["target_rank"] for name in names]
        key = "consistent_rescues_at_10" if old > 10 and all(rank <= 10 for rank in ranks) else "consistent_regressions_at_10" if old <= 10 and all(rank > 10 for rank in ranks) else None
        if key:
            q = query_by_id[query_id]
            cases[key].append({"query_id": query_id, "category": q.category, "text": q.modification_text,
                               "reference": q.reference_image_id, "target": q.target_image_ids[0],
                               "A2_rank": old, "candidate_ranks_by_seed": dict(zip(map(str, config["seeds"]), ranks, strict=True))})
    case_report = {"sampling": "first 10 query IDs, same success/failure direction in all seeds", "counts": {key: len(value) for key, value in cases.items()},
                   "examples": {key: value[:10] for key, value in cases.items()}}
    write_json(ROOT / "artifacts/metrics/learned_fusion_cases.json", case_report)
    output = {"status": "pass" if all(checks.values()) else "fail", "scope": summary["scope"],
              "checks": checks, "ranking_records": sum(len(value) for value in rows_by_model.values()),
              "replay": replay, "cases": case_report["counts"]}
    write_json(args.output, output)
    print(json.dumps(output, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
