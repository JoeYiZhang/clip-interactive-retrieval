"""Verify twelve checkpoints and report the fixed fit/dev LR/dropout design."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "artifacts/cache/matplotlib"))
from train_fashioniq_regularization import ROOT, ExperimentData, fit_supervision, read_json, sha256, write_json
from train_fashioniq_hard_negatives import evaluate
from analyze_fashioniq_hard_negatives import independently_recompute
from cir_baseline.learned_fusion import LearnedFusion
import numpy as np
import torch


def numeric_close(a, b, tolerance=1e-12):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(numeric_close(a[key], b[key], tolerance) for key in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    return a == b


def training_diagnostics(summary, curves):
    output = {}
    for name in summary["config"]["variants"]:
        runs = [row for row in summary["runs"] if row["regularization_variant"] == name]
        best_rows = [next(r for r in curves[f"{name}-seed{row['seed']}"] if r["epoch"] == row["best_epoch"]) for row in runs]
        last_rows = [curves[f"{name}-seed{row['seed']}"][-1] for row in runs]
        values = {}
        for stage, rows in (("selected", best_rows), ("last", last_rows)):
            values[stage] = {f"{part}_mean_R10": float(np.mean([row[part]["category_mean_recall@10"] for row in rows])) for part in ("fit", "dev")}
            values[stage]["mean_fit_dev_R10_gap"] = values[stage]["fit_mean_R10"] - values[stage]["dev_mean_R10"]
        values["selected_category_means"] = {cat: {f"recall@{k}": float(np.mean([row["dev"]["by_category"][cat][f"recall@{k}"] for row in runs])) for k in (10, 50)}
                                              for cat in ("dress", "shirt", "toptee")}
        values["best_epochs"] = [row["best_epoch"] for row in runs]
        values["best_at_budget_boundary_seeds"] = [row["seed"] for row in runs if row["best_epoch_at_budget_boundary"]]
        output[name] = values
    return output


def factorial_contrasts(summary):
    rows = {(row["regularization_variant"], row["seed"]): row["dev"] for row in summary["runs"]}
    pairs = {"lower_lr_at_dropout_0_1": ("low_lr", "control"),
             "lower_lr_at_dropout_0_3": ("low_lr_high_dropout", "high_dropout"),
             "higher_dropout_at_lr_0_001": ("high_dropout", "control"),
             "higher_dropout_at_lr_0_0003": ("low_lr_high_dropout", "low_lr")}
    output = {}
    for key, (a, b) in pairs.items():
        output[key] = {}
        for k in (10, 50):
            deltas = [rows[(a, seed)][f"category_mean_recall@{k}"] - rows[(b, seed)][f"category_mean_recall@{k}"] for seed in summary["config"]["seeds"]]
            output[key][f"R{k}"] = {"paired_seed_deltas": deltas, "mean_delta": float(np.mean(deltas))}
    return output


def render_curves(summary, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = {"control": "Control", "low_lr": "Lower LR", "high_dropout": "Dropout 0.3", "low_lr_high_dropout": "Lower LR + dropout 0.3"}
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    for (name, label), color in zip(labels.items(), colors, strict=True):
        rows = [curves[f"{name}-seed{seed}"] for seed in summary["config"]["seeds"]]
        x = [row["epoch"] for row in rows[0]]
        for ax, split, k in ((axes[0, 0], "dev", 10), (axes[0, 1], "dev", 50), (axes[1, 0], "fit", 10)):
            values = 100 * np.asarray([[row[split][f"category_mean_recall@{k}"] for row in trace] for trace in rows])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(x, mean, color=color, marker="o", markersize=3, label=label)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.10)
            ax.set(ylabel=f"{split.title()} category mean Recall@{k} (%)", xlabel="Epoch")
        gaps = 100 * np.asarray([[row["fit"]["category_mean_recall@10"] - row["dev"]["category_mean_recall@10"] for row in trace] for trace in rows])
        axes[1, 1].plot(x, gaps.mean(0), color=color, marker="o", markersize=3, label=label)
    axes[1, 1].set(ylabel="Fit - dev Recall@10 (percentage points)", xlabel="Epoch")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("FashionIQ LR/dropout experiment | 3 seeds per arm | bands: sample SD, not confidence intervals")
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/fusion-runs/fashioniq-regularization-dev-v1")
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact-prefix", default="fashioniq_regularization")
    args = parser.parse_args()
    if not args.artifact_prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("artifact prefix must be a simple name")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    summary = read_json(args.run_dir / "summary.json")
    protocol = read_json(args.run_dir / "protocol.json")
    curves = read_json(args.run_dir / "training_curves.json")
    config = summary["config"]
    data = ExperimentData(args.storage_root, protocol["effective_configs"]["control"])
    _, audit = fit_supervision(data)
    checks = {
        "protocol_hash": sha256(args.run_dir / "protocol.json") == summary["protocol_sha256"],
        "config_identity": config == protocol["config"],
        "cache_hashes": data.cache == protocol["cache"],
        "split_audit": data.split_audit == summary["split_audit"] == protocol["split_audit"],
        "fit_supervision": audit == protocol["supervision_audit"] == summary["supervision_audit"],
        "registered_source_snapshots": all(sha256(args.run_dir / "sources" / name) == digest for name, digest in protocol["source_sha256"].items()),
        "current_training_source_unchanged": all(sha256(ROOT / name) == digest for name, digest in protocol["source_sha256"].items()),
    }
    old_dir = ROOT / "artifacts/fusion-runs/fashioniq-hard-negatives-dev-v1"
    old_summary = read_json(old_dir / "summary.json")
    checks["previous_control_summary_unchanged"] = sha256(old_dir / "summary.json") == protocol["previous_summary_sha256"]
    rank_path = args.run_dir / "dev_rankings.jsonl.gz"
    replayed, ranks_by_name, recomputed_by_name = {}, {}, {}
    with gzip.open(rank_path, "wt", encoding="utf-8") as handle:
        for row in summary["runs"]:
            variant, seed = row["regularization_variant"], row["seed"]
            name = f"{variant}-seed{seed}"
            checkpoint = args.run_dir / variant / f"in_batch-seed{seed}.pt"
            trace_path = args.run_dir / variant / f"in_batch-seed{seed}-training.json"
            trace = read_json(trace_path)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            effective = protocol["effective_configs"][variant]
            checks[f"checkpoint_hash_{name}"] = sha256(checkpoint) == row["checkpoint_sha256"]
            checks[f"config_and_metadata_{name}"] = (
                row["effective_config"] == effective == read_json(args.run_dir / variant / "effective_config.json")
                and saved["seed"] == seed and saved["epoch"] == row["best_epoch"]
                and saved["encoder_version"] == data.encoder_version and saved["text_strategy"] == effective["text_strategy"]
                and saved["model_config"]["dropout"] == effective["dropout"])
            checks[f"trace_hash_and_budget_{name}"] = sha256(trace_path) == row["trace_sha256"] and trace == curves[name] and [r["epoch"] for r in trace] == list(range(0, 13, 2))
            selected = max(trace, key=lambda r: (r["dev"]["category_mean_recall@10"], r["dev"]["category_mean_recall@50"], -r["epoch"]))
            checks[f"checkpoint_selection_{name}"] = selected["epoch"] == row["best_epoch"] and selected["dev"] == row["dev"] and row["best_epoch_at_budget_boundary"] == (row["best_epoch"] == config["epochs"])
            model = LearnedFusion(**saved["model_config"]).to(args.device)
            model.load_state_dict(saved["state_dict"])
            observed, ranks, rankings = evaluate(model, data, "dev", args.device, rankings=True)
            checks[f"checkpoint_replay_exact_{name}"] = observed == row["dev"]
            ranking_rows = []
            for i, ranking in zip(data.positions["dev"], rankings, strict=True):
                q = data.queries[i]
                ids = ranking["ranked_image_ids"]
                if len(ids) != 50 or len(set(ids)) != 50 or q.reference_image_id in ids:
                    raise ValueError("ranking contains duplicate/reference or wrong length")
                record = {"model": name, "query_id": q.query_id, "category": q.category,
                          "target_image_ids": q.target_image_ids, "reference_image_id": q.reference_image_id, **ranking}
                ranking_rows.append(record)
                handle.write(json.dumps(record) + "\n")
            recomputed = independently_recompute(ranking_rows)
            checks[f"independent_metrics_{name}"] = numeric_close(recomputed, observed)
            replayed[name], ranks_by_name[name], recomputed_by_name[name] = observed, ranks, recomputed
            if variant == "control":
                old = next(r for r in old_summary["runs"] if r["arm"] == "in_batch" and r["seed"] == seed)
                old_saved = torch.load(old["checkpoint"], map_location="cpu", weights_only=True)
                checks[f"previous_control_trace_and_weights_{seed}"] = trace == read_json(old_dir / f"in_batch-seed{seed}-training.json") and all(torch.equal(saved["state_dict"][key], value) for key, value in old_saved["state_dict"].items())
            print(f"verified {name} epoch={saved['epoch']}", flush=True)
    means = {}
    for variant in config["variants"]:
        values = np.asarray([[recomputed_by_name[f"{variant}-seed{seed}"][f"category_mean_recall@{k}"] for k in (10, 50)] for seed in config["seeds"]])
        means[variant] = {f"{stat}_R{k}": float(getattr(values[:, j], method)(**kwargs))
                          for j, k in enumerate((10, 50)) for stat, method, kwargs in (("mean", "mean", {}), ("sd", "std", {"ddof": 1}))}
        checks[f"three_seed_aggregation_{variant}"] = numeric_close(means[variant], summary["dev_three_seed_summary"][variant])
        deltas = [replayed[f"{variant}-seed{seed}"]["category_mean_recall@10"] - replayed[f"control-seed{seed}"]["category_mean_recall@10"] for seed in config["seeds"]]
        recorded = summary["contrasts_vs_control"][variant]
        checks[f"screen_{variant}"] = recorded["screen_pass"] == (float(np.mean(deltas)) >= .005 and all(d > 0 for d in deltas) and means[variant]["mean_R50"] >= means["control"]["mean_R50"])
    checks["variant_selection"] = max(means, key=lambda n: (means[n]["mean_R10"], means[n]["mean_R50"])) == summary["selected_variant"]
    checks["deployment_unchanged"] = summary["deployed_model_changed"] is False
    for seed in config["seeds"]:
        rows = [row for row in summary["runs"] if row["seed"] == seed]
        checks[f"paired_batch_orders_{seed}"] = len({tuple(row["batch_order_sha256_by_epoch"]) for row in rows}) == 1
        checks[f"paired_candidates_{seed}"] = len({(row["negative_audit"]["samples"], row["negative_audit"]["valid_candidate_sum"], row["negative_audit"]["extra_count"]) for row in rows}) == 1 and all(row["negative_audit"]["extra_count"] == 0 for row in rows)
        checks[f"initial_metrics_identical_{seed}"] = all(curves[f"{name}-seed{seed}"][0] == curves[f"control-seed{seed}"][0] for name in config["variants"])
    np.savez_compressed(args.run_dir / "dev_ranks.npz", **ranks_by_name)
    diagnostics = training_diagnostics(summary, curves)
    diagnostics["factorial_contrasts"] = factorial_contrasts(summary)
    figure = ROOT / "artifacts/figures" / f"{args.artifact_prefix}_curves"
    render_curves(summary, curves, figure)
    source_archive = ROOT / "artifacts/provenance" / f"{args.artifact_prefix}_sources.zip"
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive, "w") as archive:
        files = {"protocol.json": args.run_dir / "protocol.json", **{f"sources/{name}": args.run_dir / "sources" / name for name in protocol["source_sha256"]}}
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    verification = {"status": "pass" if all(checks.values()) else "fail", "check_count": len(checks), "checks": checks,
                    "ranking_count": len(data.positions["dev"]) * len(summary["runs"]), "rankings_sha256": sha256(rank_path),
                    "independent_metric_absolute_tolerance": 1e-12, "source_archive_sha256": sha256(source_archive),
                    "analysis_source_sha256": {"analyze_fashioniq_regularization.py": sha256(Path(__file__)),
                                               "analyze_fashioniq_hard_negatives.py": sha256(ROOT / "scripts/analyze_fashioniq_hard_negatives.py")},
                    "scope": "fit/dev only; same dev used for selection, not independent confirmation"}
    output = ROOT / "artifacts/metrics"
    write_json(output / f"{args.artifact_prefix}_verification.json", verification)
    write_json(output / f"{args.artifact_prefix}_diagnostics.json", diagnostics)
    write_json(output / f"{args.artifact_prefix}_training_curves.json", curves)
    if not all(checks.values()):
        raise ValueError([key for key, value in checks.items() if not value])
    candidate = next(row for row in summary["runs"] if row["regularization_variant"] == summary["selected_variant"] and row["seed"] == 17)
    write_json(output / f"{args.artifact_prefix}_candidate_manifest.json", {
        "status": "research_only", "regularization_variant": summary["selected_variant"],
        "selection_source": "three-seed dev means; seed17 fixed for engineering inference, not best-seed selection",
        "development_screen_pass": summary["contrasts_vs_control"][summary["selected_variant"]]["screen_pass"],
        "inference_seed": 17, "epoch": candidate["best_epoch"], "checkpoint": candidate["checkpoint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"], "encoder_version": data.encoder_version,
        "text_strategy": candidate["effective_config"]["text_strategy"],
        "summary_sha256": sha256(args.run_dir / "summary.json"), "deployed_model_changed": False})
    print(json.dumps({"status": verification["status"], "checks": len(checks), "rankings": verification["ranking_count"]}), flush=True)


if __name__ == "__main__":
    main()
