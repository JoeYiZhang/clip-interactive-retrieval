"""Replay all dev checkpoints, audit sampling, and render research figures."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import textwrap
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "artifacts/cache/matplotlib"))
from train_fashioniq_hard_negatives import ROOT, ExperimentData, evaluate, sha256, write_json
from cir_baseline.hard_negatives import fit_image_pool
from cir_baseline.learned_fusion import LearnedFusion
import numpy as np
import torch


def independently_recompute(rows):
    categories = sorted({r["category"] for r in rows})
    values = {}
    for category in categories:
        selected = [r for r in rows if r["category"] == category]
        values[category] = {f"recall@{k}": sum(bool(set(r["target_image_ids"]) & set(r["ranked_image_ids"][:k])) for r in selected) / len(selected) for k in (10, 50)}
    return {"query_count": len(rows), "by_category": values,
            **{f"category_mean_recall@{k}": sum(r[f"recall@{k}"] for r in values.values()) / len(categories) for k in (10, 50)}}


def render_figures(summary, diagnosis, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    destination = ROOT / "artifacts/figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for row in diagnosis["runs"]:
        trace = row["trace"]
        x = [item["epoch"] for item in trace]
        ax[0].plot(x, [100 * item["dev"]["category_mean_recall@10"] for item in trace], marker="o", label=f"Seed {row['seed']}")
        ax[1].plot(x[1:], [item["loss"] for item in trace[1:]], marker="o", label=f"Seed {row['seed']}")
    ax[0].set(ylabel="Dev category mean Recall@10 (%)", title="Previous residual: dev performance")
    ax[1].set(ylabel="Training objective", title="Previous residual: loss keeps falling")
    for item in ax:
        item.set_xlabel("Epoch")
        item.grid(alpha=.2)
        item.legend()
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"fashioniq_prior_training_curves.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = {"in_batch": "#4C78A8", "random_16": "#F58518", "hard_16": "#54A24B"}
    labels = {"in_batch": "In-batch", "random_16": "Random +16", "hard_16": "Hard +16"}
    for arm in summary["config"]["arms"]:
        traces = [json.loads((run_dir / f"{arm}-seed{seed}-training.json").read_text(encoding="utf-8")) for seed in summary["config"]["seeds"]]
        x = [r["epoch"] for r in traces[0]]
        for axis, split, k in ((axes[0, 0], "dev", 10), (axes[0, 1], "dev", 50), (axes[1, 0], "fit", 10)):
            values = 100 * np.asarray([[row[split][f"category_mean_recall@{k}"] for row in trace] for trace in traces])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axis.plot(x, mean, color=colors[arm], marker="o", markersize=3, label=labels[arm])
            axis.fill_between(x, mean - sd, mean + sd, color=colors[arm], alpha=.14)
            axis.set(ylabel=f"{split.title()} category mean Recall@{k} (%)", xlabel="Epoch")
            axis.grid(alpha=.2)
            axis.legend()
    arm_names = summary["config"]["arms"]
    bars = [summary["dev_three_seed_summary"][a] for a in arm_names]
    axes[1, 1].bar([labels[a] for a in arm_names], [100 * r["mean_R10"] for r in bars],
                   yerr=[100 * r["sd_R10"] for r in bars], color=[colors[a] for a in arm_names], capsize=5)
    axes[1, 1].axhline(100 * summary["A2_fit_only_baseline"]["dev"]["category_mean_recall@10"], color="#555555", ls="--", label="Fit-only A2")
    axes[1, 1].set(ylabel="Selected dev Recall@10 (%)", title="Per-seed best epoch; mean +/- sample SD")
    axes[1, 1].legend()
    fig.suptitle("FashionIQ fit/dev only | 3 seeds per arm | bands: sample SD, not confidence intervals")
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"fashioniq_hard_negatives_curves.{ext}", dpi=180)
    plt.close(fig)
    return [str(destination / name) for name in ("fashioniq_prior_training_curves.png", "fashioniq_hard_negatives_curves.png")]


def render_cases(diagnosis, data, run_dir):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    # Representative dev-only examples stay local alongside the data-derived run.
    selected = []
    for group in ("rescued", "regressed", "outside_top50"):
        for category in ("dress", "shirt", "toptee"):
            selected.append(next(row for row in diagnosis["cases"] if row["failure_group"] == group and row["category"] == category))
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 15)
    paths = {item.image_id: item.path for item in data.bundle.gallery}
    canvas = Image.new("RGB", (710, 270 * len(selected) + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), "Dev only | Previous residual seed17 | Fixed query-ID sampling", fill="black", font=font)
    for i, row in enumerate(selected):
        y = 40 + i * 270
        for j, (label, key) in enumerate((("Reference", row["reference"]), ("Target", row["target"]), ("Residual Top-1", row["top5"][0]))):
            x = 15 + j * 230
            draw.text((x, y), label + " " + key, fill="black", font=font)
            with Image.open(paths[key]) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (210, 160))
                canvas.paste(thumb, (x + (210 - thumb.width) // 2, y + 24))
        label = f"{row['category']} | {row['failure_group']} | A2 rank {row['A2_rank']} -> residual {row['residual_rank']} (51 means >50)"
        draw.text((15, y + 188), label, fill="black", font=font)
        for k, line in enumerate(textwrap.wrap(" / ".join(row["text"]), width=90)[:3]):
            draw.text((15, y + 208 + k * 18), line, fill="#444444", font=font)
    target = run_dir / "prior_dev_cases.png"
    canvas.save(target)
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/fusion-runs/fashioniq-hard-negatives-dev-v1")
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.run_dir / "protocol.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((args.run_dir / "prior_fit_dev_diagnosis.json").read_text(encoding="utf-8"))
    config = summary["config"]
    data = ExperimentData(args.storage_root, config)
    checks = {}
    checks["protocol_hash"] = sha256(args.run_dir / "protocol.json") == summary["protocol_sha256"]
    checks["registered_source_snapshots"] = all(sha256(args.run_dir / "sources" / key) == value for key, value in protocol["source_sha256"].items())
    current_source_changes = {key: {"registered": value, "current": sha256(ROOT / key)}
                              for key, value in protocol["source_sha256"].items() if sha256(ROOT / key) != value}
    checks["cache_hashes"] = data.cache == protocol["cache"]
    checks["split_audit"] = data.split_audit == summary["split_audit"]
    bank_path = args.run_dir / "negative_banks.npz"
    checks["bank_hash"] = sha256(bank_path) == summary["negative_source_audit"]["banks_sha256"]
    with np.load(bank_path, allow_pickle=False) as banks:
        fit = data.positions["fit"]
        checks["bank_fit_order"] = np.array_equal(banks["fit_positions"], fit)
        for category in sorted(set(data.categories)):
            pool = fit_image_pool(data.partition, data.ref_positions, data.target_positions, data.categories, category)
            local = np.flatnonzero(data.categories[fit] == category)
            checks[f"fit_only_candidates_{category}"] = all(bool(np.isin(banks[name][local], pool).all()) for name in ("hard", "random"))
    rank_archive = args.run_dir / "dev_rankings.jsonl.gz"
    values_by_arm, full_ranks, training_curves, metric_deltas = {}, {}, {}, {}
    with gzip.open(rank_archive, "wt", encoding="utf-8") as handle:
        for run in summary["runs"]:
            name = f"{run['arm']}-seed{run['seed']}"
            checkpoint = Path(run["checkpoint"])
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            checks[f"checkpoint_hash_{name}"] = sha256(checkpoint) == run["checkpoint_sha256"]
            checks[f"checkpoint_metadata_{name}"] = saved["encoder_version"] == data.encoder_version and saved["epoch"] == run["best_epoch"] and saved["seed"] == run["seed"]
            trace = json.loads((args.run_dir / f"{name}-training.json").read_text(encoding="utf-8"))
            training_curves[name] = trace
            best = max(trace, key=lambda row: (row["dev"]["category_mean_recall@10"], row["dev"]["category_mean_recall@50"], -row["epoch"]))
            checks[f"dev_selection_{name}"] = best["epoch"] == run["best_epoch"] and best["dev"] == run["dev"]
            model = LearnedFusion(**saved["model_config"]).to(args.device)
            model.load_state_dict(saved["state_dict"])
            observed, ranks, hits = evaluate(model, data, "dev", args.device, rankings=True)
            rows = []
            for i, ranking in zip(data.positions["dev"], hits, strict=True):
                query = data.queries[i]
                ids = ranking["ranked_image_ids"]
                if len(ids) != 50 or len(set(ids)) != 50 or query.reference_image_id in ids:
                    raise ValueError("invalid ranking contents")
                row = {"model": name, "query_id": query.query_id, "category": query.category,
                       "reference_image_id": query.reference_image_id, "target_image_ids": query.target_image_ids, **ranking}
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
            recomputed = independently_recompute(rows)
            checks[f"checkpoint_replay_exact_metrics_{name}"] = observed == run["dev"]
            differences = [abs(recomputed[f"category_mean_recall@{k}"] - observed[f"category_mean_recall@{k}"]) for k in (10, 50)]
            differences.extend(abs(recomputed["by_category"][cat][f"recall@{k}"] - observed["by_category"][cat][f"recall@{k}"])
                               for cat in observed["by_category"] for k in (10, 50))
            metric_deltas[name] = max(differences)
            # Python sum and numpy mean can differ at ~1e-17 on identical hits.
            checks[f"independent_metrics_{name}"] = recomputed["query_count"] == observed["query_count"] and max(differences) <= 1e-12
            values_by_arm.setdefault(run["arm"], []).append([recomputed[f"category_mean_recall@{k}"] for k in (10, 50)])
            full_ranks[name] = ranks
            print(f"verified {name} epoch={saved['epoch']}", flush=True)
    for arm, values in values_by_arm.items():
        values = np.asarray(values)
        expected = summary["dev_three_seed_summary"][arm]
        checks[f"three_seed_aggregation_{arm}"] = all(abs(expected[f"{stat}_R{k}"] - float(getattr(values[:, j], method)(**kwargs))) < 1e-12
            for j, k in enumerate((10, 50)) for stat, method, kwargs in (("mean", "mean", {}), ("sd", "std", {"ddof": 1})))
    for seed in config["seeds"]:
        runs = [row for row in summary["runs"] if row["seed"] == seed]
        checks[f"paired_batch_order_{seed}"] = len({tuple(row["batch_order_sha256_by_epoch"]) for row in runs}) == 1
        audits = [row["negative_audit"] for row in runs if row["arm"] != "in_batch"]
        checks[f"matched_candidate_counts_{seed}"] = len({(a["samples"], a["extra_count"], a["valid_candidate_sum"]) for a in audits}) == 1
        checks[f"exactly_16_extras_{seed}"] = all(a["extra_count"] == 16 * a["samples"] for a in audits)
    np.savez_compressed(args.run_dir / "dev_ranks.npz", **full_ranks)
    figures = render_figures(summary, diagnosis, args.run_dir)
    cases = render_cases(diagnosis, data, args.run_dir)
    source_archive = ROOT / "artifacts/provenance/fashioniq-hard-negatives-dev-v1-sources.zip"
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        files = {"protocol.json": args.run_dir / "protocol.json",
                 **{f"sources/{key}": args.run_dir / "sources" / key for key in protocol["source_sha256"]}}
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name.replace("\\", "/"))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    verification = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
                    "check_count": len(checks), "ranking_count": len(data.positions["dev"]) * len(summary["runs"]),
                    "independent_metric_absolute_tolerance": 1e-12, "independent_metric_max_abs_deltas": metric_deltas,
                    "current_source_changes_since_registration": current_source_changes,
                    "source_change_note": "Original registered source bytes retained and hash-checked; see experiment report for any listed post-run changes.",
                    "registered_sources_archive": str(source_archive), "registered_sources_archive_sha256": sha256(source_archive),
                    "rankings_sha256": sha256(rank_archive), "figures": figures, "local_case_sheet": cases,
                    "scope": "dev-only checkpoint replay; source/hash/pairing audit; independent Recall recomputation"}
    write_json(ROOT / "artifacts/metrics/fashioniq_hard_negatives_verification.json", verification)
    write_json(ROOT / "artifacts/metrics/fashioniq_prior_fit_dev_diagnosis.json", diagnosis)
    write_json(ROOT / "artifacts/metrics/fashioniq_hard_negatives_training_curves.json", training_curves)
    if not all(checks.values()):
        raise ValueError([key for key, value in checks.items() if not value])
    print(json.dumps({"status": verification["status"], "checks": len(checks), "rankings": verification["ranking_count"]}))


if __name__ == "__main__":
    main()
