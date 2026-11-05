"""Fixed 2x2 LR/dropout experiment using the verified fit/dev-only trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from train_fashioniq_hard_negatives import (
    ROOT, ExperimentData, fit_image_pool, positive_key, sha256, train_one, write_json,
)
from train_fashioniq_fusion import objective
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def effective_configs(protocol, base):
    expected = {
        "control": {"learning_rate": .001, "dropout": .1},
        "low_lr": {"learning_rate": .0003, "dropout": .1},
        "high_dropout": {"learning_rate": .001, "dropout": .3},
        "low_lr_high_dropout": {"learning_rate": .0003, "dropout": .3},
    }
    if protocol["variants"] != expected or protocol["seeds"] != [17, 29, 43]:
        raise ValueError("unexpected preregistered factorial or seeds")
    if protocol["training_arm"] != "in_batch" or protocol["epochs"] != 12 or protocol["evaluate_every"] != 2:
        raise ValueError("unexpected training budget/arm")
    if protocol["candidate_screen"] != {"minimum_mean_R10_delta": .005, "all_paired_seeds_R10_improve": True, "mean_R50_not_worse": True}:
        raise ValueError("unexpected development screening rule")
    if base["dataset_split"] != "train" or base["split_seed"] != 20261028 or base["text_strategy"] != "symmetric_join_mean":
        raise ValueError("only the frozen train split/text strategy is allowed")
    fixed = {"batch_size": 256, "hidden_dim": 128, "weight_decay": .0001,
             "temperature": .07, "anchor_penalty": .02, "residual_scale": .2, "initial_image_weight": .3}
    if any(base[key] != value for key, value in fixed.items()):
        raise ValueError("a nonexperimental factor changed")
    active_keys = {"dataset_split", "text_strategy", "split_seed", "seeds", "epochs", "evaluate_every",
                   "batch_size", "hidden_dim", "dropout", "learning_rate", "weight_decay", "temperature",
                   "residual_scale", "anchor_penalty", "initial_image_weight"}
    active_base = {key: base[key] for key in active_keys}
    return {name: {**active_base, **changes, "experiment_id": protocol["experiment_id"],
                   "regularization_variant": name, "epochs": protocol["epochs"],
                   "evaluate_every": protocol["evaluate_every"]} for name, changes in expected.items()}


def fit_supervision(data):
    fit = data.positions["fit"]
    relation = defaultdict(set)
    for i in fit:
        relation[positive_key(data.ref_positions[i], data.queries[i].modification_texts)].add(int(data.target_positions[i]))
    positives = [relation[positive_key(data.ref_positions[i], data.queries[i].modification_texts)] for i in fit]
    pools = {}
    for category in sorted(set(data.categories)):
        pool = set(fit_image_pool(data.partition, data.ref_positions, data.target_positions, data.categories, category).tolist())
        local = np.flatnonzero(data.categories[fit] == category)
        if any(not positives[i] <= pool for i in local):
            raise ValueError("positive crosses category or fit boundary")
        pools[category] = len(pool)
    rows = [{"query_id": data.queries[i].query_id, "reference": data.references[i],
             "positives": sorted(data.ids[j] for j in positive)} for i, positive in zip(fit, positives, strict=True)]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return positives, {"query_count": len(fit), "multi_positive_queries": sum(len(p) > 1 for p in positives),
                        "same_category_fit_pool_sizes": pools, "supervision_sha256": digest,
                        "extra_negatives": 0, "source": "fit annotations only"}


def summarize_runs(runs, protocol):
    means = {}
    by_variant = {}
    for name in protocol["variants"]:
        by_variant[name] = {row["seed"]: row for row in runs if row["regularization_variant"] == name}
        if set(by_variant[name]) != set(protocol["seeds"]) or sum(row["regularization_variant"] == name for row in runs) != len(protocol["seeds"]):
            raise ValueError("missing or duplicate seed")
        values = np.asarray([objective(by_variant[name][seed]["dev"]) for seed in protocol["seeds"]])
        means[name] = {f"{stat}_R{k}": float(getattr(values[:, j], method)(**kwargs))
                       for j, k in enumerate((10, 50)) for stat, method, kwargs in
                       (("mean", "mean", {}), ("sd", "std", {"ddof": 1}))}
    selected = max(protocol["variants"], key=lambda name: (means[name]["mean_R10"], means[name]["mean_R50"]))
    contrasts = {}
    for name in protocol["variants"]:
        deltas = [objective(by_variant[name][seed]["dev"])[0] - objective(by_variant["control"][seed]["dev"])[0] for seed in protocol["seeds"]]
        delta10 = means[name]["mean_R10"] - means["control"]["mean_R10"]
        delta50 = means[name]["mean_R50"] - means["control"]["mean_R50"]
        gates = {"mean_R10_delta_at_least_0_005": delta10 >= protocol["candidate_screen"]["minimum_mean_R10_delta"],
                 "all_paired_seeds_R10_improve": all(delta > 0 for delta in deltas),
                 "mean_R50_not_worse": delta50 >= 0}
        contrasts[name] = {"paired_seed_R10_deltas": deltas, "mean_R10_delta": delta10, "mean_R50_delta": delta50,
                           "development_screen": gates, "screen_pass": all(gates.values())}
    return {"dev_three_seed_summary": means, "selected_variant": selected, "contrasts_vs_control": contrasts,
            "decision": "development_candidate_requires_independent_confirmation" if contrasts[selected]["screen_pass"] else "no_candidate_passed_development_screen",
            "deployed_model_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fashioniq_regularization.json")
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/fusion-runs/fashioniq-regularization-dev-v1")
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts/metrics/fashioniq_regularization_dev_v1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    base = read_json(ROOT / config["base_config"])
    variants = effective_configs(config, base)
    if args.output_dir.exists() and not args.resume:
        raise ValueError("run directory exists; use --resume or a fresh directory")
    if (args.output_dir / "summary.json").exists():
        raise ValueError("run complete; do not overwrite evidence")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    data = ExperimentData(args.storage_root, base)
    positives, supervision_audit = fit_supervision(data)
    old_dir = ROOT / "artifacts/fusion-runs/fashioniq-hard-negatives-dev-v1"
    old_summary, old_protocol = read_json(old_dir / "summary.json"), read_json(old_dir / "protocol.json")
    if data.cache != old_protocol["cache"] or data.split_audit != old_summary["split_audit"]:
        raise ValueError("comparison data changed")
    source_files = [Path(__file__), ROOT / "scripts/train_fashioniq_hard_negatives.py", ROOT / "scripts/train_fashioniq_fusion.py",
                    ROOT / "src/cir_baseline/learned_fusion.py", ROOT / "src/cir_baseline/hard_negatives.py"]
    protocol = {"config": config, "effective_configs": variants, "cache": data.cache, "split_audit": data.split_audit,
                "supervision_audit": supervision_audit, "previous_summary_sha256": sha256(old_dir / "summary.json"),
                "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files},
                "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "device": args.device,
                                "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None}}
    path = args.output_dir / "protocol.json"
    if args.resume and path.exists():
        previous = read_json(path)
        if {key: value for key, value in previous.items() if key != "registered_at_utc"} != protocol:
            raise ValueError("protocol/data/code/environment changed; cannot resume")
    else:
        if args.resume and any(args.output_dir.iterdir()):
            raise ValueError("cannot resume a nonempty run without a protocol")
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        for source in source_files:
            target = args.output_dir / "sources" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        write_json(path, protocol)
    runs, prior_checks, curves = [], {}, {}
    for variant, effective in variants.items():
        directory = args.output_dir / variant
        directory.mkdir(exist_ok=True)
        write_json(directory / "effective_config.json", effective)
        for seed in config["seeds"]:
            print(f"regularization_variant={variant} seed={seed}", flush=True)
            row = train_one(data, effective, "in_batch", seed, positives, {}, directory, args.device, args.resume)
            trace = read_json(directory / f"in_batch-seed{seed}-training.json")
            row = {**row, "regularization_variant": variant, "effective_config": effective,
                   "best_epoch_at_budget_boundary": row["best_epoch"] == config["epochs"],
                   "trace_sha256": sha256(directory / f"in_batch-seed{seed}-training.json")}
            runs.append(row)
            curves[f"{variant}-seed{seed}"] = trace
            if variant == "control":
                prior_trace = read_json(old_dir / f"in_batch-seed{seed}-training.json")
                previous = next(r for r in old_summary["runs"] if r["arm"] == "in_batch" and r["seed"] == seed)
                old_state = torch.load(previous["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
                new_state = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
                prior_checks[str(seed)] = {"trace_exact": trace == prior_trace, "selected_metrics_exact": row["dev"] == previous["dev"],
                                           "checkpoint_state_exact": all(torch.equal(new_state[key], old_state[key]) for key in old_state)}
                if not all(prior_checks[str(seed)].values()):
                    raise ValueError("control no longer reproduces prior experiment")
    pairing = {}
    for seed in config["seeds"]:
        paired = [row for row in runs if row["seed"] == seed]
        pairing[str(seed)] = {"batch_orders_identical": len({tuple(row["batch_order_sha256_by_epoch"]) for row in paired}) == 1,
                              "candidate_counts_identical": len({(row["negative_audit"]["samples"], row["negative_audit"]["valid_candidate_sum"], row["negative_audit"]["extra_count"]) for row in paired}) == 1}
        if not all(pairing[str(seed)].values()):
            raise ValueError("paired training conditions differ")
    summary = {"status": "pass", "scope": "fit/dev development only; no independent confirmation",
               "config": config, "protocol_sha256": sha256(path), "split_audit": data.split_audit,
               "supervision_audit": supervision_audit, "prior_control_reproduction": prior_checks, "pairing_audit": pairing,
               "A2_fit_only_baseline": old_summary["A2_fit_only_baseline"], "runs": runs, **summarize_runs(runs, config),
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds_this_invocation": round(time.perf_counter() - started, 3)}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "training_curves.json", curves)
    write_json(args.summary, summary)
    print(json.dumps({"means": summary["dev_three_seed_summary"], "selected": summary["selected_variant"], "decision": summary["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
