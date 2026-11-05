"""Train LaSCo fusion heads over fully verified, resumable embedding caches."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from cir_baseline.embedding_cache import atomic_json, read_json
from cir_baseline.lasco_runner import cache_inputs, train_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lasco_training.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    config = read_json(args.config)
    train = cache_inputs(args.train_cache)
    dev = cache_inputs(args.dev_cache) if args.dev_cache else None
    results = [train_cached(train, dev, config, seed, args.output_dir / f"seed{seed}", args.device, args.resume, args.stop_after_epoch) for seed in config["seeds"]]
    summary = {"mode": config["mode"], "results": results, "status": "pass" if all(row["status"] == "pass" for row in results) else "interrupted"}
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
