"""Build resumable frozen CLIP caches with explicit full/small-sample modes."""
import argparse
import json
import sys
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cir_baseline.dataset_plans import make_dataset_plan
from cir_baseline.embedding_cache import build_cache, digest_file
from cir_baseline.encoders import OpenCLIPEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("lasco", "circo"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--mode", choices=("full", "engineering"), default="full")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--stop-after-shards", type=int)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    version = f"open_clip:ViT-B-32:local-sha256:{digest_file(args.checkpoint)[:16]}|quick_gelu=1"
    plan = make_dataset_plan(args.storage_root, args.dataset, args.split, args.mode, version, args.sample_count)
    plan["encoder_runtime"] = {name: package_version(name) for name in ("torch", "open_clip_torch", "pillow")}
    plan["encoder_adapter_sha256"] = digest_file(ROOT / "src/cir_baseline/encoders.py")
    state = build_cache(plan, args.cache_dir, lambda: OpenCLIPEncoder(checkpoint_path=args.checkpoint, device=args.device, force_quick_gelu=True, weights_only=False),
                        args.batch_size, args.shard_size, args.stop_after_shards)
    print(json.dumps({"status": state["status"], "mode": plan["mode"], "dataset": plan["dataset"], "split": plan["split"],
                      "images": len(plan["images"]), "queries": len(plan["queries"]), "committed_shards": len(state["shards"]),
                      "plan_sha256": state["plan_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
