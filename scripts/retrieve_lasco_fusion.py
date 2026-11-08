"""Retrieve with saved LaSCo weights, a registered reference image and free text."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from cir_baseline.embedding_cache import atomic_json
from cir_baseline.encoders import OpenCLIPEncoder
from cir_baseline.lasco_inference import LaSCoRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--clip-checkpoint", type=Path, default=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-engineering", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("output exists; choose a new path to preserve prior evidence")
    torch.set_num_threads(4)
    # cuDNN can choose TF32 for a single image but FP32 for a cached image batch.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    encoder = OpenCLIPEncoder(checkpoint_path=args.clip_checkpoint, device=args.device, force_quick_gelu=True, weights_only=False)
    retriever = LaSCoRetriever(args.cache_dir, args.fusion_checkpoint, encoder, args.device, args.allow_engineering)
    result = retriever.search(args.reference_id, args.text, args.top_k)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
