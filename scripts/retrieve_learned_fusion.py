"""Run a saved research fusion head with real reference image + free text.

Uses the existing train gallery; no target ID is accepted as model input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cir_baseline.encoders import OpenCLIPEncoder
from cir_baseline.fusion import l2_normalize
from cir_baseline.index_registry import IndexRegistry
from cir_baseline.learned_fusion import LearnedFusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--category", choices=("dress", "shirt", "toptee"), required=True)
    parser.add_argument("--text", nargs="+", required=True, help="one description or two separate captions")
    parser.add_argument("--fusion-checkpoint", type=Path, default=ROOT / "artifacts/fusion-runs/fashioniq-train-v1/residual_fusion-seed17.pt")
    parser.add_argument("--clip-checkpoint", type=Path, default=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"))
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    captions = [value.strip() for value in args.text if value.strip()]
    if not 1 <= len(captions) <= 2 or not 1 <= args.top_k <= 100:
        raise ValueError("provide one or two captions and top_k between 1 and 100")
    paths = sorted((args.storage_root / "derived/embeddings").glob("fashioniq_train_official_*.npz"))
    if len(paths) != 1:
        raise ValueError("expected exactly one train gallery cache")
    registry = IndexRegistry.from_fashioniq_cache(data_root=args.storage_root / "datasets/fashioniq", cache_path=paths[0], split="train")
    entry = registry.get(f"fashioniq:train:{args.category}")
    reference = entry.embedding_by_id.get(args.reference_id)
    if reference is None:
        raise ValueError("reference is not present in the selected train category")
    saved = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=True)
    if saved["encoder_version"] != entry.index.manifest.encoder_version or saved["text_strategy"] != "symmetric_join_mean":
        raise ValueError("fusion checkpoint is incompatible with the index/text strategy")
    encoder = OpenCLIPEncoder(checkpoint_path=args.clip_checkpoint, device=args.device, force_quick_gelu=True, weights_only=False)
    if encoder.version != saved["encoder_version"]:
        raise ValueError("CLIP checkpoint does not match the trained fusion model")
    encoded = encoder.encode_texts([" and ".join(captions), " and ".join(reversed(captions))])
    text = l2_normalize(encoded[0] + encoded[1])
    model = LearnedFusion(**saved["model_config"]).to(args.device).eval()
    model.load_state_dict(saved["state_dict"])
    with torch.inference_mode():
        query = model(torch.from_numpy(np.array(reference)[None]).to(args.device), torch.from_numpy(text[None]).to(args.device)).cpu().numpy()[0]
    hits = [hit for hit in entry.index.search(query, args.top_k + 1) if hit.image_id != args.reference_id][:args.top_k]
    output = {
        "scope": "research model inference; online A2 remains unchanged", "dataset_split": "train",
        "reference_image_id": args.reference_id, "category": args.category, "captions": captions,
        "checkpoint": str(args.fusion_checkpoint.resolve()), "variant": saved["variant"], "seed": saved["seed"], "epoch": saved["epoch"],
        "encoder_version": encoder.version, "query_norm": float(np.linalg.norm(query)),
        "results": [{"rank": i, "image_id": hit.image_id, "score": hit.score, "image_path": str(entry.image_path(hit.image_id))} for i, hit in enumerate(hits, 1)],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
