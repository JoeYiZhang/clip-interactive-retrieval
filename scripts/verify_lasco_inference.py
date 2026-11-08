"""Verify raw-image/free-text inference against the frozen engineering cache."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from cir_baseline.embedding_cache import atomic_json, digest_file, load_cache
from cir_baseline.encoders import OpenCLIPEncoder
from cir_baseline.lasco_inference import LaSCoRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/cache/lasco-train-engineering-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/fusion-runs/lasco-engineering-resume-v1/seed17/final.pt")
    parser.add_argument("--clip-checkpoint", type=Path, default=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/metrics/lasco_raw_inference_verification.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import faiss
    faiss.omp_set_num_threads(4)
    plan, matrices, _ = load_cache(args.cache_dir)
    if (plan["dataset"], plan["mode"], plan["split"], len(plan["queries"]), len(plan["images"])) != ("lasco", "engineering", "train", 8, 14):
        raise ValueError("this verifier is restricted to the fixed engineering sample")
    encoder = OpenCLIPEncoder(checkpoint_path=args.clip_checkpoint, device=args.device, force_quick_gelu=True, weights_only=False)
    retriever = LaSCoRetriever(args.cache_dir, args.checkpoint, encoder, args.device, allow_engineering=True)
    lookup = {key: i for i, key in enumerate(retriever.ids)}
    checks = []
    examples = []
    for i, row in enumerate(plan["queries"]):
        raw_query = retriever.compose_query(row["reference_id"], row["text"])
        ref = matrices["images"][[lookup[row["reference_id"]]]]
        text = matrices["texts"][[i]]
        with torch.inference_mode():
            cached_query = retriever.model(torch.from_numpy(ref).to(args.device), torch.from_numpy(text).to(args.device)).cpu().numpy()[0]
        error = float(np.max(np.abs(raw_query - cached_query)))
        if error > 1e-5:
            raise ValueError("raw/cache query divergence")
        result = retriever.search(row["reference_id"], row["text"], 10)
        scores = retriever.index.embeddings @ cached_query
        expected = [retriever.ids[j] for j in np.argsort(-scores, kind="stable") if retriever.ids[j] != row["reference_id"]][:10]
        actual = [hit["image_id"] for hit in result["results"]]
        # This tiny fixture has no tied top-10 boundary. Any discrepancy needs
        # inspection rather than accepting an arbitrary rank change.
        if actual != expected:
            raise ValueError(f"raw/cache exact top10 changed for query {row['query_id']}")
        if any(abs(hit["score"] - float(scores[lookup[hit["image_id"]]])) > 1e-5 for hit in result["results"]):
            raise ValueError("exported score differs from independent dot product")
        checks.append({"query_id": row["query_id"], "query_max_abs_error": error, "exact_top10_equal": True,
                       "reference_excluded": row["reference_id"] not in actual, "score_check": "pass"})
        if i == 0:
            examples.append(result)
    # A new user-written description, absent from cached annotation texts.
    free_text = "A group of people standing outside on a grassy hill"
    if free_text in {row["text"] for row in plan["queries"]}:
        raise ValueError("free-text fixture must not reuse an annotation")
    examples.append(retriever.search(plan["queries"][0]["reference_id"], free_text))
    report = {"status": "pass", "checked_at": datetime.now().astimezone().isoformat(),
              "scope": "8 training queries and 1 new text; engineering only, NOT held-out model accuracy",
              "gallery_count": 14, "cached_query_cases": len(checks), "free_text_cases": 1,
              "precision": {"cuda_matmul_tf32": False, "cudnn_tf32": False},
              "max_query_abs_error": max(row["query_max_abs_error"] for row in checks),
              "provenance": retriever.provenance, "checks": checks, "examples": examples,
              "implementation_sha256": {name: digest_file(ROOT / name) for name in
                  ("src/cir_baseline/lasco_inference.py", "scripts/retrieve_lasco_fusion.py", "scripts/verify_lasco_inference.py", "scripts/build_additional_cache.py")}}
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("checks", "examples", "implementation_sha256")}, indent=2))


if __name__ == "__main__":
    main()
