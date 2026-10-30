"""Audit official annotations and run a deliberately small real-CLIP smoke test.

Sample galleries are engineering checks ONLY, never official dataset metrics.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import ssl
import sys
import time

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cir_baseline.encoders import OpenCLIPEncoder
from cir_baseline.fusion import l2_normalize_rows

def download(url, path):
    if path.exists():
        with Image.open(path) as image:
            image.verify()
        return
    for attempt in range(3):
        try:
            with httpx.Client(trust_env=False, verify=ssl.create_default_context(), timeout=45) as client:
                response = client.get(url)
                response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(response.content)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

def audit(rows, kind):
    query_ids = [str(row["id" if kind == "circo" else "qid"]) for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query IDs")
    invalid = []
    for query_id, row in zip(query_ids, rows):
        reason = None
        if kind == "circo":
            if not row["relative_caption"].strip() or not row["shared_concept"].strip():
                reason = "empty text"
            elif row["target_img_id"] not in row["gt_img_ids"] or row["reference_img_id"] in row["gt_img_ids"]:
                reason = "invalid positive IDs"
            elif len(row["gt_img_ids"]) != len(set(row["gt_img_ids"])):
                reason = "duplicate positives"
        else:
            if not row["query-text"].strip():
                reason = "empty text"
            elif row["query-image"][0] == row["target-image"][0]:
                reason = "reference equals target"
            for key in ("query-image", "target-image"):
                path = Path(row[key][1])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("unsafe image path")
        if reason:
            invalid.append({"query_id": query_id, "reason": reason})
    return {"status": "pass" if not invalid else "quarantine_required", "queries": len(rows),
            "usable_queries": len(rows)-len(invalid), "query_ids_unique": True, "invalid": invalid}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--dataset", choices=("circo", "lasco"), required=True)
    parser.add_argument("--sample-queries", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    annotation = (args.root / "downloads/circo-official/annotations/val.json" if args.dataset == "circo"
                  else args.root / "downloads/lasco_val.json")
    rows = json.loads(annotation.read_text("utf-8"))
    report = {"dataset": args.dataset, "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
              "audit": audit(rows, args.dataset), "scope": "sample engineering smoke ONLY; not benchmark performance"}
    jobs, queries = {}, []
    excluded = {row["query_id"] for row in report["audit"]["invalid"]}
    usable = [row for row in rows if str(row["id" if args.dataset == "circo" else "qid"]) not in excluded]
    for row in usable[:args.sample_queries]:
        if args.dataset == "circo":
            reference, targets = str(row["reference_img_id"]), list(map(str, row["gt_img_ids"]))
            text = f"{row['shared_concept']} {row['relative_caption']}"
            for key in [reference, *targets]:
                filename = f"{int(key):012}.jpg"
                jobs[key] = (f"https://s3.amazonaws.com/images.cocodataset.org/unlabeled2017/{filename}", args.root / "datasets/coco/unlabeled2017" / filename)
        else:
            reference, targets = str(row["query-image"][0]), [str(row["target-image"][0])]
            text = row["query-text"]
            for key in ("query-image", "target-image"):
                image_id, path = row[key]
                jobs[str(image_id)] = (f"https://s3.amazonaws.com/images.cocodataset.org/{path}", args.root / "datasets/coco" / path)
        queries.append({"reference": reference, "text": text})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: download(*job), jobs.values()))
    print(f"verified {len(jobs)} real images; loading CLIP", flush=True)
    encoder = OpenCLIPEncoder(checkpoint_path=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"), device=args.device, force_quick_gelu=True, weights_only=False)
    ids = sorted(jobs)
    vectors = encoder.encode_images([jobs[key][1] for key in ids])
    texts = encoder.encode_texts([row["text"] for row in queries])
    refs = vectors[[ids.index(row["reference"]) for row in queries]]
    weighted = l2_normalize_rows(.3 * refs + .7 * texts)
    outputs = {"fixed_image_weight_0.3": weighted}
    checkpoint = ROOT / "artifacts/fusion-runs/fashioniq-train-v1/residual_fusion-seed17.pt"
    if checkpoint.exists():
        import torch
        from cir_baseline.learned_fusion import LearnedFusion
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        assert saved["encoder_version"] == encoder.version
        model = LearnedFusion(**saved["model_config"]).to(args.device).eval()
        model.load_state_dict(saved["state_dict"])
        with torch.inference_mode():
            outputs["fashioniq_residual_seed17"] = model(torch.from_numpy(refs).to(args.device), torch.from_numpy(texts).to(args.device)).cpu().numpy()
    result = {}
    for name, output in outputs.items():
        assert np.isfinite(output).all() and np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-5)
        scores = output @ vectors.T
        rankings = []
        for query, row in zip(queries, scores):
            selected = [i for i in np.argsort(-row, kind="stable") if ids[i] != query["reference"]][:10]
            rankings.append({**query, "top_ids": [ids[i] for i in selected], "scores": [float(row[i]) for i in selected]})
        result[name] = rankings
    report.update(status="pass", gallery_images=len(ids), sample_queries=len(queries), encoder_version=encoder.version, results=result)
    target = ROOT / "artifacts/metrics" / f"{args.dataset}_download_smoke.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
