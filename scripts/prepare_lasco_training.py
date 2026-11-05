"""Prepare split-isolated training manifests and verify a real-image train step."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cir_baseline.lasco_training import prepare_examples, candidate_masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fetch-smoke-images", action="store_true")
    args = parser.parse_args()
    root = Path(r"D:\image_data")
    output = ROOT / "artifacts/training-data/lasco-v1"
    output.mkdir(parents=True, exist_ok=True)
    summary, image_sets = {}, {}
    for split in ("train", "val"):
        source = root / "downloads" / f"lasco_{split}.json"
        examples, positives, quarantine = prepare_examples(json.loads(source.read_text("utf-8")), split)
        destination = output / f"{split}.jsonl.gz"
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            for row in examples:
                from cir_baseline.lasco_training import query_key
                handle.write(json.dumps({**row, "known_positive_ids": sorted(positives[query_key(row['reference_id'], row['text'])])}, ensure_ascii=False) + "\n")
        image_sets[split] = {row[key] for row in examples for key in ("reference_id", "target_id")}
        summary[split] = {"examples": len(examples), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "manifest": str(destination), "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                          "multi_target_query_conditions": sum(len(value)>1 for value in positives.values()),
                          "quarantine": quarantine}
        if args.smoke and split == "train":
            if args.fetch_smoke_images:
                from concurrent.futures import ThreadPoolExecutor
                from test_additional_datasets import download
                paths = sorted({row[key] for row in examples[:8] for key in ('reference_path', 'target_path')})
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(lambda path: download(f"https://s3.amazonaws.com/images.cocodataset.org/{path}", root / 'datasets/coco' / path), paths))
            # Downloaded val images are never used for this optimizer check.
            selected = [row for row in examples if (root / "datasets/coco" / row['reference_path']).exists() and (root / "datasets/coco" / row['target_path']).exists()][:8]
            summary["smoke"] = {"status": "waiting_for_train_images", "available_train_examples": len(selected)}
            if len(selected) >= 2:
                import numpy as np
                import torch
                from cir_baseline.encoders import OpenCLIPEncoder
                from cir_baseline.learned_fusion import LearnedFusion, masked_multi_positive_loss
                torch.manual_seed(17)
                encoder = OpenCLIPEncoder(checkpoint_path=Path(r"C:\Users\25062\.cache\clip\ViT-B-32.pt"), device="cuda", force_quick_gelu=True, weights_only=False)
                refs = torch.from_numpy(encoder.encode_images([root / 'datasets/coco' / row['reference_path'] for row in selected])).cuda()
                texts = torch.from_numpy(encoder.encode_texts([row['text'] for row in selected])).cuda()
                ids = sorted({row['target_id'] for row in selected})
                paths = {row['target_id']: row['target_path'] for row in selected}
                targets = torch.from_numpy(encoder.encode_images([root / 'datasets/coco' / paths[key] for key in ids])).cuda()
                positive, valid = candidate_masks(selected, ids, positives)
                model = LearnedFusion().cuda().eval()
                before = model(refs, texts).detach().clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                loss = masked_multi_positive_loss(model(refs, texts), targets, torch.from_numpy(positive).cuda(), torch.from_numpy(valid).cuda())
                loss.backward()
                optimizer.step()
                changed = not torch.allclose(before, model(refs, texts))
                if not torch.isfinite(loss) or not changed:
                    raise ValueError("real train-step failed")
                summary['smoke'] = {"status": "pass", "queries":len(selected), "loss":float(loss.detach()), "parameters_updated":changed, "scope":"optimizer engineering check only; not model quality"}
    if image_sets['train'] & image_sets['val']:
        raise ValueError("supervised image overlap")
    summary['train_val_image_overlap'] = 0
    target = ROOT / "artifacts/metrics/lasco_training_preparation.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {a:b for a,b in v.items() if a != 'quarantine'} if isinstance(v,dict) else v for k,v in summary.items()}, indent=2))

if __name__ == "__main__":
    main()
