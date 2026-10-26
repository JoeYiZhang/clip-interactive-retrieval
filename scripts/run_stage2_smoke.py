from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.contracts import IndexManifest  # noqa: E402
from cir_baseline.fusion import l2_normalize_rows  # noqa: E402
from cir_baseline.index import NumpyFlatIPIndex  # noqa: E402
from cir_baseline.metrics import recall_at_k  # noqa: E402


def run(seed: int = 20261026) -> dict:
    rng = np.random.default_rng(seed)
    item_count = 100
    dimension = 64
    gallery = l2_normalize_rows(
        rng.standard_normal((item_count, dimension), dtype=np.float32)
    )
    ids = [f"image_{index:04d}" for index in range(item_count)]
    manifest = IndexManifest(
        index_version="synthetic-flatip-v1",
        encoder_version="synthetic-encoder-v1",
        embedding_dim=dimension,
    )
    index = NumpyFlatIPIndex(ids, gallery, manifest)

    target_indices = [3, 17, 42, 88]
    rankings: list[list[str]] = []
    relevant: list[set[str]] = []
    latencies_ms: list[float] = []
    for target in target_indices:
        query = gallery[target] + 0.01 * rng.standard_normal(dimension, dtype=np.float32)
        started = time.perf_counter()
        results = index.search(query, top_k=10)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        rankings.append([result.image_id for result in results])
        relevant.append({ids[target]})

    metrics = {
        "recall_at_1": recall_at_k(rankings, relevant, 1),
        "recall_at_5": recall_at_k(rankings, relevant, 5),
        "recall_at_10": recall_at_k(rankings, relevant, 10),
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_max": float(np.max(latencies_ms)),
    }
    return {
        "status": "pass" if metrics["recall_at_1"] == 1.0 else "fail",
        "scope": "synthetic exact-retrieval smoke test; not a research result",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "gallery_items": item_count,
        "embedding_dim": dimension,
        "query_count": len(target_indices),
        "metrics": metrics,
        "manifest": index.manifest.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "metrics" / "stage2_smoke.json",
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

