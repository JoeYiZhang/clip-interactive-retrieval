"""Real-encoder smoke test for stage 2.

This script verifies the parts of the stage-2 exit criteria that can be checked
without a licensed retrieval dataset: the project environment can import Torch,
OpenCLIP, FAISS and Pillow, the RTX GPU encodes one image and one text into
512-dim L2-normalized finite vectors, and an exact IndexFlatIP gallery returns a
manually inspectable Top-K list.

The gallery here is a small set of locally generated synthetic images. It is a
technical smoke test only and must not be reported as a research metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from cir_baseline.encoders import OpenCLIPEncoder  # noqa: E402
from cir_baseline.index import FaissFlatIPIndex, NumpyFlatIPIndex  # noqa: E402


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "clip-smoke-images"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "metrics" / "real_clip_smoke.json"
CONFIG_EMBEDDING_DIM = 512
SHAPES = ("circle", "square", "triangle")
COLORS = {
    "red": (220, 30, 30),
    "green": (30, 160, 60),
    "blue": (30, 60, 220),
    "yellow": (240, 200, 30),
    "black": (20, 20, 20),
}
QUERIES = (
    ("a red circle", "red", "circle"),
    ("a blue square", "blue", "square"),
    ("a green triangle", "green", "triangle"),
    ("a yellow square", "yellow", "square"),
)


def make_synthetic_gallery(directory: Path, size: int = 256) -> list[Path]:
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for color_name, rgb in COLORS.items():
        for shape in SHAPES:
            image = Image.new("RGB", (size, size), "white")
            draw = ImageDraw.Draw(image)
            if shape == "circle":
                draw.ellipse((48, 48, size - 48, size - 48), fill=rgb)
            elif shape == "square":
                draw.rectangle((48, 48, size - 48, size - 48), fill=rgb)
            else:
                draw.polygon(
                    [(size // 2, 40), (size - 40, size - 48), (40, size - 48)],
                    fill=rgb,
                )
            path = directory / f"{color_name}_{shape}.png"
            image.save(path)
            paths.append(path)
    return paths


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_weight_files(torch_module, max_age_seconds: int = 10800) -> list[Path]:
    """Locate encoder checkpoints downloaded during this run."""
    roots = [
        Path.home() / ".cache" / "clip",
        Path(torch_module.hub.get_dir()),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    now = time.time()
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pt", ".pth", ".bin", ".safetensors"}:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size < 10 * 1024 * 1024:
                continue
            if now - stat.st_mtime > max_age_seconds:
                continue
            candidates.append(path)
    candidates.sort(key=lambda item: item.stat().st_size, reverse=True)
    return candidates[:3]


def describe_weight_files(torch_module) -> list[dict]:
    described = []
    for path in find_weight_files(torch_module):
        started = time.perf_counter()
        described.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "hash_seconds": round(time.perf_counter() - started, 2),
            }
        )
    return described


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--images",
        nargs="*",
        type=Path,
        help="Optional explicit image list; defaults to a synthetic gallery.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="本地已校验权重文件；离线环境必须提供（open_clip 的 openai 标签会先走 HuggingFace）",
    )
    parser.add_argument(
        "--force-quick-gelu",
        dest="force_quick_gelu",
        action="store_true",
        default=True,
        help="OpenAI ViT-B 权重按 QuickGELU 训练；open_clip 3.x 不再从标签自动应用",
    )
    parser.add_argument(
        "--no-force-quick-gelu",
        dest="force_quick_gelu",
        action="store_false",
    )
    parser.add_argument(
        "--weights-only",
        dest="weights_only",
        action="store_true",
        default=False,
        help="OpenAI 原始 .pt 是 TorchScript 归档，torch>=2.6 需要 weights_only=False",
    )
    args = parser.parse_args()

    import open_clip
    import torch

    payload: dict = {
        "status": "fail",
        "scope": (
            "real OpenCLIP encoder + IndexFlatIP smoke test on synthetic images; "
            "not a research result"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    encoder = OpenCLIPEncoder(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        checkpoint_path=args.checkpoint,
        force_quick_gelu=args.force_quick_gelu,
        weights_only=args.weights_only,
    )
    device = encoder.device
    on_cuda = device.startswith("cuda")
    try:
        activation = type(
            encoder.model.visual.transformer.resblocks[0].mlp[1]
        ).__name__
    except Exception:  # noqa: BLE001 - diagnostics only
        activation = "unknown"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    image_paths = args.images or make_synthetic_gallery(args.image_dir)
    if args.images:
        for path in image_paths:
            if not path.is_file():
                raise SystemExit(f"missing image: {path}")

    started = time.perf_counter()
    image_vectors = encoder.encode_images(image_paths)
    image_seconds = time.perf_counter() - started

    started = time.perf_counter()
    text_vectors = encoder.encode_texts([text for text, _, _ in QUERIES])
    text_seconds = time.perf_counter() - started

    repeat_vectors = encoder.encode_images([image_paths[0]])
    repeat_diff = float(np.max(np.abs(repeat_vectors[0] - image_vectors[0])))

    peak_allocated = peak_reserved = None
    if on_cuda:
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

    manifest = IndexManifest(
        index_version="real-clip-smoke-flatip-v1",
        encoder_version=encoder.version,
        embedding_dim=int(image_vectors.shape[1]),
    )
    ids = [path.stem for path in image_paths]
    numpy_index = NumpyFlatIPIndex(ids, image_vectors, manifest)
    faiss_index = FaissFlatIPIndex(ids, image_vectors, manifest)

    query_results = []
    matches = 0
    expected_ranks: list[int | None] = []
    score_gaps = []
    retrieval_latencies = []
    for index, (text, color, shape) in enumerate(QUERIES):
        started = time.perf_counter()
        numpy_hits = numpy_index.search(text_vectors[index], args.top_k)
        retrieval_latencies.append((time.perf_counter() - started) * 1000)
        faiss_hits = faiss_index.search(text_vectors[index], args.top_k)
        same_order = [hit.image_id for hit in numpy_hits] == [
            hit.image_id for hit in faiss_hits
        ]
        score_gaps.append(
            max(
                abs(left.score - right.score)
                for left, right in zip(numpy_hits, faiss_hits, strict=True)
            )
        )
        expected = f"{color}_{shape}"
        top1 = numpy_hits[0].image_id if numpy_hits else ""
        matches += int(top1 == expected)
        full_hits = numpy_index.search(text_vectors[index], len(ids))
        expected_rank = next(
            (hit.rank for hit in full_hits if hit.image_id == expected), None
        )
        expected_ranks.append(expected_rank)
        query_results.append(
            {
                "text": text,
                "expected_id": expected,
                "top1": top1,
                "top1_correct": top1 == expected,
                "expected_rank": expected_rank,
                "numpy_faiss_same_order": same_order,
                "topk": [
                    {"image_id": hit.image_id, "score": round(hit.score, 6)}
                    for hit in numpy_hits
                ],
            }
        )

    image_norms = np.linalg.norm(image_vectors.astype(np.float64), axis=1)
    text_norms = np.linalg.norm(text_vectors.astype(np.float64), axis=1)
    payload["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if on_cuda else platform.processor(),
        "gpu_total_memory_mib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
            if on_cuda
            else None
        ),
        "open_clip": getattr(open_clip, "__version__", "unknown"),
    }
    payload["encoder"] = {
        "model_name": args.model,
        "pretrained": args.pretrained,
        "checkpoint_path": encoder.checkpoint_path,
        "weights_sha256": encoder.weights_sha256,
        "force_quick_gelu": args.force_quick_gelu,
        "weights_only": args.weights_only,
        "version": encoder.version,
        "embedding_dim": int(image_vectors.shape[1]),
        "configured_embedding_dim": CONFIG_EMBEDDING_DIM,
        "visual_activation": activation,
        "weight_files": (
            describe_weight_files(torch) if encoder.checkpoint_path is None else []
        ),
    }
    payload["checks"] = {
        "cuda_used": on_cuda,
        "embedding_dim_matches_config": int(image_vectors.shape[1])
        == CONFIG_EMBEDDING_DIM
        == int(text_vectors.shape[1]),
        "all_values_finite": bool(
            np.all(np.isfinite(image_vectors)) and np.all(np.isfinite(text_vectors))
        ),
        "image_norm_min": round(float(image_norms.min()), 6),
        "image_norm_max": round(float(image_norms.max()), 6),
        "text_norm_min": round(float(text_norms.min()), 6),
        "text_norm_max": round(float(text_norms.max()), 6),
        "repeat_encoding_max_abs_diff": repeat_diff,
        "faiss_topk_matches_numpy": all(
            item["numpy_faiss_same_order"] for item in query_results
        ),
        "faiss_numpy_max_score_diff": max(score_gaps) if score_gaps else None,
        "synthetic_top1_hits": f"{matches}/{len(QUERIES)}",
        "synthetic_expected_ranks": expected_ranks,
        "synthetic_median_expected_rank": (
            sorted(rank for rank in expected_ranks if rank is not None)[
                len([rank for rank in expected_ranks if rank is not None]) // 2
            ]
            if any(rank is not None for rank in expected_ranks)
            else None
        ),
        "visual_activation_is_quick_gelu": activation == "QuickGELU",
    }
    payload["gpu"] = {
        "peak_memory_allocated_mib": (
            round(peak_allocated / 1024**2, 1) if peak_allocated else None
        ),
        "peak_memory_reserved_mib": (
            round(peak_reserved / 1024**2, 1) if peak_reserved else None
        ),
    }
    payload["index"] = {
        "type": "faiss.IndexFlatIP + NumpyFlatIPIndex",
        "gallery_items": len(ids),
        "embedding_dim": int(image_vectors.shape[1]),
        "manifest": numpy_index.manifest.to_dict(),
        "mean_search_latency_ms": round(float(np.mean(retrieval_latencies)), 4),
    }
    payload["timing"] = {
        "image_encode_seconds": round(image_seconds, 3),
        "image_encode_per_image_seconds": round(image_seconds / len(image_paths), 4),
        "text_encode_seconds": round(text_seconds, 3),
    }
    payload["queries"] = query_results

    checks = payload["checks"]
    payload["status"] = (
        "pass"
        if checks["cuda_used"]
        and checks["embedding_dim_matches_config"]
        and checks["all_values_finite"]
        and abs(checks["image_norm_min"] - 1.0) < 1e-4
        and abs(checks["image_norm_max"] - 1.0) < 1e-4
        and abs(checks["text_norm_min"] - 1.0) < 1e-4
        and abs(checks["text_norm_max"] - 1.0) < 1e-4
        and checks["faiss_topk_matches_numpy"]
        else "fail"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
