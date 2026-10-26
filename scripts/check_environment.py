from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def gpu_summary() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def collect() -> dict:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_summary(),
        "modules": {
            name: available(name)
            for name in ["numpy", "torch", "open_clip", "faiss", "PIL", "pytest"]
        },
    }
    payload["stage2_core_ready"] = payload["modules"]["numpy"]
    payload["real_clip_faiss_ready"] = all(
        payload["modules"][name] for name in ["torch", "open_clip", "faiss", "PIL"]
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = collect()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["stage2_core_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

