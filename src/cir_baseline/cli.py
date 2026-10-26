from __future__ import annotations

import importlib.util
import json
import platform
import sys


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "core_ready": _available("numpy"),
        "optional": {
            "torch": _available("torch"),
            "open_clip": _available("open_clip"),
            "faiss": _available("faiss"),
            "PIL": _available("PIL"),
            "pytest": _available("pytest"),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["core_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

