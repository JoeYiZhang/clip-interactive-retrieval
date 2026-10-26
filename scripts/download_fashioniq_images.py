"""Download official FashionIQ split images from the repository-provided URL map.

The script is resumable: valid existing images are skipped, downloads use a
temporary ``.part`` file, and failures are recorded in JSON for retry.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


_local = threading.local()


def _opener() -> urllib.request.OpenerDirector:
    if not hasattr(_local, "opener"):
        _local.opener = urllib.request.build_opener()
        _local.opener.addheaders = [("User-Agent", "FashionIQ-research-downloader/1.0")]
    return _local.opener


def read_url_map(metadata_root: Path, categories: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for category in categories:
        path = metadata_root / f"asin2url.{category}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"URL map not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                result[fields[0]] = fields[1]
    return result


def read_split_ids(
    dataset_root: Path, categories: tuple[str, ...], splits: tuple[str, ...]
) -> set[str]:
    result: set[str] = set()
    for category in categories:
        for split in splits:
            path = dataset_root / "image_splits" / f"split.{category}.{split}.json"
            if not path.is_file():
                raise FileNotFoundError(f"split file not found: {path}")
            result.update(str(value) for value in json.loads(path.read_text("utf-8")))
    return result


def _valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:  # noqa: BLE001 - invalid files are retried
        return False


def _download_one(
    image_id: str, url: str, output_dir: Path, timeout: int, retries: int
) -> dict:
    destination = output_dir / f"{image_id}.jpg"
    if _valid_image(destination):
        return {"image_id": image_id, "status": "skipped", "bytes": destination.stat().st_size}
    temporary = output_dir / f"{image_id}.jpg.part"
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url)
            with _opener().open(request, timeout=timeout) as response:
                payload = response.read()
            temporary.write_bytes(payload)
            if not _valid_image(temporary):
                raise ValueError("response is not a valid image")
            os.replace(temporary, destination)
            return {
                "image_id": image_id,
                "status": "downloaded",
                "bytes": len(payload),
                "attempt": attempt,
            }
        except Exception as exc:  # noqa: BLE001 - captured for a retry manifest
            last_error = f"{type(exc).__name__}: {exc}"
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {"image_id": image_id, "status": "failed", "error": last_error, "url": url}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=("dress", "shirt", "toptee"))
    parser.add_argument("--splits", nargs="+", default=("val",))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Connectivity/debug limit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    categories = tuple(args.categories)
    splits = tuple(args.splits)
    url_map = read_url_map(args.metadata_root, categories)
    requested = sorted(read_split_ids(args.dataset_root, categories, splits))
    missing_urls = [image_id for image_id in requested if image_id not in url_map]
    jobs = [(image_id, url_map[image_id]) for image_id in requested if image_id in url_map]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    output_dir = args.dataset_root / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.dataset_root / "download_report.json"
    payload: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "categories": categories,
        "splits": splits,
        "requested_count": len(requested),
        "mapped_count": len(requested) - len(missing_urls),
        "missing_url_count": len(missing_urls),
        "missing_url_samples": missing_urls[:50],
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        payload["status"] = "pass" if not missing_urls else "partial"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if not missing_urls else 2

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[dict] = []
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                image_id,
                url,
                output_dir,
                args.timeout,
                args.retries,
            ): image_id
            for image_id, url in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            counts[result["status"]] += 1
            processed += 1
            if result["status"] == "failed":
                failures.append(result)
            if processed % 250 == 0 or processed == len(jobs):
                print(
                    f"processed={processed}/{len(jobs)} downloaded={counts['downloaded']} "
                    f"skipped={counts['skipped']} failed={counts['failed']}",
                    flush=True,
                )
    payload.update(counts)
    payload["attempted_count"] = len(jobs)
    payload["failures"] = failures
    payload["status"] = "pass" if not failures and not missing_urls else "partial"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "failures"}, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
