"""Resume official COCO archives with checked HTTP ranges, then CRC-check/extract.

The S3 URLs address the same official bucket linked by images.cocodataset.org.
Completed chunks are recorded separately from the preallocated file size.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import zipfile
import ssl

import httpx

ARCHIVES = {
    "train2014": 13510573713,
    "val2014": 6645013297,
    "unlabeled2017": 20126613414,
}

def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

def acquire(name, root, proxy, workers, chunk_size):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "downloads" / "coco" / f"{name}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://s3.amazonaws.com/images.cocodataset.org/zips/{name}.zip"
    def session():
        return httpx.Client(trust_env=False, proxy=proxy or None, verify=ssl.create_default_context(), follow_redirects=True, timeout=120)
    with session() as client:
        response = client.head(url, timeout=40)
        response.raise_for_status()
        size, etag = int(response.headers["Content-Length"]), response.headers["ETag"]
    part, state_path = archive.with_suffix(".zip.part"), archive.with_suffix(".zip.progress.json")
    signature = {"url": url, "size": size, "etag": etag, "chunk_size": chunk_size}
    state = json.loads(state_path.read_text()) if state_path.exists() else {**signature, "done": []}
    if any(state.get(k) != v for k, v in signature.items()):
        raise ValueError("remote identity or chunk size changed; preserve old evidence and use a fresh directory")
    done = set(state["done"])
    if not archive.exists():
        if not part.exists():
            if done:
                raise ValueError("progress exists but partial archive is missing")
            with part.open("wb") as handle:
                handle.truncate(size)
        if part.stat().st_size != size:
            raise ValueError("partial archive size mismatch")
        lock = threading.Lock()
        started = time.monotonic()
        def fetch(index):
            start, stop = index * chunk_size, min((index + 1) * chunk_size, size) - 1
            for attempt in range(5):
                try:
                    with session() as client, client.stream("GET", url, headers={"Range": f"bytes={start}-{stop}"}) as result:
                        result.raise_for_status()
                        expected = f"bytes {start}-{stop}/{size}"
                        if result.status_code != 206 or result.headers.get("Content-Range") != expected:
                            raise ValueError(f"server did not honor range: {result.status_code} {result.headers.get('Content-Range')}")
                        received = 0
                        with part.open("r+b") as handle:
                            handle.seek(start)
                            for block in result.iter_bytes(1 << 20):
                                if received + len(block) > stop - start + 1:
                                    raise ValueError("oversized range response")
                                handle.write(block)
                                received += len(block)
                            handle.flush()
                            os.fsync(handle.fileno())
                        if received != stop - start + 1:
                            raise ValueError("truncated range response")
                    with lock:
                        done.add(index)
                        state["done"] = sorted(done)
                        save(state_path, state)
                        completed = sum(min(chunk_size, size - i * chunk_size) for i in done)
                        print(f"{name}: {completed}/{size} bytes ({100*completed/size:.1f}%) elapsed={time.monotonic()-started:.0f}s", flush=True)
                    return
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 15))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, i) for i in range((size + chunk_size - 1)//chunk_size) if i not in done]
            for future in as_completed(futures):
                future.result()
        part.replace(archive)
    if archive.stat().st_size != size:
        raise ValueError("archive size differs from remote")
    print(f"{name}: verifying ZIP CRC and extracting", flush=True)
    destination = root / "datasets" / "coco"
    destination.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    count = 0
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("unsafe ZIP member path")
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reading the entire member verifies its ZIP CRC even on repeated runs.
            with zipped.open(info) as source, target.open("wb") as output:
                while block := source.read(1 << 20):
                    output.write(block)
            count += 1
    report = {**signature, "archive": str(archive), "sha256": digest, "image_count": count,
              "extracted_to": str(destination / name), "zip_crc": "pass", "status": "complete"}
    save(archive.with_suffix(".zip.verified.json"), report)
    print(json.dumps(report), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--archives", nargs="+", choices=tuple(ARCHIVES), default=list(ARCHIVES))
    parser.add_argument("--proxy", default="")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for name in args.archives:
        acquire(name, args.root, args.proxy, args.workers, 8 << 20)

if __name__ == "__main__":
    main()
