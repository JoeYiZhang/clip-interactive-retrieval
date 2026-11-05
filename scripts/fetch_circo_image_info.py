"""Fetch the official COCO unlabeled2017 image inventory (small metadata ZIP)."""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import zipfile

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=Path(r"D:\image_data"))
    parser.add_argument("--proxy", default="")
    args = parser.parse_args()
    folder = args.storage_root / "downloads/coco"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "image_info_unlabeled2017.zip"
    url = "https://s3.amazonaws.com/images.cocodataset.org/annotations/image_info_unlabeled2017.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.tmp")
        with httpx.Client(trust_env=False, proxy=args.proxy or None, verify=ssl.create_default_context(), timeout=60, follow_redirects=True) as client:
            head = client.head(url)
            head.raise_for_status()
            size = int(head.headers["Content-Length"])
            for attempt in range(5):
                start = temporary.stat().st_size if temporary.exists() else 0
                if start == size:
                    break
                if start > size:
                    raise ValueError("metadata partial file exceeds source size")
                try:
                    with client.stream("GET", url, headers={"Range": f"bytes={start}-{size-1}", "If-Range": head.headers["ETag"]}) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{size-1}/{size}":
                            raise ValueError("metadata server did not honor range")
                        with temporary.open("ab") as handle:
                            for block in response.iter_bytes(1 << 18):
                                handle.write(block)
                except httpx.HTTPError:
                    if attempt == 4:
                        raise
            if temporary.stat().st_size != size:
                raise ValueError("metadata download incomplete")
        with zipfile.ZipFile(temporary) as source:
            if source.testzip() is not None:
                raise ValueError("metadata ZIP CRC failure")
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:
            raise ValueError("metadata ZIP CRC failure")
        payload = source.read("annotations/image_info_unlabeled2017.json")
    metadata = json.loads(payload)
    images = metadata["images"]
    if not images or len({row["id"] for row in images}) != len(images):
        raise ValueError("empty or duplicate official image IDs")
    for row in images:
        if row["file_name"] != f"{int(row['id']):012}.jpg":
            raise ValueError("unexpected image path")
    destination = folder / "image_info_unlabeled2017.json"
    destination.write_bytes(payload)
    report = {"url": url, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "json_sha256": hashlib.sha256(payload).hexdigest(), "image_count": len(images),
              "zip_crc": "pass", "status": "complete"}
    (folder / "image_info_unlabeled2017.source.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
