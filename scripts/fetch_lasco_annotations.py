"""Fetch official LaSCo JSON atomically; keep originals and source hashes."""
import hashlib
import json
from pathlib import Path
import ssl
import time
import httpx

root = Path(r"D:\image_data\downloads")
manifest = []
for name in ("lasco_val.json", "lasco_train.json"):
    target = root / name
    url = f"https://raw.githubusercontent.com/levymsn/LaSCo/main/downloads/{name}"
    if not target.exists():
        temporary = target.with_suffix(".json.part")
        for attempt in range(12):
            try:
                proxy = None if attempt % 2 == 0 else "http://127.0.0.1:7897"
                with httpx.Client(trust_env=False, proxy=proxy, verify=ssl.create_default_context(), timeout=120, follow_redirects=True) as client:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as handle:
                            for block in response.iter_bytes(1 << 20):
                                handle.write(block)
                                print(f"{name}: {handle.tell()} bytes", flush=True)
                json.loads(temporary.read_text("utf-8"))
                break
            except Exception as error:
                print(f"{name}: attempt {attempt+1} failed: {error}", flush=True)
                if attempt == 11:
                    raise
                time.sleep(10)
        temporary.replace(target)
    rows = json.loads(target.read_text("utf-8"))
    manifest.append({"url": url, "path": str(target), "queries": len(rows), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (root / "lasco_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest[-1], flush=True)
