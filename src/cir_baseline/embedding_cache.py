"""Versioned, bounded-batch embedding shards with crash-safe continuation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_path(plan, row):
    root = Path(plan["image_root"]).resolve()
    relative = Path(row["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("image path escapes declared root")
    return path


def validate_plan(plan):
    if plan["mode"] not in ("full", "engineering") or plan["dimension"] < 1:
        raise ValueError("invalid cache scope/dimension")
    images, queries = plan["images"], plan["queries"]
    image_ids = {row["id"] for row in images}
    if not images or not queries or len(image_ids) != len(images) or len({r["query_id"] for r in queries}) != len(queries):
        raise ValueError("empty or duplicate cache IDs")
    for row in images:
        source_path(plan, row)
    for row in queries:
        if row["reference_id"] not in image_ids or not row["text"].strip():
            raise ValueError("query missing reference/text")
        if not set(row.get("positive_ids", [])) <= image_ids and plan["mode"] == "full":
            raise ValueError("full gallery lacks positive images")
        if row.get("target_id") is not None and row["target_id"] not in image_ids:
            raise ValueError("gallery lacks annotated target")


def shard_specs(plan, size):
    for kind, rows, id_key in (("images", plan["images"], "id"), ("texts", plan["queries"], "query_id")):
        for start in range(0, len(rows), size):
            batch = rows[start:start + size]
            yield f"{kind}-{start:09}.npz", kind, start, batch, [row[id_key] for row in batch]


def validate_vectors(vectors, count, dimension):
    if vectors.shape != (count, dimension) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("invalid embedding matrix")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("embeddings must be normalized")


def read_shard(path, record, ids, dimension):
    if digest_file(path) != record["sha256"]:
        raise ValueError("cache shard checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        if data["ids"].astype(str).tolist() != ids:
            raise ValueError("cache shard IDs mismatch")
        vectors = data["vectors"]
    validate_vectors(vectors, len(ids), dimension)
    return vectors


def build_cache(plan, output, encoder_factory, batch_size=32, shard_size=256, stop_after_shards=None):
    from filelock import FileLock
    validate_plan(plan)
    if batch_size < 1 or shard_size < 1 or (stop_after_shards is not None and stop_after_shards < 1):
        raise ValueError("invalid cache batch/shard budget")
    missing = [str(source_path(plan, row)) for row in plan["images"] if not source_path(plan, row).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} required images missing; no subset fallback: {missing[:5]}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".writer.lock"), timeout=0):
        signature = {"plan_sha256": digest_json(plan), "encoder_version": plan["encoder_version"],
                     "batch_size": batch_size, "shard_size": shard_size}
        state_path = output / "state.json"
        state = read_json(state_path) if state_path.exists() else {**signature, "status": "building", "shards": {}}
        if any(state[key] != value for key, value in signature.items()):
            raise ValueError("cache plan/encoder/batch configuration changed; use a new cache")
        if (output / "plan.json").exists() and read_json(output / "plan.json") != plan:
            raise ValueError("persisted cache plan changed")
        atomic_json(output / "plan.json", plan)
        atomic_json(state_path, state)
        encoder, written = None, 0
        for filename, kind, start, rows, ids in shard_specs(plan, shard_size):
            paths = [source_path(plan, row) for row in rows] if kind == "images" else []
            fingerprints = [digest_file(path) for path in paths]
            if filename in state["shards"]:
                record = state["shards"][filename]
                read_shard(output / filename, record, ids, plan["dimension"])
                if record["image_content_sha256"] != fingerprints:
                    raise ValueError("source image content changed; use a new cache")
                continue
            if encoder is None:
                encoder = encoder_factory()
                if encoder.version != plan["encoder_version"]:
                    raise ValueError("actual encoder differs from registered cache encoder")
            chunks = []
            for offset in range(0, len(rows), batch_size):
                values = (encoder.encode_images(paths[offset:offset + batch_size]) if kind == "images"
                          else encoder.encode_texts([row["text"] for row in rows[offset:offset + batch_size]]))
                chunks.append(values)
            vectors = np.concatenate(chunks).astype(np.float32)
            validate_vectors(vectors, len(ids), plan["dimension"])
            if paths and fingerprints != [digest_file(path) for path in paths]:
                raise ValueError("image changed during encoding")
            path = output / filename
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, ids=np.asarray(ids, dtype=str), vectors=vectors)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            state["shards"][filename] = {"sha256": digest_file(path), "count": len(ids), "image_content_sha256": fingerprints}
            atomic_json(state_path, state)
            written += 1
            print(f"cache {kind}: {start + len(rows)} rows committed", flush=True)
            if stop_after_shards is not None and written >= stop_after_shards:
                return state
        state["status"] = "complete"
        atomic_json(state_path, state)
        return state


def load_cache(output):
    output = Path(output)
    plan, state = read_json(output / "plan.json"), read_json(output / "state.json")
    validate_plan(plan)
    if state["status"] != "complete" or state["plan_sha256"] != digest_json(plan) or state["encoder_version"] != plan["encoder_version"]:
        raise ValueError("cache incomplete or version mismatch")
    specifications = list(shard_specs(plan, state["shard_size"]))
    if set(state["shards"]) != {row[0] for row in specifications}:
        raise ValueError("cache shard coverage mismatch")
    matrices = {"images": np.empty((len(plan["images"]), plan["dimension"]), dtype=np.float32),
                "texts": np.empty((len(plan["queries"]), plan["dimension"]), dtype=np.float32)}
    for filename, kind, start, rows, ids in specifications:
        matrices[kind][start:start + len(rows)] = read_shard(output / filename, state["shards"][filename], ids, plan["dimension"])
    return plan, matrices, digest_file(output / "state.json")
