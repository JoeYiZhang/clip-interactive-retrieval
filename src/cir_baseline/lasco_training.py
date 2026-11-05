"""Stable LaSCo training examples and annotation-derived positive masks."""
from __future__ import annotations
from collections import defaultdict
from pathlib import PurePosixPath
import numpy as np

def query_key(reference_id, text):
    return int(reference_id), " ".join(text.split())

def prepare_examples(rows, split):
    if split not in ("train", "val"):
        raise ValueError("invalid split")
    examples, quarantine, paths = [], [], {}
    positives = defaultdict(set)
    for position, row in enumerate(rows):
        reference, target = row["query-image"], row["target-image"]
        text = row["query-text"].strip()
        for image_id, relative in (reference, target):
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
                raise ValueError("unsafe image path")
            expected = "train2014" if split == "train" else "val2014"
            if len(path.parts) != 2 or path.parts[0] != expected or path.name != f"COCO_{expected}_{int(image_id):012}.jpg":
                raise ValueError("image ID/path/split mismatch")
            if paths.setdefault(int(image_id), relative) != relative:
                raise ValueError("conflicting image paths")
        identity = f"lasco:{split}:{position}"
        reason = "empty text" if not text else "reference equals target" if reference[0] == target[0] else None
        if reason:
            quarantine.append({"example_id": identity, "source_qid": row["qid"], "reason": reason})
            continue
        example = {"example_id": identity, "source_row_index": position, "source_qid": row["qid"],
                   "reference_id": int(reference[0]), "target_id": int(target[0]), "text": text,
                   "reference_path": reference[1], "target_path": target[1]}
        examples.append(example)
        positives[query_key(reference[0], text)].add(int(target[0]))
    return examples, dict(positives), quarantine

def candidate_masks(examples, candidate_ids, positives):
    """Exclude the reference; every known correct target is positive, never negative.

    Positives are only known within the same reference and whitespace-normalized
    text. Sharing a reference alone does not establish a positive relationship.
    """
    ids = np.asarray(candidate_ids)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError("candidates must be unique image IDs")
    valid = np.stack([ids != row["reference_id"] for row in examples])
    positive = np.stack([np.isin(ids, list(positives[query_key(row["reference_id"], row["text"])])) for row in examples])
    positive &= valid
    if not positive.any(axis=1).all():
        raise ValueError("each query requires at least one positive candidate")
    return positive, valid
