"""CIRCO AP@K and designated-target recall, matching the official equations."""
from collections import defaultdict


def evaluate_circo(queries, predictions, gallery_ids, ks=(5, 10, 25, 50)):
    if not queries or not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("nonempty queries and positive unique cutoffs required")
    query_ids = [row["query_id"] for row in queries]
    if len(set(query_ids)) != len(query_ids) or set(predictions) != set(query_ids):
        raise ValueError("predictions must cover exactly every query once")
    gallery = set(gallery_ids)
    if len(gallery) != len(gallery_ids):
        raise ValueError("duplicate gallery IDs")
    per_query, semantic = [], defaultdict(list)
    for row in queries:
        reference, target = row["reference_id"], row.get("target_id")
        positive = set(row["positive_ids"])
        if not positive or len(positive) != len(row["positive_ids"]) or reference in positive or target not in positive:
            raise ValueError("validation ground truth invalid or unavailable")
        if not positive <= gallery or reference not in gallery:
            raise ValueError("gallery missing a reference/positive")
        ranking = predictions[row["query_id"]]
        if len(ranking) < max(ks) or len(set(ranking)) != len(ranking) or reference in ranking or not set(ranking) <= gallery:
            raise ValueError("ranking truncated, duplicated, unknown, or contains reference")
        precision_sum, hits, ap = 0., 0, {}
        for rank, image_id in enumerate(ranking[:max(max(ks), 10)], 1):
            if image_id in positive:
                hits += 1
                precision_sum += hits / rank
            if rank in ks or rank == 10:
                ap[rank] = precision_sum / min(len(positive), rank)
        values = {"query_id": row["query_id"],
                  **{f"AP@{k}": ap[k] for k in ks},
                  **{f"target_recall@{k}": float(target in ranking[:k]) for k in ks}}
        per_query.append(values)
        if 10 in ap:
            for aspect in set(row.get("semantic_aspects", [])):
                semantic[aspect].append(ap[10])
    return {"query_count": len(queries), "units": "fraction in [0,1]",
            **{f"mAP@{k}": sum(row[f"AP@{k}"] for row in per_query) / len(queries) for k in ks},
            **{f"target_recall@{k}": sum(row[f"target_recall@{k}"] for row in per_query) / len(queries) for k in ks},
            "semantic_mAP@10": {key: sum(values) / len(values) for key, values in sorted(semantic.items())},
            "per_query": per_query}
