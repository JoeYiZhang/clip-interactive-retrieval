from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("vector contains NaN or infinity")
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix contains NaN or infinity")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("matrix contains at least one zero vector")
    return values / norms


def fuse_weighted(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
    image_weight: float = 0.5,
    text_weight: float = 0.5,
) -> np.ndarray:
    image = l2_normalize(image_embedding)
    text = l2_normalize(text_embedding)
    if image.shape != text.shape:
        raise ValueError(
            f"image/text dimensions differ: {image.shape} versus {text.shape}"
        )
    if image_weight < 0 or text_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    if image_weight + text_weight <= 0:
        raise ValueError("at least one fusion weight must be positive")
    return l2_normalize(image_weight * image + text_weight * text)

