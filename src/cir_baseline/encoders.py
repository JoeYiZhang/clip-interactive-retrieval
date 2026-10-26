from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .fusion import l2_normalize, l2_normalize_rows


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DeterministicHashEncoder:
    """Dependency-free test encoder. It is not a scientific baseline."""

    def __init__(self, dimension: int = 64, version: str = "hash-smoke-v1") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.version = version

    def _encode_one(self, namespace: str, value: str) -> np.ndarray:
        digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        return l2_normalize(rng.standard_normal(self.dimension, dtype=np.float32))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._encode_one("text", text) for text in texts])

    def encode_images(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        return np.stack(
            [self._encode_one("image", str(Path(path))) for path in image_paths]
        )


class OpenCLIPEncoder:
    """Thin optional adapter around open_clip_torch.

    The import happens at construction time so the core baseline and tests remain
    runnable before machine-learning dependencies and model weights are installed.

    Offline use: pass ``checkpoint_path`` with a verified local weight file. The
    weight file's SHA256 is recorded in ``weights_sha256`` and folded into
    ``version`` so the encoder/index contract stays traceable. For OpenAI CLIP
    checkpoints (TorchScript archives) also pass ``weights_only=False``;
    ``force_quick_gelu=True`` reproduces the activation those weights were
    trained with, which open_clip 3.x no longer applies from the ``openai`` tag
    on its own.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
        force_quick_gelu: bool = False,
        weights_only: bool = True,
    ) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP dependencies are missing. Install the project with the "
                "'ml' extra before constructing OpenCLIPEncoder."
            ) from exc

        self._torch = torch
        self._Image = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path: str | None = None
        self.weights_sha256: str | None = None

        pretrained_arg = pretrained
        version_tag = pretrained
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint file not found: {path}")
            self.checkpoint_path = str(path)
            self.weights_sha256 = _sha256_of(path)
            pretrained_arg = self.checkpoint_path
            version_tag = f"local-sha256:{self.weights_sha256[:16]}"

        self.version = f"open_clip:{model_name}:{version_tag}"
        if force_quick_gelu:
            self.version += "|quick_gelu=1"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_arg,
            force_quick_gelu=force_quick_gelu,
            weights_only=weights_only,
        )
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self.tokenizer(list(texts)).to(self.device)
        with self._torch.inference_mode():
            vectors = self.model.encode_text(tokens)
        return l2_normalize_rows(vectors.float().cpu().numpy())

    def encode_images(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        tensors = []
        for path in image_paths:
            with self._Image.open(path) as image:
                tensors.append(self.preprocess(image.convert("RGB")))
        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.inference_mode():
            vectors = self.model.encode_image(batch)
        return l2_normalize_rows(vectors.float().cpu().numpy())
