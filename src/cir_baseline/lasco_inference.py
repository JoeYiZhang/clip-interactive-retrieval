"""Inference from a registered reference image and free text, without labels."""
from pathlib import Path

import numpy as np
import torch

from .contracts import IndexManifest
from .embedding_cache import digest_file, load_cache, read_json, source_path, validate_vectors
from .index import FaissFlatIPIndex
from .learned_fusion import LearnedFusion


class LaSCoRetriever:
    def __init__(self, cache_dir, checkpoint, encoder, device="cpu", allow_engineering=False):
        plan, matrices, identity = load_cache(cache_dir)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if plan["dataset"] != "lasco" or plan["text_policy"] != "query-text-verbatim-stripped-v1":
            raise ValueError("LaSCo single-text gallery required")
        if saved.get("variant") != "lasco_cached_multi_positive" or saved.get("text_strategy") != "single_text":
            raise ValueError("LaSCo single-text checkpoint required")
        if plan["encoder_version"] != encoder.version or saved["encoder_version"] != encoder.version:
            raise ValueError("encoder/cache/checkpoint version mismatch")
        if saved["model_config"]["dimension"] != plan["dimension"]:
            raise ValueError("fusion/gallery dimension mismatch")
        precision = plan.get("encoder_precision")
        if precision is not None and precision != {"cuda_matmul_tf32": False, "cudnn_tf32": False}:
            raise ValueError("gallery precision contract is incompatible with raw inference")
        if saved.get("mode") not in ("full", "engineering"):
            raise ValueError("checkpoint scope missing")
        self.engineering = plan["mode"] == "engineering" or saved["mode"] == "engineering"
        if self.engineering and not allow_engineering:
            raise ValueError("small-sample weights/gallery require explicit allow_engineering")
        # Keep only image metadata. Query annotations and cached texts are neither
        # retained by the retriever nor passed into the fusion model.
        self.plan = {key: plan[key] for key in ("image_root", "dimension", "encoder_version")}
        self.rows = {row["id"]: row for row in plan["images"]}
        self.ids = list(self.rows)
        state = read_json(Path(cache_dir) / "state.json")
        fingerprints = []
        for filename in sorted(name for name in state["shards"] if name.startswith("images-")):
            fingerprints.extend(state["shards"][filename]["image_content_sha256"])
        if len(fingerprints) != len(self.ids):
            raise ValueError("image fingerprint coverage mismatch")
        self.image_hashes = dict(zip(self.ids, fingerprints, strict=True))
        self.device, self.encoder = device, encoder
        self.model = LearnedFusion(**saved["model_config"]).to(device).eval()
        self.model.load_state_dict(saved["state_dict"])
        self.index = FaissFlatIPIndex(self.ids, matrices["images"],
            IndexManifest("lasco-inference-" + identity[:12], encoder.version, plan["dimension"]))
        self.provenance = {"cache_sha256": identity, "checkpoint_sha256": digest_file(checkpoint),
                           "encoder_version": encoder.version, "seed": saved["seed"], "epoch": saved["epoch"],
                           "cache_encoder_precision": precision or "legacy unspecified; validate raw/cache parity before use",
                           "inference_precision": {"cuda_matmul_tf32": False, "cudnn_tf32": False},
                           "gallery_split": plan["split"], "gallery_mode": plan["mode"], "checkpoint_mode": saved["mode"]}

    def image_path(self, image_id):
        path = source_path(self.plan, self.rows[image_id])
        if digest_file(path) != self.image_hashes[image_id]:
            raise ValueError("source image changed since gallery encoding")
        return path

    def compose_query(self, reference_id, text):
        if self.device.startswith("cuda") and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError("disable CUDA matmul and cuDNN TF32 before raw-image inference")
        if reference_id not in self.rows:
            raise ValueError("reference ID is not in the registered gallery")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("provide a nonempty modification description")
        path = self.image_path(reference_id)
        reference = self.encoder.encode_images([path])
        encoded_text = self.encoder.encode_texts([text.strip()])
        validate_vectors(reference, 1, self.plan["dimension"])
        validate_vectors(encoded_text, 1, self.plan["dimension"])
        self.image_path(reference_id)  # Detect edits during encoding.
        with torch.inference_mode():
            query = self.model(torch.from_numpy(reference).to(self.device), torch.from_numpy(encoded_text).to(self.device)).cpu().numpy()
        validate_vectors(query, 1, self.plan["dimension"])
        return query[0]

    def search(self, reference_id, text, top_k=10):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        query = self.compose_query(reference_id, text)
        hits = [hit for hit in self.index.search(query, top_k + 1) if hit.image_id != reference_id][:top_k]
        return {"scope": "engineering inference only; NOT benchmark performance" if self.engineering else "research model inference; NOT a benchmark evaluation",
                "reference_id": reference_id, "reference_image_path": str(self.image_path(reference_id)), "text": text.strip(),
                "gallery_count": len(self.ids), "requested_top_k": top_k, "returned_count": len(hits),
                "query_norm": float(np.linalg.norm(query)), "provenance": self.provenance,
                "results": [{"rank": rank, "image_id": hit.image_id, "score": hit.score, "image_path": str(self.image_path(hit.image_id))}
                            for rank, hit in enumerate(hits, 1)]}
