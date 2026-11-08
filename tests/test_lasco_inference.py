import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from cir_baseline.embedding_cache import atomic_json, build_cache, digest_json, read_json
from cir_baseline.lasco_inference import LaSCoRetriever
from cir_baseline.learned_fusion import LearnedFusion


class Encoder:
    version = "inference-tiny-v1"
    def __init__(self):
        self.image_calls, self.text_calls = [], []
    def encode_images(self, paths):
        self.image_calls.extend(str(path) for path in paths)
        return np.asarray([np.eye(3)[int(Path(path).stem)] for path in paths], dtype=np.float32)
    def encode_texts(self, texts):
        self.text_calls.extend(texts)
        return np.asarray([np.eye(3)[1 if text == "left" else 2] for text in texts], dtype=np.float32)


def fixture(root):
    for i in range(3):
        (root / f"{i}.jpg").write_bytes(f"fixture{i}".encode())
    plan = {"dataset": "lasco", "split": "train", "mode": "engineering", "dimension": 3,
            "image_root": str(root), "encoder_version": Encoder.version, "text_policy": "query-text-verbatim-stripped-v1",
            "images": [{"id": str(i), "path": f"{i}.jpg"} for i in range(3)],
            "queries": [{"query_id": "q", "reference_id": "0", "target_id": "1", "positive_ids": ["1"], "text": "left"}]}
    cache = root / "cache"
    build_cache(plan, cache, Encoder, 3, 3)
    model = LearnedFusion(dimension=3, hidden_dim=4)
    saved = {"state_dict": model.state_dict(), "model_config": model.config, "encoder_version": Encoder.version,
             "text_strategy": "single_text", "variant": "lasco_cached_multi_positive", "mode": "engineering", "seed": 17, "epoch": 3}
    checkpoint = root / "model.pt"
    torch.save(saved, checkpoint)
    return cache, checkpoint, saved


class LaSCoInferenceTests(unittest.TestCase):
    def test_conflicting_cache_precision_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, checkpoint, _ = fixture(Path(folder))
            plan = read_json(cache / "plan.json")
            plan["encoder_precision"] = {"cuda_matmul_tf32": False, "cudnn_tf32": True}
            atomic_json(cache / "plan.json", plan)
            state = read_json(cache / "state.json")
            state["plan_sha256"] = digest_json(plan)
            atomic_json(cache / "state.json", state)
            with self.assertRaisesRegex(ValueError, "precision contract"):
                LaSCoRetriever(cache, checkpoint, Encoder(), allow_engineering=True)

    def test_implicit_tf32_in_raw_inference_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, checkpoint, _ = fixture(Path(folder))
            retriever = LaSCoRetriever(cache, checkpoint, Encoder(), allow_engineering=True)
            retriever.device = "cuda"  # Guard runs before any GPU operation.
            previous = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = True
                with self.assertRaisesRegex(ValueError, "TF32"):
                    retriever.compose_query("0", "left")
            finally:
                torch.backends.cudnn.allow_tf32 = previous

    def test_free_text_changes_results_and_reference_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, checkpoint, _ = fixture(Path(folder))
            encoder = Encoder()
            retriever = LaSCoRetriever(cache, checkpoint, encoder, allow_engineering=True)
            a, b = retriever.search("0", "left", 100), retriever.search("0", "right", 100)
            self.assertEqual(a["results"][0]["image_id"], "1")
            self.assertEqual(b["results"][0]["image_id"], "2")
            self.assertEqual(a["returned_count"], 2)
            self.assertNotIn("0", [row["image_id"] for row in a["results"]])
            self.assertEqual(encoder.text_calls, ["left", "right"])
            self.assertEqual({Path(path).name for path in encoder.image_calls}, {"0.jpg"})

    def test_label_changes_do_not_change_query_or_ranking(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, checkpoint, _ = fixture(Path(folder))
            before = LaSCoRetriever(cache, checkpoint, Encoder(), allow_engineering=True).search("0", "left")
            plan = read_json(cache / "plan.json")
            plan["queries"][0].update(target_id="2", positive_ids=["2"], text="unrelated annotation")
            atomic_json(cache / "plan.json", plan)
            state = read_json(cache / "state.json")
            state["plan_sha256"] = digest_json(plan)
            atomic_json(cache / "state.json", state)
            after = LaSCoRetriever(cache, checkpoint, Encoder(), allow_engineering=True).search("0", "left")
            self.assertEqual(before["results"], after["results"])
            self.assertEqual(before["query_norm"], after["query_norm"])

    def test_scope_encoder_text_policy_and_dimension_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            cache, checkpoint, saved = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "allow_engineering"):
                LaSCoRetriever(cache, checkpoint, Encoder())
            for key, value in (("encoder_version", "wrong"), ("text_strategy", "symmetric_join_mean"),
                               ("model_config", {**saved["model_config"], "dimension": 4})):
                changed = copy.deepcopy(saved)
                changed[key] = value
                torch.save(changed, checkpoint)
                with self.assertRaises(ValueError):
                    LaSCoRetriever(cache, checkpoint, Encoder(), allow_engineering=True)

    def test_invalid_input_and_changed_source_images_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, checkpoint, _ = fixture(root)
            retriever = LaSCoRetriever(cache, checkpoint, Encoder(), allow_engineering=True)
            for reference, text, k in (("missing", "left", 2), ("0", "  ", 2), ("0", "left", 0), ("0", "left", True)):
                with self.assertRaises(ValueError):
                    retriever.search(reference, text, k)
            (root / "1.jpg").write_bytes(b"changed result image")
            with self.assertRaisesRegex(ValueError, "source image changed"):
                retriever.search("0", "left", 1)
            (root / "0.jpg").write_bytes(b"changed reference")
            with self.assertRaisesRegex(ValueError, "source image changed"):
                retriever.compose_query("0", "left")
