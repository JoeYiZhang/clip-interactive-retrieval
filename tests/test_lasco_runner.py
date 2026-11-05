import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from cir_baseline.lasco_runner import train_cached, validate_training_inputs


def fixture():
    rng = np.random.default_rng(11)
    images = rng.normal(size=(12, 4)).astype(np.float32)
    texts = rng.normal(size=(6, 4)).astype(np.float32)
    images /= np.linalg.norm(images, axis=1, keepdims=True)
    texts /= np.linalg.norm(texts, axis=1, keepdims=True)
    queries = [{"query_id": str(i), "reference_id": str(i), "target_id": str(i+6), "text": f"query {i}", "positive_ids": [str(i+6)]} for i in range(6)]
    data = {"plan": {"dataset": "lasco", "split": "train", "mode": "engineering", "dimension": 4, "encoder_version": "tiny", "text_policy": "single", "queries": queries},
            "matrices": {"images": images, "texts": texts}, "identity": "tiny-cache-v1", "ids": list(map(str, range(12))),
            "references": np.arange(6), "targets": np.arange(6, 12)}
    config = {"mode": "engineering", "epochs": 3, "batch_size": 3, "hidden_dim": 4, "dropout": .1, "initial_image_weight": .3,
              "residual_scale": .2, "learning_rate": .0003, "weight_decay": .0001, "temperature": .07, "anchor_penalty": .02}
    return data, config


class LaSCoRunnerTests(unittest.TestCase):
    def test_full_validation_selects_and_restores_best_checkpoint(self):
        data, config = fixture()
        data["plan"]["mode"] = "full"
        dev = copy.deepcopy(data)
        dev["plan"]["split"] = "val"
        dev["ids"] = [str(int(key) + 100) for key in dev["ids"]]
        dev["identity"] = "independent-val"
        for row in dev["plan"]["queries"]:
            for key in ("reference_id", "target_id"):
                row[key] = str(int(row[key]) + 100)
            row["positive_ids"] = [row["target_id"]]
        config = {**config, "mode": "full", "epochs": 2}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "full-validation-synthetic"
            result = train_cached(data, dev, config, 17, output, "cpu")
            self.assertEqual(result["checkpoint_file"], "best.pt")
            expected_epoch = max(result["history"], key=lambda row: (row["dev"]["target_R10"], row["dev"]["target_R50"]))["epoch"]
            self.assertEqual(result["best_epoch"], expected_epoch)
            saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["epoch"], expected_epoch)
            self.assertEqual(train_cached(data, dev, config, 17, output, "cpu", resume=True), result)

    def test_incomplete_scope_or_val_overlap_is_rejected(self):
        data, config = fixture()
        with self.assertRaises(ValueError):
            validate_training_inputs(data, None, {**config, "mode": "full"})
        data["plan"]["mode"] = "full"
        with self.assertRaises(ValueError):
            validate_training_inputs(data, None, {**config, "mode": "full"})
        dev = copy.deepcopy(data)
        dev["plan"]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_training_inputs(data, dev, {**config, "mode": "full"})

    def _resume(self, device):
        data, config = fixture()
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                full, interrupted = Path(folder) / "完整", Path(folder) / "恢复"
                train_cached(data, None, config, 17, full, device)
                result = train_cached(data, None, config, 17, interrupted, device, stop_after_epoch=1)
                self.assertEqual(result["status"], "interrupted_at_epoch_boundary")
                with self.assertRaises(ValueError):
                    train_cached(data, None, {**config, "learning_rate": .01}, 17, interrupted, device, resume=True)
                train_cached(data, None, config, 17, interrupted, device, resume=True)
                a, b = [torch.load(path / "resume.pt", map_location="cpu", weights_only=True) for path in (full, interrupted)]
                self.assertEqual(a["history"], b["history"])
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(old_threads)

    def test_cpu_epoch_resume_is_exact(self):
        self._resume("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_epoch_resume_is_exact(self):
        self._resume("cuda")
