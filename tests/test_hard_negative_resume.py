"""Check exact epoch-boundary continuation using a small real CPU optimizer."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_fashioniq_hard_negatives as trainer


class TinyData:
    def __init__(self):
        rng = np.random.default_rng(5)
        vectors = rng.normal(size=(30, 6)).astype(np.float32)
        self.gallery = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.ref_positions = np.arange(8)
        self.target_positions = np.arange(8, 16)
        self.ref_vectors = self.gallery[self.ref_positions]
        self.texts = self.gallery[16:24]
        self.categories = np.asarray(["a"] * 8)
        self.positions = {"fit": np.arange(6), "dev": np.arange(6, 8)}
        self.encoder_version = "test-frozen-encoder"

    def search(self, positions, vectors, keep_rankings=False):
        scores = vectors @ self.gallery.T
        scores[np.arange(len(positions)), self.ref_positions[positions]] = -np.inf
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.asarray([np.flatnonzero(row == self.target_positions[i])[0] + 1 for i, row in zip(positions, order)])
        return ranks, None, None


class ResumeTests(unittest.TestCase):
    def test_holdout_evaluation_is_rejected(self):
        with self.assertRaises(ValueError):
            trainer.evaluate(None, None, "holdout", "cpu")

    def test_interrupted_epoch_resumes_identical_weights_and_trace(self):
        self._verify_resume("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_epoch_resume_restores_rng_and_weights(self):
        self._verify_resume("cuda")

    def _verify_resume(self, device):
        data = TinyData()
        config = {"hidden_dim": 4, "initial_image_weight": .3, "dropout": .1,
                  "residual_scale": .2, "learning_rate": .001, "weight_decay": .0001,
                  "batch_size": 3, "temperature": .07, "anchor_penalty": .02,
                  "epochs": 2, "evaluate_every": 1, "text_strategy": "test"}
        positives = [{int(i)} for i in data.target_positions[:6]]
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                full, interrupted = Path(folder) / "完整训练", Path(folder) / "中断恢复"
                full.mkdir()
                interrupted.mkdir()
                trainer.train_one(data, config, "in_batch", 17, positives, {}, full, device, False)
                save = trainer.atomic_torch_save

                def save_then_interrupt(payload, path):
                    save(payload, path)
                    if str(path).endswith("-resume.pt") and payload["epoch"] == 1:
                        raise InterruptedError("simulated shutdown after an epoch")

                with patch.object(trainer, "atomic_torch_save", side_effect=save_then_interrupt):
                    with self.assertRaises(InterruptedError):
                        trainer.train_one(data, config, "in_batch", 17, positives, {}, interrupted, device, False)
                torch.manual_seed(999)
                trainer.train_one(data, config, "in_batch", 17, positives, {}, interrupted, device, True)
                # Completed-arm skip must also handle UTF-8 paths on Windows.
                completed = trainer.train_one(data, config, "in_batch", 17, positives, {}, interrupted, device, True)
                self.assertEqual(completed["seed"], 17)
                a, b = [torch.load(path / "in_batch-seed17-resume.pt", weights_only=True, map_location="cpu") for path in (full, interrupted)]
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
                for key in ("trace", "batch_orders", "best", "best_epoch", "stats", "numpy_rng"):
                    self.assertEqual(a[key], b[key])
                torch.testing.assert_close(a["torch_rng"], b["torch_rng"], rtol=0, atol=0)
                for left, right in zip(a["cuda_rng"], b["cuda_rng"], strict=True):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        finally:
            torch.set_num_threads(original_threads)


if __name__ == "__main__":
    unittest.main()
