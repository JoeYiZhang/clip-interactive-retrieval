from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from cir_baseline.learned_fusion import LearnedFusion, image_disjoint_split, multi_positive_contrastive_loss


class LearnedFusionTests(unittest.TestCase):
    def test_connected_images_never_cross_partitions_and_order_is_irrelevant(self):
        refs = [f"ref-{i}" for i in range(200)] + ["ref-0", "target-0", "ref-10"]
        targets = [f"target-{i}" for i in range(200)] + ["ref-1", "ref-2", "target-1"]
        split, groups = image_disjoint_split(refs, targets)
        membership = {}
        for ref, target, part in zip(refs, targets, split, strict=True):
            for item in (ref, target):
                self.assertEqual(membership.setdefault(item, part), part)
        self.assertEqual(len(set(split)), 3)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[10])
        reverse, reverse_groups = image_disjoint_split(list(reversed(refs)), list(reversed(targets)))
        np.testing.assert_array_equal(split, reverse[::-1])
        np.testing.assert_array_equal(groups, reverse_groups[::-1])

    def test_zero_initialized_heads_match_weighted_baseline(self):
        torch.manual_seed(17)
        images, texts = F.normalize(torch.randn(8, 16), dim=-1), F.normalize(torch.randn(8, 16), dim=-1)
        expected = F.normalize(.3 * images + .7 * texts, dim=-1)
        for residual in (False, True):
            model = LearnedFusion(dimension=16, hidden_dim=8, residual=residual).eval()
            output = model(images, texts)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(output.norm(dim=-1), torch.ones(8))
            self.assertFalse(torch.allclose(output, model(images.flip(0), texts)))
            self.assertFalse(torch.allclose(output, model(images, texts.flip(0))))

    def test_duplicate_targets_do_not_create_false_negatives(self):
        candidates = torch.eye(3)
        queries = torch.eye(3)
        ids = torch.arange(3)
        original = multi_positive_contrastive_loss(queries, candidates, ids, ids)
        duplicated = multi_positive_contrastive_loss(
            queries, torch.cat((candidates, candidates[:1])), ids, torch.tensor([0, 1, 2, 0])
        )
        torch.testing.assert_close(original, duplicated)
        with self.assertRaises(ValueError):
            multi_positive_contrastive_loss(queries, candidates, torch.tensor([0, 1, 99]), ids)

    def test_training_step_updates_fusion_but_not_frozen_inputs(self):
        torch.manual_seed(29)
        images, texts, targets = (F.normalize(torch.randn(12, 16), dim=-1) for _ in range(3))
        model = LearnedFusion(dimension=16, hidden_dim=8).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model(images, texts).detach().clone()
        loss = multi_positive_contrastive_loss(model(images, texts), targets, torch.arange(12), torch.arange(12))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(images.grad)
        self.assertIsNone(texts.grad)
        self.assertIsNone(targets.grad)
        optimizer.step()
        self.assertFalse(torch.allclose(before, model(images, texts)))


if __name__ == "__main__":
    unittest.main()
