import unittest

import numpy as np
import torch

from cir_baseline.hard_negatives import batch_candidate_masks, fit_image_pool, positive_key
from cir_baseline.learned_fusion import masked_multi_positive_loss


class HardNegativeTests(unittest.TestCase):
    def test_pool_is_fit_only_and_rejects_cross_split_images(self):
        args = (["fit", "dev", "holdout", "fit"], [0, 2, 4, 6], [1, 3, 5, 7], ["a", "a", "a", "b"], "a")
        np.testing.assert_array_equal(fit_image_pool(*args), [0, 1])
        with self.assertRaises(ValueError):
            fit_image_pool(args[0], [0, 0, 4, 6], *args[2:])

    def test_known_positives_reference_and_other_queries_extras(self):
        ids, positive, valid, extras = batch_candidate_masks(
            [1, 9], [{2, 3}, {1}], [2, 1], [[1, 2, 3, 7, 7, 8], [1, 2, 9, 5, 6]], 2)
        col = {key: j for j, key in enumerate(ids)}
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(valid[0, col[1]])  # reference despite another query's target
        self.assertTrue(positive[0, col[3]])  # extra known positive outside batch targets
        self.assertFalse(valid[0, col[5]])  # another query's extra is invisible
        np.testing.assert_array_equal(extras, [[7, 8], [5, 6]])
        self.assertFalse((positive & ~valid).any())

    def test_random_and_hard_have_matched_counts_and_masked_gradients(self):
        common = ([0, 1], [{2}, {3}], [2, 3])
        random = batch_candidate_masks(*common, [[4, 5], [5, 6]], 2)
        hard = batch_candidate_masks(*common, [[7, 8], [8, 9]], 2)
        np.testing.assert_array_equal(random[2].sum(1), hard[2].sum(1))
        ids, positive, valid, _ = hard
        torch.manual_seed(17)
        queries = torch.randn(2, 6, requires_grad=True)
        candidates = torch.randn(len(ids), 6, requires_grad=True)
        # Single query's loss must be invariant to excluded candidates.
        loss = masked_multi_positive_loss(queries[:1], candidates, torch.tensor(positive[:1]), torch.tensor(valid[:1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(candidates.grad[~torch.tensor(valid[0])], torch.zeros_like(candidates.grad[~torch.tensor(valid[0])])))

    def test_insufficient_distinct_extras_and_bad_positives_fail(self):
        with self.assertRaises(ValueError):
            batch_candidate_masks([0], [{1}], [1], [[0, 1, 2, 2]], 2)
        with self.assertRaises(ValueError):
            batch_candidate_masks([0], [{0, 1}], [1])

    def test_caption_order_and_whitespace_are_normalized_only(self):
        self.assertEqual(positive_key(1, ["red  shirt", "long"]), positive_key(1, ["long", "red shirt"]))
        self.assertNotEqual(positive_key(1, ["red shirt"]), positive_key(1, ["blue shirt"]))
        self.assertNotEqual(positive_key(1, ["red shirt"]), positive_key(2, ["red shirt"]))


if __name__ == "__main__":
    unittest.main()
