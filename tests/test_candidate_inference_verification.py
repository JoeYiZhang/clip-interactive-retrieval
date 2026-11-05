import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_fashioniq_candidate_inference import compare_rankings


class CandidateInferenceTests(unittest.TestCase):
    def test_identical_vector_reordering_keeps_exact_membership(self):
        vectors = {"a": np.array([1., 0.]), "b": np.array([1., 0.])}
        result = compare_rankings(["a", "b"], [.5, .5], ["b", "a"], [.5, .5], vectors)
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["exact_id_order_equal"])

    def test_equal_scores_do_not_hide_different_vectors(self):
        vectors = {"a": np.array([1., 0.]), "b": np.array([0., 1.])}
        result = compare_rankings(["a", "b"], [.5, .5], ["b", "a"], [.5, .5], vectors)
        self.assertFalse(result["checks"]["order_difference_only_at_identical_vectors"])

    def test_duplicate_or_different_ids_are_not_accepted_as_ties(self):
        vectors = {key: np.array([1., 0.]) for key in ("a", "b", "c")}
        duplicate = compare_rankings(["a", "a"], [.5, .5], ["a", "b"], [.5, .5], vectors)
        changed = compare_rankings(["a", "c"], [.5, .5], ["a", "b"], [.5, .5], vectors)
        self.assertFalse(duplicate["checks"]["unique_candidates"])
        self.assertFalse(changed["checks"]["same_topk_image_set"])


if __name__ == "__main__":
    unittest.main()
