import unittest
from cir_baseline.circo_evaluation import evaluate_circo


class CircoMetricsTests(unittest.TestCase):
    def fixture(self):
        query = {"query_id": "q", "reference_id": "r", "target_id": "c", "positive_ids": ["a", "b", "c"]}
        return [query], {"q": ["a", "x", "b", "y", "c"]}, ["r", "a", "b", "c", "x", "y"]

    def test_hand_calculated_ap_and_target_recall_are_distinct(self):
        q, p, g = self.fixture()
        result = evaluate_circo(q, p, g, ks=(1, 3, 5))
        self.assertEqual(result["mAP@1"], 1.)  # denominator min(3,1), not 3
        self.assertAlmostEqual(result["mAP@3"], (1 + 2/3) / 3)
        self.assertAlmostEqual(result["mAP@5"], (1 + 2/3 + 3/5) / 3)
        self.assertEqual(result["target_recall@3"], 0.)
        self.assertEqual(result["target_recall@5"], 1.)

    def test_duplicate_reference_unknown_and_truncated_rankings_fail(self):
        q, _, g = self.fixture()
        for ranking in (["a", "a", "b"], ["a", "r", "b"], ["a", "unknown", "b"], ["a"]):
            with self.assertRaises(ValueError):
                evaluate_circo(q, {"q": ranking}, g, ks=(3,))

    def test_missing_queries_or_positive_gallery_cannot_be_scored(self):
        q, p, g = self.fixture()
        with self.assertRaises(ValueError):
            evaluate_circo(q, {}, g, ks=(1,))
        with self.assertRaises(ValueError):
            evaluate_circo(q, p, [v for v in g if v != "c"], ks=(1,))
        q[0]["positive_ids"] = []
        with self.assertRaises(ValueError):
            evaluate_circo(q, p, g, ks=(1,))
