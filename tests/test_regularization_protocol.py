import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_fashioniq_regularization import ROOT, effective_configs, read_json, summarize_runs


class RegularizationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = read_json(ROOT / "configs/fashioniq_regularization.json")
        self.base = read_json(ROOT / self.protocol["base_config"])

    def test_only_lr_dropout_and_identity_differ_between_arms(self):
        configs = effective_configs(self.protocol, self.base)
        for config in configs.values():
            changed = {key for key in config if config.get(key) != configs["control"].get(key)}
            self.assertLessEqual(changed, {"learning_rate", "dropout", "regularization_variant"})
        bad = copy.deepcopy(self.protocol)
        bad["variants"]["low_lr"]["weight_decay"] = .1
        with self.assertRaises(ValueError):
            effective_configs(bad, self.base)
        bad_base = {**self.base, "split_seed": 1}
        with self.assertRaises(ValueError):
            effective_configs(self.protocol, bad_base)

    def sample_runs(self):
        return [{"regularization_variant": name, "seed": seed,
                 "dev": {"category_mean_recall@10": .2, "category_mean_recall@50": .4}}
                for name in self.protocol["variants"] for seed in self.protocol["seeds"]]

    def test_good_average_cannot_hide_one_regressing_seed(self):
        runs = self.sample_runs()
        rows = [row for row in runs if row["regularization_variant"] == "low_lr"]
        for row, recall in zip(rows, [.23, .23, .199], strict=True):
            row["dev"]["category_mean_recall@10"] = recall
        result = summarize_runs(runs, self.protocol)
        self.assertEqual(result["selected_variant"], "low_lr")
        self.assertFalse(result["contrasts_vs_control"]["low_lr"]["screen_pass"])
        self.assertFalse(result["deployed_model_changed"])

    def test_passing_screen_never_changes_deployment_and_missing_seed_fails(self):
        runs = self.sample_runs()
        for row in runs:
            if row["regularization_variant"] == "low_lr":
                row["dev"]["category_mean_recall@10"] = .21
        result = summarize_runs(runs, self.protocol)
        self.assertTrue(result["contrasts_vs_control"]["low_lr"]["screen_pass"])
        self.assertFalse(result["deployed_model_changed"])
        with self.assertRaises(ValueError):
            summarize_runs(runs[:-1], self.protocol)


if __name__ == "__main__":
    unittest.main()
