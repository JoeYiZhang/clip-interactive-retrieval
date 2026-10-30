import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("additional_smoke", SCRIPTS / "test_additional_datasets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class AdditionalDataTests(unittest.TestCase):
    def row(self):
        return {"qid": 1, "query-text": "make it blue", "query-image": [1, "train2014/a.jpg"], "target-image": [2, "train2014/b.jpg"]}

    def test_empty_text_is_quarantined_with_original_id(self):
        row = self.row()
        row["query-text"] = " "
        result = module.audit([row], "lasco")
        self.assertEqual(result["usable_queries"], 0)
        self.assertEqual(result["invalid"], [{"query_id": "1", "reason": "empty text"}])

    def test_duplicate_query_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            module.audit([self.row(), self.row()], "lasco")

    def test_image_path_cannot_escape_data_root(self):
        row = self.row()
        row["target-image"][1] = "../private.jpg"
        with self.assertRaises(ValueError):
            module.audit([row], "lasco")
