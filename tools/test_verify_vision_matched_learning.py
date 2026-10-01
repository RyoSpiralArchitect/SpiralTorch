"""Verify archived coverage and negative cases without importing ML libraries."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verifier", ROOT / "tools/verify_vision_matched_learning.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
RESULTS = ROOT / "benchmarks/results/2026-10-01-vision-matched-learning"


class EvidenceChecks(unittest.TestCase):
    def test_all_original_conditions(self):
        for path in RESULTS.glob("*-report.json"):
            with self.subTest(path=path.name):
                result = verifier.verify(json.loads(path.read_text()))
                self.assertEqual(result["parameter_values"], 5530)
                self.assertEqual(result["parameter_tensors"], 24)
        self.assertEqual(len(list(RESULTS.glob("*-report.json"))), 6)

    def test_rejects_incomplete_or_mislabelled_evidence(self):
        original = json.loads((RESULTS / "expanded-mps-report.json").read_text())
        def drop_weight(r):
            r["runs"][0]["admission"]["updated_weights"].pop()
        def repeat_role(r):
            r["runs"][0]["admission"]["parameter_gradients"][1]["name"] = "convnext.stem::weight"
        def drop_epoch(r):
            r["runs"][1]["epochs"].pop()
        def wrong_clock(r):
            r["runs"][0]["epochs"][0]["accepted_updates"] += 1
        def wrong_value_count(r):
            r["runs"][0]["admission"]["updated_weights"][0]["values"] -= 1
        def missing_seed(r):
            r["runs"].pop()
        def bad_metric(r):
            r["runs"][0]["initial_evaluation"]["torch"]["loss"] = float("nan")
        def duplicate_sample(r):
            r["data"]["train_indices"][1] = r["data"]["train_indices"][0]
        def error_over_bound(r):
            r["runs"][0]["admission"]["logits"]["max_scaled_error"] = 2e-4
        for mutation in (drop_weight, repeat_role, drop_epoch, wrong_clock, wrong_value_count,
                         missing_seed, bad_metric, duplicate_sample, error_over_bound):
            report = copy.deepcopy(original)
            mutation(report)
            with self.subTest(mutation=mutation.__name__), self.assertRaises(ValueError):
                verifier.verify(report)

    def test_rejects_changed_fixed_recipe(self):
        original = json.loads((RESULTS / "expanded-mps-report.json").read_text())
        for location, key, value in ((None, "rate", 1e9), (None, "batch_size", 8),
                                     (None, "torch_threads", 8), (None, "torch_device", "cuda"),
                                     ("data", "name", "different"), ("data", "official_archive_md5", "0" * 32),
                                     ("data", "augmentation", "random_flip"),
                                     ("data", "normalization", dict(input_divisor=1, mean=[0.] * 3, std=[1.] * 3))):
            report = copy.deepcopy(original)
            target = report if location is None else report[location]
            target[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                verifier.verify(report)


if __name__ == "__main__":
    unittest.main()
