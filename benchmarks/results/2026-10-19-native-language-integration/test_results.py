"""Check publication identity and completeness, not original runtime execution."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest


class PublishedResults(unittest.TestCase):
    def test_all_predeclared_arms_and_unmodified_gates_are_retained(self):
        raw = Path(__file__).with_name("comparison.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "3769449c6cc23a6280a2faa7c2e972022bf5cb67a53985023550b3cc755f8123")
        result = json.loads(raw)
        self.assertIs(result["passed"], True)
        self.assertEqual(result["nll_atol"], 5e-4)
        self.assertEqual(result["context_length"], 32)
        self.assertEqual(result["reference_relation"],
                         "retained FP32 Torch reference on identical pinned inputs, not rerun")
        scores = result["scores"]
        expected = set(itertools.product(("base", "trained"), ("f32", "f64"), (1, 2)))
        self.assertEqual(len(scores), len(expected))
        self.assertEqual({(r["model"], r["readout"], r["batch"]) for r in scores}, expected)
        for row in scores:
            score = row["score"]
            self.assertTrue(math.isfinite(score["nll"]))
            self.assertEqual(score["target_tokens"], 1302)
            self.assertEqual(score["readout_accumulator"], row["readout"])
            self.assertEqual(row["absolute_error"], abs(score["nll"] - row["reference_nll"]))
            self.assertLessEqual(row["absolute_error"], result["nll_atol"])
            self.assertIs(row["passed"], True)
        generation = result["generation"]
        expected = set(itertools.product(("base", "trained"), ("f32", "f64"), ("full_prefix", "kv_cache")))
        self.assertEqual(len(generation), len(expected))
        self.assertEqual({(r["model"], r["readout"], r["execution"]["kind"]) for r in generation}, expected)
        for row in generation:
            self.assertIs(row["passed"], True)
            self.assertEqual(row["resumed_steps"], 3)
            self.assertRegex(row["step_reports_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["final_checkpoint_id"], r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
