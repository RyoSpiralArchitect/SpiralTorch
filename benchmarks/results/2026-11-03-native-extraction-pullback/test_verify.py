"""Only the frozen scalar export is opened. No model, source corpus, or old evaluation is needed."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify as v  # noqa: E402


class Archive(unittest.TestCase):
    def test_all_rows_and_both_primary_gains_are_reproducible(self):
        report = v.verify(v.m.HERE / "receipts", archived=True)
        self.assertEqual(report["calls_verified"], 864)
        self.assertEqual(report["model_executions_by_verifier"], 0)
        for summary in report["per_order"].values():
            self.assertEqual(summary["smaller_fixed_residual_docs"], 96)
            self.assertEqual(summary["smaller_fixed_correction_docs"], 96)
            self.assertEqual(summary["min_max_fixed_affine_threshold_crossings_per_doc"], [1, 1])

    def test_relabeling_or_adding_a_hidden_vector_cannot_pass(self):
        root = v.m.HERE / "receipts"
        row = json.loads(next(v.measurements(root, True)))
        with self.assertRaisesRegex(ValueError, "unexpected site payload"):
            changed = copy.deepcopy(row)
            changed["observation"]["sites"][v.m.SITE]["hidden_states"] = [3.]
            v.scalar_payload(changed)
        plan = v.m.read(root / "manifest.json")["plan"][0]
        e = v.m.boundaries()[v.m.ORDERS[0],v.m.ARMS[0]]
        with self.assertRaisesRegex(ValueError, "unpaired window"):
            v.m.validate_row(row, v.m.ORDERS[0], v.m.ARMS[0], 1, e, plan)
        row["observation"]["sites"][v.m.SITE]["pullback"]["relative_correction_rms"] = 1.
        with self.assertRaisesRegex(ValueError, "denominator"):
            v.m.validate_row(row, v.m.ORDERS[0], v.m.ARMS[0], 0, e, plan)


if __name__ == "__main__":
    unittest.main()
