from copy import deepcopy
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import PATHS, PLAN, check_observation, first_separation, summarize_sites
from replay import ORIGINAL


class Diagnostics(unittest.TestCase):
    def test_pool_sums_then_take_ratio_and_leave_absent_gradients_absent(self):
        a = dict(
            elements=1,
            input_rms=1,
            residual_rms=0.5,
            trainable_gradient_l2=1,
            log_radius_gradient=None,
        )
        b = dict(a, elements=9, input_rms=3, residual_rms=0.3)
        expected = math.sqrt((0.25 + 9 * 0.09) / (1 + 9 * 9))
        row = summarize_sites([a, b])
        self.assertAlmostEqual(row["pooled_relative_residual_rms"], expected)
        self.assertNotAlmostEqual(row["pooled_relative_residual_rms"], (0.5 + 0.1) / 2)
        self.assertIsNone(row["mean_abs_log_radius_gradient"])
        with self.assertRaises(ValueError):
            summarize_sites([a, dict(b, log_radius_gradient=0.0)])
        row = summarize_sites(
            [dict(a, log_radius_gradient=0.0), dict(b, log_radius_gradient=-2.0)]
        )
        self.assertEqual(row["negative_log_radius_gradient_fraction"], 0.5)
        self.assertEqual(row["mean_abs_log_radius_gradient"], 1)
        self.assertEqual(row["mean_signed_log_radius_gradient"], -1)

    def test_milestones_are_predeclared_and_not_a_continuous_onset(self):
        values = {str(u): [0, 0, 0] for u in PLAN["evaluation_updates"]}
        values["0"] = [0.5, 0.5, 0.5]
        values["8"] = [0.002, 0.002, 0.001]
        self.assertIsNone(first_separation(values))
        values["32"] = [0.002, 0.002, 0.002]
        self.assertEqual(first_separation(values), 32)

    def test_reject_double_count_mixed_data_and_invented_fixed_radius_gradient(self):
        step = {"update": 1}
        site = dict(
            elements=2 * 63 * 768,
            input_rms=2,
            residual_rms=1,
            relative_residual_rms=0.5,
            trainable_gradient_l2=1,
            pre_log_radius=1.0,
            log_radius_gradient=None,
        )
        record = dict(
            schema="spiraltorch.native_lm.observed_training_step.v1",
            base_model_identity=ORIGINAL["base_id"],
            data_id="same",
            step=step,
            sites={p: dict(site) for p in PATHS},
        )
        check_observation(record, step, "same", "fixed4")
        for field, value in (
            ("elements", site["elements"] * 2),
            ("log_radius_gradient", 0.0),
            ("relative_residual_rms", 0.25),
        ):
            other = deepcopy(record)
            other["sites"][PATHS[0]][field] = value
            with self.assertRaises(ValueError):
                check_observation(other, step, "same", "fixed4")
        with self.assertRaises(ValueError):
            check_observation(record, step, "other", "fixed4")


if __name__ == "__main__":
    unittest.main()
