"""Model-free coverage, no-selection and public-boundary regressions."""

import copy
import json
from pathlib import Path
import runpy
import unittest

HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "protocol.py"))
PLAN = json.loads((HERE / "plan.json").read_text())


def layers(value):
    return [{"path": path, "relative_residual_rms": value} for path in API["SITES"]]


class ProtocolTests(unittest.TestCase):
    def test_complete_ordered_case_set(self):
        cases = API["cases_for"](PLAN)
        self.assertEqual(len(cases), 27)
        self.assertEqual(sum(c["mode"] == "reference" for c in cases), 15)
        for seed in PLAN["seeds"]:
            group = [c for c in cases if c["seed"] == seed]
            self.assertEqual(group[0], {"arm": "ordinary", "seed": seed, "mode": "reference"})
            self.assertEqual([c["arm"] for c in group[1:]], [a for a in API["ARMS"] for _ in range(2)])
            self.assertEqual([c["mode"] for c in group[1:]], ["reference", "calibrated_ordinary"] * 4)

    def test_changed_selection_budget_device_or_boolean_rejected(self):
        for key, value in (("seeds", [41]), ("geometry_arms", API["ARMS"][:1]),
                           ("case_count", 3), ("calibration_windows", 32), ("train_windows", 100),
                           ("calibration_relative_tolerance", .1), ("device", "cpu"),
                           ("optimizer_updates", False), ("kv_cache", 0)):
            changed = copy.deepcopy(PLAN)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                API["cases_for"](changed)

    def test_fixed_training_selection_is_distinct_and_bounded(self):
        selected = API["calibration_indices"](1047, 64)
        self.assertEqual(selected, [i * 1047 // 64 for i in range(64)])
        self.assertEqual(len(set(selected)), 64)
        self.assertEqual(selected, sorted(selected))
        self.assertTrue(all(0 <= i < 1047 for i in selected))
        for size, count in ((10, 11), (1, 0), (True, 1), (10, 1.5)):
            with self.assertRaises(ValueError):
                API["calibration_indices"](size, count)

    def test_scalar_calibration_matches_ratio_without_clipping_or_search(self):
        self.assertAlmostEqual(API["fitted_strength"](.1, .002, .01), .02)
        self.assertAlmostEqual(API["fitted_strength"](.1, .2, .1), .2)
        for values in ((.1, 0., .1), (.1, .1, 0.), (0., .1, .1), (True, .1, .1),
                       (.1, float("nan"), .1), (.1, .1, float("inf")), (.5, 1., .1),
                       (.1, -.1, .1), (.1, 1e308, 1e-308)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                API["fitted_strength"](*values)

    def test_only_recipe_strength_is_excluded_from_state_equality(self):
        original = {"placements": [{"path": path, "recipe": {"strength": .1, "features": 768},
                                    "tensors": [{"name": "gate", "values": [1., 2.]}]}
                                   for path in API["SITES"]]}
        changed = copy.deepcopy(original)
        changed["placements"][0]["recipe"]["strength"] = .02
        self.assertEqual(API["without_strength"](original), API["without_strength"](changed))
        self.assertEqual(original["placements"][0]["recipe"]["strength"], .1)
        changed["placements"][0]["tensors"][0]["values"][0] += .001
        self.assertNotEqual(API["without_strength"](original), API["without_strength"](changed))
        changed["placements"].reverse()
        with self.assertRaises(ValueError):
            API["without_strength"](changed)

    def test_public_allowlist_does_not_expose_native_radii_or_values(self):
        private = [{"path": path, "calls": 32, "rows": 8192, "relative_residual_energy": .01,
                    "native_conditioning": {"log_radius": 1.23, "projection_radius": 3.42},
                    "residual_max_abs": 123., "values": [999.]} for path in API["SITES"]]
        public = API["public_layers"](private)
        self.assertTrue(all(set(item) == {"path", "calls", "rows", "relative_residual_rms"} for item in public))
        for value in (None, 0., float("nan"), float("inf")):
            changed = copy.deepcopy(private)
            changed[0]["relative_residual_energy"] = value
            with self.assertRaises(ValueError):
                API["public_layers"](changed)

    def test_development_mismatch_is_reported_not_silently_refitted(self):
        comparison = API["mismatch"](layers(.02), layers(.01))
        self.assertEqual([row["relative_rms_error"] for row in comparison], [1.] * 4)
        self.assertEqual(API["mismatch"](layers(.01), layers(.01))[0]["relative_rms_error"], 0.)
        for actual in (layers(.01)[:3], layers(0.), layers(float("nan"))):
            with self.assertRaises(ValueError):
                API["mismatch"](actual, layers(.01))

    def test_public_projection_removes_reference_and_intermediate_fit_operands(self):
        stage = {"stage": "sample", "layers": [dict(row, calls=32, rows=8192) for row in layers(.01)]}
        for mode in ("reference", "calibrated_ordinary"):
            original = {"mode": mode, "calibration_stages": [copy.deepcopy(stage)]
                        if mode == "calibrated_ordinary" else [], "calibration": copy.deepcopy(stage),
                        "development": copy.deepcopy(stage), "development_loss": 4.,
                        "calibration_mismatch": API["mismatch"](layers(.01), layers(.01)),
                        "development_mismatch": API["mismatch"](layers(.012), layers(.01))}
            before = copy.deepcopy(original)
            result = API["public_record"](original)
            self.assertEqual(original, before)
            stages = [*result["calibration_stages"], result["calibration"]]
            if mode == "calibrated_ordinary":
                stages.append(result["development"])
                self.assertNotIn("relative_residual_rms", json.dumps(result))
            else:
                self.assertEqual(result["development"], original["development"])
            self.assertNotIn("relative_rms_error", json.dumps(result))
            self.assertTrue(all(row["within_0_1_percent"] for row in result["calibration_mismatch"]))
            self.assertTrue(all(not row["within_5_percent"] for row in result["development_mismatch"]))
            for public_stage in stages:
                self.assertTrue(all(set(row) == {"path", "calls", "rows"} for row in public_stage["layers"]))


if __name__ == "__main__":
    unittest.main()
