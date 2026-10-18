"""Pure-stdlib checks: no pretrained model, corpus or private artifacts."""

import copy
import json
import math
from pathlib import Path
import runpy
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "replay.py"))
BOOK = HERE.parent / "2026-10-18-hf-book-controls"


def trace(native=True):
    events = []
    for rows, source, delta, norm in ((2, 3., 1., 1.), (1, 4., 2., 4.)):
        for site in ("a", "b", "c", "d"):
            report = {"schema": "spiraltorch.wave_gate_conditioning.v2", "rows": rows,
                      "features": 768, "affine_nonfinite_values": 0,
                      "affine_outside_values": rows * 384, "gate_outside_values": 1,
                      "affine_abs_slope_mean": norm, "dimensionless_norm_mean": norm,
                      "relative_radial_gain_mean": norm, "relative_tangential_gain_mean": norm,
                      "relative_radial_gain_min": .2, "relative_tangential_gain_min": .4,
                      "dimensionless_norm_max": 5., "projection_radius": 4., "log_radius": 1.3862944}
            events.append({"path": site, "training": False, "grad_enabled": False,
                           "status": "observed", "shape": [rows, 768], "device": "mps:0",
                           "native_reports": [report] if native else [], "native_reports_omitted": 0,
                           "residual": {"finite": True, "input_l2": source,
                                        "residual_l2": delta, "residual_max_abs": .3}})
    return {"schema": "spiraltorch.geometry_observation.v1", "coverage_complete": True,
            "calls_seen": 8, "events": events}


class ReplayTests(unittest.TestCase):
    def test_wheel_bytes_bind_all_top_level_python_and_native_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("spiraltorch/__init__.py", b"example")
                archive.writestr("spiraltorch/native.so", b"native")
                archive.writestr("unrelated/entry.py", b"not package code")
            result = API["wheel_package_identity"](path)
            self.assertEqual(set(result), {"__init__.py", "native.so"})
            self.assertEqual(result["native.so"], "sha256:" + API["hashlib"].sha256(b"native").hexdigest())

    def aggregate(self, report, native=True):
        return API["aggregate"](report, ["a", "b", "c", "d"], native=native,
                                expected_calls=8, expected_rows=3)

    def test_energy_and_native_means_are_weighted_by_rows(self):
        for row in self.aggregate(trace()):
            self.assertAlmostEqual(row["relative_residual_energy"], math.sqrt(5 / 25))
            self.assertEqual(row["native_conditioning"]["dimensionless_norm_mean"], 2.)
            self.assertEqual(row["native_conditioning"]["affine_outside_fraction"], .5)
            self.assertEqual(row["native_conditioning"]["gate_outside_values_max"], 1)

    def test_ordinary_has_no_invented_geometric_jacobian(self):
        self.assertTrue(all(r["native_conditioning"] is None
                            for r in self.aggregate(trace(False), native=False)))

    def test_incomplete_nonfinite_wrong_shape_and_training_fail_closed(self):
        for field, value in (("training", True), ("grad_enabled", True), ("shape", [2, 767]),
                             ("device", "cpu"), ("native_reports", []),
                             ("native_reports_omitted", 1), ("status", "skipped_value_budget")):
            with self.subTest(field=field):
                report = trace()
                report["events"][0][field] = value
                with self.assertRaises(ValueError):
                    self.aggregate(report)
        for mutation in (lambda r: r.update(coverage_complete=False),
                         lambda r: r["events"].pop(),
                         lambda r: r["events"][0]["residual"].update(finite=False),
                         lambda r: r["events"][0]["native_reports"][0].update(rows=3)):
            report = trace()
            mutation(report)
            with self.assertRaises(ValueError):
                self.aggregate(report)

    def test_changed_native_recipe_is_not_pooled(self):
        report = trace()
        report["events"][0]["native_reports"][0]["projection_radius"] = 2.
        with self.assertRaises(ValueError):
            self.aggregate(report)

    def test_all_28_cases_and_external_state_pins_are_retained(self):
        results = json.loads((BOOK / "results.json").read_text())
        saved = json.loads((BOOK / "saved-state-check.json").read_text())
        preflight = json.loads((BOOK / "preflight-qualified.json").read_text())
        cases = API["cases_for"](results, saved, preflight)
        self.assertEqual(len(cases), 28)
        self.assertEqual([c["seed"] for c in cases[:4]], [None] * 4)
        self.assertEqual([c["step"] for c in cases[4:]], [256, 512] * 12)
        for index, record in enumerate(results["runs"]):
            mid, final = cases[4 + 2 * index:6 + 2 * index]
            self.assertEqual(mid["artifact_id"], saved["checkpoints"][2 * index]["private_file_ids"]["spiraltorch-geometry.json"])
            self.assertEqual(final["artifact_id"], record["artifact_id"])
            self.assertEqual(final["expected_loss"], record["trained_geometry_eval_loss"])
        for source, key in ((results, "runs"), (saved, "checkpoints")):
            original = copy.deepcopy(source)
            source[key].reverse()
            with self.assertRaises(ValueError):
                API["cases_for"](results, saved, preflight)
            source.update(original)

    def test_published_results_retain_every_case_and_qualified_boundary(self):
        results = json.loads((HERE / "results.json").read_text())
        execution = json.loads((HERE / "execution.json").read_text())
        self.assertEqual(results["status"], "completed")
        self.assertEqual(results["execution_id"], API["digest"](HERE / "execution.json"))
        self.assertEqual(results["plan_id"], API["digest"](HERE / "plan.json"))
        self.assertFalse(results["reserve_opened"])
        self.assertEqual(results["optimizer_updates"], 0)
        self.assertEqual(results["quality_status"], "posthoc_mechanism_diagnosis_not_quality_confirmation")
        declared = API["cases_for"](
            json.loads((BOOK / "results.json").read_text()),
            json.loads((BOOK / "saved-state-check.json").read_text()),
            json.loads((BOOK / "preflight-qualified.json").read_text()))
        self.assertEqual(len(results["cases"]), len(declared))
        self.assertEqual(len(execution["cases"]), len(declared))
        sites = json.loads((BOOK / "plan.json").read_text())["sites"]
        for expected, case, original in zip(declared, results["cases"], execution["cases"]):
            self.assertEqual({k: case[k] for k in expected}, expected)
            self.assertEqual({k: original[k] for k in expected}, expected)
            self.assertTrue(math.isfinite(case["eval_loss"]))
            if expected["expected_loss"] is not None:
                self.assertEqual(case["eval_loss"], expected["expected_loss"])
            for field in ("observed_loss_exact", "first_window_logits_exact", "rng_exact",
                          "base_and_geometry_unchanged", "coverage_complete"):
                self.assertIs(case[field], True)
            self.assertEqual(case["optimizer_updates"], 0)
            self.assertEqual([layer["path"] for layer in case["layers"]], sites)
            for layer in case["layers"]:
                self.assertEqual((layer["calls"], layer["rows"]), (80, 20352))
                self.assertEqual(layer["native_conditioning"] is None, case["arm"] == "ordinary")
        json.dumps(results, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
