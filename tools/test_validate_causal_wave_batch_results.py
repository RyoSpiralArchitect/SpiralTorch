"""Publication checks that need neither a GPU nor private experiment tensors."""
import copy
import itertools
import math
import unittest

import validate_causal_wave_batch_results as validator


def wave_fixture():
    controls = []
    for exponent, order, channel in itertools.product((24, 48, 96), itertools.permutations(range(3)), (0, 1)):
        expected = validator.f32(validator.f32(.99) * (.25 if channel == 0 else .5 * validator.f32(math.pi)))
        controls.append({"exponent": exponent, "order": list(order), "terminal_channel": channel,
            "expected": expected, "cpu": expected, "gpu": expected, "exact_f32": True, "finite_family": True})
    return {"schema": "spiraltorch.causal_zspace_wave.validation.v1", "passed": True,
        "adapter": "test-only", "scope": "synthetic validator fixture, not runtime evidence",
        "checks": [{"name": f"case{i}_{'features' if i == 0 else 'terminal' if i == 1 else 'both'}",
            "logical_shapes": True, "all_inputs_and_cotangents_strided": True, "cpu_max_abs_error": 0.,
            "errors": dict.fromkeys(("features", "final_state", "drive_vjp", "raw_decay_vjp", "raw_phase_vjp", "initial_state_vjp"), 0.)} for i in range(10)],
        "guards": {"forward_operand_guards": 4, "cotangent_guards": 2, "atomic_rejections": 2,
            "rejection_negative_controls": 4, "good_bad_good": True, "retained_gradients": True,
            "late_reduction_whole_family": True, "large_state_chart": True},
        "streaming": {"feature_max_abs_error": 0., "state_max_abs_error": 0., "vjp_max_abs_error": 0.,
            "prefix_max_abs_error": 0., "future_gradient_zero": True, "sensitivity_control": True, "retained_output": True},
        "radial": {"f64_analytic_reference": True, "one_ulp_controls": 2, "large_exponent_nonzero": True,
            "cases": [{"name": name, "all_four_vjps": True, "cpu_and_gpu_max_abs_error": 0., "expected_phase_gradient": 0.}
                for name in ("axis", "diagonal", "oblique", "oblique_up", "oblique_down", "oblique_negative", "large_exponent", "oblique_phase", "mixed_exponents")]},
        "learning": {"steps": 16, "observation_after_all_updates": True,
            "trace": [{"revision": i + 1, "feature_loss": .1, "terminal_loss": .1, "parameter_and_gradient_max_abs_error": 0.} for i in range(16)]},
        "batch_reduction": {"analytic_reference": True, "scope": "test-only", "checks": controls,
            "overflow": [{"terminal_channel": i, "opposite_partials_cancel": True,
                "finite_family": True, "same_sign_whole_family_rejected": True} for i in range(2)]}}


def model_fixture():
    checks = [{"name": f"case{i}", "output": [1., 2.], "qualification_failures": [],
        "resume_trajectory": {"exact_parameters": True},
        "learning": {"trace": [{"loss": .1, "parameters": [[.25, .5]]} for _ in range(16)]}} for i in range(6)]
    checks[1]["qualification_failures"] = [{"gate": "geometry_relative_l2", "geometry_slot": 3,
        "step": 12, "threshold": .002, "error": "retained failure"}]
    return {"schema": "spiraltorch.resident_byte_geometry_projective.validation.v1", "passed": False, "checks": checks}


class WaveBatchPublicationTests(unittest.TestCase):
    def test_complete_wave_controls_pass(self):
        self.assertTrue(validator.wave(wave_fixture())["passed"])

    def test_rejects_duplicate_or_missing_cases(self):
        for key in ("checks", "overflow"):
            for duplicate in (False, True):
                report = wave_fixture()
                rows = report["batch_reduction"][key]
                if duplicate:
                    rows[-1] = copy.deepcopy(rows[0])
                else:
                    rows.pop()
                with self.subTest(key=key, duplicate=duplicate), self.assertRaises(ValueError):
                    validator.wave(report)

    def test_recomputes_analytic_result_despite_pass_flags(self):
        for value in (.25, 0., float("nan")):
            report = wave_fixture()
            for key in ("expected", "cpu", "gpu"):
                report["batch_reduction"]["checks"][0][key] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validator.wave(report)

    def test_rejects_lost_overflow_guard_or_nonfinite_measurement(self):
        for section, key, value in (("guards", "atomic_rejections", 0),
                                    ("streaming", "retained_output", False),
                                    ("streaming", "vjp_max_abs_error", float("nan"))):
            report = wave_fixture()
            report[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validator.wave(report)
        report = wave_fixture()
        report["batch_reduction"]["overflow"][0]["same_sign_whole_family_rejected"] = False
        with self.assertRaises(ValueError):
            validator.wave(report)

    def test_publication_drops_extra_raw_payloads(self):
        report = wave_fixture()
        expected = validator.wave(report)
        report["parameters"] = [[42.]]
        report["checks"][0]["raw_gradient"] = [42.]
        report["batch_reduction"]["checks"][0]["checkpoint"] = {"raw": [42.]}
        self.assertEqual(expected, validator.wave(report))
        report["checks"][0]["cpu_max_abs_error"] = [42.]
        with self.assertRaises(ValueError):
            validator.wave(report)

    def test_numeric_fields_require_finite_numbers_not_booleans_or_strings(self):
        for value in ("NaN", "0.0", False, -1., float("inf")):
            for field in ("vjp_max_abs_error", "feature_loss"):
                report = wave_fixture()
                section = report["streaming"] if field == "vjp_max_abs_error" else report["learning"]["trace"][0]
                section[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validator.wave(report)

    def test_query_errors_cannot_publish_raw_payloads(self):
        for value in ([42.], {"checkpoint": [42.]}, None, False):
            baseline = model_fixture()
            baseline["checks"][4]["query_controls_error"] = "geometry derivative is inert or inaccurate: norm=0.01, actual=0.009, relative=0.003"
            for target in ("current", "baseline"):
                report, old = copy.deepcopy(baseline), copy.deepcopy(baseline)
                (report if target == "current" else old)["checks"][4]["query_controls_error"] = value
                with self.subTest(target=target, value=value), self.assertRaises(ValueError):
                    validator.model(report, old)

    def test_changed_query_measurement_is_retained_but_different_failure_is_rejected(self):
        baseline = model_fixture()
        baseline["checks"][4]["query_controls_error"] = "geometry derivative is inert or inaccurate: norm=0.01, actual=0.009, relative=0.003"
        report = copy.deepcopy(baseline)
        report["checks"][4]["query_controls_error"] = "geometry derivative is inert or inaccurate: norm=0.01, actual=0.008, relative=0.004"
        self.assertEqual(validator.model(report, baseline)["qualified_cases"], 4)
        for error in ("case4 step 12 geometry.query_projection.0.weight: contextual update inert or inaccurate: ref=0.1, rel=0.2",
                      "invalid geometry gradient family", "geometry derivative is inert or inaccurate: norm=NaN, actual=0.1, relative=0.3"):
            report["checks"][4]["query_controls_error"] = error
            with self.subTest(error=error), self.assertRaises(ValueError):
                validator.model(report, baseline)

    def test_retains_failed_qualification_without_parameters(self):
        baseline = model_fixture()
        result = validator.model(copy.deepcopy(baseline), baseline)
        self.assertEqual(result["qualified_cases"], 5)
        self.assertFalse(result["cases"][1]["qualified"])
        self.assertEqual(result["cases"][1]["qualification_failures"], baseline["checks"][1]["qualification_failures"])
        self.assertNotIn("parameters", result["cases"][0]["trace"][0])

    def test_rejects_new_gate_failure_or_changed_threshold(self):
        for field, value in (("threshold", .003), ("step", 13), ("geometry_slot", 4)):
            baseline = model_fixture()
            report = copy.deepcopy(baseline)
            report["checks"][1]["qualification_failures"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator.model(report, baseline)

    def test_rejects_forward_resume_or_trajectory_regression(self):
        for mutation in ("forward", "resume", "trace", "nonfinite", "passed", "query"):
            baseline = model_fixture()
            report = copy.deepcopy(baseline)
            first = report["checks"][0]
            if mutation == "forward":
                first["output"][0] += 1e-7
            elif mutation == "resume":
                first["resume_trajectory"]["exact_parameters"] = False
            elif mutation == "trace":
                first["learning"]["trace"].pop()
            elif mutation == "nonfinite":
                first["learning"]["trace"][0]["parameters"][0][0] = float("inf")
            elif mutation == "passed":
                report["passed"] = True
            else:
                first["query_controls_error"] = "new failure"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validator.model(report, baseline)


if __name__ == "__main__":
    unittest.main()
