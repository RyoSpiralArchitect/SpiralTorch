"""Small hostile-input checks, independent of GPU and archived raw files."""
import copy
import unittest

import validate_resident_query_pullback as validator


def valid():
    names = ["token", "position", "geometry.projection.weight", "geometry.raw_decay",
             "geometry.raw_phase", "geometry.raw_gain.0", "geometry.query_projection.0.weight", "head.weight"]
    reference = {"name": "query", "parameters": [{"name": n} for n in names],
        "parameter_gradients": [[.25, .5] for _ in names],
        "embedding_output_gradient": [1.25, -.5],
        "query_controls": {"router_detached_embedding_gradient": [1., -.25]}}
    stopped = {"name": "query", "embedding": [1., -.25],
               "parameters": [[.125, .5] if i < 5 else [.25, .5] for i in range(len(names))]}
    actual = {"name": "query", "passed": True, "paired_relative_l2": 0.,
        "full_embedding_gradient": [1.25, -.5], "stopped_embedding_gradient": [1., -.25],
        "full_parameter_gradients": copy.deepcopy(reference["parameter_gradients"]),
        "stopped_parameter_gradients": copy.deepcopy(stopped["parameters"]),
        "unchanged_parameter_slots": [5, 6, 7], "legacy_mixed_runtime_passed": True,
        "negative_controls_rejected": 2, "forward_unchanged": True,
        "parameters_unchanged": True, "full_repeated_exact": True}
    return ({"schema": "spiraltorch.resident_byte_query_pullback.validation.v1", "passed": True,
             "adapter": "test-only", "checks": [copy.deepcopy(actual), actual]},
            {"cases": [{}, {}, {}, {}, copy.deepcopy(reference), reference]},
            {"cases": [copy.deepcopy(stopped), stopped]})


class PairedValidatorTests(unittest.TestCase):
    def test_complete_nonzero_control_passes(self):
        result = validator.validate_runtime(*valid())
        self.assertTrue(result["paired_passed"])
        self.assertEqual(len(result["checks"]), 2)

    def test_rejects_missing_family_or_control(self):
        for field in ("full_parameter_gradients", "stopped_parameter_gradients", "unchanged_parameter_slots"):
            args = valid()
            args[0]["checks"][0][field].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator.validate_runtime(*args)

    def test_rejects_changed_unrelated_parameter_within_absolute_tolerance(self):
        args = valid()
        args[0]["checks"][0]["stopped_parameter_gradients"][7][0] += 1e-7
        with self.assertRaisesRegex(ValueError, "unrelated"):
            validator.validate_runtime(*args)

    def test_rejects_inert_geometry_even_below_absolute_tolerance(self):
        args = valid()
        args[0]["checks"][0]["stopped_parameter_gradients"][2] = [0., 0.]
        args[2]["cases"][0]["parameters"][2] = [1e-7, 1e-7]
        with self.assertRaisesRegex(ValueError, "inert"):
            validator.validate_runtime(*args)

    def test_does_not_trust_pass_flags_or_error_numbers(self):
        for field, value in [("paired_relative_l2", 0.001), ("legacy_mixed_runtime_passed", False),
                             ("negative_controls_rejected", 0), ("forward_unchanged", False),
                             ("full_repeated_exact", False), ("passed", False)]:
            args = valid()
            args[0]["checks"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator.validate_runtime(*args)

    def test_relative_checker_rejects_zero_wrong_sign_nan_and_truncation(self):
        for bad in ([0., 0.], [-1., 2.], [float("nan"), -2.], [1.]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validator.relative(bad, [1., -2.])

    def test_original_report_equality_is_bitwise_not_tolerant(self):
        self.assertFalse(validator.exact({"v": [0.]}, {"v": [-0.]}))
        self.assertFalse(validator.exact([.1], [.100000000000001]))
        self.assertFalse(validator.exact([float("nan")], [float("nan")]))
        self.assertFalse(validator.exact({"a": 1}, {}))
        self.assertTrue(validator.exact({"v": [0., .1]}, {"v": [0., .1]}))


if __name__ == "__main__":
    unittest.main()
