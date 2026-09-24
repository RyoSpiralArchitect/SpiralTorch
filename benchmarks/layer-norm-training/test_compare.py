"""The public comparison gate rejects missing work and mismatched fixtures."""

import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("layer_norm_training_compare",
                                              Path(__file__).with_name("compare.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def reports():
    native = {"schema": "spiraltorch.layer_norm.training_residency_exploratory.v1",
              "adapter": "device_type: IntegratedGpu, backend: Metal",
              "routes": compare.RUST_ROUTES, "stage_routes": compare.STAGES,
              "steps": 32, "warmup": 3, "iterations": 9, "epsilon": 1e-5,
              "rate": "-0.1 * cols", "scaled_tolerance": compare.TOLERANCE,
              "cases": []}
    torch = {"schema": "spiraltorch.layer_norm.training_residency_torch.v1",
             "status": "passed", "devices": compare.TORCH_ROUTES,
             "mps_fallback": False, "compiled": False,
             "intra_op_threads": 4, "inter_op_threads": 1,
             "steps": 32, "warmup": 3, "iterations": 9,
             "epsilon": 1e-5, "rate": "-0.1 * cols",
             "scaled_tolerance": compare.TOLERANCE, "cases": []}
    for rows, cols in compare.SHAPES:
        output = [[0.1]] + [[1.0] * cols for _ in range(4)]
        base = {"rows": rows, "cols": cols, "initial_loss": 1.0,
                "input_fnv64": "same-input",
                "target_fnv64": "same-target"}
        native_case = dict(base, terminal_maps=[0, 1, 1, 1],
                           max_scaled_error=[0.0] * 4,
                           final_outputs=[copy.deepcopy(output) for _ in range(4)],
                           intervals=[{"iteration": i, "route": route, "ms": 1.0}
                                      for i in range(9) for route in range(4)],
                           stage_intervals=[{"iteration": i, "stage": stage, "ms": 1.0}
                                            for i in range(9) for stage in range(7)])
        torch_case = dict(base, max_scaled_error={"cpu": 0.0, "mps": 0.0},
                          final_outputs={route: copy.deepcopy(output)
                                         for route in compare.TORCH_ROUTES},
                          intervals=[{"iteration": i, "route": route, "ms": 1.0}
                                     for i in range(9) for route in compare.TORCH_ROUTES])
        native["cases"].append(native_case)
        torch["cases"].append(torch_case)
    return native, torch


class ComparisonTests(unittest.TestCase):
    def test_complete_matched_reports_pass(self):
        native, torch = reports()
        summary = compare.validate(native, torch)
        self.assertEqual(summary["status"], "validated")
        self.assertEqual(summary["update_execution"], "sequential")

    def test_update_execution_is_validated_and_preserved(self):
        native, torch = reports()
        for execution in compare.UPDATE_EXECUTIONS:
            native["update_execution"] = execution
            self.assertEqual(compare.validate(
                native, torch, expected_update_execution=execution)["update_execution"],
                execution)
        native["update_execution"] = "fused"
        with self.assertRaisesRegex(ValueError, "Rust update execution"):
            compare.validate(native, torch, expected_update_execution="batched")
        native["update_execution"] = "unrecognized"
        with self.assertRaisesRegex(ValueError, "Rust update execution"):
            compare.validate(native, torch)

    def test_mismatched_fixture_is_rejected(self):
        native, torch = reports()
        torch["cases"][3]["target_fnv64"] = "different"
        with self.assertRaisesRegex(ValueError, "target_fnv64"):
            compare.validate(native, torch)

    def test_missing_stage_interval_is_rejected(self):
        native, torch = reports()
        native["cases"][0]["stage_intervals"].pop()
        with self.assertRaisesRegex(ValueError, "interval count"):
            compare.validate(native, torch)

    def test_cross_runtime_drift_is_rejected(self):
        native, torch = reports()
        torch["cases"][1]["final_outputs"]["mps"][1][0] = 2.0
        with self.assertRaisesRegex(ValueError, "differs from Rust CPU"):
            compare.validate(native, torch)

    def test_native_gpu_drift_is_rejected(self):
        native, torch = reports()
        native["cases"][2]["final_outputs"][2][1][0] = 2.0
        with self.assertRaisesRegex(ValueError, "wgpu_preloaded_all differs from Rust CPU"):
            compare.validate(native, torch)


if __name__ == "__main__":
    unittest.main()
