"""Validate matched Rust/PyTorch training reports and derive all route medians."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


SHAPES = [(2, 3), (8, 257), (32, 256), (64, 768), (128, 1025), (256, 256)]
RUST_ROUTES = ["rust_cpu_all", "wgpu_host_to_host_all", "wgpu_preloaded_all",
               "wgpu_preloaded_affine_only"]
STAGES = ["readback_only", "forward", "mse", "backward_all",
          "backward_input", "backward_affine", "update"]
TORCH_ROUTES = ["cpu", "mps"]
TOLERANCE = 5e-4
UPDATE_EXECUTIONS = ("sequential", "batched", "fused", "grouped_fused")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_nonnegative(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and value >= 0


def validate_outputs(outputs, cols):
    require(len(outputs) == 5, "final output count")
    require([len(part) for part in outputs] == [1, cols, cols, cols, cols],
            "final output shapes")
    require(all(math.isfinite(value) for part in outputs for value in part),
            "nonfinite final output")


def scaled_error(actual, expected):
    return max(abs(value - reference) / (TOLERANCE * (1.0 + abs(reference)))
               for part, reference_part in zip(actual, expected)
               for value, reference in zip(part, reference_part))


def validate_intervals(intervals, key, routes, count):
    require(len(intervals) == len(routes) * count, "interval count")
    require({(item["iteration"], item[key]) for item in intervals} ==
            {(iteration, route) for iteration in range(count) for route in routes},
            "interval identities")
    require(all(finite_nonnegative(item["ms"]) and item["ms"] > 0
                for item in intervals), "invalid time")
    return [statistics.median(item["ms"] for item in intervals if item[key] == route)
            for route in routes]


def validate(native, torch, expected_update_execution="sequential"):
    require(native["schema"] == "spiraltorch.layer_norm.training_residency_exploratory.v1",
            "Rust schema")
    update_execution = native.get("update_execution", "sequential")
    require(expected_update_execution in UPDATE_EXECUTIONS and
            update_execution == expected_update_execution, "Rust update execution")
    require(torch["schema"] == "spiraltorch.layer_norm.training_residency_torch.v1"
            and torch["status"] == "passed", "PyTorch schema/status")
    require("device_type: IntegratedGpu" in native["adapter"] and
            "backend: Metal" in native["adapter"], "native GPU identity")
    require(torch["devices"] == TORCH_ROUTES and torch["mps_fallback"] is False,
            "PyTorch device policy")
    require(torch["compiled"] is False and torch["intra_op_threads"] == 4 and
            torch["inter_op_threads"] == 1, "PyTorch execution policy")
    for report in (native, torch):
        require((report["steps"], report["warmup"], report["iterations"]) == (32, 3, 9),
                "training protocol")
        require(struct.pack("<f", report["epsilon"]) == struct.pack("<f", 1e-5)
                and report["rate"] == "-0.1 * cols"
                and report["scaled_tolerance"] == TOLERANCE, "numerical protocol")
        require([(case["rows"], case["cols"]) for case in report["cases"]] == SHAPES,
                "shape coverage/order")
    require(native["routes"] == RUST_ROUTES and native["stage_routes"] == STAGES,
            "Rust route coverage")
    rows = []
    max_cross_error = {route: 0.0 for route in TORCH_ROUTES}
    max_native_error = {route: 0.0 for route in RUST_ROUTES}
    for rust_case, torch_case in zip(native["cases"], torch["cases"]):
        shape = (rust_case["rows"], rust_case["cols"])
        require(shape == (torch_case["rows"], torch_case["cols"]), "shape mismatch")
        for key in ("input_fnv64", "target_fnv64"):
            require(rust_case[key] == torch_case[key], f"{shape} {key} mismatch")
        for case in (rust_case, torch_case):
            require(finite_nonnegative(case["initial_loss"]) and case["initial_loss"] > 0,
                    "initial loss")
        require(abs(rust_case["initial_loss"] - torch_case["initial_loss"]) /
                (TOLERANCE * (1.0 + abs(rust_case["initial_loss"]))) <= 1,
                "initial loss differs")
        require(rust_case["terminal_maps"] == [0, 1, 1, 1], "terminal maps")
        require(len(rust_case["max_scaled_error"]) == len(RUST_ROUTES) and
                all(finite_nonnegative(v) and v <= 1
                    for v in rust_case["max_scaled_error"]), "Rust numerical gate")
        require(set(torch_case["max_scaled_error"]) == set(TORCH_ROUTES) and
                all(finite_nonnegative(v) and v <= 1
                    for v in torch_case["max_scaled_error"].values()),
                "PyTorch numerical gate")
        require(len(rust_case["final_outputs"]) == len(RUST_ROUTES),
                "Rust final route count")
        for route, output in zip(RUST_ROUTES, rust_case["final_outputs"]):
            validate_outputs(output, shape[1])
            require(output[0][0] < rust_case["initial_loss"], "Rust loss did not decrease")
            error = scaled_error(output, rust_case["final_outputs"][0])
            require(error <= 1, f"{shape} {route} differs from Rust CPU")
            max_native_error[route] = max(max_native_error[route], error)
        for route in TORCH_ROUTES:
            validate_outputs(torch_case["final_outputs"][route], shape[1])
            require(torch_case["final_outputs"][route][0][0] < torch_case["initial_loss"],
                    "PyTorch loss did not decrease")
            error = scaled_error(torch_case["final_outputs"][route],
                                 rust_case["final_outputs"][0])
            require(error <= 1, f"{shape} {route} differs from Rust CPU")
            max_cross_error[route] = max(max_cross_error[route], error)
        rows.append({"rows": shape[0], "cols": shape[1],
                     "rust_median_ms": validate_intervals(
                         rust_case["intervals"], "route", list(range(4)), 9),
                     "stage_median_ms": validate_intervals(
                         rust_case["stage_intervals"], "stage", list(range(7)), 9),
                     "torch_median_ms": validate_intervals(
                         torch_case["intervals"], "route", TORCH_ROUTES, 9),
                     "initial_loss": rust_case["initial_loss"],
                     "final_loss": {"rust_cpu": rust_case["final_outputs"][0][0][0],
                                    "torch_cpu": torch_case["final_outputs"]["cpu"][0][0],
                                    "torch_mps": torch_case["final_outputs"]["mps"][0][0]}})
    return {"schema": "spiraltorch.layer_norm.training_residency_comparison.v1",
            "status": "validated", "update_execution": update_execution, "shapes": rows,
            "routes": {"rust": RUST_ROUTES, "stages": STAGES, "pytorch": TORCH_ROUTES},
            "max_native_scaled_error": max_native_error,
            "max_cross_scaled_error": max_cross_error,
            "boundary": "Same f32 input/target bits and 32-step update equation; separate Rust and PyTorch host stacks, no universal speed claim. Affine-only Rust route omits dx and is not work-matched to all-gradient controls."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_json", type=Path)
    parser.add_argument("torch_json", type=Path)
    parser.add_argument("--expected-update-execution", choices=UPDATE_EXECUTIONS,
                        default="sequential")
    args = parser.parse_args()
    paths = [args.native_json, args.torch_json]
    reports = [json.loads(path.read_text()) for path in paths]
    summary = validate(*reports, expected_update_execution=args.expected_update_execution)
    summary["inputs"] = [{"name": path.name,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in paths]
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
