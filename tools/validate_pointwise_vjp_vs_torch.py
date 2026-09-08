#!/usr/bin/env python3
"""Replay frozen native/browser pointwise and upstream NN VJPs in real Torch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rank_vs_torch as audit

RECIPES = (
    ((2, 3, 4), (4,), 17),
    ((2, 3, 4), (1, 3, 1), 29),
    ((513, 3), (3,), 43),
    ((65537, 1), (), 47),
    ((), (), 53),
    ((2, 0, 3), (3,), 59),
)
STEPS = [
    {"op": "multiply", "rhs": 1},
    {"op": "gelu", "rhs": None},
    {"op": "add", "rhs": 0},
    {"op": "multiply", "rhs": 2},
    {"op": "relu", "rhs": None},
]
GUARDS = (
    "zero_seed_does_not_hide_bad_forward",
    "derivative_overflow_rejected",
    "late_reduction_invalidates_all_slots",
    "empty_failed_seed_preserved",
    "shape_count_mixed_host_rejected",
    "foreign_device_rejected",
    "strided_cotangent",
    "immutable_gradients",
)


def validate_fixture(source):
    fixture = source["pointwise_vjp"]
    if (
        fixture.get("schema") != "spiraltorch.pointwise_vjp.fixture.v1"
        or fixture.get("status") != "passed"
        or fixture["guards"].get("status") != "passed"
        or any(fixture["guards"].get(k) is not True for k in GUARDS)
    ):
        raise ValueError("incomplete VJP status or guards")
    cases = fixture["cases"]
    if len(cases) != len(RECIPES):
        raise ValueError("incomplete VJP recipes")
    for case, (shape, gain_shape, seed) in zip(cases, RECIPES):
        inputs = [
            {
                "shape": list(shape),
                "values": [
                    ((i * 13 + seed) % 61) / 32.0 - 0.875
                    for i in range(math.prod(shape))
                ],
            },
            {
                "shape": list(gain_shape),
                "values": [0.5 + (i % 7) / 8.0 for i in range(math.prod(gain_shape))],
            },
            {"shape": [], "values": [0.75]},
        ]
        if (
            case["shape"] != list(shape)
            or case["gain_shape"] != list(gain_shape)
            or case["seed"] != seed
            or case["steps"] != STEPS
            or case["inputs"] != inputs
            or case["cotangent"]
            != [((i * 7 + seed) % 17) / 16.0 - 0.5 for i in range(math.prod(shape))]
        ):
            raise ValueError("changed VJP fixture inputs/recipe")
        if [g["shape"] for g in case["gradients"]] != [i["shape"] for i in inputs]:
            raise ValueError("VJP gradient slots or shapes changed")
    bridge = fixture["nn_bridge"]
    expected_inputs = [
        {
            "shape": [2, 2, 4],
            "values": [
                ((old_axis * 2 + first_axis) * 4 + col) / 16.0 - 0.5
                for first_axis in range(2)
                for old_axis in (1, 2)
                for col in range(4)
            ],
        },
        {"shape": [4], "values": [0.5, 1.25, -0.5, 0.75]},
        {"shape": [], "values": [1.0]},
    ]
    layers = []
    cursor = 0
    for inner, cols, gelu in ((4, 5, True), (5, 2, False)):
        weights = [
            (i * 11 % 23 - 11) / 32.0 for i in range(cursor, cursor + inner * cols)
        ]
        cursor += inner * cols
        bias = [(i * 11 % 23 - 11) / 32.0 for i in range(cursor, cursor + cols)]
        cursor += cols
        layers.append(
            dict(inner=inner, cols=cols, weights=weights, bias=bias, gelu=gelu)
        )
    if (
        bridge["inputs"] != expected_inputs
        or bridge["layers"] != layers
        or bridge["targets"] != [0.125] * 8
    ):
        raise ValueError("changed NN bridge inputs, parameters or targets")
    if (
        bridge.get("status") != "passed"
        or bridge["steps"] != STEPS
        or bridge.get("snapshots_survive_reuse") is not True
        or bridge.get("stale_gradient_rejected") is not True
        or bridge.get("rejected_nn_step_invalidates_upstream_gradients") is not True
        or bridge["shape"] != [2, 2, 4]
        or bridge["output_shape"] != [2, 2, 2]
        or len(bridge["layers"]) != 2
        or [g["shape"] for g in bridge["gradients"]] != [[2, 2, 4], [4], []]
    ):
        raise ValueError("changed or incomplete NN bridge")
    return fixture


def forward(torch, inputs):
    x, gain, scale = inputs
    return ((torch.nn.functional.gelu(x * gain, approximate="tanh") + x) * scale).relu()


def replay(torch, np, fixture, device):
    checks = []
    for index, case in enumerate(fixture["cases"] + [fixture["nn_bridge"]]):
        error = 0.0

        def compare(actual, expected):
            nonlocal error
            a = np.asarray(actual, dtype=np.float32).reshape(-1)
            b = expected.detach().cpu().numpy().reshape(-1)
            if (
                a.shape != b.shape
                or not np.isfinite(a).all()
                or not np.isfinite(b).all()
                or not np.allclose(a, b, atol=2e-5, rtol=2e-4)
            ):
                raise ValueError(f"VJP mismatch in case {index}, device {device}")
            error = max(error, float(np.max(np.abs(a - b))) if a.size else 0.0)

        inputs = [
            torch.tensor(item["values"], dtype=torch.float32, device=device)
            .reshape(item["shape"])
            .requires_grad_(True)
            for item in case["inputs"]
        ]
        y = forward(torch, inputs)
        if index < len(RECIPES):
            cotangent = torch.tensor(
                case["cotangent"], dtype=torch.float32, device=device
            ).reshape(y.shape)
            compare(case["output"], y)
            gradients = torch.autograd.grad(y, inputs, grad_outputs=cotangent)
        else:
            compare(case["processed"], y)
            y.retain_grad()
            out = y
            for layer in case["layers"]:
                w = torch.tensor(
                    layer["weights"], dtype=torch.float32, device=device
                ).reshape(layer["inner"], layer["cols"])
                b = torch.tensor(layer["bias"], dtype=torch.float32, device=device)
                out = out @ w + b
                if layer["gelu"]:
                    out = torch.nn.functional.gelu(out, approximate="tanh")
            target = torch.tensor(
                case["targets"], dtype=torch.float32, device=device
            ).reshape(out.shape)
            (out - target).square().mean().backward()
            compare(case["input_cotangent"], y.grad)
            gradients = [item.grad for item in inputs]
        if len(gradients) != len(case["gradients"]):
            raise ValueError("missing gradients")
        for actual, expected in zip(case["gradients"], gradients):
            if actual["shape"] != list(expected.shape):
                raise ValueError("gradient shape mismatch")
            compare(actual["values"], expected)
        checks.append(
            {
                "device": device,
                "case": index,
                "shape": case["shape"],
                "nn_bridge": index == len(RECIPES),
                "max_abs_error": error,
            }
        )
    return checks


def validate(paths, devices, output, source_ref):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch

    binding = {
        "commit": audit.git_bytes("rev-parse", source_ref).decode().strip(),
        "tree": audit.git_bytes("rev-parse", source_ref + "^{tree}").decode().strip(),
        "tracked_dirty": False,
    }
    report = {
        "schema": "spiraltorch.pointwise_vjp.torch_validation.v1",
        "status": "error",
        "source": binding,
        "checks": [],
        "torch_version": torch.__version__,
        "fallback_enabled": False,
        "boundary": "logical input slot VJPs, summed broadcast axes; tanh GELU; mean-MSE NN probe; no optimizer or storage-view adjoint claim",
    }
    with output.open("x", encoding="utf-8") as handle:
        try:
            for path in paths:
                raw = path.read_bytes()
                source = json.loads(raw)
                if (
                    source.get("schema") != "spiraltorch.resident_nd_tensor.fixture.v1"
                    or source.get("status") != "passed"
                ):
                    raise ValueError("wrong outer fixture")
                identity = {
                    "schema": "spiraltorch.native_build_identity.v1",
                    "manifest": source["build_manifest"],
                }
                if not audit.validate_source_binding(identity, binding)["valid"]:
                    raise ValueError("fixture is not bound to the frozen source")
                fixture = validate_fixture(source)
                for device in devices:
                    if device == "mps" and not torch.backends.mps.is_available():
                        raise RuntimeError("MPS unavailable; no fallback")
                    for check in replay(torch, np, fixture, device):
                        report["checks"].append(
                            dict(
                                check,
                                source=str(path),
                                sha256=hashlib.sha256(raw).hexdigest(),
                            )
                        )
            report["status"] = "passed"
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": len(report["checks"]),
                "max_abs_error": max(c["max_abs_error"] for c in report["checks"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "mps"), default=["cpu", "mps"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="HEAD")
    args = parser.parse_args()
    validate(args.reports, args.devices, args.output, args.source)
