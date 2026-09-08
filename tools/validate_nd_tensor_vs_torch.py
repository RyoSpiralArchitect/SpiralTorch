#!/usr/bin/env python3
"""Replay captured N-D preprocessing, NN inference and eight SGD steps in Torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rank_vs_torch as audit


def validate(paths, devices, output, source_ref="HEAD"):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch

    source_binding = {
        "commit": audit.git_bytes("rev-parse", source_ref).decode().strip(),
        "tree": audit.git_bytes("rev-parse", source_ref + "^{tree}").decode().strip(),
        "tracked_dirty": False,
    }

    expected_recipes = {((2, 3, 4), 17, 0), ((3, 5, 7), 29, 1), ((2, 4, 11), 43, 20)}
    report = {
        "schema": "spiraltorch.nd_tensor.torch_validation.v1",
        "status": "error",
        "torch_version": torch.__version__,
        "checks": [],
        "fallback_enabled": False,
        "source": source_binding,
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
                    raise ValueError("wrong fixture schema/status")
                identity = {
                    "schema": "spiraltorch.native_build_identity.v1",
                    "manifest": source.get("build_manifest"),
                }
                if not audit.validate_source_binding(identity, source_binding)["valid"]:
                    raise ValueError(
                        "fixture build is not bound to the supplied frozen source"
                    )
                recipes = [
                    (tuple(c["shape"]), c["seed"], c["iterations"])
                    for c in source["cases"]
                ]
                if len(recipes) != 3 or set(recipes) != expected_recipes:
                    raise ValueError("incomplete or changed recipes")
                pointwise = source.get("pointwise_cases", [])
                if "pointwise_cases" in source:
                    expected = {
                        (shape, seed, iterations, mode)
                        for shape, seed, iterations in expected_recipes
                        for mode in ("Sequential", "Batched", "Fused")
                    }
                    actual = [
                        (
                            tuple(c["shape"]),
                            c["seed"],
                            c["iterations"],
                            c["pointwise_mode"],
                        )
                        for c in pointwise
                    ]
                    if len(actual) != 9 or set(actual) != expected:
                        raise ValueError("incomplete pointwise recipes")
                for device in devices:
                    if device == "mps" and not torch.backends.mps.is_available():
                        raise RuntimeError("MPS unavailable; no fallback")
                    for case in source["cases"] + pointwise:
                        error = 0.0

                        def compare(actual, expected):
                            nonlocal error
                            a = np.asarray(actual, dtype=np.float32).reshape(-1)
                            b = expected.detach().cpu().numpy().reshape(-1)
                            if (
                                a.shape != b.shape
                                or not np.isfinite(a).all()
                                or not np.isfinite(b).all()
                                or not np.allclose(a, b, atol=1e-5, rtol=1e-4)
                            ):
                                raise ValueError("captured tensor disagrees with Torch")
                            error = max(
                                error, float(np.max(np.abs(a - b))) if a.size else 0.0
                            )

                        shape = case["shape"]
                        if (
                            case["gain"] != 0.75
                            or case["steps"] != 8
                            or (
                                case["learning_rate"] != float(np.float32(0.02))
                                and case["learning_rate"] != 0.02
                            )
                        ):
                            raise ValueError(
                                "changed optimizer or preprocessing recipe"
                            )
                        x = (
                            torch.tensor(
                                case["input"], dtype=torch.float32, device=device
                            )
                            .reshape(shape)
                            .permute(1, 0, 2)
                            .narrow(0, 1, shape[1] - 1)
                        )
                        bias = torch.tensor(
                            case["bias"], dtype=torch.float32, device=device
                        )
                        gain = torch.tensor(0.75, dtype=torch.float32, device=device)
                        for _ in range(case["iterations"]):
                            x = torch.nn.functional.gelu(
                                (x + bias) * gain, approximate="tanh"
                            )
                        if list(x.shape) != case["processed_shape"]:
                            raise ValueError("processed shape mismatch")
                        compare(case["processed"], x)
                        layers = []
                        for layer in case["layers"]:
                            w = (
                                torch.tensor(
                                    layer["weights"], dtype=torch.float32, device=device
                                )
                                .reshape(layer["inner"], layer["cols"])
                                .requires_grad_(True)
                            )
                            b = torch.tensor(
                                layer["bias"], dtype=torch.float32, device=device
                            ).requires_grad_(True)
                            layers.append((w, b, layer["gelu"]))

                        def forward(value):
                            for w, b, gelu in layers:
                                value = value @ w + b
                                if gelu:
                                    value = torch.nn.functional.gelu(
                                        value, approximate="tanh"
                                    )
                            return value

                        with torch.no_grad():
                            predicted = forward(x).mul(gain).permute(1, 0, 2).relu()
                        if list(predicted.shape) != case["output_shape"]:
                            raise ValueError("output shape mismatch")
                        compare(case["output"], predicted)
                        x = x.detach().requires_grad_(True)
                        target = torch.tensor(
                            case["targets"], dtype=torch.float32, device=device
                        ).reshape(*x.shape[:-1], layers[-1][0].shape[1])
                        for step in range(8):
                            x.grad = None
                            for w, b, _ in layers:
                                w.grad = None
                                b.grad = None
                            predicted = forward(x)
                            loss = (predicted - target).square().mean()
                            loss.backward()
                            if step == 7:
                                final = case["training"]
                                compare([final["loss"]], loss)
                                compare(final["prediction"], predicted)
                                compare(final["input_gradient"], x.grad)
                                if len(final["parameter_gradients"]) != len(
                                    layers
                                ) or len(final["parameters"]) != len(layers):
                                    raise ValueError("parameter count mismatch")
                                for (w, b, _), g in zip(
                                    layers, final["parameter_gradients"]
                                ):
                                    compare(g["weights"], w.grad)
                                    compare(g["bias"], b.grad)
                            with torch.no_grad():
                                for w, b, _ in layers:
                                    w.add_(w.grad, alpha=-0.02)
                                    b.add_(b.grad, alpha=-0.02)
                        for (w, b, _), p in zip(layers, case["training"]["parameters"]):
                            compare(p["weights"], w)
                            compare(p["bias"], b)
                        report["checks"].append(
                            {
                                "source": str(path),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "device": device,
                                "shape": shape,
                                "seed": case["seed"],
                                "max_abs_error": error,
                            }
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
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "mps"), default=["cpu", "mps"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        default="HEAD",
        help="immutable source ref used to build both fixtures",
    )
    args = parser.parse_args()
    validate(args.reports, args.devices, args.output, args.source)
