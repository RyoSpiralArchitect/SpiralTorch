#!/usr/bin/env python3
"""Independent single-thread CPU Torch comparison for the native shared NN probe.

Both paths produce input and summed feature-gate VJPs and accumulate the gate
gradient into a preallocated zero buffer. Only Rust includes semantic audits.
This measures no optimizer updates, file transport, gradient reset or quality.
"""

import argparse
import array
import hashlib
import json
import math
import sys
import time
from pathlib import Path


FIELDS = ("input", "gate", "upstream", "output", "grad_input", "grad_gate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(receipt, data):
    require(receipt["schema"] == "spiraltorch.topos_shared_nn_probe.v1", "schema")
    require(receipt["status"] == "measured" and receipt["backend"] == "cpu", "status/backend")
    require(receipt["dtype"] == "float32" and receipt["gate_layout"] == "shared_rows", "dtype/layout")
    require(receipt["gate_gradient_reduction"] == "sum_without_additional_mean", "reduction")
    require(receipt["gradient_storage"] == "preallocated_zero_accumulator_for_reference_and_all_rounds", "gradient storage")
    shape = receipt["shape"]
    require(isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape), "shape")
    volume = math.prod(shape)
    require(volume <= 1_048_576, "volume budget")
    shapes = [shape, [1, shape[1]], shape, shape, shape, [1, shape[1]]]
    require(receipt["vector_order"] == list(FIELDS), "vector order")
    require(receipt["vector_shapes"] == shapes and all(type(n) is int for s in receipt["vector_shapes"] for n in s), "vector shapes")
    config = receipt["config"]
    require(type(config["iterations"]) is int and 1 <= config["iterations"] <= 64, "iteration budget")
    require(all(finite_number(config[n]) for n in ("coupling", "saturation", "porosity")), "finite config")
    require(0 <= config["coupling"] < 1 and config["saturation"] > 0 and 0 <= config["porosity"] <= 1, "config domain")
    orders = receipt["round_order"]
    require(isinstance(orders, list) and 2 <= len(orders) <= 256 and len(orders) % 2 == 0, "round budget")
    require(all(type(route) is int for order in orders for route in order), "integer route order")
    require(orders == [[0, 1] if i % 2 == 0 else [1, 0] for i in range(len(orders))], "balanced route order")
    require(type(receipt["warmup_per_route"]) is int and receipt["warmup_per_route"] == 2, "warmup")
    require(set(receipt["measurements_ms"]) == {"forward", "forward_backward"}, "timing routes")
    for values in receipt["measurements_ms"].values():
        require(len(values) == len(orders) and all(finite_number(v) and v > 0 for v in values), "finite timings")
    counts = [math.prod(s) for s in shapes]
    require(len(data) == sum(counts) * 4 and digest(data) == receipt["vectors_sha256"], "vector identity")
    chunks, offset = [], 0
    for count in counts:
        chunk = data[offset:offset + count * 4]
        chunks.append(chunk)
        offset += count * 4
    require([digest(c) for c in chunks] == receipt["vector_sha256"], "per-field identity")
    for chunk in chunks:
        values = array.array("f")
        values.frombytes(chunk)
        if sys.byteorder != "little":
            values.byteswap()
        require(all(math.isfinite(v) for v in values), "finite vectors")
    return shapes, chunks


def compare(receipt, data):
    shapes, chunks = validate(receipt, data)
    import torch
    from benchmark_topos_module_reference import reference

    torch.set_num_threads(1)
    vectors = []
    for shape, chunk in zip(shapes, chunks):
        values = array.array("f")
        values.frombytes(chunk)
        if sys.byteorder != "little":
            values.byteswap()
        vectors.append(torch.tensor(values, dtype=torch.float32, device="cpu").reshape(shape))
    value, gate, upstream, *expected = vectors
    accumulator = torch.zeros_like(gate)
    rounds = len(receipt["round_order"])
    timings = [[], []]
    errors = dict.fromkeys(("output", "grad_input", "grad_gate"), 0.0)
    ratios = dict.fromkeys(errors, 0.0)
    for round_index in range(rounds + 2):
        for route in ([0, 1] if round_index % 2 == 0 else [1, 0]):
            x, g = (v.detach().requires_grad_() for v in (value, gate))
            accumulator.zero_()
            start = time.perf_counter_ns()
            output = reference(x, g, receipt["config"])
            if route:
                dx, dg = torch.autograd.grad(output, (x, g), upstream)
                accumulator.add_(dg)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if round_index >= 2:
                timings[route].append(elapsed)
            actual = [output, dx, accumulator] if route else [output]
            for name, observed, wanted in zip(errors, actual, expected):
                require(bool(torch.isfinite(observed).all()), f"nonfinite Torch {name}")
                torch.testing.assert_close(observed, wanted, rtol=5e-4, atol=3e-5)
                delta = (observed.detach().to(torch.float64) - wanted.to(torch.float64)).abs()
                ratio = delta / (3e-5 + 5e-4 * wanted.to(torch.float64).abs())
                errors[name] = max(errors[name], float(delta.max()))
                ratios[name] = max(ratios[name], float(ratio.max()))
                require(ratios[name] <= 1.0, f"Torch {name} tolerance ratio")
            # Do not charge destruction of the previous forward graph to the
            # next sample. Rust drops its outputs and invalidates its tape
            # outside the measured interval as well.
            del actual, observed, output, x, g
            if route:
                del dx, dg
    return {
        "schema": "spiraltorch.topos_shared_nn_torch_reference.v1", "status": "passed",
        "torch": str(torch.__version__), "threads": 1, "device": "cpu", "dtype": "float32",
        "shape": receipt["shape"], "config": receipt["config"],
        "gate_layout": "shared_rows", "gate_gradient_reduction": "sum_without_additional_mean",
        "warmup_per_route": 2, "round_order": receipt["round_order"],
        "gradient_storage": receipt["gradient_storage"],
        "measurements_ms": {"forward": timings[0], "forward_backward": timings[1]},
        "rtol": 5e-4, "atol": 3e-5, "max_abs_error": errors,
        "max_tolerance_ratio": ratios,
        "vectors_sha256": digest(data),
        "scope": "Independent CPU Torch finite Picard recurrence, shared-gate broadcast and both VJPs, gate gradient accumulation. Only Rust includes semantic audits. No optimizer update, file transport or gradient reset in timings. Not accelerator or quality evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    require(args.receipt.stat().st_size <= 256_000, "receipt size budget")
    vectors_path = args.receipt.with_suffix(".f32le")
    require(vectors_path.stat().st_size <= 6 * 1_048_576 * 4, "vector size budget")
    raw = args.receipt.read_bytes()
    report = compare(json.loads(raw), vectors_path.read_bytes())
    report.update(receipt_sha256=digest(raw), client_sha256=digest(Path(__file__).read_bytes()),
                  reference_sha256=digest(Path(__file__).with_name("benchmark_topos_module_reference.py").read_bytes()))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "max_abs_error": report["max_abs_error"]}), flush=True)


if __name__ == "__main__":
    main()
