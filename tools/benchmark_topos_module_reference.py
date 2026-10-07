#!/usr/bin/env python3
"""Independent CPU Torch check/timing of an actual Rust NN Topos probe.

Both routes request a full per-element gate gradient, not a broadcast reduction.
Rust includes its semantic audit. Timings exclude file transport and validation.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(value, gate, config):
    drive = value * gate
    state = torch.zeros_like(drive)
    limit = config["saturation"]
    absorb = min(config["porosity"] * 0.25, 1.0)
    for _ in range(config["iterations"]):
        raw = drive + config["coupling"] * state
        tail = raw.sign() * limit * (1.0 - absorb * (raw.abs() - limit) / (raw.abs() + limit))
        state = torch.where(raw.abs() <= limit, raw, tail)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    raw_receipt = args.receipt.read_bytes()
    receipt = json.loads(raw_receipt)
    require(receipt["schema"] == "spiraltorch.topos_nn_probe.v1", "receipt schema")
    require(receipt["status"] == "measured" and receipt["backend"] == "cpu", "receipt status/backend")
    require(receipt["dtype"] == "float32", "receipt dtype")
    shape = receipt["shape"]
    require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape), "receipt shape")
    volume = math.prod(shape)
    require(volume <= 1_048_576, "volume budget")
    config = receipt["config"]
    require(type(config["iterations"]) is int and 1 <= config["iterations"] <= 64, "iteration budget")
    for name in ("coupling", "saturation", "porosity"):
        require(type(config[name]) in (int, float) and math.isfinite(config[name]), "nonfinite config")
    require(0 <= config["coupling"] < 1 and config["saturation"] > 0 and 0 <= config["porosity"] <= 1, "config domain")
    require(receipt["vector_order"] == ["input", "gate", "upstream", "output", "grad_input", "grad_gate"], "vector order")
    data = args.receipt.with_suffix(".f32le").read_bytes()
    digest = lambda value: hashlib.sha256(value).hexdigest()
    require(len(data) == volume * 4 * 6 and digest(data) == receipt["vectors_sha256"], "vector identity")
    chunks = [data[i * volume * 4:(i + 1) * volume * 4] for i in range(6)]
    require([digest(chunk) for chunk in chunks] == receipt["vector_sha256"], "per-field identity")
    # The file contract is little-endian; byteswap explicitly on other hosts.
    import array
    import sys
    vectors = []
    for chunk in chunks:
        values = array.array("f")
        values.frombytes(chunk)
        if sys.byteorder != "little":
            values.byteswap()
        tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
        require(bool(torch.isfinite(tensor).all()), "nonfinite vector")
        vectors.append(tensor)
    value, gate, upstream, *expected = vectors
    rounds = len(receipt["round_order"])
    require(2 <= rounds <= 256 and rounds % 2 == 0, "round budget")
    require(receipt["round_order"] == [[0, 1] if i % 2 == 0 else [1, 0] for i in range(rounds)], "route order")
    torch.set_num_threads(2)
    timings = [[], []]
    errors = [0.0, 0.0, 0.0]
    for round_index in range(rounds + 2):
        for route in ([0, 1] if round_index % 2 == 0 else [1, 0]):
            x, g = (v.detach().requires_grad_() for v in (value, gate))
            start = time.perf_counter_ns()
            output = reference(x, g, config)
            actual = [output]
            if route:
                actual.extend(torch.autograd.grad(output, (x, g), upstream))
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if round_index >= 2:
                timings[route].append(elapsed)
            for i, (observed, wanted) in enumerate(zip(actual, expected)):
                require(bool(torch.isfinite(observed).all()), "nonfinite comparison")
                torch.testing.assert_close(observed, wanted, rtol=5e-4, atol=3e-5)
                errors[i] = max(errors[i], float((observed.detach() - wanted).abs().max()))
    report = {
        "schema": "spiraltorch.topos_nn_torch_reference.v1", "status": "passed",
        "receipt_sha256": digest(raw_receipt), "vectors_sha256": digest(data),
        "client_sha256": digest(Path(__file__).read_bytes()),
        "torch": str(torch.__version__), "threads": 2, "shape": shape, "config": config,
        "warmup_per_route": 2, "round_order": receipt["round_order"],
        "measurements_ms": {"forward": timings[0], "forward_backward": timings[1]},
        "max_abs_error": dict(zip(("output", "grad_input", "grad_gate"), errors)),
        "scope": "Independent Torch finite Picard map with both full per-element VJPs. CPU float32, rtol=5e-4, atol=3e-5. No Rust audit work in Torch, no transport or process startup in timings. Not model quality or accelerator performance.",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "max_abs_error": report["max_abs_error"]}), flush=True)


if __name__ == "__main__":
    main()
