#!/usr/bin/env python3
"""Matched finite Topos unroll: CPU forward plus input and broadcast-gate VJPs.

Torch math is an independent benchmark reference, never a production backend.
Compare old/new native hashes before interpreting sequential binary timings.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.geometry_autograd as bridge
import spiraltorch.spiraltorch as native

NAMES = ("output", "input_gradient", "gate_gradient")


class Recomputed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gate, kernel, buffers):
        ctx.kernel, ctx.buffers = kernel, buffers
        ctx.rows, ctx.features = value.numel() // value.shape[-1], value.shape[-1]
        ctx.save_for_backward(value, gate)
        transport = bridge._buffer_values if buffers else bridge._values
        forward = kernel.forward_buffer if buffers else kernel.forward
        result = forward(transport(value), transport(gate), ctx.rows, ctx.features)
        return bridge._transport_output(result, value, buffers)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        value, gate = ctx.saved_tensors
        transport = bridge._buffer_values if ctx.buffers else bridge._values
        backward = ctx.kernel.backward_buffer if ctx.buffers else ctx.kernel.backward
        dx, dg = backward(transport(value), transport(gate), transport(upstream), ctx.rows, ctx.features)
        return (bridge._transport_output(dx, value, ctx.buffers),
                bridge._transport_output(dg, gate, ctx.buffers), None, None)


def torch_reference(value, gate, config):
    drive = value * gate
    state = torch.zeros_like(drive)
    limit = config["saturation"]
    absorb = min(config["porosity"] * .25, 1.)
    for _ in range(config["iterations"]):
        raw = drive + config["coupling"] * state
        # Safe in the unselected branch too, including zero drive.
        tail = raw.sign() * limit * (1. - absorb * (raw.abs() - limit) / (raw.abs() + limit))
        state = torch.where(raw.abs() <= limit, raw, tail)
    return state


def routes(kernel):
    result = ["rust_list", "rust_public", "torch_reference"]
    if hasattr(kernel, "backward_buffer"):
        result.insert(1, "rust_buffer_recomputed")
    return result


def run_route(name, values, upstream, kernel, config):
    value, gate = (v.detach().requires_grad_() for v in values)
    if name in ("rust_list", "rust_buffer_recomputed"):
        output = Recomputed.apply(value, gate.expand_as(value), kernel, name != "rust_list")
    elif name == "rust_public":
        output = st.topos_resonator_autograd(value, gate, kernel=kernel)
    elif name == "torch_reference":
        output = torch_reference(value, gate, config)
    else:
        raise ValueError("unknown route")
    return (output.detach(), *(g.detach() for g in torch.autograd.grad(output, (value, gate), upstream)))


def tensor_bytes(value):
    return value.contiguous().numpy().tobytes()


def check_result(actual, reference, *, exact):
    if len(actual) != 3 or len(reference) != 3:
        raise ValueError("gradient arity mismatch")
    result = {}
    for name, value, expected in zip(NAMES, actual, reference):
        if value.shape != expected.shape:
            raise ValueError("gradient shape mismatch")
        if not bool(torch.isfinite(value).all() and torch.isfinite(expected).all()):
            raise ValueError("nonfinite comparison")
        if exact:
            if tensor_bytes(value) != tensor_bytes(expected):
                raise ValueError("Rust path changed output/gradient bits")
        else:
            torch.testing.assert_close(value, expected, rtol=5e-4, atol=3e-5)
        result[name] = {"sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
                        "max_abs_error": float((value - expected).abs().max()) if value.numel() else 0.}
    return result


def round_orders(names, rounds):
    if rounds % len(names):
        raise ValueError("rounds must balance route positions")
    return [names[i % len(names):] + names[:i % len(names)] for i in range(rounds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[2, 128, 768])
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--coupling", type=float, default=.25)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--native-profile", choices=["dev", "release", "unknown"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if (min(args.shape) <= 0 or math.prod(args.shape) > 1_048_576 or args.threads < 1
            or args.warmup < 1 or args.rounds < 1 or not 1 <= args.iterations <= 64
            or not 0 <= args.coupling < 1):
        parser.error("invalid shape, configuration or benchmark budget")
    torch.set_num_threads(args.threads)
    rng = torch.Generator().manual_seed(args.seed)
    values = [torch.randn(args.shape, generator=rng), torch.randn(args.shape[-1], generator=rng)]
    upstream = torch.randn(args.shape, generator=rng)
    kernel = st.ToposResonatorKernel(coupling=args.coupling, iterations=args.iterations, porosity=.2)
    config = json.loads(kernel.configuration_json())
    names = routes(kernel)
    orders = round_orders(names, args.rounds)
    reference = run_route("rust_list", values, upstream, kernel, config)
    correctness = {n: check_result(run_route(n, values, upstream, kernel, config), reference,
                                    exact=n != "torch_reference") for n in names}
    for _ in range(args.warmup):
        for name in names:
            run_route(name, values, upstream, kernel, config)
    timings = {n: [] for n in names}
    for order in orders:
        for name in order:
            start = time.perf_counter_ns()
            result = run_route(name, values, upstream, kernel, config)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            check_result(result, reference, exact=name != "torch_reference")
            timings[name].append(elapsed)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = {"schema": "spiraltorch.topos_learning_benchmark.v1", "status": "measured",
              "shape": args.shape, "config": config, "seed": args.seed, "dtype": "float32", "device": "cpu",
              "threads": args.threads, "all_inputs_require_grad": True, "capture_available": hasattr(kernel, "capture"),
              "native_profile_declared": args.native_profile, "native_sha256": digest(native.__file__),
              "bridge_sha256": digest(bridge.__file__), "benchmark_sha256": digest(__file__),
              "transport_sha256": digest(Path(bridge.__file__).with_name("_torch_transport.py")),
              "input_sha256": [hashlib.sha256(tensor_bytes(v)).hexdigest() for v in values],
              "upstream_sha256": hashlib.sha256(tensor_bytes(upstream)).hexdigest(),
              "torch": str(torch.__version__), "machine": platform.machine(), "correctness": correctness,
              "warmup_per_route": args.warmup, "round_order": orders, "measurements_ms": timings,
              "median_ms": {n: statistics.median(v) for n, v in timings.items()},
              "scope": "CPU forward + both VJPs including host transport and Torch broadcast reduction. Same finite Picard map; independent Torch arithmetic within rtol=5e-4, atol=3e-5. Not model quality, accelerator or general-library throughput."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "median_ms": report["median_ms"]}), flush=True)


if __name__ == "__main__":
    main()
