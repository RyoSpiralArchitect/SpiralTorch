#!/usr/bin/env python3
"""Matched CPU WaveGate forward + all VJPs, including native host transport.

The differentiable Torch formula is a benchmark reference, not a production
geometry backend. Old/new native reports must have identical output/gradient
hashes before comparing times. No model-quality or accelerator claim is made.
"""

import argparse
import hashlib
import itertools
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

ROUTES = ("rust_list", "rust_public", "torch_reference")
NAMES = ("output", "input_gradient", "gate_gradient", "bias_gradient", "radius_gradient")


class ListTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gate, bias, log_radius, kernel):
        ctx.has_radius = log_radius is not None
        args = (bridge._values(value), bridge._values(gate), bridge._values(bias),
                value.numel() // value.shape[-1], value.shape[-1])
        ctx.snapshot = (kernel.forward_with_log_radius(*args, log_radius.detach().item())
                        if ctx.has_radius else kernel.forward(*args))
        ctx.save_for_backward(value, gate, bias, log_radius)
        return torch.tensor(ctx.snapshot.output, dtype=value.dtype).reshape_as(value)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        value, gate, bias, radius = ctx.saved_tensors
        gradients = (ctx.snapshot.vjp_with_log_radius(bridge._values(upstream))
                     if ctx.has_radius else ctx.snapshot.vjp(bridge._values(upstream)))
        parameters = (value, gate, bias, radius) if ctx.has_radius else (value, gate, bias)
        result = tuple(torch.tensor(g, dtype=p.dtype).reshape_as(p)
                       for g, p in zip(gradients, parameters))
        return result + ((None,) if ctx.has_radius else (None, None))


def torch_reference(value, gate, bias, log_radius):
    def saturate(v):
        tail = v.sign() * (1.0 - (0.2 * 0.25) * (v.abs() - 1.0) / (v.abs() + 1.0))
        return torch.where(v.abs() <= 1.0, v, tail)

    z = saturate(value * saturate(gate) + bias).double()
    scale = torch.tensor(0.7, dtype=torch.float32).sqrt().double()
    norm = z.norm(dim=-1, keepdim=True)
    radius = log_radius.double().exp() if log_radius is not None else 1.0
    a = norm / (scale * radius)
    safe = torch.where(a == 0, torch.ones_like(a), a)
    gain = torch.where(a == 0, torch.ones_like(a), safe.tanh() / safe)
    return (z * gain / scale).float()


def run_route(name, values, upstream, kernel):
    parameters = tuple(v.detach().requires_grad_() for v in values)
    args = (*parameters[:3], parameters[3] if len(parameters) == 4 else None)
    if name == "rust_list":
        output = ListTransport.apply(*args, kernel)
    elif name == "rust_public":
        output = st.wave_gate_autograd(*args[:3], log_radius=args[3], kernel=kernel)
    elif name == "torch_reference":
        output = torch_reference(*args)
    else:
        raise ValueError("unknown route")
    return (output.detach(), *(g.detach() for g in torch.autograd.grad(output, parameters, upstream)))


def tensor_bytes(value):
    return value.contiguous().numpy().tobytes()


def check_result(actual, reference, *, exact):
    if len(actual) != len(reference):
        raise ValueError("gradient arity mismatch")
    result = {}
    for name, value, expected in zip(NAMES, actual, reference):
        if not bool(torch.isfinite(value).all() and torch.isfinite(expected).all()):
            raise ValueError("nonfinite comparison")
        if value.shape != expected.shape:
            raise ValueError("gradient shape mismatch")
        if exact:
            if tensor_bytes(value) != tensor_bytes(expected):
                raise ValueError("Rust transport changed output/gradient bits")
        else:
            torch.testing.assert_close(value, expected, rtol=5e-4, atol=3e-5)
        result[name] = {"sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
                        "max_abs_error": float((value - expected).abs().max())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[2, 128, 768])
    parser.add_argument("--log-radius", type=float)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--native-profile", choices=["dev", "release", "unknown"], required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if (min(args.shape) <= 0 or math.prod(args.shape) > 1_048_576 or args.threads <= 0
            or args.warmup < 1 or args.rounds < 1
            or args.log_radius is not None and (not math.isfinite(args.log_radius) or abs(args.log_radius) > 80)):
        parser.error("invalid shape, radius or benchmark budget")
    torch.set_num_threads(args.threads)
    rng = torch.Generator().manual_seed(args.seed)
    values = [torch.randn(args.shape, generator=rng), torch.randn(args.shape[-1], generator=rng),
              torch.randn(args.shape[-1], generator=rng) * .2]
    if args.log_radius is not None:
        values.append(torch.tensor(args.log_radius, dtype=torch.float32))
    upstream = torch.randn(args.shape, generator=rng)
    kernel = st.WaveGateKernel(curvature=-.7, porosity=.2)
    reference = run_route(ROUTES[0], values, upstream, kernel)
    correctness = {name: check_result(run_route(name, values, upstream, kernel), reference,
                                      exact=name != "torch_reference") for name in ROUTES}
    timings, order = {name: [] for name in ROUTES}, []
    if not args.validate_only:
        for _ in range(args.warmup):
            for name in ROUTES:
                run_route(name, values, upstream, kernel)
        for names in itertools.islice(itertools.cycle(itertools.permutations(ROUTES)), args.rounds):
            order.append(list(names))
            for name in names:
                start = time.perf_counter_ns()
                result = run_route(name, values, upstream, kernel)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                check_result(result, reference, exact=name != "torch_reference")
                timings[name].append(elapsed)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    transport = Path(bridge.__file__).with_name("_torch_transport.py")
    report = {"schema": "spiraltorch.wave_gate_learning_benchmark.v1", "shape": args.shape,
              "log_radius": args.log_radius, "seed": args.seed, "dtype": "float32", "device": "cpu",
              "threads": args.threads, "all_inputs_require_grad": True,
              "status": "validated_only" if args.validate_only else "measured",
              "native_profile_declared": args.native_profile, "native_sha256": digest(native.__file__),
              "bridge_sha256": digest(bridge.__file__), "benchmark_sha256": digest(__file__),
              "transport_sha256": digest(transport) if transport.exists() else None,
              "buffer_method_available": hasattr(kernel, "forward_buffer"),
              "input_sha256": [hashlib.sha256(tensor_bytes(v)).hexdigest() for v in values],
              "upstream_sha256": hashlib.sha256(tensor_bytes(upstream)).hexdigest(),
              "torch": str(torch.__version__), "machine": platform.machine(), "correctness": correctness,
              "warmup_per_route": 0 if args.validate_only else args.warmup, "round_order": order,
              "measurements_ms": timings, "median_ms": {n: statistics.median(v) for n, v in timings.items() if v},
              "scope": "CPU forward + joint input/gate/bias/radius VJP including host transport. Same map and gradient requests; rounding/accumulation differ for Torch within rtol=5e-4, atol=3e-5. Not model throughput, quality, GPU or a general-library speed claim. Build profile is declared."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "median_ms": report["median_ms"]}), flush=True)


if __name__ == "__main__":
    main()
