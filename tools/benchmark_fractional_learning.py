#!/usr/bin/env python3
"""Matched CPU GL forward + order VJP, including the Python/native transport.

The Torch polynomial/convolution is a benchmark reference, not a production
fractional backend. Use an idle machine and record an optimized native build
separately before interpreting timings. No model-quality claim is made here.
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
import spiraltorch.spiraltorch as native
import spiraltorch.fractional_autograd as bridge

ROUTES = ("rust_joint", "rust_selective", "torch_conv1d")


class JointBackward(bridge._FractionalGlFunction):
    @staticmethod
    def backward(ctx, upstream):
        value, alpha = ctx.saved_tensors
        dx, da = ctx.snapshot.vjp(bridge._values(upstream))
        return (None, torch.tensor(dx, dtype=value.dtype, device=value.device).reshape_as(value),
                torch.tensor(da, dtype=alpha.dtype, device=alpha.device), None, None)


def torch_reference(value, alpha, kernel_len, step, history):
    coefficients = [torch.ones_like(alpha)]
    for k in range(1, kernel_len):
        coefficients.append(coefficients[-1] * (k - 1 - alpha) / k)
    if history:
        coefficients[0] = alpha * 0
    weights = torch.stack(coefficients).flip(0).reshape(1, 1, kernel_len) * step ** (-alpha)
    b, t, f = value.shape
    lanes = value.transpose(1, 2).reshape(b * f, 1, t)
    result = torch.nn.functional.conv1d(torch.nn.functional.pad(lanes, (kernel_len - 1, 0)), weights)
    return result.reshape(b, f, t).transpose(1, 2)


def run_route(name, value, upstream, *, alpha, kernel, kernel_len, step, history):
    order = torch.tensor(alpha, dtype=torch.float32, requires_grad=True)
    if name == "rust_joint":
        output = JointBackward.apply(kernel, value, order, 1, history)
    elif name == "rust_selective":
        operation = st.fractional_gl_history_autograd if history else st.fractional_gl_autograd
        output = operation(value, order, axis=1, kernel=kernel)
    elif name == "torch_conv1d":
        output = torch_reference(value, order, kernel_len, step, history)
    else:
        raise ValueError("unknown route")
    gradient = torch.autograd.grad(output, order, upstream)[0]
    return output.detach(), gradient.detach()


def check_result(actual, reference, *, exact):
    if not all(bool(torch.isfinite(v).all()) for v in actual + reference):
        raise ValueError("nonfinite comparison")
    if exact:
        if not all(torch.equal(a, b) for a, b in zip(actual, reference)):
            raise ValueError("selective/joint results differ")
    else:
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    return {"max_abs_output_error": float((actual[0] - reference[0]).abs().max()),
            "abs_alpha_gradient_error": float((actual[1] - reference[1]).abs()),
            "alpha_gradient": float(actual[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[2, 128, 768], metavar=("B", "T", "F"))
    parser.add_argument("--kernel-len", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=.9)
    parser.add_argument("--step", type=float, default=1.)
    parser.add_argument("--map", choices=["full", "history"], default="history")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--native-profile", choices=["dev", "release", "unknown"], required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if (min(args.shape) <= 0 or args.kernel_len <= 0 or args.threads <= 0
            or args.warmup < 1 or args.rounds < 1
            or not all(math.isfinite(v) and v > 0 for v in (args.alpha, args.step))
            or math.prod(args.shape) > 1_048_576
            or math.prod(args.shape) * args.kernel_len > 16_777_216):
        parser.error("invalid dimensions, finite domain or comparison budget")
    torch.set_num_threads(args.threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    value = torch.randn(args.shape, dtype=torch.float32, generator=generator)
    upstream = torch.randn(args.shape, dtype=torch.float32, generator=generator)
    options = {"alpha": args.alpha, "kernel_len": args.kernel_len, "step": args.step,
               "history": args.map == "history",
               "kernel": st.FractionalGlKernel(kernel_len=args.kernel_len, step=args.step)}
    reference = run_route("rust_joint", value, upstream, **options)
    correctness = {name: check_result(run_route(name, value, upstream, **options), reference,
                                      exact=name != "torch_conv1d") for name in ROUTES}
    measurements, order = {name: [] for name in ROUTES}, []
    if not args.validate_only:
        for _ in range(args.warmup):
            for name in ROUTES:
                run_route(name, value, upstream, **options)
        # Rotate all permutations to reduce fixed-order warmup/cache drift.
        for names in itertools.islice(itertools.cycle(itertools.permutations(ROUTES)), args.rounds):
            order.append(list(names))
            for name in names:
                start = time.perf_counter_ns()
                result = run_route(name, value, upstream, **options)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                check_result(result, reference, exact=name != "torch_conv1d")
                measurements[name].append(elapsed_ms)
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    report = {"schema": "spiraltorch.fractional_learning_benchmark.v1",
              "status": "validated_only" if args.validate_only else "measured",
              "shape": args.shape, "kernel_len": args.kernel_len, "alpha": args.alpha,
              "step": args.step, "map": args.map, "seed": args.seed,
              "input_requires_grad": False, "alpha_requires_grad": True,
              "dtype": "float32", "device": "cpu", "threads": args.threads,
              "native_profile_declared": args.native_profile,
              "native_sha256": digest(Path(native.__file__).read_bytes()),
              "bridge_sha256": digest(Path(bridge.__file__).read_bytes()),
              "benchmark_sha256": digest(Path(__file__).read_bytes()),
              "input_sha256": digest(value.numpy().tobytes()),
              "upstream_sha256": digest(upstream.numpy().tobytes()),
              "torch": str(torch.__version__), "machine": platform.machine(),
              "correctness": correctness, "round_order": order,
              "warmup_per_route": 0 if args.validate_only else args.warmup,
              "measurements_ms": measurements,
              "median_ms": {name: statistics.median(v) for name, v in measurements.items() if v},
              "scope": "Same finite GL map and requested order VJP; forward, AD and native host transport included. CPU float32 interfaces; backend accumulation differs within declared tolerance. Not a quality, GPU, adapter/model throughput or general-library speed claim. Native profile is declared, not inferred from the binary."}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "correctness": correctness,
                      "median_ms": report["median_ms"]}), flush=True)


if __name__ == "__main__":
    main()
