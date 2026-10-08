#!/usr/bin/env python3
"""Matched CPU GL forward + order VJP, including the Python/native transport.

The Torch polynomial/convolution is a benchmark reference, not a production
fractional backend. Use an idle machine and record an optimized native build
separately before interpreting timings. No model-quality claim is made here.
Use --compare-transport for explicit list/buffer routes sharing one Rust kernel.
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
TRANSPORT_ROUTES = ("rust_list", "rust_buffer", "torch_conv1d")


class JointBackward(bridge._FractionalGlFunction):
    @staticmethod
    def backward(ctx, upstream):
        value, alpha = ctx.saved_tensors
        if getattr(ctx, "buffer_transport", False):
            dx, da = ctx.snapshot.vjp_buffer(bridge._buffer_values(upstream))
            dx = bridge._transport_output(dx, value, True)
        else:
            dx, da = ctx.snapshot.vjp(bridge._values(upstream))
            dx = torch.tensor(dx, dtype=value.dtype, device=value.device).reshape_as(value)
        return (None, dx,
                torch.tensor(da, dtype=alpha.dtype, device=alpha.device), None, None)


class ListTransport(bridge._FractionalGlFunction):
    """Keep the original list forward explicit when comparing transport only."""

    @staticmethod
    def forward(ctx, kernel, value, alpha, axis, history):
        ctx.buffer_transport = False
        operation = kernel.forward_history if history else kernel.forward
        ctx.snapshot = operation(bridge._values(value), list(value.shape), axis, alpha.detach().item())
        ctx.save_for_backward(value, alpha)
        ctx.save_for_forward(value, alpha)
        return torch.tensor(ctx.snapshot.output, dtype=value.dtype, device=value.device).reshape_as(value)


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
    elif name == "rust_list":
        output = ListTransport.apply(kernel, value, order, 1, history)
    elif name in ("rust_selective", "rust_buffer"):
        operation = st.fractional_gl_history_autograd if history else st.fractional_gl_autograd
        output = operation(value, order, axis=1, kernel=kernel)
        if name == "rust_buffer" and not getattr(output.grad_fn, "buffer_transport", False):
            raise ValueError("buffer comparison requires working Torch/NumPy interop")
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
            raise ValueError("Rust route results differ")
    else:
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    return {"max_abs_output_error": float((actual[0] - reference[0]).abs().max()),
            "abs_alpha_gradient_error": float((actual[1] - reference[1]).abs()),
            "alpha_gradient": float(actual[1]),
            "output_sha256": hashlib.sha256(actual[0].detach().numpy().tobytes()).hexdigest(),
            "alpha_gradient_sha256": hashlib.sha256(actual[1].detach().numpy().tobytes()).hexdigest()}


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
    parser.add_argument("--compare-transport", action="store_true",
                        help="compare explicit list vs buffer forward/order VJP with the same Rust kernel")
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
    routes = TRANSPORT_ROUTES if args.compare_transport else ROUTES
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    value = torch.randn(args.shape, dtype=torch.float32, generator=generator)
    upstream = torch.randn(args.shape, dtype=torch.float32, generator=generator)
    options = {"alpha": args.alpha, "kernel_len": args.kernel_len, "step": args.step,
               "history": args.map == "history",
               "kernel": st.FractionalGlKernel(kernel_len=args.kernel_len, step=args.step)}
    reference = run_route(routes[0], value, upstream, **options)
    correctness = {name: check_result(run_route(name, value, upstream, **options), reference,
                                      exact=name != "torch_conv1d") for name in routes}
    measurements, order = {name: [] for name in routes}, []
    if not args.validate_only:
        for _ in range(args.warmup):
            for name in routes:
                run_route(name, value, upstream, **options)
        # Rotate all permutations to reduce fixed-order warmup/cache drift.
        for names in itertools.islice(itertools.cycle(itertools.permutations(routes)), args.rounds):
            order.append(list(names))
            for name in names:
                start = time.perf_counter_ns()
                result = run_route(name, value, upstream, **options)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                check_result(result, reference, exact=name != "torch_conv1d")
                measurements[name].append(elapsed_ms)
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    report = {"schema": "spiraltorch.fractional_learning_benchmark.v1",
              "comparison": "list_vs_buffer" if args.compare_transport else "joint_vs_selective",
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
