#!/usr/bin/env python3
"""Same-math CPU full-normalization history window, including host transport.

Torch is an independent benchmark oracle, not a production geometry backend.
Both routes compute the entire K normalization before selecting the same taps.
The Torch convolution uses only the selected window, not a zero-padded K filter.
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
import spiraltorch.spiraltorch as native
import spiraltorch.fractional_autograd as bridge

ROUTES = ("rust_buffer", "torch_window_conv1d")
RTOL = ATOL = 3e-5


def torch_reference(value, alpha, log_gain, kernel_len, window):
    # Cancel -alpha analytically; normalize in f64, then expose f32 taps.
    order = alpha.double()
    polynomial = [order * 0, order * 0 + 1]
    for lag in range(2, kernel_len):
        polynomial.append(polynomial[-1] * (lag - 1 - order) / lag)
    polynomial = torch.stack(polynomial)
    coefficients = (-log_gain.exp().double() * polynomial / polynomial.norm()).float()
    start, end = window
    weights = coefficients[start:end].flip(0).reshape(1, 1, end - start)
    batch, length, features = value.shape
    lanes = value.transpose(1, 2).reshape(batch * features, 1, length)
    output = torch.nn.functional.conv1d(torch.nn.functional.pad(lanes, (end - 1, 0)), weights)
    return output[:, :, :length].reshape(batch, features, length).transpose(1, 2)


def run_route(name, value, upstream, *, alpha, log_gain, kernel, kernel_len, window):
    order = torch.tensor(alpha, dtype=torch.float32, requires_grad=True)
    amplitude = torch.tensor(log_gain, dtype=torch.float32, requires_grad=True)
    if name == "rust_buffer":
        output = st.fractional_gl_history_log_gain_autograd(
            value, order, amplitude, axis=1, kernel=kernel, lag_window=window)
        if not getattr(output.grad_fn, "buffer_transport", False):
            raise ValueError("benchmark requires working Torch/NumPy buffer transport")
    elif name == "torch_window_conv1d":
        output = torch_reference(value, order, amplitude, kernel_len, window)
    else:
        raise ValueError("unknown route")
    inputs = (order, amplitude, value) if value.requires_grad else (order, amplitude)
    gradients = torch.autograd.grad(output, inputs, upstream)
    return (output.detach(), *(gradient.detach() for gradient in gradients))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def check_result(actual, reference, *, exact=False):
    if len(actual) != len(reference) or len(actual) not in (3, 4):
        raise ValueError("gradient request mismatch")
    errors, hashes = {}, {}
    for name, observed, expected in zip(("output", "alpha", "log_gain", "input"), actual, reference):
        if not bool(torch.isfinite(observed).all() and torch.isfinite(expected).all()):
            raise ValueError("nonfinite comparison")
        if observed.shape != expected.shape:
            raise ValueError("shape mismatch")
        observed_hash = digest(observed.numpy().tobytes())
        if exact:
            if observed_hash != digest(expected.numpy().tobytes()):
                raise ValueError("repeat differs bitwise")
        else:
            torch.testing.assert_close(observed, expected, rtol=RTOL, atol=ATOL)
        errors[name] = float((observed - expected).abs().max())
        hashes[name] = observed_hash
    return {"max_abs_error": errors, "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[2, 128, 768])
    parser.add_argument("--kernel-len", type=int, default=32)
    parser.add_argument("--window", nargs=2, type=int, required=True)
    parser.add_argument("--alpha", type=float, default=.55)
    parser.add_argument("--log-gain", type=float, default=math.log(5.))
    parser.add_argument("--input-gradient", action="store_true")
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
    if (min(args.shape) <= 0 or args.kernel_len < 2 or args.threads < 1
            or args.warmup < 1 or args.rounds < 2
            or not 1 <= args.window[0] < args.window[1] <= args.kernel_len
            or not math.isfinite(args.alpha) or args.alpha <= 0
            or not math.isfinite(args.log_gain)
            or math.prod(args.shape) > 1_048_576
            or math.prod(args.shape) * args.kernel_len > 16_777_216):
        parser.error("invalid shape, support, finite domain or comparison budget")
    torch.set_num_threads(args.threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    value = torch.randn(args.shape, generator=generator).requires_grad_(args.input_gradient)
    upstream = torch.randn(args.shape, generator=generator)
    options = {"alpha": args.alpha, "log_gain": args.log_gain, "kernel_len": args.kernel_len,
               "window": tuple(args.window), "kernel": st.FractionalGlKernel(kernel_len=args.kernel_len)}
    references = {name: run_route(name, value, upstream, **options) for name in ROUTES}
    correctness = {name: check_result(result, references[ROUTES[0]]) for name, result in references.items()}
    measurements, ordering = {name: [] for name in ROUTES}, []
    if not args.validate_only:
        for _ in range(args.warmup):
            for name in ROUTES:
                check_result(run_route(name, value, upstream, **options), references[name], exact=True)
        for index in range(args.rounds):
            names = ROUTES if index % 2 == 0 else ROUTES[::-1]
            ordering.append(list(names))
            for name in names:
                start = time.perf_counter_ns()
                result = run_route(name, value, upstream, **options)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                check_result(result, references[name], exact=True)
                measurements[name].append(elapsed)
    report = {
        "schema": "spiraltorch.fractional_window_benchmark.v1",
        "status": "validated_only" if args.validate_only else "measured",
        "shape": args.shape, "kernel_len": args.kernel_len, "window": args.window,
        "normalization": "full_declared_kernel_before_window",
        "alpha": args.alpha, "log_gain": args.log_gain, "seed": args.seed,
        "input_requires_grad": args.input_gradient, "alpha_requires_grad": True, "log_gain_requires_grad": True,
        "dtype": "float32", "device": "cpu", "threads": args.threads,
        "native_profile_declared": args.native_profile,
        "native_sha256": digest(Path(native.__file__).read_bytes()),
        "bridge_sha256": digest(Path(bridge.__file__).read_bytes()),
        "benchmark_sha256": digest(Path(__file__).read_bytes()),
        "input_sha256": digest(value.detach().numpy().tobytes()), "upstream_sha256": digest(upstream.numpy().tobytes()),
        "torch": str(torch.__version__), "python": platform.python_version(), "machine": platform.machine(),
        "correctness_tolerance": {"rtol": RTOL, "atol": ATOL}, "correctness": correctness,
        "round_order": ordering, "warmup_per_route": 0 if args.validate_only else args.warmup,
        "measurements_ms": measurements,
        "median_ms": {name: statistics.median(values) for name, values in measurements.items() if values},
        "scope": "Same full-K normalized window and requested scalar/input VJPs. Includes forward, AD and host transport; Torch uses a support-length convolution. Backend accumulation differs within the fixed tolerance. Local CPU operator evidence, not end-to-end model, GPU/browser speed, quality or universal library superiority. Native profile is declared, not inferred."
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "median_ms": report["median_ms"]}), flush=True)


if __name__ == "__main__":
    main()
