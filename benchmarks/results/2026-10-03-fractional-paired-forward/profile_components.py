"""Diagnostic stages of CPU GL transport; not an isolated kernel benchmark."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.spiraltorch as native
from spiraltorch.geometry_autograd import _values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", choices=["full", "history"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(239)
    value = torch.randn(2, 128, 768, generator=generator)
    upstream = torch.randn(value.shape, generator=generator)
    kernel = st.FractionalGlKernel(kernel_len=32)
    operation = kernel.forward_history if args.map == "history" else kernel.forward
    public = st.fractional_gl_history_autograd if args.map == "history" else st.fractional_gl_autograd
    alpha = torch.tensor(.9, requires_grad=True)
    reference_output = public(value, alpha, axis=1, kernel=kernel)
    reference_gradient = torch.autograd.grad(reference_output, alpha, upstream)[0]

    def run():
        measurements = {}

        def stage(name, function):
            start = time.perf_counter_ns()
            result = function()
            measurements[name] = (time.perf_counter_ns() - start) / 1e6
            return result

        source = stage("input_to_list", lambda: _values(value))
        snapshot = stage("native_forward_including_input_conversion", lambda: operation(source, list(value.shape), 1, .9))
        values = stage("native_output_to_list", lambda: snapshot.output)
        output = stage("list_to_output_tensor", lambda: torch.tensor(values, dtype=value.dtype, device=value.device).reshape_as(value))
        direction = stage("upstream_to_list", lambda: _values(upstream))
        gradient = stage("native_order_vjp_including_direction_conversion", lambda: snapshot.vjp_alpha(direction))
        order_gradient = stage("scalar_to_tensor", lambda: torch.tensor(gradient, dtype=alpha.dtype, device=alpha.device))
        assert torch.equal(output, reference_output)
        assert torch.equal(order_gradient, reference_gradient)
        return measurements

    for _ in range(2):
        run()
    rows = [run() for _ in range(12)]
    report = {"schema": "spiraltorch.fractional_stage_diagnostic.v1",
              "map": args.map, "shape": list(value.shape), "kernel_len": 32,
              "alpha": .9, "step": 1, "seed": 239, "threads": 2,
              "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              "profile_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "public_output_and_order_gradient_exact": True,
              "measurements_ms": rows,
              "stage_median_ms": {name: statistics.median(r[name] for r in rows) for name in rows[0]},
              "scope": "Instrumented staged route under ambient load, not a speed claim or additive model of public autograd latency. Native call stages include PyO3 list extraction; pure Rust timings are needed to isolate convolution."}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["stage_median_ms"]), flush=True)


if __name__ == "__main__":
    main()
