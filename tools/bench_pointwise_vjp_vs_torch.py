#!/usr/bin/env python3
"""Matched resident forward and all-input VJP, alternating with eager Torch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rank_vs_torch as audit
from bench_nd_tensor_vs_torch import Worker
from bench_resident_nn_vs_torch import admit_device, match_adapter


def recipes():
    return [
        dict(shape=list(shape), seed=seed, iterations=8)
        for shape in ((8, 16, 64), (16, 32, 128))
        for seed in (17, 29, 43)
    ]


def validate_fixture(value, config, identity):
    shape, seed = config["shape"], config["seed"]
    logical = [shape[1] - 1, shape[0], shape[2]]
    if (
        value.get("schema") != "spiraltorch.pointwise_vjp.bench_fixture.v1"
        or any(value.get(k) != v for k, v in config.items())
        or value.get("identity") != identity
        or value.get("logical_shape") != logical
        or value.get("scale") != 0.75
        or value.get("input")
        != [((i * 13 + seed) % 61) / 64.0 - 0.46875 for i in range(math.prod(shape))]
        or value.get("gain") != [0.5 + (i % 7) / 16.0 for i in range(shape[2])]
        or value.get("cotangent")
        != [((i * 7 + seed) % 17) / 16.0 - 0.5 for i in range(math.prod(logical))]
    ):
        raise ValueError("fixture recipe or identity mismatch")


def validate_sample(value, shape, capture, rust):
    elapsed = value.get("elapsed_ms")
    if (
        value.get("shape") != shape
        or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or rust
        and value.get("finite_checked") is not True
    ):
        raise ValueError("invalid interval")
    values = value.get("values")
    if not capture:
        if values is not None:
            raise ValueError("unexpected capture")
        return
    sizes = [math.prod(shape), math.prod(shape), shape[-1], 1]
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("missing output or gradient slot")
    for values, size in zip(values, sizes):
        if (
            not isinstance(values, list)
            or len(values) != size
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in values)
        ):
            raise ValueError("invalid capture")


def reaggregate(report):
    if (
        report.get("schema") != "spiraltorch.pointwise_vjp.torch_bench.v1"
        or report.get("status") != "passed"
        or report.get("warmups") != 2
        or report.get("samples") != 8
        or len(report["cases"]) != 6
        or report.get("source_binding", {}).get("valid") is not True
        or report.get("fallback_enabled") is not False
    ):
        raise ValueError("incomplete benchmark")
    summaries = []
    for index, (case, config) in enumerate(zip(report["cases"], recipes())):
        validate_fixture(case["fixture"], config, report["identity"])
        shape = case["fixture"]["logical_shape"]
        if len(case["intervals"]) != 10:
            raise ValueError("missing intervals")
        for i, interval in enumerate(case["intervals"]):
            order = ["rust", "torch"] if (i + index) % 2 == 0 else ["torch", "rust"]
            if (
                interval.get("iteration") != i
                or interval.get("warmup") is not (i < 2)
                or interval.get("order") != order
            ):
                raise ValueError("interval order changed")
            for lane in order:
                validate_sample(interval[lane], shape, i == 2, lane == "rust")
            if i == 2:
                maximum = 0.0
                for av, bv in zip(
                    interval["rust"]["values"], interval["torch"]["values"]
                ):
                    for a, b in zip(av, bv):
                        if abs(a - b) > 2e-5 + 2e-4 * abs(b):
                            raise ValueError("capture mismatch")
                        maximum = max(maximum, abs(a - b))
                if not math.isclose(maximum, case["max_abs_error"], abs_tol=1e-12):
                    raise ValueError("incorrect error summary")
        medians = {
            lane: statistics.median(
                interval[lane]["elapsed_ms"] for interval in case["intervals"][2:]
            )
            for lane in ("rust", "torch")
        }
        ratio = medians["torch"] / medians["rust"]
        if medians != case["median_ms"] or ratio != case["torch_over_rust"]:
            raise ValueError("incorrect timing summary")
        summaries.append(dict(config, median_ms=medians, torch_over_rust=ratio))
    return summaries


def run(binary, output, device):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no fallback")
    before, product = audit.source_identity(), audit.file_identity(binary)
    identity = audit.read_native_build_identity(binary)
    binding = audit.validate_source_binding(identity, before)
    if not binding["valid"] or audit.git_bytes(
        "ls-files", "--others", "--exclude-standard"
    ):
        raise ValueError("freeze the source and binary before measurement")
    admission = admit_device(device) if device == "mps" else {"device": "cpu"}
    torch.set_num_threads(1)
    report = {
        "schema": "spiraltorch.pointwise_vjp.torch_bench.v1",
        "status": "error",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": before,
        "product": product,
        "identity": identity,
        "source_binding": binding,
        "torch_version": torch.__version__,
        "torch_device": device,
        "device_admission": admission,
        "fallback_enabled": False,
        "warmups": 2,
        "samples": 8,
        "cases": [],
        "boundary": "resident strided logical input, channel gain and scalar; 8x(mul gain,tanh GELU,original-input residual,mul scalar),ReLU; forward plus explicit cotangent VJP to all 3 slots; terminal contiguous host output and all gradients included; upload/views/plan construction/JSON excluded; Rust recomputes forward and checks every intermediate, Torch eager autograd retains a tape and has no equivalent guards; no NN optimizer/browser/CUDA/torch.compile claim; uncontrolled OS load",
    }
    with output.open("x", encoding="utf-8") as handle, output.with_suffix(
        ".stderr.log"
    ).open("x") as stderr:
        worker = None
        try:
            worker = Worker([str(binary)], stderr)
            for index, config in enumerate(recipes()):
                fixture = worker.request({"op": "init", "config": config})
                validate_fixture(fixture, config, identity)
                if device == "mps":
                    match_adapter(fixture["adapter"], device, admission["name"])
                shape = config["shape"]
                x = torch.tensor(
                    fixture["input"], dtype=torch.float32, device=device
                ).reshape(shape)
                x = (
                    x.permute(1, 0, 2)
                    .narrow(0, 1, shape[1] - 1)
                    .detach()
                    .requires_grad_(True)
                )
                gain = torch.tensor(
                    fixture["gain"],
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )
                scale = torch.tensor(
                    0.75, dtype=torch.float32, device=device, requires_grad=True
                )
                seed = torch.tensor(
                    fixture["cotangent"], dtype=torch.float32, device=device
                ).reshape(x.shape)
                if device == "mps":
                    torch.mps.synchronize()

                def torch_sample(capture):
                    start = time.perf_counter()
                    y = x
                    for _ in range(config["iterations"]):
                        y = (
                            torch.nn.functional.gelu(y * gain, approximate="tanh") + x
                        ) * scale
                    y = y.relu()
                    gradients = torch.autograd.grad(
                        y, [x, gain, scale], grad_outputs=seed
                    )
                    hosts = [
                        value.detach().contiguous().cpu().numpy()
                        for value in (y, *gradients)
                    ]
                    elapsed = (time.perf_counter() - start) * 1000.0
                    if any(not np.isfinite(host).all() for host in hosts):
                        raise ValueError("non-finite Torch output")
                    return {
                        "elapsed_ms": elapsed,
                        "shape": list(y.shape),
                        "values": (
                            [host.reshape(-1).tolist() for host in hosts]
                            if capture
                            else None
                        ),
                    }

                case = {"fixture": fixture, "intervals": []}
                report["cases"].append(case)
                for i in range(10):
                    capture = i == 2
                    order = (
                        ["rust", "torch"] if (i + index) % 2 == 0 else ["torch", "rust"]
                    )
                    interval = {"iteration": i, "warmup": i < 2, "order": order}
                    case["intervals"].append(interval)
                    for lane in order:
                        value = (
                            worker.request({"op": "sample", "capture": capture})
                            if lane == "rust"
                            else torch_sample(capture)
                        )
                        validate_sample(value, list(x.shape), capture, lane == "rust")
                        interval[lane] = value
                    if capture:
                        maximum = 0.0
                        for actual, expected in zip(
                            interval["rust"]["values"], interval["torch"]["values"]
                        ):
                            a, b = np.asarray(actual), np.asarray(expected)
                            if not np.allclose(a, b, atol=2e-5, rtol=2e-4):
                                raise ValueError("Rust/Torch gradient mismatch")
                            maximum = max(maximum, float(np.max(np.abs(a - b))))
                        case["max_abs_error"] = maximum
                case["median_ms"] = {
                    lane: statistics.median(
                        i[lane]["elapsed_ms"] for i in case["intervals"][2:]
                    )
                    for lane in ("rust", "torch")
                }
                case["torch_over_rust"] = (
                    case["median_ms"]["torch"] / case["median_ms"]["rust"]
                )
                print(
                    json.dumps(
                        dict(
                            config,
                            median_ms=case["median_ms"],
                            max_abs_error=case["max_abs_error"],
                        )
                    ),
                    flush=True,
                )
            worker.finish()
            if (
                audit.source_identity() != before
                or audit.file_identity(binary) != product
            ):
                raise RuntimeError("source or binary changed")
            if device == "mps" and admit_device(device) != admission:
                raise RuntimeError("GPU admission changed")
            report["status"] = "passed"
            reaggregate(report)
        except BaseException as error:
            report["status"] = "error"
            report["error"] = repr(error)
            if worker is not None:
                worker.abort()
            raise
        finally:
            try:
                if worker is not None:
                    worker.close_streams()
            finally:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(reaggregate(json.loads(args.verify.read_text())), indent=2))
    elif args.binary and args.output:
        run(args.binary.resolve(), args.output, args.torch_device)
    else:
        parser.error("provide --verify or --binary and --output")
