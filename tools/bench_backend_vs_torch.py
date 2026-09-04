#!/usr/bin/env python3
"""Matched host-to-host backend benchmarks; resident Torch is a separate boundary."""

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


def parse_sizes(spec):
    shapes = []
    for chunk in spec.split(";"):
        if "x" in chunk:
            left, right = chunk.split("x")
            m, k = map(int, left.split(","))
            other, n = map(int, right.split(","))
            if k != other:
                raise ValueError("inner dimensions must match")
        else:
            m, k = map(int, chunk.split(","))
            if m != k:
                raise ValueError("use M,KxK,N for rectangular shapes")
            n = k
        if min(m, k, n) <= 0:
            raise ValueError("dimensions must be positive")
        shapes.append((m, k, n))
    return shapes


def fixture(count, seed):
    rng = random.Random(seed)
    values = array("f", (rng.uniform(-0.5, 0.5) for _ in range(count)))
    return values.tolist(), hashlib.sha256(values.tobytes()).hexdigest()


def timed(fn, sync):
    sync()
    start = time.perf_counter_ns()
    result = fn()
    sync()
    return result, (time.perf_counter_ns() - start) / 1e6


def summarize(samples):
    ordered = sorted(samples)
    return {"n": len(samples), "median_ms": statistics.median(samples),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "min_ms": ordered[0], "samples_ms": samples}


def paired_timings(functions, warmup, iterations, sync, seed):
    for fn in functions.values():
        for _ in range(warmup):
            timed(fn, sync)
    rng = random.Random(seed)
    samples = {name: [] for name in functions}
    order = list(functions)
    for _ in range(iterations):
        rng.shuffle(order)
        for name in order:
            _, elapsed = timed(functions[name], sync)
            samples[name].append(elapsed)
    return {name: summarize(values) for name, values in samples.items()}


def correctness(actual, expected, torch, rtol, atol):
    actual = actual.detach().cpu().to(torch.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"output shape mismatch: {actual.shape} != {expected.shape}")
    error = (actual - expected).abs()
    ok = bool(torch.isfinite(actual).all()) and bool((error <= atol + rtol * expected.abs()).all())
    result = {"passed": ok, "max_abs_error": error.max().item(), "rtol": rtol, "atol": atol}
    if not ok:
        raise ValueError(f"correctness gate failed: {result}")
    return result


def benchmark_case(operation, shape, seed, args, st, torch, device):
    m, k, n = shape
    values, digest = fixture((m if operation == "scatter" else k) * n if operation != "matmul" else m * k, seed)
    ids = torch.tensor([(i * 17 + seed) % k for i in range(m)], dtype=torch.int64)
    ids_list = ids.tolist()
    hashes = [digest]
    if operation == "matmul":
        other, digest = fixture(k * n, seed + 1)
        hashes.append(digest)
        hosts = [torch.tensor(values, dtype=torch.float32).reshape(m, k),
                 torch.tensor(other, dtype=torch.float32).reshape(k, n)]
        native = [st.Tensor(m, k, values), st.Tensor(k, n, other)]
        st_fn = lambda: native[0].matmul(native[1], backend=args.st_backend)
        torch_op = lambda inputs, _: torch.mm(*inputs)
    else:
        rows = m if operation == "scatter" else k
        hosts = [torch.tensor(values, dtype=torch.float32).reshape(rows, n)]
        native = st.Tensor(rows, n, values)
        util_backend = "cpu" if args.st_backend == "faer" else args.st_backend
        if operation == "gather":
            st_fn = lambda: native.gather_rows(ids_list, backend=util_backend)
            torch_op = lambda inputs, index: inputs[0].index_select(0, index)
        else:
            st_fn = lambda: native.scatter_add_rows(ids_list, k, backend=util_backend)
            torch_op = lambda inputs, index: inputs[0].new_zeros((k, n)).index_add_(0, index, inputs[0])
        hashes.append(hashlib.sha256(array("q", ids_list).tobytes()).hexdigest())

    reference = torch_op([value.double() for value in hosts], ids)
    residents = [value.to(device) for value in hosts]
    resident_ids = ids.to(device) if operation != "matmul" else None
    if device.type == "cuda":
        sync = torch.cuda.synchronize
    elif device.type == "mps":
        sync = torch.mps.synchronize
    else:
        sync = lambda: None

    def torch_host():
        # Every GPU invocation includes fresh uploads and result readback, as ST does.
        inputs = [value.to(device) for value in hosts]
        index = ids.to(device) if operation != "matmul" else None
        return torch_op(inputs, index).cpu()

    functions = {"spiraltorch_host_to_host": st_fn, "torch_host_to_host": torch_host}
    if device.type != "cpu":
        functions["torch_device_resident"] = lambda: torch_op(residents, resident_ids)
    cold, checks = {}, {}
    for name, fn in functions.items():
        output, cold[name] = timed(fn, sync)
        if name.startswith("spiraltorch"):
            output = torch.tensor(output.tolist(), dtype=torch.float64)
        checks[name] = correctness(output, reference, torch, args.rtol, args.atol)
    timings = paired_timings(functions, args.warmup, args.iters, sync, seed)
    st_ms = timings["spiraltorch_host_to_host"]["median_ms"]
    torch_ms = timings["torch_host_to_host"]["median_ms"]
    return {"operation": operation, "shape_m_k_n": shape, "seed": seed,
            "input_sha256": hashes, "status": "passed", "correctness": checks,
            "first_call_ms": cold, "timings": timings,
            "torch_over_st_host_to_host": torch_ms / st_ms,
            "resident_speed_ratio": None}


def run(args):
    # Pin CPU work before importing either runtime; do not alter system configuration.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["SPIRALTORCH_STRICT_GPU"] = "1"
    os.environ["SPIRALTORCH_WGPU_ALLOW_INT8"] = "0"
    # Autotuning must not read or update another workload's persistent cache.
    os.environ["SPIRALTORCH_AUTOTUNE_STORE"] = str(args.output.with_suffix(".autotune.json"))
    import spiraltorch as st
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.torch_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no fallback")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; no fallback")
    build = st.build_info()
    if args.st_backend == "wgpu" and not build["features"]["wgpu"]:
        raise RuntimeError("native wheel has no WGPU feature; refusing an auto fallback")
    native_path = Path(st._rs.__file__).resolve()
    if args.native_prefix and not native_path.is_relative_to(Path(args.native_prefix).resolve()):
        raise RuntimeError(f"native import is outside expected prefix: {native_path}")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    report = {"schema": "spiraltorch.matched_backend_bench.v1", "git_revision": revision.stdout.strip() or None,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "host": platform.node(), "platform": platform.platform(), "byteorder": sys.byteorder,
              "python": sys.version, "torch": torch.__version__, "torch_file": torch.__file__,
              "torch_device": str(device), "torch_cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "spiraltorch_build": build, "spiraltorch_native": str(native_path),
              "spiraltorch_native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
              "requested_st_backend": args.st_backend, "threads": args.threads,
              "autotune_store": os.environ["SPIRALTORCH_AUTOTUNE_STORE"],
              "vulkan_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
              "wgpu_adapter_identity": "not exposed by this native wheel",
              "dtype": "float32", "tf32": False, "warmup": args.warmup, "iterations": args.iters,
              "st_int8_opt_in": False,
              "boundary": "host tensor -> operation -> host tensor; list conversion excluded",
              "resident_boundary": "diagnostic only; not used for the host-to-host speed ratio",
              "cases": []}
    for operation in args.operations.split(","):
        if operation not in ("matmul", "gather", "scatter"):
            raise ValueError(f"unknown operation: {operation}")
        for shape in parse_sizes(args.sizes):
            for seed in args.seeds:
                try:
                    case = benchmark_case(operation, shape, seed, args, st, torch, device)
                except Exception as error:
                    case = {"operation": operation, "shape_m_k_n": shape, "seed": seed,
                            "status": "error", "error": f"{type(error).__name__}: {error}"}
                report["cases"].append(case)
                print(json.dumps(case, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="64,64;256,256;512,512")
    parser.add_argument("--operations", default="matmul,gather,scatter")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--st-backend", choices=["cpu", "faer", "wgpu"], default="wgpu")
    parser.add_argument("--torch-device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--native-prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("positive iterations/threads and nonnegative warmup required")
    if any(not math.isfinite(x) or x < 0 for x in (args.rtol, args.atol)):
        parser.error("finite nonnegative tolerances required")
    parse_sizes(args.sizes)
    # Reserve a new result path before starting expensive work; never overwrite a run.
    with args.output.open("x") as stream:
        try:
            report = run(args)
        except Exception as error:
            report = {"status": "setup_error", "error": f"{type(error).__name__}: {error}"}
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if "cases" not in report or any(case["status"] != "passed" for case in report["cases"]):
        print(json.dumps({"status": "failed", "output": str(args.output)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
