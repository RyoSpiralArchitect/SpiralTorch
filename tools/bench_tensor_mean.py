#!/usr/bin/env python3
"""Same input-order f64 mean in native SpiralTorch and PyTorch CPU/CUDA.

Inputs are allocated before timing; each timed call returns a new output.
SpiralTorch validates inputs every call; Torch inputs are prevalidated outside
timing (a conservative advantage for Torch). CUDA timings include synchronization
but exclude host transfers. This is not a float32 stack.mean or training benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch


def checksum(values):
    result = 2166136261
    for value in np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32):
        result = ((result ^ int(value)) * 16777619) & 0xFFFFFFFF
    return result


def ordered_mean(partials, scale):
    result = torch.zeros_like(partials[0], dtype=torch.float64)
    for partial in partials:
        result.add_(partial)
    return result.div_(len(partials)).mul_(float(np.float32(scale))).float()


def cuda_availability():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    foreign = [pid for pid in pids if pid != os.getpid()]
    if foreign:
        raise RuntimeError(f"foreign CUDA processes observed: {foreign}")
    return {"time_ns": time.time_ns(), "compute_pids": pids, "own_pid": os.getpid()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--native-results", type=Path)
    parser.add_argument("--torch-only", action="store_true")
    args = parser.parse_args()
    # Exclusive creation prevents accidentally overwriting an earlier run.
    with args.output.open("x") as output:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        availability = []
        if args.device == "cuda":
            availability.append(cuda_availability())
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        st = None
        if not args.torch_only:
            import spiraltorch as st
            assert callable(st.mean_tensors_scaled)
        native = None
        if args.native_results:
            native = json.loads(args.native_results.read_text())
            assert native["contract"] == "ordered-f64-mean-scaled.v1"
            native = {
                (c["rows"], c["cols"], c["count"], c["seed"]): c["checksum"]
                for c in native["cases"] if not c["mixed_col_major"]
            }
            assert len(native) == 27
        report = {
            "schema": "spiraltorch.tensor_mean_torch.v1", "torch_version": torch.__version__,
            "torch_file": torch.__file__, "device": args.device, "host": platform.platform(),
            "torch_execution": args.device + "_ordered_f64",
            "spiraltorch_execution": "native_cpu_ordered_f64" if st is not None else None,
            "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.machine(),
            "torch_threads": torch.get_num_threads(), "boundary": __doc__,
            "warmup": 3, "samples": 16, "cases": [],
            "gpu_availability": availability,
        }
        if st is not None:
            report["spiraltorch_file"] = st.__file__
        if args.native_results:
            report["native_results_sha256"] = hashlib.sha256(args.native_results.read_bytes()).hexdigest()
        try:
            with torch.inference_mode():
                for seed in [17, 29, 43]:
                    for count in [4, 16, 64]:
                        for rows, cols in [(1, 1025), (32, 2048), (128, 2048)]:
                            if args.device == "cuda":
                                availability.append(cuda_availability())
                            index = np.arange(rows * cols, dtype=np.int64)
                            data = [(((index * 17 + p * 131 + seed * 73) % 4093 - 2046) / 64).astype(np.float32).reshape(rows, cols) for p in range(count)]
                            partials = [torch.from_numpy(values).to(args.device) for values in data]
                            scale = 1.25
                            reference = np.zeros((rows, cols), dtype=np.float64)
                            for values in data:
                                reference += values
                            reference = (reference / count * scale).astype(np.float32)
                            expected_checksum = checksum(reference)
                            if native is not None:
                                assert native[(rows, cols, count, seed)] == expected_checksum
                            spiral = [st.Tensor(rows, cols, values.ravel().tolist()) for values in data] if st else None
                            arms = ["torch", "spiraltorch"] if st else ["torch"]
                            samples = {arm + "_ms": [] for arm in arms}
                            for sample in range(report["warmup"] + report["samples"]):
                                for arm in arms if sample % 2 == 0 else arms[::-1]:
                                    if args.device == "cuda":
                                        torch.cuda.synchronize()
                                    start = time.perf_counter()
                                    actual = ordered_mean(partials, scale) if arm == "torch" else st.mean_tensors_scaled(spiral, scale)
                                    if arm == "torch" and args.device == "cuda":
                                        torch.cuda.synchronize()
                                    elapsed_ms = (time.perf_counter() - start) * 1000
                                    values = actual.cpu().numpy() if arm == "torch" else np.asarray(actual.tolist(), dtype=np.float32)
                                    assert np.array_equal(values.view(np.uint32), reference.view(np.uint32))
                                    if sample >= report["warmup"]:
                                        samples[arm + "_ms"].append(elapsed_ms)
                            case = dict(rows=rows, cols=cols, count=count, seed=seed, scale=scale, checksum=expected_checksum, **samples)
                            report["cases"].append(case)
                            if args.device == "cuda":
                                availability.append(cuda_availability())
                            print(seed, count, rows, cols, {arm: statistics.median(samples[arm + "_ms"]) for arm in arms}, flush=True)
                report["status"] = "passed"
        except BaseException as exc:
            report["status"] = "error"
            report["error"] = repr(exc)
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
