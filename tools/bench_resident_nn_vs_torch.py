#!/usr/bin/env python3
"""Source-bound host-to-host Sequential versus resident WGPU versus eager PyTorch."""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile
from time import perf_counter

import bench_rank_vs_torch as audit
from bench_resident_rank_vs_torch import require_uncontended_gpu


def requests():
    for seed in (17, 29, 43):
        for shape, depth in (([2, 3, 7], 2), ([2, 8, 64], 16), ([4, 8, 128], 16)):
            yield dict(shape=shape, depth=depth, seed=seed)


def fixture(request):
    state = request["seed"]

    def value():
        nonlocal state
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        return (state % 65 - 32) / 256

    width, depth = request["shape"][-1], request["depth"]
    parameters = [dict(weight=[value() for _ in range(width * width)],
                       bias=[value() for _ in range(width)], gelu=i + 1 < depth)
                  for i in range(depth)]
    return [value() for _ in range(math.prod(request["shape"]))], parameters


def validate_native(case, request):
    if (case.get("status") != "passed"
            or any(case.get(key) != value for key, value in request.items())
            or case.get("source_operations") != request["depth"] * 2 - 1
            or case.get("gpu_stages") != request["depth"]):
        raise ValueError("native case identity or lowering differs")
    expected_input, expected_parameters = fixture(request)
    if case.get("input") != expected_input or case.get("parameters") != expected_parameters:
        raise ValueError("native fixture differs from independent input/parameter generation")
    if len(case.get("reference", [])) != len(expected_input) or not all(
            math.isfinite(x) for x in case["reference"]):
        raise ValueError("invalid native reference")
    samples = case.get("samples", [])
    if len(samples) != 12:
        raise ValueError("native sample cardinality differs")
    for i, sample in enumerate(samples):
        expected_order = [0, 1] if (i + 2 + request["seed"]) % 2 == 0 else [1, 0]
        if sample.get("order") != expected_order:
            raise ValueError("native control order differs")
        if not all(math.isfinite(sample[key]) and sample[key] > 0
                   for key in ("legacy_ms", "resident_ms")):
            raise ValueError("invalid native timing")
    for name in ("legacy", "resident"):
        actual = statistics.mean(sample[name + "_ms"] for sample in samples)
        if not math.isclose(actual, case[name + "_mean_ms"], rel_tol=1e-12):
            raise ValueError("native summary differs from raw samples")


def admit_device(device):
    if device == "cuda":
        require_uncontended_gpu()
        return dict(device="cuda", contention="no foreign GPU compute processes observed")
    if device != "mps" or platform.system() != "Darwin":
        raise RuntimeError("MPS comparison requires macOS")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("disable PYTORCH_ENABLE_MPS_FALLBACK for a GPU comparison")
    result = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"],
                            check=True, capture_output=True, text=True, timeout=30)
    devices = json.loads(result.stdout)["SPDisplaysDataType"]
    if (len(devices) != 1 or devices[0].get("spdisplays_vendor") != "sppci_vendor_Apple"
            or not devices[0].get("sppci_model", "").startswith("Apple ")):
        raise RuntimeError("MPS comparison requires exactly one Apple GPU")
    return dict(device="mps", name=devices[0]["sppci_model"],
                contention="unknown: macOS does not expose CUDA-style GPU process accounting; diagnostic only")


def match_adapter(adapter, device, name):
    if (adapter.get("name") != name or adapter.get("device_type") == "Cpu"
            or device == "mps" and adapter.get("backend") != "Metal"):
        raise RuntimeError("WGPU and PyTorch GPU identities differ")


def run(executable, state, device="cuda"):
    admission = admit_device(device)
    state["device_admission"] = admission
    before, original = audit.source_identity(), audit.file_identity(executable)
    identity = audit.read_native_build_identity(executable)
    binding = audit.validate_source_binding(identity, before)
    state.update(source=before, native_build_identity=identity, build_source_binding=binding,
                 executable=original)
    if not binding["valid"]:
        raise RuntimeError("native executable is not bound to the clean measured source")
    matrix = list(requests())
    payload = "".join(json.dumps(request, separators=(",", ":")) + "\n" for request in matrix)
    with tempfile.TemporaryDirectory(prefix="spiraltorch-nn-") as directory:
        image = Path(directory) / "resident-mlp"
        image.write_bytes(executable.read_bytes())
        image.chmod(0o555)
        copied = audit.file_identity(image)
        if copied["sha256"] != original["sha256"]:
            raise RuntimeError("immutable image differs")
        completed = subprocess.run([str(image)], input=payload, text=True, capture_output=True, timeout=600)
        state["native_stderr"] = completed.stderr
        if completed.returncode:
            state["native_partial_stdout"] = completed.stdout
            raise RuntimeError(f"native failed: {completed.stderr[-4000:]}")
        cases = [json.loads(line) for line in completed.stdout.splitlines()]
        state["cases"] = cases
        if len(cases) != len(matrix) or audit.file_identity(image) != copied:
            raise RuntimeError("native cardinality or executable identity changed")
    if admit_device(device) != admission:
        raise RuntimeError("GPU admission changed")
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("matched comparison requires CUDA, no fallback")
        name, synchronize = torch.cuda.get_device_name(), torch.cuda.synchronize
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("matched comparison requires MPS, no fallback")
        name, synchronize = admission["name"], torch.mps.synchronize
    with torch.inference_mode():
        for case, request in zip(cases, matrix):
            validate_native(case, request)
            match_adapter(case["adapter"], device, name)
            width = request["shape"][-1]
            host = torch.tensor(case["input"], dtype=torch.float32).reshape(-1, width)
            expected = torch.tensor(case["reference"], dtype=torch.float32).reshape(host.shape)
            weights = [(torch.tensor(p["weight"], dtype=torch.float32, device=device).reshape(width, width),
                        torch.tensor(p["bias"], dtype=torch.float32, device=device), p["gelu"])
                       for p in case["parameters"]]
            gpu_input = torch.empty_like(host, device=device)
            times = []
            case["torch_samples_ms"] = times
            for block in range(14):
                synchronize()
                start = perf_counter()
                gpu_input.copy_(host, non_blocking=True)
                value = gpu_input
                for weight, bias, gelu in weights:
                    value = torch.addmm(bias, value, weight)
                    if gelu:
                        value = torch.nn.functional.gelu(value, approximate="tanh")
                actual = value.cpu()
                elapsed = (perf_counter() - start) * 1000
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("invalid PyTorch latency")
                if block >= 2:
                    times.append(elapsed)
            case["torch_mean_ms"] = statistics.mean(times)
            case["resident_over_legacy"] = case["resident_mean_ms"] / case["legacy_mean_ms"]
            case["resident_over_torch"] = case["resident_mean_ms"] / case["torch_mean_ms"]
    if admit_device(device) != admission:
        raise RuntimeError("GPU admission changed")
    if audit.source_identity() != before or audit.file_identity(executable) != original:
        raise RuntimeError("source or native executable changed during the run")
    return dict(schema="spiraltorch.resident_nn_comparison.v1", status="passed",
                source=before, native_build_identity=identity, build_source_binding=binding,
                executable=original, torch=torch.__version__, torch_device=device,
                device_admission=admission, cases=cases,
                native_stderr=completed.stderr,
                request_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                boundary="host-to-host eager inference; fixed model parameters, native legacy prepack/cache behavior unchanged; resident/Torch weights and bias device-persistent; fresh input upload and final readback each sample; native controls interleaved, PyTorch measured afterward; 2 warmups + 12 samples; compilation/setup excluded; no fastest-PyTorch claim",
                semantics="Rust preserves intermediate finite guards on GPU; eager PyTorch does not add per-stage guards for these frozen finite fixtures; every output is checked outside timing; not backward/training throughput")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    args = parser.parse_args()
    report = dict(schema="spiraltorch.resident_nn_comparison.v1", status="error")
    try:
        report.update(run(args.executable.resolve(strict=True), report, args.device))
    except Exception as error:
        report.update(status="error", error=str(error))
    audit.write_report_exclusive(args.output, report)
    print(json.dumps(dict(status=report["status"], cases=len(report.get("cases", [])), error=report.get("error"))))
    raise SystemExit(report["status"] != "passed")


if __name__ == "__main__":
    main()
