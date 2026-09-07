#!/usr/bin/env python3
"""Source-bound rank diagnostics; CPU/API overhead is included, not GPU-only time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import bench_rank_vs_torch as audit
import torch_rank_reference as cuda_reference


def foreign_gpu_processes():
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=10,
    )
    return sorted({int(line.strip()) for line in completed.stdout.splitlines()
                   if line.strip()} - {os.getpid()})


def require_uncontended_gpu():
    foreign = foreign_gpu_processes()
    if foreign:
        raise RuntimeError(f"timing admission blocked by other GPU compute processes: {foreign}")


def requests_for(bench, suite):
    if suite == "standard":
        geometries = [(cols, k, tile) for cols, k in [(256, 8), (2048, 16)]
                      for tile in [128, 256, 512]]
    elif suite == "midk-boundary":
        geometries = [(cols, 7, tile) for cols, tile in
                      [(1024, 32), (1025, 32), (1025, 256), (4097, 128),
                       (8193, 256), (257, 1), (1025, 8)]]
    elif suite == "active-lanes":
        geometries = [(tiles * 32 - 1, k, 32)
                      for tiles in [1, 2, 3, 5, 17, 33, 65, 129, 257]
                      for k in [1, 7, min(65, tiles * 32 - 1)]]
    else:
        raise ValueError(f"unknown rank suite: {suite}")
    for seed in [17, 29, 43]:
        for kind in ["topk", "midk", "bottomk"]:
            for cols, k, tile in geometries:
                values, _ = bench.fixture(2 * cols, seed)
                if suite != "standard" and seed == 43:
                    # Ties without mixed signed zeros, whose PyTorch order differs.
                    values = [float(int(value * 8)) for value in values]
                yield dict(kind=kind, rows=2, cols=cols, k=k, tile=tile, input=values, seed=seed)


def validate_native_result(result, request, resident_only):
    if (result.get("status") != "passed" or
        any(result.get(k) != request[k] for k in ("rows", "cols", "k", "kind", "seed")) or
        result.get("tile") != min(request["tile"], request["cols"])):
        raise RuntimeError(f"native result mismatch: {result}")
    if resident_only and (result.get("mode") != "resident_only" or
                          set(result.get("samples_ms", {})) != {"resident_dispatch_fence_per_op"}):
        raise RuntimeError("native resident-only boundary mismatch")


def cuda_rank_operation(request):
    if request["kind"] == "midk":
        return "stable_sort"
    cols = request["cols"]
    for row in range(request["rows"]):
        values = request["input"][row * cols:(row + 1) * cols]
        if len(set(values)) != cols:
            return "stable_sort"
    return "topk"


def require_canonical_indices(actual, expected):
    if actual != expected:
        raise RuntimeError("CUDA rank differs from canonical stable source indices")


def run(executable, suite="standard", resident_only=False):
    require_uncontended_gpu()
    import torch

    bench = audit.load_bench_module()
    before = audit.source_identity()
    original = audit.file_identity(executable)
    if not torch.cuda.is_available():
        raise RuntimeError("this matched device comparison requires PyTorch CUDA")
    requests = list(requests_for(bench, suite))
    payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r in requests)
    with tempfile.TemporaryDirectory(prefix="rank-execution-", dir=executable.parent) as directory:
        image = Path(directory) / "resident_rank_bench"
        os.link(executable, image)
        image_before = audit.file_identity(image)
        identity = audit.read_native_build_identity(image)
        binding = audit.validate_source_binding(identity, before)
        if not binding["valid"]:
            raise RuntimeError(f"source/build identity mismatch: {binding}")
        native = subprocess.run([str(image)] + (["--resident-only"] if resident_only else []),
                                input=payload, text=True, capture_output=True, timeout=300)
        if native.returncode or native.stderr:
            raise RuntimeError(f"native benchmark failed: {native.returncode} {native.stderr[-3000:]} {native.stdout[-3000:]}")
        results = [json.loads(line) for line in native.stdout.splitlines()]
        if len(results) != len(requests):
            raise RuntimeError("native result cardinality mismatch")
        report = {"schema": "spiraltorch.resident_rank_comparison.v2", "status": "passed",
                  "suite": suite, "resident_only_requested": resident_only,
                  "comparison": "separate-process wrapper diagnostics, not interleaved cross-framework or GPU-event timings",
                  "boundaries": {
                      "host_api": "upload, allocate outputs/scratch, two submits, two maps; pipelines prebuilt",
                      "resident_host_to_host": "reuse device buffers; upload, one compute submit, combined snapshot/map",
                      "resident_dispatch_fence_per_op": "16 resident rank pairs, one submit then completion fence; divided by 16; no upload/readback",
                      "torch_resident_per_op": "hindsight best eligible CUDA control; 16 resident operations including key encoding/repair/gather then completion; divided by 16; no upload/readback",
                  }, "torch": torch.__version__, "torch_device": torch.cuda.get_device_name(),
                  "request_sha256": hashlib.sha256(payload.encode()).hexdigest(), "source": before,
                  "native_build_identity": identity, "build_source_binding": binding, "cases": []}
        with torch.inference_mode():
            for r, result in zip(requests, results):
                validate_native_result(result, r, resident_only)
                if result["adapter"]["name"] != torch.cuda.get_device_name():
                    raise RuntimeError("WGPU and PyTorch must select the same named GPU")
                host = torch.tensor(r["input"], dtype=torch.float32).reshape(r["rows"], r["cols"])
                expected = cuda_reference.contract(r)
                actual = torch.tensor(result["values"], dtype=torch.float32)
                expected_bits = torch.tensor(expected["values"], dtype=torch.float32).view(torch.int32)
                if not torch.equal(actual.view(torch.int32), expected_bits) or result["indices"] != expected["indices"]:
                    raise RuntimeError("native output differs from canonical PyTorch reference")
                device = host.cuda()
                controls = cuda_reference.measure(r, device, torch, bench.summarize)
                operation = controls["best_fixed"]
                samples = {key: bench.summarize(v) for key, v in result["samples_ms"].items()}
                samples["torch_resident_per_op"] = controls["timings"][operation]
                report["cases"].append({"request": {k:v for k,v in r.items() if k != "input"},
                    "native": result, "timings": samples, "torch_operation": operation,
                    "torch_controls": controls, "legacy_torch_operation": cuda_rank_operation(r),
                    "torch_canonical_indices": "checked before and after all timing intervals"})
        after = audit.source_identity()
        require_uncontended_gpu()
        report["gpu_process_gate"] = "no foreign compute PIDs at preflight/postflight; not an exclusive reservation"
        report["provenance"] = {"valid": before == after and original == audit.file_identity(executable)
                                 and image_before == audit.file_identity(image), "source_after": after,
                                 "executable": original, "execution_image": image_before}
        if not report["provenance"]["valid"]:
            raise RuntimeError("source or executable changed during measurement")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["standard", "midk-boundary", "active-lanes"], default="standard")
    parser.add_argument("--resident-only", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.executable.resolve(strict=True), args.suite, args.resident_only)
    except Exception as error:
        report = {"schema": "spiraltorch.resident_rank_comparison.v2", "status": "error", "error": str(error)}
    audit.write_report_exclusive(args.output, report)
    print(json.dumps({"status": report["status"], "cases": len(report.get("cases", [])), "error": report.get("error")}))
    raise SystemExit(report["status"] != "passed")


if __name__ == "__main__":
    main()
