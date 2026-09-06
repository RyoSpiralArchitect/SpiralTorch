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


def run(executable):
    import torch

    bench = audit.load_bench_module()
    before = audit.source_identity()
    original = audit.file_identity(executable)
    if not torch.cuda.is_available():
        raise RuntimeError("this matched device comparison requires PyTorch CUDA")
    requests = []
    for seed in [17, 29, 43]:
        for kind in ["topk", "midk", "bottomk"]:
            for cols, k in [(256, 8), (2048, 16)]:
                values, _ = bench.fixture(2 * cols, seed)
                for tile in [128, 256, 512]:
                    requests.append(dict(kind=kind, rows=2, cols=cols, k=k, tile=tile, input=values, seed=seed))
    payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r in requests)
    with tempfile.TemporaryDirectory(prefix="rank-execution-", dir=executable.parent) as directory:
        image = Path(directory) / "resident_rank_bench"
        os.link(executable, image)
        image_before = audit.file_identity(image)
        identity = audit.read_native_build_identity(image)
        binding = audit.validate_source_binding(identity, before)
        if not binding["valid"]:
            raise RuntimeError(f"source/build identity mismatch: {binding}")
        native = subprocess.run([str(image)], input=payload, text=True, capture_output=True, timeout=180)
        if native.returncode or native.stderr:
            raise RuntimeError(f"native benchmark failed: {native.returncode} {native.stderr[-3000:]} {native.stdout[-3000:]}")
        results = [json.loads(line) for line in native.stdout.splitlines()]
        if len(results) != len(requests):
            raise RuntimeError("native result cardinality mismatch")
        report = {"schema": "spiraltorch.resident_rank_comparison.v1", "status": "passed",
                  "comparison": "separate-process wrapper diagnostics, not interleaved cross-framework or GPU-event timings",
                  "boundaries": {
                      "host_api": "upload, allocate outputs/scratch, two submits, two maps; pipelines prebuilt",
                      "resident_host_to_host": "reuse device buffers; upload, one compute submit, combined snapshot/map",
                      "resident_dispatch_fence_per_op": "16 resident rank pairs, one submit then completion fence; divided by 16; no upload/readback",
                      "torch_resident_per_op": "16 Python enqueue calls with preallocated CUDA outputs then synchronize; divided by 16; no upload/readback",
                  }, "torch": torch.__version__, "torch_device": torch.cuda.get_device_name(),
                  "request_sha256": hashlib.sha256(payload.encode()).hexdigest(), "source": before,
                  "native_build_identity": identity, "build_source_binding": binding, "cases": []}
        with torch.inference_mode():
            for r, result in zip(requests, results):
                if result.get("status") != "passed" or any(result.get(k) != r[k] for k in ("rows", "cols", "k", "kind", "seed")):
                    raise RuntimeError(f"native result mismatch: {result}")
                if result["adapter"]["name"] != torch.cuda.get_device_name():
                    raise RuntimeError("WGPU and PyTorch must select the same named GPU")
                host = torch.tensor(r["input"], dtype=torch.float32).reshape(r["rows"], r["cols"])
                expected, expected_ids = host.sort(dim=1, descending=r["kind"] == "topk", stable=True)
                start = (r["cols"] - r["k"]) // 2 if r["kind"] == "midk" else 0
                expected = expected[:, start:start + r["k"]]
                expected_ids = expected_ids[:, start:start + r["k"]]
                if result["values"] != expected.flatten().tolist() or result["indices"] != expected_ids.flatten().tolist():
                    raise RuntimeError("native output differs from canonical PyTorch reference")
                device = host.cuda()
                shape = device.shape if r["kind"] == "midk" else (r["rows"], r["k"])
                out_values = torch.empty(shape, dtype=torch.float32, device="cuda")
                out_ids = torch.empty(shape, dtype=torch.int64, device="cuda")

                def op():
                    if r["kind"] == "midk":
                        torch.sort(device, dim=1, stable=True, out=(out_values, out_ids))
                    else:
                        torch.topk(device, r["k"], dim=1, largest=r["kind"] == "topk", sorted=True, out=(out_values, out_ids))

                def repeated():
                    for _ in range(16):
                        op()

                op()
                view = out_values[:, start:start + r["k"]] if r["kind"] == "midk" else out_values
                bench.correctness(view, expected.double(), torch, 0, 0)
                if not torch.equal(device.gather(1, out_ids), out_values):
                    raise RuntimeError("PyTorch rank indices do not point to returned values")
                timing = bench.paired_timings({"torch": repeated}, 2, 12, torch.cuda.synchronize, r["seed"])["torch"]
                samples = {key: bench.summarize(v) for key, v in result["samples_ms"].items()}
                samples["torch_resident_per_op"] = bench.summarize([v / 16 for v in timing["samples_ms"]])
                report["cases"].append({"request": {k:v for k,v in r.items() if k != "input"}, "native": result, "timings": samples})
        after = audit.source_identity()
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
    args = parser.parse_args()
    try:
        report = run(args.executable.resolve(strict=True))
    except Exception as error:
        report = {"schema": "spiraltorch.resident_rank_comparison.v1", "status": "error", "error": str(error)}
    audit.write_report_exclusive(args.output, report)
    print(json.dumps({"status": report["status"], "cases": len(report.get("cases", [])), "error": report.get("error")}))
    raise SystemExit(report["status"] != "passed")


if __name__ == "__main__":
    main()
