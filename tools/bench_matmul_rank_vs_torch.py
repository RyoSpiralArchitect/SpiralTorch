#!/usr/bin/env python3
"""Small resident projection-head diagnostics; not full-model or GPU-event time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile

import bench_rank_vs_torch as audit
from bench_resident_rank_vs_torch import require_uncontended_gpu


def require_canonical_indices(actual, expected):
    if actual != expected:
        raise RuntimeError("CUDA rank indices differ from the canonical stable selection")


def run_native_pass(image, payload, probe_only=False, resident_only=False):
    if probe_only and resident_only:
        raise ValueError("a native pass cannot be both comparison and probe")
    flags = ["--readback-probe"] if probe_only else (["--resident-only"] if resident_only else [])
    native = subprocess.run([str(image)] + flags,
        input=payload, text=True, capture_output=True, timeout=180)
    if native.returncode or native.stderr:
        raise RuntimeError(f"native benchmark failed: {native.returncode} {native.stderr[-3000:]} {native.stdout[-3000:]}")
    return [json.loads(line) for line in native.stdout.splitlines()]


def collect_readback_diagnostics(image, payload, requests, comparisons):
    results = run_native_pass(image, payload, probe_only=True)
    if len(results) != len(requests) or len(comparisons) != len(requests):
        raise RuntimeError("native diagnostic cardinality mismatch")
    for request, result, comparison in zip(requests, results, comparisons):
        if (result.get("status") != "passed" or result.get("mode") != "probe_only"
                or result.get("samples_ms") is not None
                or any(result.get(key) != request[key] for key in ("rows", "inner", "cols", "k", "kind", "seed"))
                or any(result.get(key) != comparison[key] for key in ("values", "indices", "adapter"))
                or not isinstance(result.get("readback_probe"), dict)
                or result["readback_probe"].get("status") != "passed"):
            raise RuntimeError("native diagnostic contract mismatch")
    return dict(boundary="separate native process after ALL native and CUDA comparison samples; not comparison timing",
                results=results)


def run(executable, readback_probe=False, resident_only=False):
    require_uncontended_gpu()
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    bench = audit.load_bench_module()
    before, original = audit.source_identity(), audit.file_identity(executable)
    if not torch.cuda.is_available():
        raise RuntimeError("matched comparison requires PyTorch CUDA")
    requests = []
    for seed in [17, 29, 43]:
        for rows, inner, cols in [(2, 32, 257), (3, 128, 1025)]:
            rng = random.Random(seed)
            lhs = [float(rng.randint(-2, 2)) for _ in range(rows * inner)]
            rhs = [float(rng.randint(-3, 3)) for _ in range(inner * cols)]
            for kind in ["topk", "midk", "bottomk"]:
                requests.append(dict(rows=rows, inner=inner, cols=cols, k=7, kind=kind, seed=seed, lhs=lhs, rhs=rhs))
    payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r in requests)
    with tempfile.TemporaryDirectory(prefix="matmul-rank-execution-", dir=executable.parent) as directory:
        image = Path(directory) / "resident_matmul_rank_bench"
        os.link(executable, image)
        image_before = audit.file_identity(image)
        identity = audit.read_native_build_identity(image)
        binding = audit.validate_source_binding(identity, before)
        if not binding["valid"]:
            raise RuntimeError(f"source/build mismatch: {binding}")
        results = run_native_pass(image, payload, resident_only=resident_only)
        if len(results) != len(requests):
            raise RuntimeError("native result cardinality mismatch")
        report = dict(schema="spiraltorch.matmul_rank_comparison.v1", status="passed",
            source=before, native_build_identity=identity, build_source_binding=binding,
            request_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            torch=str(torch.__version__), torch_device=torch.cuda.get_device_name(),
            dtype="float32", tf32=bool(torch.backends.cuda.matmul.allow_tf32), torch_cpu_threads=torch.get_num_threads(),
            comparison="bounded integer projection heads; frameworks measured in separate blocks",
            readback_probe_requested=readback_probe,
            resident_only_requested=resident_only,
            boundaries={
                "host_bridge":"resident operands; matmul, full intermediate map, rank upload/dispatch, final rank map",
                "device_copy_bridge":"resident operands; matmul, GPU-local intermediate copy, rank dispatch, final rank map",
                "resident_copy_bridge_per_op":"16 matmul/copy/rank chains then completion fence; divided by 16, no maps",
                "single_submit_bridge":"resident operands; one matmul/copy/rank submission, final rank map",
                "resident_single_submit_per_op":"16 calls of one matmul/copy/rank submission then completion fence; divided by 16, no maps",
                "resident_batched_submit_per_op":"16 complete matmul/copy/rank chains in one submission then completion fence; divided by 16, no maps",
                "torch_resident_per_op":"16 matmul/stable-sort chains with preallocated CUDA outputs then synchronize; divided by 16, no maps",
            }, cases=[])
        with torch.inference_mode():
            for r, result in zip(requests, results):
                if result.get("status") != "passed" or any(result.get(k) != r[k] for k in ("rows", "inner", "cols", "k", "kind", "seed")):
                    raise RuntimeError("native result/request mismatch")
                if resident_only and (result.get("mode") != "resident_only"
                        or set(result.get("samples_ms", {})) != {"resident_single_submit_per_op"}):
                    raise RuntimeError("native resident-only boundary mismatch")
                if result["adapter"]["name"] != torch.cuda.get_device_name():
                    raise RuntimeError("WGPU and CUDA must select the same named GPU")
                left = torch.tensor(r["lhs"], dtype=torch.float32).reshape(r["rows"], r["inner"])
                right = torch.tensor(r["rhs"], dtype=torch.float32).reshape(r["inner"], r["cols"])
                reference = left @ right
                ordered, ordered_ids = reference.sort(dim=1, descending=r["kind"] == "topk", stable=True)
                start = (r["cols"] - r["k"]) // 2 if r["kind"] == "midk" else 0
                expected = ordered[:, start:start+r["k"]]
                expected_ids = ordered_ids[:, start:start+r["k"]]
                if result["values"] != expected.flatten().tolist() or result["indices"] != expected_ids.flatten().tolist():
                    raise RuntimeError("native projection/rank differs from canonical PyTorch reference")
                left, right = left.cuda(), right.cuda()
                logits = torch.empty_like(reference, device="cuda")
                shape = logits.shape
                values = torch.empty(shape, dtype=torch.float32, device="cuda")
                indices = torch.empty(shape, dtype=torch.int64, device="cuda")

                def op():
                    torch.mm(left, right, out=logits)
                    torch.sort(logits, dim=1, descending=r["kind"] == "topk", stable=True, out=(values, indices))

                def repeated():
                    for _ in range(16):
                        op()

                op()
                bench.correctness(logits, reference.double(), torch, 0, 0)
                selected = values[:, start:start+r["k"]]
                bench.correctness(selected, expected.double(), torch, 0, 0)
                require_canonical_indices(indices[:, start:start+r["k"]].cpu().tolist(), expected_ids.tolist())
                if not torch.equal(logits.gather(1, indices), values):
                    raise RuntimeError("PyTorch indices do not refer to returned logits")
                timing = bench.paired_timings({"torch":repeated}, 2, 12, torch.cuda.synchronize, r["seed"])["torch"]
                samples = {name:bench.summarize(v) for name,v in result["samples_ms"].items()}
                samples["torch_resident_per_op"] = bench.summarize([v/16 for v in timing["samples_ms"]])
                report["cases"].append(dict(request={k:v for k,v in r.items() if k not in ("lhs","rhs")}, native=result, timings=samples,
                    cuda_rank_contract="stable sort; selected values and indices match the canonical reference"))
        if readback_probe:
            require_uncontended_gpu()
            report["readback_diagnostics"] = collect_readback_diagnostics(image, payload, requests, results)
        after = audit.source_identity()
        require_uncontended_gpu()
        report["gpu_process_gate"] = "no foreign compute PIDs at preflight/postflight; not an exclusive reservation"
        report["provenance"] = dict(valid=before == after and original == audit.file_identity(executable)
            and image_before == audit.file_identity(image), source_after=after, executable=original, execution_image=image_before)
        if not report["provenance"]["valid"]:
            raise RuntimeError("source or execution image changed during measurement")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readback-probe", action="store_true",
                        help="run separate diagnostic process AFTER all native/CUDA samples; not comparison timing")
    parser.add_argument("--resident-only", action="store_true",
                        help="measure only composed resident chains; no maps/uploads between timed intervals")
    args = parser.parse_args()
    try:
        report = run(args.executable.resolve(strict=True), args.readback_probe, args.resident_only)
    except Exception as error:
        report = dict(schema="spiraltorch.matmul_rank_comparison.v1", status="error", error=str(error))
    audit.write_report_exclusive(args.output, report)
    print(json.dumps(dict(status=report["status"], cases=len(report.get("cases", [])), error=report.get("error"))))
    raise SystemExit(report["status"] != "passed")


if __name__ == "__main__":
    main()
