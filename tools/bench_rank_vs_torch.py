#!/usr/bin/env python3
"""Strict rank parity plus diagnostic Rust/Python host-boundary timings."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location("backend_bench", Path(__file__).with_name("bench_backend_vs_torch.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def requests():
    for backend in ("wgpu", "cuda"):
        for seed in (17, 29, 43):
            for kind in ("topk", "bottomk", "midk"):
                for cols, k, pattern in ((256, 8, "random"), (2048, 16, "concentrated")):
                    values, _ = bench.fixture(2 * cols, seed)
                    if pattern == "concentrated":
                        # All winners map to one heap thread; uniform inputs hide this.
                        for row in range(2):
                            for i in range(16):
                                values[row * cols + i * 128] = (100.0 + i) * (-1 if kind == "bottomk" else 1)
                    scripts = ["u2: false;"] if backend == "cuda" else ["u2: true;"]
                    if backend == "wgpu" and cols == 256:
                        scripts.insert(0, "u2: false;")
                    yield {"backend": backend, "kind": kind, "rows": 2, "cols": cols, "k": k,
                           "input": values, "iterations": 12, "warmup": 2, "seed": seed,
                           "scripts": scripts}, pattern


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["RAYON_NUM_THREADS"] = "1"
        import torch
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no fallback")
        cases = list(requests())
        payload = "".join(json.dumps(request) + "\n" for request, _ in cases)
        native = subprocess.run([str(args.executable)], input=payload, text=True, capture_output=True)
        results = [json.loads(line) for line in native.stdout.splitlines()]
        if len(results) != len(cases):
            raise RuntimeError(f"native process returned {len(results)}/{len(cases)} cases: {native.stderr[-4000:]}")
        report = {"schema": "spiraltorch.rank_backend_bench.v1", "torch": torch.__version__,
                  "device": torch.cuda.get_device_name(0), "request_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                  "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  "native_returncode": native.returncode, "native_stderr": native.stderr,
                  "comparison": "diagnostic unpaired timings; Rust and Python wrappers differ; no speed ratio", "cases": []}
        for (request, pattern), result in zip(cases, results):
            host = torch.tensor(request["input"], dtype=torch.float32).reshape(2, request["cols"])
            k, kind = request["k"], request["kind"]

            def torch_op():
                tensor = host.cuda()
                if kind == "midk":
                    values, indices = tensor.sort(dim=1, stable=True)
                    start = (request["cols"] - k) // 2
                    return values[:, start:start + k].cpu(), indices[:, start:start + k].cpu()
                values, indices = tensor.topk(k, dim=1, largest=kind == "topk", sorted=True)
                return values.cpu(), indices.cpu()

            expected, expected_ids = host.sort(dim=1, descending=kind == "topk", stable=True)
            start = (request["cols"] - k) // 2 if kind == "midk" else 0
            expected, expected_ids = expected[:, start:start + k], expected_ids[:, start:start + k]
            actual, _ = torch_op()
            bench.correctness(actual, expected.double(), torch, 0, 0)
            if result["status"] == "passed":
                if result["values"] != expected.flatten().tolist() or result["indices"] != expected_ids.flatten().tolist():
                    result = {"status": "error", "error": "native/PyTorch canonical rank mismatch", "native": result}
            timing = bench.paired_timings({"torch_cuda_host_to_host": torch_op}, 2, 12, torch.cuda.synchronize, request["seed"])
            case = {"request": {key: value for key, value in request.items() if key != "input"},
                    "pattern": pattern, "native": result, "torch_timings": timing}
            report["cases"].append(case)
            print(json.dumps({"backend": request["backend"], "kind": kind, "cols": request["cols"],
                              "seed": request["seed"], "status": result["status"]}), flush=True)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return int(native.returncode != 0 or any(case["native"]["status"] != "passed" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
