#!/usr/bin/env python3
"""Compare actual single-threaded WASM tensors with PyTorch CPU, not WebGPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess

spec = importlib.util.spec_from_file_location("backend_bench", Path(__file__).with_name("bench_backend_vs_torch.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def run(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    requests, expected, torch_fns = [], [], []
    operations = ["matmul", "gather", "scatter"]
    if args.prepacked:
        operations.insert(1, "matmul_prepacked")
    if args.reverse_operations:
        operations.reverse()
    for operation in operations:
        for size in (32, 64, 128):
            for seed in (17, 29, 43):
                values, lhs_sha = bench.fixture(size * size, seed)
                other, rhs_sha = bench.fixture(size * size, seed + 1)
                # Duplicate IDs exercise actual reduction, not just permutation.
                ids = [(i * 17 + seed) % (size // 2) for i in range(size)]
                request = {"operation": operation, "input_rows": size, "input_cols": size,
                           "output_rows": size, "output_cols": size, "values": values,
                           "other": other, "ids": ids, "iterations": 20, "warmup": 5, "seed": seed,
                           "input_sha256": [lhs_sha, rhs_sha]}
                requests.append(request)
                a = torch.tensor(values, dtype=torch.float32).reshape(size, size)
                b = torch.tensor(other, dtype=torch.float32).reshape(size, size)
                index = torch.tensor(ids, dtype=torch.int64)
                if operation in ("matmul", "matmul_prepacked"):
                    fn = lambda a=a, b=b: a @ b
                    reference = a.double() @ b.double()
                elif operation == "gather":
                    fn = lambda a=a, index=index: a.index_select(0, index)
                    reference = fn().double()
                else:
                    fn = lambda a=a, index=index: torch.zeros_like(a).index_add_(0, index, a)
                    reference = torch.zeros_like(a, dtype=torch.float64).index_add_(0, index, a.double())
                expected.append(reference)
                torch_fns.append(fn)
    payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r in requests)
    child = subprocess.run([args.node, str(root / "bindings/st-wasm/tests/backend_bench.cjs"), str(args.module.resolve())],
                           input=payload, text=True, capture_output=True)
    results = [json.loads(line) for line in child.stdout.splitlines()]
    if len(results) != len(requests):
        raise RuntimeError(f"WASM returned {len(results)}/{len(requests)} results: {child.stderr[-4000:]}")
    report = {"schema": "spiraltorch.wasm_cpu_bench.v2", "host": platform.node(),
              "platform": platform.platform(), "torch": torch.__version__, "threads": 1,
              "module": str(args.module.resolve()), "wasm_sha256": hashlib.sha256(args.module.with_name("spiraltorch_wasm_bg.wasm").read_bytes()).hexdigest(),
              "request_sha256": hashlib.sha256(payload.encode()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "node_harness_sha256": hashlib.sha256((root / "bindings/st-wasm/tests/backend_bench.cjs").read_bytes()).hexdigest(),
              "operation_order": operations,
              "native_stderr": child.stderr, "wasm_returncode": child.returncode,
              "comparison": "unpaired runtime call timings; WASM CPU, not browser WebGPU", "cases": []}
    for r, result, reference, fn in zip(requests, results, expected, torch_fns):
        case = {"operation": r["operation"], "size": r["input_rows"], "seed": r["seed"],
                "input_sha256": r["input_sha256"], "wasm": result}
        if result["status"] == "passed":
            values = result.pop("values")
            case["wasm_correctness"] = bench.correctness(torch.tensor(values, dtype=torch.float32).reshape(reference.shape), reference, torch, 1e-4, 1e-5)
            result["timings"] = bench.summarize(result.pop("samples_ms"))
        case["torch_correctness"] = bench.correctness(fn(), reference, torch, 1e-4, 1e-5)
        case["torch_timings"] = bench.paired_timings({"torch_cpu": fn}, 5, 20, lambda: None, r["seed"])
        report["cases"].append(case)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True, help="wasm-bindgen nodejs spiraltorch_wasm.js")
    parser.add_argument("--node", default="node")
    parser.add_argument("--prepacked", action="store_true", help="also exercise the Rust-owned autograd packed RHS")
    parser.add_argument("--reverse-operations", action="store_true", help="reverse operation order for warmup/order diagnostics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        try:
            report = run(args)
        except Exception as error:
            report = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return int("cases" not in report or report.get("wasm_returncode", 1) != 0
               or any(c["wasm"]["status"] != "passed" for c in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
