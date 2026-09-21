"""Verify published bytes and recorded gates/aggregations, not numerical replay."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

SHAPES = [(1, 64), (8, 3072), (32, 3072), (64, 1024), (17, 195), (65, 97)]
NAMES = ["warm-baseline", "warm-candidate", "baseline-a", "candidate-a", "candidate-b", "baseline-b"]
STAGES = ["format", "tensor", "nn", "clippy", "wasm-clippy", "wasm-build", "bindgen",
          "wasm-gelu", "wasm-autograd", "python-build", "python-gelu", "python-autograd", "wgpu-layout"]
RUNTIME = ["crates/st-tensor/src/pure.rs", "crates/st-nn/src/layers/gelu.rs"]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())


def groups(rows, group_key):
    result = {}
    for name in sorted({group_key(row) for row in rows}):
        values = [row["ratio"] for row in rows if group_key(row) == name]
        result[name] = {"geomean": math.exp(statistics.mean(map(math.log, values))),
                        "min": min(values), "max": max(values), "favorable": sum(v > 1 for v in values), "count": len(values)}
    return result


def summarize(root):
    expected = {(r, c, b) for r, c in SHAPES for b in [False, True]}
    indexes = {}
    for name in NAMES:
        report = read(root / "gelu" / (name + ".json"))
        assert (report["warmups"], report["intervals"], report["repetitions"]) == (3, 15, 8)
        index = {(c["rows"], c["cols"], c["backward"]): c for c in report["cases"]}
        assert len(report["cases"]) == 12 and set(index) == expected
        for case in index.values():
            assert case["valid"] is True and len(case["elapsed_ns"]) == 15
            assert all(math.isfinite(v) and v > 0 for v in case["elapsed_ns"])
            assert len(case["output_sha256"]) == 64
        indexes[name] = index
    for condition in expected:
        assert len({index[condition]["output_sha256"] for index in indexes.values()}) == 1
    comparisons = []
    for condition in sorted(expected):
        for round_id in ["a", "b"]:
            a, b = [indexes[lane + "-" + round_id][condition] for lane in ["baseline", "candidate"]]
            assert (a["allocation_calls"], b["allocation_calls"]) == (4, 3)
            assert a["allocated_bytes"] - b["allocated_bytes"] == condition[0] * condition[1] * 4
            comparisons.append({"condition": condition, "round": round_id,
                                "ratio": statistics.median(a["elapsed_ns"]) / statistics.median(b["elapsed_ns"])})
    torch = read(root / "torch.json")
    assert torch["threads"] == 4 and torch["interop_threads"] == 1
    assert (torch["warmups"], torch["intervals"], torch["repetitions"]) == (3, 15, 8)
    assert [r["round"] for r in torch["reports"]] == ["warm", "a", "b"]
    torch_comparisons = []
    for report in torch["reports"]:
        assert len(report["cases"]) == 12
        assert {(c["rows"], c["cols"], c["backward"]) for c in report["cases"]} == expected
        for case in report["cases"]:
            assert case["valid"] is True and len(case["elapsed_ns"]) == 15
            assert math.isfinite(case["max_abs"])
            assert all(math.isfinite(v) and v > 0 for v in case["elapsed_ns"])
            if report["round"] != "warm":
                key = (case["rows"], case["cols"], case["backward"])
                candidate = indexes["candidate-" + report["round"]][key]
                torch_comparisons.append({"condition": key, "round": report["round"],
                                          "ratio": statistics.median(case["elapsed_ns"]) / statistics.median(candidate["elapsed_ns"])})
    path = root / "reproduction/nn_measure.py"
    spec = importlib.util.spec_from_file_location("nn_measure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nn = module.compare(root / "nn", 90)
    nn_expected = set()
    for rows, inner, cols in [(1, 64, 64), (8, 768, 3072), (32, 768, 3072), (64, 256, 1024), (17, 137, 195), (65, 1025, 97)]:
        nn_expected.update((op, rows, inner, cols, None, None) for op in ["pack", "transpose", "column_transpose_pack"])
        nn_expected.update((op, rows, inner, cols, backend, invalid) for op in ["linear", "mlp"] for backend in ["auto", "faer", "cpu_simd"] for invalid in [False, True])
    assert {tuple(row["condition"]) for row in nn["comparisons"]} == nn_expected
    return {"gelu": {"comparisons": comparisons, "groups": groups(comparisons, lambda row: str(row["condition"][2]))},
            "torch": {"comparisons": torch_comparisons, "groups": groups(torch_comparisons, lambda row: str(row["condition"][2]))},
            "nn_groups": nn["groups"]}


def verify(root):
    manifest = read(root / "manifest.json")
    seen = set()
    for row in manifest["files"]:
        path = (root / row["path"]).resolve(strict=True)
        assert path.is_relative_to(root.resolve()) and path.relative_to(root.resolve()).as_posix() == row["path"]
        assert row["path"] not in seen
        seen.add(row["path"])
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], row["path"]
    assert seen == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"manifest.json"}
    provenance = read(root / "provenance.json")
    followup = provenance["ci_followup"]
    assert followup["parent"] == provenance["validated_source"]
    assert followup["sha256"] == sha(root / "ci-followup.patch")
    changed_lines = [line for line in (root / "ci-followup.patch").read_text().splitlines()
                     if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    assert changed_lines == ['+          node bindings/st-wasm/tests/gelu_host.cjs "$module"',
                             '+          python -P bindings/st-py/tests/test_gelu_host.py']
    baseline, candidate = [read(root / (lane + "-build/receipt.json")) for lane in ["baseline", "candidate"]]
    for lane, receipt in [("baseline", baseline), ("candidate", candidate)]:
        assert receipt["exit_code"] == 0 and receipt["source_unchanged"] and receipt["source"]["status"] == ""
        assert receipt["source"]["commit"] == provenance[lane + "_source"]
    for path in ["crates/st-bench/examples/cpu_gelu.rs", "crates/st-bench/examples/cpu_nn_layout.rs"]:
        assert baseline["source"]["files"][path] == candidate["source"]["files"][path]
        assert sha(root / "reproduction" / Path(path).name) == candidate["source"]["files"][path]
    stages = read(root / "verification-final/stages.json")
    assert stages == [{"name": name, "exit_code": 0} for name in STAGES]
    for name in STAGES:
        receipt = read(root / "verification-final" / name / "receipt.json")
        assert receipt["exit_code"] == 0 and receipt["source_unchanged"]
        assert receipt["source"]["commit"] == provenance["validated_source"]
        assert all(receipt["source"]["files"][f] == candidate["source"]["files"][f] for f in RUNTIME)
    gpu = read(root / "verification-final/wgpu-layout/receipt.json")
    assert "SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1" in gpu["command"]
    assert "strict_wgpu_backward_pairs_logical_layouts ... ok" in (root / "verification-final/wgpu-layout/stdout.log").read_text()
    before = read(root / "regression-before/receipt.json")
    assert before["source"]["commit"] == provenance["baseline_source"] and before["exit_code"] == 101
    assert "0 passed; 2 failed" in (root / "regression-before/stdout.log").read_text()
    assert read(root / "verification/python-gelu/receipt.json")["exit_code"] == 1
    assert "ValueError: non-finite" in (root / "verification/python-gelu/stderr.log").read_text()
    for directory, worker in [("gelu", "cpu_gelu"), ("nn", "cpu_nn_layout")]:
        receipt = read(root / directory / "receipt.json")
        hashes = receipt["workers"] if directory == "gelu" else receipt["sha256"]
        assert all(hashes[lane] == provenance["products"][lane + "/" + worker] for lane in ["baseline", "candidate"])
        assert len(receipt["steps"]) == 6 and all(step["exit_code"] == 0 for step in receipt["steps"])
    computed = json.loads(json.dumps(summarize(root)))
    assert computed == read(root / "summary.json")
    return {"status": "passed", "files": len(seen), "gelu_conditions": 48, "nn_conditions": 360,
            "torch_conditions": 24, "verification_stages": 13, "numerical_replay": False}


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O; checks require assertions.")
    print(json.dumps(verify(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent), indent=2))
