"""Verify archive bytes, full grids and recorded validation, not fresh numerics."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

def read(root, name):
    return json.loads((root / name).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(root, filename, name):
    spec = importlib.util.spec_from_file_location(name, root / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify(root):
    manifest = read(root, "manifest.json")
    assert manifest["schema"] == "spiraltorch.cpu_nn_layout.archive.v1"
    assert not any(path.is_symlink() for path in root.rglob("*"))
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != root / "manifest.json"}
    assert actual == set(manifest["files"]), "archive file set mismatch"
    for name, item in manifest["files"].items():
        path = root / name
        assert not Path(name).is_absolute() and ".." not in Path(name).parts and not path.is_symlink()
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], name
    measurement = load(root, "measure.py", "layout_measurement")
    native_shapes = [(1, 64, 64), (8, 768, 3072), (32, 768, 3072), (64, 256, 1024), (17, 137, 195), (65, 1025, 97)]
    native_grid = {(op, r, k, c, None, None) for r, k, c in native_shapes for op in ["pack", "transpose", "column_transpose_pack"]}
    native_grid |= {(op, r, k, c, backend, invalidate) for r, k, c in native_shapes for op in ["linear", "mlp"]
                    for backend in ["auto", "faer", "cpu_simd"] for invalidate in [False, True]}
    wasm_grid = {(op, r, None, c, None, None) for r, c in [(64, 64), (768, 3072), (3072, 768), (256, 1024), (137, 195), (1025, 97)] for op in ["pack", "transpose"]}
    for directory, grid in [("native", native_grid), ("preflight/native", native_grid), ("preflight/native-repeat", native_grid), ("wasm", wasm_grid), ("preflight/wasm", wasm_grid)]:
        result = measurement.compare(root / directory, len(grid))
        assert json.loads(json.dumps(result)) == read(root, directory + "/comparison.json"), "aggregation mismatch"
        for name in ["baseline-a", "candidate-a", "baseline-b", "candidate-b", "warm-baseline", "warm-candidate"]:
            cases = read(root, f"{directory}/{name}.json")["cases"]
            assert {measurement.key(case) for case in cases} == grid, "condition grid mismatch"
            for case in cases:
                if "max_abs" in case:
                    assert math.isfinite(case["max_abs"]) and case["max_abs"] >= 0
                    if case["operation"] in ["pack", "transpose", "column_transpose_pack"]:
                        assert case["f32_reference_bits_equal"] is True
                else:
                    assert re.fullmatch("[0-9a-f]{64}", case["output_sha256"])
        receipt = read(root, directory + "/receipt.json")
        assert len(receipt["steps"]) == 6 and all(step["exit_code"] == 0 for step in receipt["steps"])
    torch = read(root, "torch.json")
    assert {(case["operation"], case["rows"], case["inner"], case["cols"]) for case in torch["cases"]} == {
        (op, r, k, c) for r, k, c in native_shapes for op in ["linear", "mlp"]}
    for case in torch["cases"]:
        assert case["valid"] is True and math.isfinite(case["max_abs"])
        assert len(case["elapsed_ns"]) == 9 and all(math.isfinite(v) and v > 0 for v in case["elapsed_ns"])
    assert read(root, "torch-receipt.json")["exit_code"] == 0
    summary = load(root, "summarize_torch.py", "layout_torch_summary").summarize(root)
    assert summary == read(root, "torch-comparison.json")
    provenance = read(root, "provenance.json")
    verification = read(root, "verification/receipt.json")
    required = {"format", "tensor-cpu", "linear", "linear-layout", "native-clippy", "wasm-clippy", "wasm-build", "wasm-bindgen",
                "wasm-numerics", "wasm-autograd", "nn-cpu", "python-build", "python-autograd", "wgpu-probe", "wgpu-prepacked"}
    assert verification["status"] == "passed" and len(verification["steps"]) == len(required)
    assert {step["name"] for step in verification["steps"]} == required
    assert all(step["exit_code"] == 0 for step in verification["steps"])
    assert verification["commit"] == provenance["candidate_commit"]
    assert verification["source_hashes"] == provenance["source_hashes"]
    for name, build in [("baseline", "build-baseline"), ("candidate", "build-candidate")]:
        receipt = read(root, build + "/receipt.json")
        assert receipt["exit_code"] == 0
        assert receipt["worker_sha256"] == read(root, "native/receipt.json")["sha256"][name]
        assert receipt["source"]["commit"] == provenance[name + "_commit"]
        if name == "candidate":
            assert all(provenance["source_hashes"][file] == value for file, value in receipt["source"]["files"].items())
    baseline_build = read(root, "build-baseline/receipt.json")
    assert baseline_build["source"]["files"]["crates/st-bench/examples/cpu_nn_layout.rs"] == digest(root / "cpu_nn_layout.rs")
    for directory, build in [("preflight/native", "preflight/build"), ("preflight/native-repeat", "preflight/build-repeat")]:
        assert read(root, directory + "/receipt.json")["sha256"]["baseline"] == baseline_build["worker_sha256"]
        assert read(root, directory + "/receipt.json")["sha256"]["candidate"] == read(root, build + "/receipt.json")["worker_sha256"]
    assert read(root, "preflight/wasm/receipt.json")["sha256"]["candidate_wasm"] == read(root, "preflight/verification/receipt.json")["products"]["wasm/spiraltorch_wasm_bg.wasm"]
    assert digest(root / "cpu_nn_layout.rs") == provenance["source_hashes"]["crates/st-bench/examples/cpu_nn_layout.rs"]
    assert digest(root / "cpu_layout_bench.cjs") == provenance["source_hashes"]["bindings/st-wasm/tests/cpu_layout_bench.cjs"]
    hashes = read(root, "wasm/receipt.json")["sha256"]
    assert hashes["candidate_wasm"] == verification["products"]["wasm/spiraltorch_wasm_bg.wasm"]
    assert hashes["baseline_wasm"] == provenance["baseline_wasm_sha256"]
    assert hashes["candidate_wasm"] != hashes["baseline_wasm"]
    assert hashes["harness"] == digest(root / "cpu_layout_bench.cjs")
    return {"status": "passed", "files": len(actual), "native_conditions": 360, "native_preconditioning": 180,
            "retained_preflight_native_conditions": 720, "retained_preflight_wasm_conditions": 48,
            "wasm_conditions": 48, "wasm_preconditioning": 24,
            "torch_conditions": 12, "numerical_replay": False}

if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent), indent=2))
