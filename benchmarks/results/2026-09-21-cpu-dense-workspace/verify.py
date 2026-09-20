"""Verify archived bytes, complete condition grids and recorded gates, not runtime replay."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

sys.dont_write_bytecode = True

def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "spiraltorch.cpu_dense_workspace.archive.v1":
        raise ValueError("Unknown archive schema")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    if files != set(manifest["files"]):
        raise ValueError("Archive file set mismatch")
    for name, record in manifest["files"].items():
        part = PurePosixPath(name)
        if part.is_absolute() or ".." in part.parts or "\\" in name or PureWindowsPath(name).drive or part.as_posix() != name:
            raise ValueError("Unsafe archive path")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("Symlink escapes archive")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Hash/size mismatch: " + name)
    dense = load(root / "measure_dense.py").report(root / "dense")
    crosscut = load(root / "measure_crosscut.py").report(root / "crosscut")
    provenance = json.loads((root / "provenance.json").read_text())
    receipt = json.loads((root / "verification/receipt.json").read_text())
    names = ["format", "cpu-suite", "cpu-serial", "selfsup-cpu", "native-clippy", "wasm-clippy",
             "wasm-build", "wasm-bindgen", "wasm-numerics", "wasm-dense-autograd", "python-build",
             "python-numerics", "selfsup-wgpu", "benchmark-build"]
    if receipt["status"] != "passed" or receipt["source"]["commit"] != provenance["candidate_commit"]:
        raise ValueError("Verification source/status mismatch")
    if [s["name"] for s in receipt["steps"]] != names or any(s["exit_code"] != 0 for s in receipt["steps"]):
        raise ValueError("Incomplete verification steps")
    extra = json.loads((root / "verification/portable-tests.json").read_text())
    if extra["source"] != provenance["candidate_commit"] or extra["exit_code"] != 0:
        raise ValueError("Portable dispatch verification mismatch")
    return dict(files=len(files), dense_bitwise_condition_runs=dense["bitwise_passes"],
                crosscut_candidate_numerical_passes=crosscut["candidate_numerical_passes"],
                baseline_numerical_failures=crosscut["baseline_numerical_failures"],
                numerical_replay=False, status="passed")

if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
