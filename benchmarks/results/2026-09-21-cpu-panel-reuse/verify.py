"""Check archive integrity and recorded gates; this does not rerun numerical kernels."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

sys.dont_write_bytecode = True

def require(value, message):
    if not value:
        raise ValueError(message)

def read(root, name):
    return json.loads((root / name).read_text())

def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify(root):
    root = root.resolve()
    manifest = read(root, "manifest.json")
    require(manifest["schema"] == "spiraltorch.cpu_panel_reuse.archive.v1", "Wrong schema")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    require(files == set(manifest["files"]), "Archive file set mismatch")
    for name, record in manifest["files"].items():
        part = PurePosixPath(name)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and not PureWindowsPath(name).drive and part.as_posix() == name, "Unsafe path")
        path = root / name
        require(path.resolve().is_relative_to(root), "Symlink escapes archive")
        data = path.read_bytes()
        require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], "Hash/size mismatch: "+name)
    extended = load(root / "measure_extended.py").report(root / "extended")
    require(extended == read(root, "extended/comparison.json"), "Extended summary mismatch")
    dense = load(root / "measure_dense.py").report(root / "dense-initial")
    crosscut = load(root / "measure_crosscut.py").report(root / "crosscut")
    require(crosscut == read(root, "crosscut/comparison.json"), "Crosscut summary mismatch")
    provenance = read(root, "provenance.json")
    receipt = read(root, "verification/receipt.json")
    names = ["format", "cpu-suite", "cpu-serial", "selfsup-cpu", "native-clippy", "wasm-clippy",
             "wasm-build", "wasm-bindgen", "wasm-numerics", "wasm-dense-autograd", "python-build",
             "python-numerics", "selfsup-wgpu", "benchmark-build"]
    require(receipt["status"] == "passed" and receipt["source"]["commit"] == provenance["candidate_commit"], "Verification source/status mismatch")
    require([s["name"] for s in receipt["steps"]] == names and all(s["exit_code"] == 0 for s in receipt["steps"]), "Incomplete verification")
    extra = read(root, "verification/portable-tests.json")
    require(extra["source"] == provenance["candidate_commit"] and extra["exit_code"] == 0, "Portable verification mismatch")
    build = read(root, "build/receipt.json")
    require(build["status"] == "passed" and [s["label"] for s in build["steps"]] == ["baseline", "candidate"], "Wrong build receipt")
    baseline, candidate = build["steps"]
    require(baseline["commit"] == provenance["baseline_commit"] and candidate["commit"] == provenance["candidate_commit"], "Build source mismatch")
    require(all(s["exit_code"] == 0 for s in build["steps"]) and baseline["binary_sha256"] != candidate["binary_sha256"], "Identical binaries or failed build")
    require({s["binary_sha256"] for s in build["steps"]} == set(read(root, "extended/receipt.json")["binaries"].values()), "Measurement/build binding mismatch")
    require(candidate["source_sha256"] == provenance["cpu_dense_sha256"], "Production source hash mismatch")
    require(build["harness_sha256"] == provenance["harness_hashes"]["cpu_dense_extended.rs"], "Harness identity mismatch")
    crosscut_receipt = read(root, "crosscut/measurement-receipt.json")
    require(crosscut_receipt["hashes"]["candidate-worker"] == receipt["products"]["candidate-crosscut"], "Crosscut worker mismatch")
    probe = read(root, "probe/receipt.json")
    require(probe["products"]["candidate-dense"] == receipt["products"]["candidate-dense"], "Initial/verified worker mismatch")
    dense_receipt = read(root, "dense-initial/receipt.json")
    dense_candidate = [value for name, value in dense_receipt["binaries"].items() if name.endswith("/candidate-dense")]
    require(dense_candidate == [receipt["products"]["candidate-dense"]], "Dense worker mismatch")
    return dict(status="passed", files=len(files), numerical_replay=False,
                dense_bitwise_condition_runs=dense["bitwise_passes"],
                extended_measured_runs=extended["measured_rust_condition_runs"],
                extended_preconditioning_runs=extended["preconditioning_condition_runs"],
                extended_torch_runs=extended["torch_condition_runs"],
                crosscut_candidate_passes=crosscut["candidate_numerical_passes"],
                crosscut_baseline_failures=crosscut["baseline_numerical_failures"])

if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
