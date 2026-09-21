"""Verify immutable bytes and recorded gates, not a fresh numerical replay."""
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
    manifest = read(root,"manifest.json")
    require(manifest["schema"] == "spiraltorch.cpu_row_major.archive.v1", "Unknown archive schema")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root/"manifest.json"}
    require(files == set(manifest["files"]), "File set mismatch")
    for name, record in manifest["files"].items():
        part = PurePosixPath(name)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and not PureWindowsPath(name).drive and part.as_posix() == name, "Unsafe archive path")
        path = root/name
        require(path.resolve().is_relative_to(root), "Escaping symlink")
        data = path.read_bytes()
        require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], "Hash/size mismatch: "+name)
    extended = load(root/"measure_extended.py").report(root/"extended")
    dense = load(root/"measure_dense.py").report(root/"dense")
    wasm = load(root/"measure_wasm.py").report(root/"wasm")
    crosscut = load(root/"measure_crosscut.py").report(root/"crosscut")
    for name, result in [("extended",extended),("dense",dense),("wasm",wasm),("crosscut",crosscut)]:
        require(result == read(root,name+"/comparison.json"), "Recorded summary mismatch: "+name)
    provenance = read(root,"provenance.json")
    verification = read(root,"verification/receipt.json")
    names = ["format", "cpu-suite", "cpu-serial", "selfsup-cpu", "native-clippy", "wasm-clippy",
             "wasm-build", "wasm-bindgen", "wasm-numerics", "wasm-dense-autograd", "python-build",
             "python-numerics", "selfsup-wgpu", "benchmark-build"]
    require(verification["status"] == "passed" and verification["source"]["commit"] == provenance["candidate_commit"], "Verification source/status mismatch")
    require([s["name"] for s in verification["steps"]] == names and all(s["exit_code"] == 0 for s in verification["steps"]), "Incomplete verification")
    for name in ["portable-tests","simd"]:
        extra = read(root,"verification/"+name+".json")
        require(extra["source"] == provenance["candidate_commit"] and extra["exit_code"] == 0, "Additional test source/status mismatch")
    probe = read(root,"probe/receipt.json")
    require(probe["status"] == "passed" and all(s["exit_code"] == 0 for s in probe["steps"]), "Probe failed")
    require(probe["source_hashes"] == provenance["native_source_hashes"], "Selected probe source mismatch")
    for worker in ["candidate-dense","candidate-crosscut"]:
        require(probe["products"][worker] == verification["products"][worker], "Verified/measured worker mismatch")
    for directory, worker, base in [("extended","candidate-extended","baseline-extended"),("dense","candidate-dense","baseline-dense")]:
        receipt = read(root,directory+"/receipt.json")
        hashes = receipt["binaries"]
        require([value for path,value in hashes.items() if path.endswith("/"+worker)] == [probe["products"][worker]], "Candidate worker binding mismatch")
        require([value for path,value in hashes.items() if path.endswith("/"+base)] == [provenance["baseline_workers"][base]], "Baseline worker binding mismatch")
    require(read(root,"crosscut/measurement-receipt.json")["hashes"]["candidate-worker"] == verification["products"]["candidate-crosscut"], "Crosscut worker mismatch")
    wasm_receipt = read(root,"wasm/receipt.json")
    require(wasm_receipt["harness_sha256"] == provenance["wasm_harness_sha256"], "WASM harness mismatch")
    require(wasm_receipt["modules"]["candidate"]["wasm_sha256"] == provenance["wasm_products"]["candidate"], "WASM product mismatch")
    require(wasm_receipt["modules"]["baseline"]["wasm_sha256"] == provenance["wasm_products"]["baseline"], "WASM baseline mismatch")
    return dict(status="passed", files=len(files), numerical_replay=False,
        dense_condition_runs=dense["bitwise_passes"], extended_condition_runs=extended["measured_rust_condition_runs"],
        extended_preconditioning_runs=extended["preconditioning_condition_runs"],torch_conditions=extended["torch_condition_runs"],
        wasm_condition_runs=wasm["measured_condition_runs"],wasm_preconditioning_runs=wasm["preconditioning_condition_runs"],
        crosscut_candidate_passes=crosscut["candidate_numerical_passes"],crosscut_baseline_failures=crosscut["baseline_numerical_failures"])

if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
