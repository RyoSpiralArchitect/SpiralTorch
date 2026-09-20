"""Verify and freeze a clean CPU workspace candidate, without touching global settings."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root, target, output = map(Path, sys.argv[1:4])
output.mkdir()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
source = {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}")}
report = {"source": source, "steps": [], "products": {}, "status": "running"}
environment = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", PYTHONNOUSERSITE="1",
                   SPIRAL_DETERMINISTIC="0", SPIRAL_DETERMINISTIC_REDUCTION="0", RAYON_NUM_THREADS="4")

def check_source():
    assert git("rev-parse", "HEAD") == source["commit"]
    assert not git("status", "--porcelain"), "Source changed during verification"

def save():
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")

def run(name, command, overrides=None):
    check_source()
    command = list(map(str, command))
    env = dict(environment)
    env.pop("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS", None)
    env.update(overrides or {})
    print("START", name, flush=True)
    start = time.monotonic()
    with (output / (name + ".log")).open("x") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    report["steps"].append({"name": name, "command": command, "exit_code": result.returncode,
                            "seconds": time.monotonic() - start, "overrides": overrides or {}})
    save()
    print("END", name, result.returncode, flush=True)
    result.check_returncode()
    check_source()

def freeze(path, name):
    destination = output / name
    shutil.copy2(path, destination)
    report["products"][name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    save()
    return destination

check_source()
report["compiler"] = subprocess.check_output(["rustc", "+1.98.0", "--version", "--verbose"], text=True)
report["environment"] = {k: environment[k] for k in ["CARGO_BUILD_JOBS", "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION", "RAYON_NUM_THREADS"]}
save()
cargo = ["cargo", "+1.98.0"]
try:
    run("format", ["cargo", "+nightly-2026-04-15", "fmt", "--all", "--", "--check"])
    run("cpu-suite", cargo + ["test", "--locked", "-p", "st-tensor", "--no-default-features", "--features", "cpu,faer", "--lib", "--test", "prepacked_cpu_dispatch", "--test", "autograd_observability", "--test", "autograd_sgd", "--test", "classification", "--", "--test-threads=1"])
    run("cpu-serial", cargo + ["test", "--locked", "--release", "-p", "st-tensor", "--no-default-features", "--features", "cpu", "--lib", "backend::cpu_dense", "--", "--test-threads=1"], {"SPIRAL_DETERMINISTIC": "1", "SPIRAL_DETERMINISTIC_REDUCTION": "1", "RAYON_NUM_THREADS": "1"})
    run("selfsup-cpu", cargo + ["test", "--locked", "-p", "spiral-selfsup"])
    run("native-clippy", cargo + ["clippy", "--locked", "-p", "st-tensor", "-p", "spiral-selfsup", "-p", "st-bench", "--all-targets", "--no-deps", "--", "-D", "warnings"])
    run("wasm-clippy", cargo + ["clippy", "--locked", "-p", "st-tensor", "--target", "wasm32-unknown-unknown", "--all-targets", "--no-default-features", "--features", "cpu,wgpu", "--", "-D", "warnings"])
    run("wasm-build", cargo + ["build", "--locked", "-p", "spiraltorch-wasm", "--target", "wasm32-unknown-unknown", "--release"])
    bindgen = Path.home() / "Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.104/wasm-bindgen"
    run("wasm-bindgen", [bindgen, target / "wasm32-unknown-unknown/release/spiraltorch_wasm.wasm", "--target", "nodejs", "--out-dir", output / "wasm"])
    run("wasm-numerics", ["node", root / "bindings/st-wasm/tests/numeric_boundaries.cjs", output / "wasm/spiraltorch_wasm.js"])
    run("wasm-dense-autograd", ["node", root / "bindings/st-wasm/tests/cpu_dense_matmul.cjs", output / "wasm/spiraltorch_wasm.js"])
    run("python-build", cargo + ["build", "--locked", "--release", "-p", "spiraltorch-py", "--no-default-features", "--features", "python-default"])
    library = freeze(target / "release/libspiraltorch.dylib", "python-cpu.dylib")
    loader = Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py"
    run("python-numerics", [sys.executable, "-I", loader, library, root, "bindings/st-py/tests/test_source_crosscut.py"])
    run("selfsup-wgpu", cargo + ["test", "--locked", "--release", "-p", "spiral-selfsup", "--features", "wgpu", "--test", "info_nce_tensor", "--", "--test-threads=1"], {"SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS": "1"})
    run("benchmark-build", cargo + ["build", "--locked", "--release", "-p", "st-bench", "--example", "cpu_dense_workspace", "--example", "source_crosscut"])
    freeze(target / "release/examples/source_crosscut", "candidate-crosscut")
    freeze(target / "release/examples/cpu_dense_workspace", "candidate-dense")
    report["status"] = "passed"
except Exception as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    raise
finally:
    save()
