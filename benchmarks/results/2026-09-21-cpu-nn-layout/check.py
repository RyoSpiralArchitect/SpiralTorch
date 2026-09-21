"""Run recorded validation without interpreting a successful build as test coverage."""
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
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4",
           SPIRAL_DETERMINISTIC="0", SPIRAL_DETERMINISTIC_REDUCTION="0",
           SPIRALTORCH_AUTOTUNE_STORE=str(output / "tuning.json"), SPIRALTORCH_AUTOTUNE="1")
env.pop("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS", None)
report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
          "steps": [], "status": "running"}
files = ["crates/st-tensor/src/pure.rs", "crates/st-tensor/src/backend/faer_dense.rs",
         "crates/st-tensor/src/backend/transpose.rs", "crates/st-tensor/src/backend/mod.rs",
         "crates/st-nn/tests/linear_layout_contract.rs", "crates/st-bench/examples/cpu_nn_layout.rs",
         "bindings/st-py/tests/test_autograd_prepacked.py", "bindings/st-wasm/tests/cpu_layout_bench.cjs"]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report["source_hashes"] = {file: sha(root / file) for file in files}
report["compiler"] = subprocess.check_output(["rustc", "+1.98.0", "-Vv"], text=True)
report["environment"] = {name: env[name] for name in ["RAYON_NUM_THREADS", "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION", "SPIRALTORCH_AUTOTUNE", "SPIRALTORCH_AUTOTUNE_STORE"]}
def check_source():
    assert all(sha(root / file) == expected for file, expected in report["source_hashes"].items())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == report["commit"]

cargo = ["cargo", "+1.98.0"]
steps = [
    ("format", ["cargo", "+nightly-2026-04-15", "fmt", "--all", "--", "--check"]),
    ("tensor-cpu", cargo + ["test", "--locked", "-p", "st-tensor", "--no-default-features", "--features", "cpu,faer", "--lib", "--test", "autograd_observability", "--test", "autograd_sgd", "--test", "classification", "--", "--test-threads=1"]),
    ("linear", cargo + ["test", "--locked", "-p", "st-nn", "--lib", "layers::linear::tests", "--", "--test-threads=1"]),
    ("linear-layout", cargo + ["test", "--locked", "-p", "st-nn", "--test", "linear_layout_contract", "--", "--test-threads=1"]),
    ("native-clippy", cargo + ["clippy", "--locked", "-p", "st-tensor", "-p", "st-bench", "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("wasm-clippy", cargo + ["clippy", "--locked", "-p", "st-tensor", "--target", "wasm32-unknown-unknown", "--all-targets", "--no-default-features", "--features", "cpu,wgpu", "--", "-D", "warnings"]),
    ("wasm-build", cargo + ["build", "--locked", "-p", "spiraltorch-wasm", "--target", "wasm32-unknown-unknown", "--release"]),
    ("wasm-bindgen", [str(Path.home() / "Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.104/wasm-bindgen"), str(target / "wasm32-unknown-unknown/release/spiraltorch_wasm.wasm"), "--target", "nodejs", "--out-dir", str(output / "wasm")]),
    ("wasm-numerics", ["node", str(root / "bindings/st-wasm/tests/numeric_boundaries.cjs"), str(output / "wasm/spiraltorch_wasm.js")]),
    ("wasm-autograd", ["node", str(root / "bindings/st-wasm/tests/cpu_dense_matmul.cjs"), str(output / "wasm/spiraltorch_wasm.js")]),
    ("nn-cpu", cargo + ["test", "--locked", "-p", "st-nn", "--lib", "--", "--test-threads=1"]),
    ("python-build", cargo + ["build", "--locked", "--release", "-p", "spiraltorch-py", "--no-default-features", "--features", "python-default"]),
    ("python-autograd", [sys.executable, "-I", str(Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py"), str(output / "python-cpu.dylib"), str(root), "bindings/st-py/tests/test_autograd_prepacked.py"]),
    ("wgpu-probe", cargo + ["test", "--locked", "--release", "-p", "spiral-selfsup", "--features", "wgpu", "--test", "info_nce_tensor", "--", "--test-threads=1"]),
    ("wgpu-prepacked", cargo + ["test", "--locked", "--release", "-p", "st-tensor", "--no-default-features", "--features", "cpu,faer,wgpu", "--lib", "pure::tests::matmul_prepacked", "--", "--test-threads=1"]),
]
try:
    for name, command in steps:
        check_source()
        start = time.monotonic()
        print("START", name, flush=True)
        stage_env = dict(env)
        if name.startswith("wgpu-"):
            stage_env["SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"] = "1"
        with (output / (name + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=root, env=stage_env, stdout=log, stderr=subprocess.STDOUT)
        report["steps"].append({"name": name, "command": command, "exit_code": result.returncode, "seconds": time.monotonic() - start})
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print("END", name, result.returncode, flush=True)
        result.check_returncode()
        if name == "python-build":
            shutil.copy2(target / "release/libspiraltorch.dylib", output / "python-cpu.dylib")
        check_source()
    report["products"] = {str(path.relative_to(output)): sha(path) for path in
                          [output / "python-cpu.dylib", output / "wasm/spiraltorch_wasm.js", output / "wasm/spiraltorch_wasm_bg.wasm"]}
    report["status"] = "passed"
except Exception as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    raise
finally:
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
