"""Serial source-bound checks; each stage retains stdout, stderr and its receipt."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root, target, output = map(Path, sys.argv[1:4])
output.mkdir()
driver = Path(__file__).with_name("run.py")
cargo = ["cargo", "+1.98.0"]
steps = [
    ("format", ["cargo", "+nightly-2026-04-15", "fmt", "--all", "--", "--check"]),
    ("tensor", cargo + ["test", "--locked", "-p", "st-tensor", "--no-default-features", "--features", "cpu,faer", "--lib", "--test", "autograd_observability", "--test", "autograd_sgd", "--test", "classification", "--", "--test-threads=1"]),
    ("nn", cargo + ["test", "--locked", "-p", "st-nn", "--lib", "--test", "gelu_layout_contract", "--test", "linear_layout_contract", "--", "--test-threads=1"]),
    ("clippy", cargo + ["clippy", "--locked", "-p", "st-tensor", "-p", "st-bench", "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("wasm-clippy", cargo + ["clippy", "--locked", "-p", "st-tensor", "--target", "wasm32-unknown-unknown", "--all-targets", "--no-default-features", "--features", "cpu,wgpu", "--", "-D", "warnings"]),
    ("wasm-build", cargo + ["build", "--locked", "-p", "spiraltorch-wasm", "--target", "wasm32-unknown-unknown", "--release"]),
    ("bindgen", [str(Path.home() / "Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.104/wasm-bindgen"), str(target / "wasm32-unknown-unknown/release/spiraltorch_wasm.wasm"), "--target", "nodejs", "--out-dir", str(output / "wasm")]),
    ("wasm-gelu", ["node", str(root / "bindings/st-wasm/tests/gelu_host.cjs"), str(output / "wasm/spiraltorch_wasm.js")]),
    ("wasm-autograd", ["node", str(root / "bindings/st-wasm/tests/cpu_dense_matmul.cjs"), str(output / "wasm/spiraltorch_wasm.js")]),
    ("python-build", cargo + ["build", "--locked", "--release", "-p", "spiraltorch-py", "--no-default-features", "--features", "python-default"]),
    ("python-gelu", [sys.executable, "-I", str(Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py"), str(output / "python-cpu.dylib"), str(root), "bindings/st-py/tests/test_gelu_host.py"]),
    ("python-autograd", [sys.executable, "-I", str(Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py"), str(output / "python-cpu.dylib"), str(root), "bindings/st-py/tests/test_autograd_prepacked.py"]),
    ("wgpu-layout", ["env", "SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"] + cargo + ["test", "--locked", "--release", "-p", "st-nn", "--features", "wgpu", "--test", "gelu_layout_contract", "--", "--test-threads=1"]),
]
completed = []
for name, command in steps:
    print("START", name, flush=True)
    result = subprocess.run([sys.executable, "-B", "-I", str(driver), str(root), str(target), str(output / name), *command], stdout=subprocess.DEVNULL)
    completed.append({"name": name, "exit_code": result.returncode})
    (output / "stages.json").write_text(json.dumps(completed, indent=2) + "\n")
    result.check_returncode()
    if name == "python-build":
        shutil.copy2(target / "release/libspiraltorch.dylib", output / "python-cpu.dylib")
    print("END", name, flush=True)
