"""Serial checks; full output and source identity retained at each boundary."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root, target, output = map(Path, sys.argv[1:4])
output.mkdir(exist_ok=True)
driver = Path(__file__).with_name("run.py")
client_target = target / "positional-geometry-clients-v1"
cargo = ["cargo", "+1.98.0"]
steps = [
    ("format", ["cargo", "+nightly-2026-04-15", "fmt", "--all", "--", "--check"]),
    ("vision-clippy", cargo + ["clippy", "--locked", "-p", "st-vision", "--features", "nerf",
        "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("contracts", cargo + ["test", "--locked", "--release", "-p", "st-vision", "-p", "st-nn", "-p", "st-core",
        "--features", "st-vision/nerf", "--lib", "--test", "nerf_geometry", "--test", "nerf_regression",
        "--test", "linear_layout_contract", "--", "--nocapture", "--test-threads=1"]),
    ("changed-crates-clippy", cargo + ["clippy", "--locked", "-p", "st-nn", "-p", "st-core",
        "--all-targets", "--no-deps"]),
    ("vision-wasm", cargo + ["check", "--locked", "-p", "st-vision", "--features", "nerf",
        "--target", "wasm32-unknown-unknown"]),
    ("wasm-build", cargo + ["build", "--locked", "--release", "-p", "spiraltorch-wasm",
        "--target", "wasm32-unknown-unknown"]),
    ("bindgen", [str(Path.home() / "Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.104/wasm-bindgen"),
        str(client_target / "wasm32-unknown-unknown/release/spiraltorch_wasm.wasm"),
        "--target", "nodejs", "--out-dir", str(output / "wasm")]),
    ("wasm-gelu", ["node", str(root / "bindings/st-wasm/tests/gelu_host.cjs"),
        str(output / "wasm/spiraltorch_wasm.js")]),
    ("wasm-autograd", ["node", str(root / "bindings/st-wasm/tests/cpu_dense_matmul.cjs"),
        str(output / "wasm/spiraltorch_wasm.js")]),
    ("python-build", cargo + ["build", "--locked", "--release", "-p", "spiraltorch-py",
        "--no-default-features", "--features", "python-default"]),
    ("python-gelu", [sys.executable, "-I",
        str(Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py"),
        str(output / "python-cpu.dylib"), str(root), "bindings/st-py/tests/test_gelu_host.py"]),
    ("python-autograd", [sys.executable, "-I",
        str(Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py"),
        str(output / "python-cpu.dylib"), str(root), "bindings/st-py/tests/test_autograd_prepacked.py"]),
    ("wgpu-linear", ["env", "SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"] + cargo + ["test",
        "--locked", "--release", "-p", "st-nn", "--features", "wgpu", "--lib", "layers::linear",
        "--", "--nocapture", "--test-threads=1"]),
    ("wgpu-resident", ["env", "SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"] + cargo + ["test",
        "--locked", "--release", "-p", "st-nn", "--features", "wgpu", "--test", "resident_graph_forward",
        "--", "--nocapture", "--test-threads=1"]),
]
completed = []
for name, command in steps:
    previous = output / name / "receipt.json"
    if previous.exists():
        receipt = json.loads(previous.read_text())
        assert receipt["exit_code"] == 0 and receipt["source_unchanged"] is True
        assert receipt["source"]["commit"] == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        assert all(hashlib.sha256((root / f).read_bytes()).hexdigest() == digest
                   for f, digest in receipt["source"]["files"].items())
        completed.append({"name": name, "exit_code": 0})
        print("REUSE verified stage", name, flush=True)
        continue
    print("START", name, flush=True)
    selected_target = client_target if name in {
        "wasm-build", "bindgen", "wasm-gelu", "wasm-autograd", "python-build",
        "python-gelu", "python-autograd", "wgpu-linear", "wgpu-resident"} else target
    result = subprocess.run([sys.executable, "-B", "-I", str(driver), str(root), str(selected_target),
        str(output / name), *command], stdout=subprocess.DEVNULL)
    completed.append({"name": name, "exit_code": result.returncode})
    (output / "stages.json").write_text(json.dumps(completed, indent=2) + "\n")
    result.check_returncode()
    if name == "python-build":
        shutil.copy2(client_target / "release/libspiraltorch.dylib", output / "python-cpu.dylib")
    print("END", name, flush=True)
