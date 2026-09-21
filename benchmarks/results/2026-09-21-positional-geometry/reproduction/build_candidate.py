"""Build and freeze only successful candidate workers in the private target."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

root, target, log = map(Path, sys.argv[1:4])
stages = json.loads((log / "verification/stages.json").read_text())
assert len(stages) == 14 and all(s["exit_code"] == 0 for s in stages)
def run(name, command):
    subprocess.run([sys.executable, "-B", "-I", str(log / "run.py"), str(root), str(target),
                    str(log / name), *command], check=True)
cargo = ["cargo", "+1.98.0", "build", "--locked", "--release", "--manifest-path", str(log / "fixture/Cargo.toml")]
run("candidate-native-build", cargo + ["--bin", "positional-geometry-fixture"])
frozen = log / "candidate-build"
frozen.mkdir()
shutil.copy2(target / "release/positional-geometry-fixture", frozen / "positional-geometry-fixture")
run("candidate-wasm-build", cargo + ["--target", "wasm32-unknown-unknown", "--lib"])
run("candidate-bindgen", [str(Path.home() / "Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.104/wasm-bindgen"),
    str(target / "wasm32-unknown-unknown/release/positional_geometry_fixture.wasm"),
    "--target", "nodejs", "--out-dir", str(frozen / "wasm")])
