"""Serial source-bound validation, with fixture builds in an isolated target."""
from pathlib import Path
import json
import subprocess
import sys

root, target, fixture_target, output = map(Path, sys.argv[1:5])
output.mkdir()
driver = Path(__file__).with_name("run.py")
cargo = ["cargo", "+1.98.0"]
manifest = "benchmarks/nerf-ray-integral/Cargo.toml"
steps = [
    ("format", target, ["cargo", "+nightly-2026-04-15", "fmt", "--all", "--", "--check"]),
    ("fixture-format", target, ["cargo", "+nightly-2026-04-15", "fmt", "--manifest-path", manifest, "--", "--check"]),
    ("clippy", target, cargo + ["clippy", "--locked", "-p", "st-vision", "--features", "nerf", "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("vision-tests", target, cargo + ["test", "--locked", "--release", "-p", "st-vision", "--features", "nerf", "--lib", "--test", "nerf_geometry", "--test", "nerf_regression", "--test", "nerf_ray_integral", "--test", "temporal_channels", "--", "--nocapture", "--test-threads=1"]),
    ("vision-no-default", target, cargo + ["test", "--locked", "--release", "-p", "st-vision", "--no-default-features", "--lib", "--", "--test-threads=1"]),
    ("vision-wasm", target, cargo + ["check", "--locked", "-p", "st-vision", "--features", "nerf", "--target", "wasm32-unknown-unknown"]),
    ("fixture-native", fixture_target, cargo + ["build", "--locked", "--release", "--manifest-path", manifest]),
    ("fixture-wasm", fixture_target, cargo + ["build", "--locked", "--release", "--lib", "--target", "wasm32-unknown-unknown", "--manifest-path", manifest]),
    ("fixture-bindgen", fixture_target, [str(Path.home() / "Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.104/wasm-bindgen"),
        str(fixture_target / "wasm32-unknown-unknown/release/nerf_ray_integral_fixture.wasm"), "--target", "nodejs", "--out-dir", str(output / "wasm")]),
    ("native-preflight", fixture_target, [str(fixture_target / "release/nerf-ray-integral-fixture")]),
    ("wasm-preflight", fixture_target, ["node", str(root / "benchmarks/nerf-ray-integral/wasm.cjs"), str(output / "wasm/nerf_ray_integral_fixture.js")]),
    ("torch-preflight", fixture_target, [sys.executable, "-I", str(root / "benchmarks/nerf-ray-integral/torch_reference.py"), str(output / "native-preflight/stdout.log")]),
]
stages = []
for name, selected, command in steps:
    print("START", name, flush=True)
    result = subprocess.run([sys.executable, "-I", "-B", str(driver), str(root), str(selected),
                             str(output / name), *command], stdout=subprocess.DEVNULL)
    stages.append({"stage": name, "exit_code": result.returncode})
    (output / "stages.json").write_text(json.dumps(stages, indent=2) + "\n")
    print("END", name, result.returncode, flush=True)
    result.check_returncode()
