"""Preserve serial validation commands, source hashes and unmodified output."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root, target, output = map(Path, sys.argv[1:4])
command = sys.argv[4:]
output.mkdir(parents=True)
def git(*args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
def identity():
    fixture = root / "benchmarks/nerf-ray-integral"
    paths = [*sorted((root / "crates/st-vision").rglob("*.rs")),
             *sorted(p for p in fixture.rglob("*") if p.is_file() and p.suffix in {".rs", ".py", ".cjs", ".toml", ".lock"}),
             root / "crates/st-vision/Cargo.toml", root / "Cargo.lock",
             root / ".github/workflows/ci.yml"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before = identity()
source = {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"), "files": before}
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4")
for key in ["SPIRALTORCH_AUTOTUNE_STORE", "SPIRALTORCH_AUTOTUNE", "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION"]:
    env.pop(key, None)
start = time.monotonic()
with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
    result = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
receipt = {"source": source, "command": command, "exit_code": result.returncode,
           "seconds": time.monotonic() - start,
           "environment": {k: env[k] for k in ["CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "RAYON_NUM_THREADS"]},
           "source_unchanged": before == identity()}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "source"}))
sys.exit(result.returncode or (0 if receipt["source_unchanged"] else 1))
