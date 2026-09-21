"""Exercise CPU-only dispatch tests that are gated off when faer is enabled."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root, target, output = map(Path, sys.argv[1:4])
git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
source = git("rev-parse", "HEAD")
assert not git("status", "--porcelain")
command = ["cargo", "+1.98.0", "test", "--locked", "-p", "st-tensor", "--no-default-features", "--features", "cpu",
           "--test", "prepacked_cpu_dispatch", "--test", "autograd_sgd", "--test", "autograd_observability",
           "--test", "classification", "--", "--test-threads=1"]
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4", SPIRAL_DETERMINISTIC="0")
start = time.monotonic()
with output.with_suffix(".log").open("x") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
output.with_suffix(".json").write_text(json.dumps(dict(source=source, command=command, exit_code=result.returncode,
    seconds=time.monotonic()-start, environment={k:env[k] for k in ["RAYON_NUM_THREADS","SPIRAL_DETERMINISTIC"]}), indent=2)+"\n")
result.check_returncode()
assert source == git("rev-parse", "HEAD") and not git("status", "--porcelain")
assert "1 passed" in output.with_suffix(".log").read_text(), "Portable dispatch test did not run"
print("Portable CPU dispatch and autograd tests passed", flush=True)
