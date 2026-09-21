"""Record bounded commands and source identity without normalizing their output."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root, target, output = map(Path, sys.argv[1:4])
command = sys.argv[4:]
output.mkdir()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
files = ["crates/st-tensor/src/pure.rs", "crates/st-nn/src/layers/gelu.rs",
         "crates/st-nn/tests/gelu_layout_contract.rs", "crates/st-bench/examples/cpu_gelu.rs",
         "crates/st-bench/examples/cpu_nn_layout.rs", "bindings/st-py/tests/test_gelu_host.py",
         "bindings/st-wasm/tests/gelu_host.cjs", ".github/workflows/ci.yml", "Cargo.lock"]
source = {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
          "files": {f: sha(root / f) for f in files}}
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4")
for key in ["SPIRALTORCH_AUTOTUNE_STORE", "SPIRALTORCH_AUTOTUNE", "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION"]:
    env.pop(key, None)
start = time.monotonic()
with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
    result = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
receipt = {"source": source, "command": command, "exit_code": result.returncode,
           "seconds": time.monotonic() - start, "environment": {k: env[k] for k in ["CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "RAYON_NUM_THREADS"]},
           "source_unchanged": source["files"] == {f: sha(root / f) for f in files}}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
sys.exit(result.returncode or (0 if receipt["source_unchanged"] else 1))
