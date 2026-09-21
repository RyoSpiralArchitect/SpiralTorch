"""Freeze a matching-root benchmark, retaining source identity and compiler output."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root, target, directory = map(Path, sys.argv[1:4])
directory.mkdir()
files = ["crates/st-bench/examples/cpu_nn_layout.rs", "crates/st-tensor/src/pure.rs",
         "crates/st-tensor/src/backend/faer_dense.rs", "crates/st-tensor/src/backend/mod.rs"]
transpose = "crates/st-tensor/src/backend/transpose.rs"
if (root / transpose).exists():
    files.append(transpose)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
source = {"commit": git("rev-parse", "HEAD"), "files": {f: sha(root / f) for f in files},
          "status": git("status", "--porcelain")}
(directory / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root))
environment = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4")
command = ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "st-bench", "--example", "cpu_nn_layout"]
subprocess.run(["touch", "crates/st-tensor/src/lib.rs", "crates/st-bench/src/lib.rs"], cwd=root, check=True)
start = time.monotonic()
with (directory / "build.log").open("x") as log:
    result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
receipt = {"source": source, "command": command, "exit_code": result.returncode, "seconds": time.monotonic() - start}
(directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
result.check_returncode()
log = (directory / "build.log").read_text()
for crate in ["st-tensor", "st-bench"]:
    assert f"({root}/crates/{crate})" in log, f"missing matching-root compilation: {crate}"
assert all(sha(root / f) == source["files"][f] for f in files)
worker = directory / "worker"
shutil.copy2(target / "release/examples/cpu_nn_layout", worker)
receipt["worker_sha256"] = sha(worker)
(directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
