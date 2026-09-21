"""Verify current kernels and freeze source-bound workers before measurement."""
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
sources = ["crates/st-tensor/src/backend/cpu_dense.rs", "crates/st-bench/examples/cpu_dense_workspace.rs",
           "crates/st-bench/examples/cpu_dense_extended.rs", "crates/st-bench/examples/source_crosscut.rs"]
hashes = lambda: {name:hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
report = dict(commit=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain"),
              source_hashes=hashes(), steps=[], products={}, status="running")
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4",
           SPIRAL_DETERMINISTIC="0", SPIRAL_DETERMINISTIC_REDUCTION="0")
def save():
    (output / "receipt.json").write_text(json.dumps(report, indent=2)+"\n")
try:
    subprocess.run(["touch", str(root / sources[0])], check=True)
    for name, command in [
        ("cpu-kernels", ["cargo", "+1.98.0", "test", "--locked", "-p", "st-tensor", "--no-default-features", "--features", "cpu", "--lib", "backend::cpu_dense", "--", "--test-threads=1"]),
        ("benchmark-build", ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "st-bench", "--example", "cpu_dense_workspace", "--example", "cpu_dense_extended", "--example", "source_crosscut"]),
    ]:
        assert hashes() == report["source_hashes"]
        start = time.monotonic()
        print("START", name, flush=True)
        with (output / (name+".log")).open("x") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["steps"].append(dict(name=name, command=command, exit_code=result.returncode, seconds=time.monotonic()-start))
        save()
        result.check_returncode()
        assert hashes() == report["source_hashes"]
        print("END", name, flush=True)
    assert "Compiling st-tensor v0.1.0 ("+str(root / "crates/st-tensor")+")" in (output / "benchmark-build.log").read_text()
    for source, name in [("cpu_dense_workspace", "candidate-dense"), ("cpu_dense_extended", "candidate-extended"), ("source_crosscut", "candidate-crosscut")]:
        path = output / name
        shutil.copy2(target / "release/examples" / source, path)
        report["products"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["status"] = "passed"
except Exception as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    raise
finally:
    save()
