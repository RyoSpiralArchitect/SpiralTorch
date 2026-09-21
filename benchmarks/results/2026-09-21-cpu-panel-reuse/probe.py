"""Freeze source, numerical probe, and benchmark executables before comparison."""
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
report = {"source": {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}")}, "steps": [], "products": {}}
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", SPIRAL_DETERMINISTIC="0", RAYON_NUM_THREADS="4")
def save():
    (output / "receipt.json").write_text(json.dumps(report, indent=2)+"\n")
def check():
    assert git("rev-parse", "HEAD") == report["source"]["commit"] and not git("status", "--porcelain")
try:
    for name, command in [
        ("cpu-kernels", ["cargo", "+1.98.0", "test", "--locked", "-p", "st-tensor", "--no-default-features", "--features", "cpu", "--lib", "backend::cpu_dense", "--", "--test-threads=1"]),
        ("benchmark-build", ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "st-bench", "--example", "cpu_dense_workspace", "--example", "source_crosscut"]),
    ]:
        check()
        start = time.monotonic()
        print("START", name, flush=True)
        with (output / (name+".log")).open("x") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["steps"].append(dict(name=name, command=command, exit_code=result.returncode, seconds=time.monotonic()-start))
        save()
        result.check_returncode()
        check()
        print("END", name, flush=True)
    for source, name in [("cpu_dense_workspace", "candidate-dense"), ("source_crosscut", "candidate-crosscut")]:
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
