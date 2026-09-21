"""Build identical extended harnesses on baseline/changed production sources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

baseline, candidate, target, output = map(Path, sys.argv[1:5])
output.mkdir()
harness = Path("crates/st-bench/examples/cpu_dense_extended.rs")
assert (baseline / harness).read_bytes() == (candidate / harness).read_bytes()
report = {"harness_sha256": hashlib.sha256((candidate / harness).read_bytes()).hexdigest(), "steps": []}
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4")
try:
    for label, root in [("baseline", baseline), ("candidate", candidate)]:
        git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        assert status == ("?? " + str(harness) if label == "baseline" else ""), status
        command = ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "st-bench", "--example", "cpu_dense_extended"]
        print("START", label, flush=True)
        start = time.monotonic()
        with (output / (label+".log")).open("x") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        step = dict(label=label, root=str(root), commit=commit, status=status, command=command,
                    exit_code=result.returncode, seconds=time.monotonic()-start)
        report["steps"].append(step)
        (output / "receipt.json").write_text(json.dumps(report, indent=2)+"\n")
        result.check_returncode()
        assert commit == git("rev-parse", "HEAD") and status == git("status", "--porcelain")
        destination = output / (label+"-extended")
        shutil.copy2(target / "release/examples/cpu_dense_extended", destination)
        step["binary_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        print("END", label, flush=True)
    report["status"] = "passed"
except Exception as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    raise
finally:
    (output / "receipt.json").write_text(json.dumps(report, indent=2)+"\n")
