"""Run measurements and verification sequentially so builds do not overlap timers."""
from pathlib import Path
import shutil
import subprocess
import sys

local = Path(__file__).resolve().parent
root = Path(sys.argv[1])
target = Path(sys.argv[2])
previous = local.parent / "cpu-dense-workspace-20260921"
python = sys.executable
site = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"
def run(*args):
    subprocess.run(list(map(str, args)), check=True, cwd=root)

run(python, "-B", "-I", "-S", local / "measure_extended.py", "measure",
    "--directory", local / "extended-b", "--baseline", local / "extended-build-b/baseline-extended",
    "--candidate", local / "extended-build-b/candidate-extended", "--torch-python", python, "--torch-site", site)
run(python, "-B", "-I", "-S", previous / "verify.py", root, target, local / "verified-a")
run(python, "-B", "-I", "-S", previous / "verify_portable.py", root, target, local / "portable-tests")
crosscut = local / "crosscut-a"
crosscut.mkdir()
shutil.copy2(local / "baseline-crosscut", crosscut / "baseline-worker")
shutil.copy2(local / "verified-a/candidate-crosscut", crosscut / "candidate-worker")
run(python, "-B", "-I", "-S", root / "benchmarks/results/2026-09-21-cpu-dense-workspace/measure_crosscut.py",
    "measure", "--directory", crosscut, "--torch-python", python, "--torch-site", site)
