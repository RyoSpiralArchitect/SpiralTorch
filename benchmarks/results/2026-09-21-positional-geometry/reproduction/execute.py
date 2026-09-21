"""Timing never overlaps another build or measurement launched by this driver."""
from pathlib import Path
import subprocess
import sys

root, target, log = map(Path, sys.argv[1:4])
py = sys.executable
def call(*args):
    subprocess.run(args, check=True)
for backend in ["native", "wasm"]:
    call(py, "-B", "-I", str(log / "measure.py"), str(log), backend)
call(py, "-B", "-I", str(log / "run.py"), str(root), str(target), str(log / "torch"),
     py, "-B", "-I", str(log / "torch_encoding.py"))
call(py, "-B", "-I", str(log / "check.py"), str(root), str(target), str(log / "verification"))
