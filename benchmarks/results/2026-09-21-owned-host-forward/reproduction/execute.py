"""Measurements and checks are serial: our builds do not overlap timing."""
from pathlib import Path
import subprocess
import sys

root, target, log = map(Path, sys.argv[1:4])
py = sys.executable
def call(*args):
    subprocess.run(args, check=True)
call(py, "-B", "-I", str(log / "measure.py"), str(log), "native")
call(py, "-B", "-I", str(log / "measure.py"), str(log), "wasm")
call(py, "-B", "-I", str(log / "nn_measure.py"), str(log / "nn"),
     str(log / "baseline-build/cpu_nn_layout"), str(log / "candidate-build/cpu_nn_layout"))
call(py, "-B", "-I", str(log / "run.py"), str(root), str(target), str(log / "torch"),
     py, "-B", "-I", str(log / "torch_chain.py"))
call(py, "-B", "-I", str(log / "check.py"), str(root), str(target), str(log / "verification"))
