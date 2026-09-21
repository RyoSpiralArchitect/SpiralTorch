"""Do not overlap local measurements and compilation/test activity."""
from pathlib import Path
import subprocess
import sys

root, target, out = map(Path, sys.argv[1:4])
commands = [
    [sys.executable, "-B", "-I", str(out / "measure.py"), str(out)],
    [sys.executable, "-B", "-I", str(out.parent / "cpu-nn-layout-20260921/measure.py"), str(out / "nn"), str(out / "baseline-build/cpu_nn_layout"), str(out / "candidate-build/cpu_nn_layout")],
    [sys.executable, "-B", "-I", str(out / "check.py"), str(root), str(target), str(out / "verification")],
]
for command in commands:
    subprocess.run(command, check=True)
