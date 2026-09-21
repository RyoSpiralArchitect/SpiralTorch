"""Measure the unchanged native grids sequentially, keeping every condition."""
from pathlib import Path
import subprocess
import sys

root, local, variant = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
previous = root / "benchmarks/results/2026-09-21-cpu-panel-reuse"
python = sys.executable
site = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"
def run(*args):
    subprocess.run(list(map(str, args)), cwd=root, check=True)
run(python, "-B", "-I", "-S", previous / "measure_extended.py", "measure",
    "--directory", local / ("extended-"+variant), "--baseline", local / "baseline-extended",
    "--candidate", local / ("probe-"+variant) / "candidate-extended",
    "--torch-python", python, "--torch-site", site)
run(python, "-B", "-I", "-S", previous / "measure_dense.py", "measure", local / ("dense-"+variant),
    "--baseline", local / "baseline-dense", "--candidate", local / ("probe-"+variant) / "candidate-dense")
