"""Final measurement after the small-layout correction; retain all earlier runs."""
import json
import os
from pathlib import Path
import subprocess
import sys

root, target, output = map(Path, sys.argv[1:4])
python = sys.executable
subprocess.run([python, "-B", "-I", str(output / "build.py"), str(root), str(target), str(output / "candidate-c")], check=True)
subprocess.run([python, "-B", "-I", str(output / "measure.py"), str(output / "native-c"), str(output / "baseline/worker"), str(output / "candidate-c/worker")], check=True)
subprocess.run([python, "-B", "-I", str(output / "measure.py"), str(output / "wasm-b"),
                str(output.parent / "cpu-row-major-20260921/verified-a/wasm/spiraltorch_wasm.js"),
                str(output / "verified-b/wasm/spiraltorch_wasm.js"), "--wasm-harness", str(root / "bindings/st-wasm/tests/cpu_layout_bench.cjs")], check=True)
environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
command = [python, "-B", "-I", "-S", str(output / "torch_compare.py"), "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"]
with (output / "torch-b.json").open("x") as out, (output / "torch-b.stderr").open("x") as err:
    result = subprocess.run(command, env=environment, stdout=out, stderr=err)
(output / "torch-b-receipt.json").write_text(json.dumps({"command": command, "exit_code": result.returncode,
    "environment": {key: environment[key] for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}, indent=2) + "\n")
result.check_returncode()
print("Selected comparisons completed", flush=True)
