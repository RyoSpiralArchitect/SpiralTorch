"""Build once, then measure without overlapping our builds or other benchmarks."""
import json
import os
from pathlib import Path
import subprocess
import sys

root, target, output = map(Path, sys.argv[1:4])
python = sys.executable
subprocess.run([python, "-B", "-I", str(output / "build.py"), str(root), str(target), str(output / "candidate-b")], check=True)
subprocess.run([python, "-B", "-I", str(output / "measure.py"), str(output / "native-b"), str(output / "baseline/worker"), str(output / "candidate-b/worker")], check=True)
subprocess.run([python, "-B", "-I", str(output / "measure.py"), str(output / "wasm-a"),
                str(output.parent / "cpu-row-major-20260921/verified-a/wasm/spiraltorch_wasm.js"),
                str(output / "verified-a/wasm/spiraltorch_wasm.js"), "--wasm-harness", str(root / "bindings/st-wasm/tests/cpu_layout_bench.cjs")], check=True)
environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
command = [python, "-B", "-I", "-S", str(output / "torch_compare.py"), "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"]
with (output / "torch.json").open("x") as out, (output / "torch.stderr").open("x") as err:
    result = subprocess.run(command, env=environment, stdout=out, stderr=err)
(output / "torch-receipt.json").write_text(json.dumps({"command": command, "exit_code": result.returncode,
    "environment": {key: environment[key] for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}, indent=2) + "\n")
result.check_returncode()
print("All comparison processes completed", flush=True)
