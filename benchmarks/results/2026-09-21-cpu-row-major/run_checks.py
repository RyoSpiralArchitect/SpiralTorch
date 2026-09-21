"""Verify selected source and measure its clients without overlapping heavy work."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root, target = map(Path, sys.argv[1:3])
local = Path(__file__).resolve().parent
previous = root / "benchmarks/results/2026-09-21-cpu-panel-reuse"
python = sys.executable
site = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"
source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
def run(*args):
    subprocess.run(list(map(str,args)), cwd=root, check=True)

command = ["cargo", "+nightly-2026-04-15", "test", "--locked", "-p", "st-tensor", "--no-default-features",
    "--features", "cpu,simd", "--lib", "backend::cpu_dense", "--", "--test-threads=1"]
environment = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4",
    SPIRAL_DETERMINISTIC="0", SPIRAL_DETERMINISTIC_REDUCTION="0",
    SPIRALTORCH_AUTOTUNE_STORE=str(local / "simd-tuning.json"), SPIRALTORCH_AUTOTUNE="1")
start = time.monotonic()
print("START optional portable SIMD tests",flush=True)
with (local / "simd.log").open("x") as log:
    result = subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
(local / "simd.json").write_text(json.dumps(dict(source=source,command=command,exit_code=result.returncode,
    seconds=time.monotonic()-start,environment={k:environment[k] for k in ["RAYON_NUM_THREADS","SPIRAL_DETERMINISTIC","SPIRALTORCH_AUTOTUNE","SPIRALTORCH_AUTOTUNE_STORE"]}),indent=2)+"\n")
result.check_returncode()
run(python,"-B","-I","-S",local / "verification_driver.py",root,target,local / "verified-a")
run(python,"-B","-I","-S",previous / "verify_portable.py",root,target,local / "portable-tests")
run(python,"-B","-I","-S",local / "measure_wasm.py","measure","--directory",local / "wasm-a",
    "--harness",root / "bindings/st-wasm/tests/cpu_dense_bench.cjs",
    "--baseline",local.parent / "cpu-panel-reuse-20260921/verified-a/wasm/spiraltorch_wasm.js",
    "--candidate",local / "verified-a/wasm/spiraltorch_wasm.js")
crosscut = local / "crosscut-a"
crosscut.mkdir()
shutil.copy2(local / "baseline-crosscut",crosscut / "baseline-worker")
shutil.copy2(local / "verified-a/candidate-crosscut",crosscut / "candidate-worker")
run(python,"-B","-I","-S",previous / "measure_crosscut.py","measure","--directory",crosscut,
    "--torch-python",python,"--torch-site",site)
assert source == subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
assert not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()
