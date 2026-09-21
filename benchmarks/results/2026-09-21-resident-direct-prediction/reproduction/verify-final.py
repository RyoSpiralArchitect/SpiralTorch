"""Serial clean-source verification and frozen runtime products."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/Users/ryospiralarchitect/\U0001f300SpiralReality\U0001f300/_wt/spiraltorch-resident-graph-forward-v1")
TARGET = ROOT.parent / "spiraltorch-spiralk-golden-blackcat-contract-v1/target"
P = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
LOADER = "/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py"
BINDGEN = "/Users/ryospiralarchitect/Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.104/wasm-bindgen"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = Path(__file__).parent / sys.argv[1]
OUT.mkdir()
git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
assert not git("status", "--porcelain")
source = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
env = dict(os.environ, CARGO_BUILD_JOBS="4", CARGO_TARGET_DIR=str(TARGET),
    PYTHONNOUSERSITE="1", PYTORCH_ENABLE_MPS_FALLBACK="0",
    NODE_PATH="/Users/ryospiralarchitect/Library/Logs/SpiralTorch/midk-seek-20260906/browser-deps/node_modules")
env.pop("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS", None)
report = dict(status="running", pid=os.getpid(), source=source, steps=[], products={})
start = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save():
    temp = OUT / "receipt.json.tmp"
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(OUT / "receipt.json")


def run(name, command, gpu=False):
    command = list(map(str, command))
    active_env = dict(env)
    if gpu:
        active_env["SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"] = "1"
    began = time.monotonic()
    print("START", name, flush=True)
    with (OUT / (name + ".log")).open("xb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=active_env, stdout=log, stderr=subprocess.STDOUT)
        report["active"] = dict(name=name, pid=child.pid, command=command)
        save()
        code = child.wait()
    report.pop("active")
    report["steps"].append(dict(name=name, command=command, exit_code=code,
        seconds=time.monotonic() - began, gpu_opt_in=gpu))
    save()
    print("END", name, code, flush=True)
    if code:
        raise RuntimeError(name)
    assert not git("status", "--porcelain") and git("rev-parse", "HEAD") == source["commit"]


def freeze(source_path, name):
    target = OUT / name
    shutil.copyfile(source_path, target)
    shutil.copymode(source_path, target)
    report["products"][name] = sha(target)
    save()
    return target


try:
    run("format", ["cargo", "+nightly-2026-04-15", "fmt", "--all", "--", "--check"])
    run("admission", [P, "-S", "-m", "unittest", "discover", "-s", "tests", "-p", "test_resident_training_bench.py"])
    run("backend-clippy", ["cargo", "+1.98.0", "clippy", "--locked", "-p", "st-backend-wgpu", "--all-targets", "--", "-D", "warnings"])
    run("backend-wasm-clippy", ["cargo", "+1.98.0", "clippy", "--locked", "-p", "st-backend-wgpu", "--target", "wasm32-unknown-unknown", "--all-targets", "--", "-D", "warnings"])
    run("backend-gpu", ["cargo", "+1.98.0", "test", "--locked", "--release", "-p", "st-backend-wgpu", "--", "--test-threads=1"], gpu=True)
    run("nn-gpu", ["cargo", "+1.98.0", "test", "--locked", "--release", "-p", "st-nn", "--features", "wgpu", "--lib", "--", "--test-threads=1"], gpu=True)
    run("native-build", ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "st-nn", "--features", "wgpu", "--example", "resident_training_bench"])
    freeze(TARGET / "release/examples/resident_training_bench", "native-worker")
    for mode in ["gpu", "cpu"]:
        args = [] if mode == "gpu" else ["--no-default-features", "--features", "python-default"]
        run("python-" + mode + "-build", ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "spiraltorch-py", *args])
        library = freeze(TARGET / "release/libspiraltorch.dylib", "python-" + mode + ".dylib")
        for test in ["test_nn_pointwise_cotangent", "test_nn_resident_graph_autograd", "test_nn_resident_graph_learner",
                     "test_wgpu_pointwise", "test_nn_resident_exports", "test_runtime_imports"]:
            directory = "tests/" if test == "test_runtime_imports" else "bindings/st-py/tests/"
            run(mode + "-" + test, [P, "-I", LOADER, library, ROOT, directory + test + ".py"], gpu=mode == "gpu")
    run("wasm-client-build", ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "spiraltorch-wasm",
        "--target", "wasm32-unknown-unknown", "--features", "webgpu"])
    run("wasm-client-bindgen", [BINDGEN, "--target", "web", "--out-dir", OUT / "module", "--out-name",
        "spiraltorch_wasm", TARGET / "wasm32-unknown-unknown/release/spiraltorch_wasm.wasm"])
    run("wasm-bench-build", ["cargo", "+1.98.0", "build", "--locked", "--release", "-p", "st-nn",
        "--target", "wasm32-unknown-unknown", "--features", "wgpu", "--example", "resident_training_bench_browser"])
    run("wasm-bench-bindgen", [BINDGEN, "--target", "web", "--out-dir", OUT / "benchmark", "--out-name",
        "spiraltorch_wasm", TARGET / "wasm32-unknown-unknown/release/examples/resident_training_bench_browser.wasm"])
    for module in ["module", "benchmark"]:
        for path in sorted((OUT / module).rglob("*")):
            if path.is_file():
                report["products"][str(path.relative_to(OUT))] = sha(path)
    save()
    for fixture in ["nn-learner-clients", "nn-autograd-clients", "pointwise-clients"]:
        run("browser-" + fixture, ["node", "tools/test_resident_browser.cjs", OUT / "module", CHROME,
            OUT / (fixture + ".json"), "", "", "", "", fixture], gpu=True)
    report["status"] = "passed"
except BaseException as error:
    report.update(status="error", error=repr(error))
finally:
    report["seconds"] = time.monotonic() - start
    save()
print(json.dumps(dict(status=report["status"], error=report.get("error"))), flush=True)
raise SystemExit(report["status"] != "passed")
