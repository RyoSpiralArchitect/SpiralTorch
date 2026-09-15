"""Two complete route-isolated matrices, with immutable products and all outcomes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/Users/ryospiralarchitect/\U0001f300SpiralReality\U0001f300/_wt/spiraltorch-resident-graph-forward-v1")
LOG = Path(__file__).parent
PRODUCT = LOG / sys.argv[1]
OUT = LOG / sys.argv[2]
OUT.mkdir()
P = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
TORCH = "/Users/ryospiralarchitect/Library/Logs/SpiralTorch/rank-finite-bound-20260907/venv/bin/python"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
assert not git("status", "--porcelain")
source = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
verified = json.loads((PRODUCT / "receipt.json").read_bytes())
assert verified["status"] == "passed" and "active" not in verified
worker_source = verified["source"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


for name, expected in verified["products"].items():
    assert sha(PRODUCT / name) == expected
helpers = ["tools/bench_resident_training_vs_torch.py", "tools/bench_resident_training_browser.cjs",
    "tools/validate_resident_training_bench.py", "tools/resident_learner_bench_reference.py",
    "bindings/st-wasm/tests/resident_training_bench.html",
    "crates/st-nn/examples/support/resident_learner_bench.rs"]
for name in helpers:
    assert subprocess.check_output(["git", "show", worker_source["commit"] + ":" + name], cwd=ROOT) == (ROOT / name).read_bytes()
env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS="1",
    NODE_PATH="/Users/ryospiralarchitect/Library/Logs/SpiralTorch/midk-seek-20260906/browser-deps/node_modules")
report = dict(status="running", pid=os.getpid(), source=source, worker_source=worker_source,
    verification_receipt_sha256=sha(PRODUCT / "receipt.json"), products=verified["products"],
    runner_sha256=sha(Path(__file__)), steps=[], results={},
    boundary="Same executable/module in both Rust lanes, with identical fused pointwise seed programs. "
    "Materialized baseline vs direct candidate; two exact VJPs and ordered weighted updates unchanged. "
    "27 standard recipes per round; two complete rounds, reversed optimizer and client order. "
    "Eight updates per interval, two warmups/eight retained blocks per cadence. "
    "Torch eager MPS is not an equivalent finite-guard/atomic-rollback runtime. "
    "Owned GPU work serial; host exclusivity and physical browser adapter identity UNKNOWN. "
    "No CUDA, fastest-Torch, FT-quality or GPU-kernel-time claim.")
start = time.monotonic()


def save():
    temp = OUT / "receipt.json.tmp"
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(OUT / "receipt.json")


def run(name, command):
    command = list(map(str, command))
    began = time.monotonic()
    print("START", name, flush=True)
    with (OUT / (name + ".log")).open("xb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["active"] = dict(name=name, pid=child.pid, command=command)
        save()
        code = child.wait()
    report.pop("active")
    report["steps"].append(dict(name=name, command=command, exit_code=code, seconds=time.monotonic() - began))
    save()
    print("END", name, code, flush=True)
    if code:
        raise RuntimeError(name)
    assert not git("status", "--porcelain") and git("rev-parse", "HEAD") == source["commit"]


try:
    for round_name, modes, clients in [
        ("a", ["plain", "topos_ema", "clipped_topos_ema"], ["native", "browser"]),
        ("b", ["clipped_topos_ema", "topos_ema", "plain"], ["browser", "native"]),
    ]:
        for mode in modes:
            prefix = round_name + "-" + mode
            native = OUT / (prefix + "-native.json")
            browser = OUT / (prefix + "-browser.json")
            commands = {
                "native": [TORCH, "-I", ROOT / "tools/bench_resident_training_vs_torch.py",
                    "--baseline", PRODUCT / "native-worker", "--baseline-source", worker_source["commit"],
                    "--candidate", PRODUCT / "native-worker", "--candidate-source", worker_source["commit"],
                    "--device", "mps", "--graph", "--learner", "--direct-learner-seeds",
                    *(["--learner-optimizer", mode] if mode != "plain" else []), "--output", native],
                "browser": ["node", ROOT / "tools/bench_resident_training_browser.cjs",
                    PRODUCT / "benchmark", PRODUCT / "benchmark", CHROME, browser,
                    "learner", "standard", "direct-learner-seeds", "none" if mode == "plain" else mode],
            }
            for client in clients:
                run(prefix + "-" + client, commands[client])
            validation = OUT / (prefix + "-validation.json")
            run(prefix + "-validate", [P, "-I", ROOT / "tools/validate_resident_training_bench.py",
                "--native", native, "--browser", browser, "--browser-progress", str(browser) + ".progress.jsonl",
                "--baseline-source", worker_source["commit"], "--candidate-source", worker_source["commit"],
                "--browser-harness-source", source["commit"], "--output", validation])
            for path in [native, browser, validation]:
                data = json.loads(path.read_bytes())
                assert data["status"] == "passed" and data["direct_learner_seeds"] is True and len(data["cases"]) == 9
                report["results"][path.name] = sha(path)
    for name, expected in verified["products"].items():
        assert sha(PRODUCT / name) == expected
    assert len(report["steps"]) == 18
    report["status"] = "passed"
except BaseException as error:
    report.update(status="error", error=repr(error))
finally:
    report["seconds"] = time.monotonic() - start
    save()
print(json.dumps(dict(status=report["status"], error=report.get("error"))), flush=True)
raise SystemExit(report["status"] != "passed")
