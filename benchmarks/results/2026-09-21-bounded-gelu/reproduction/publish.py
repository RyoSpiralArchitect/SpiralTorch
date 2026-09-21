"""Publish compact evidence only; binaries and output arrays remain local."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

root, log, archive = map(Path, sys.argv[1:4])
archive.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, data):
    (archive / name).write_text(json.dumps(data, indent=2) + "\n")
def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
builds = [lane + "-" + kind for lane in ["baseline", "candidate"]
          for kind in ["native-build", "wasm-build", "bindgen"]]
for directory in [*builds, "baseline-contract", "native", "wasm", "nn", "torch"]:
    for file in (log / directory).iterdir():
        if file.is_file() and not file.name.endswith(".raw.json"):
            copy(file, archive / directory / file.name)
stages = json.loads((log / "verification/stages.json").read_text())
assert len(stages) == 13 and all(s["exit_code"] == 0 for s in stages)
copy(log / "verification/stages.json", archive / "verification/stages.json")
for stage in stages:
    for file in (log / "verification" / stage["name"]).iterdir():
        if file.is_file():
            copy(file, archive / "verification" / stage["name"] / file.name)
for name in ["run.py", "check.py", "measure.py", "nn_measure.py", "torch_gelu.py", "wasm_gelu.cjs", "execute.py", "publish.py"]:
    copy(log / name, archive / "reproduction" / name)
for name in ["Cargo.toml", "Cargo.lock", "src/lib.rs"]:
    copy(log / "wasm-shim" / name, archive / "reproduction/wasm-shim" / name)
copy(Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py",
     archive / "reproduction/python-client.py")
for name in ["crates/st-bench/examples/cpu_gelu.rs", "crates/st-bench/examples/cpu_nn_layout.rs",
             "crates/st-nn/tests/gelu_layout_contract.rs"]:
    copy(root / name, archive / "reproduction" / Path(name).name)
baseline = json.loads((log / "baseline-native-build/receipt.json").read_text())["source"]["commit"]
candidate = json.loads((log / "candidate-native-build/receipt.json").read_text())["source"]["commit"]
(archive / "runtime.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", baseline, candidate], cwd=root))
identities = {}
for key, command in {"rustc": ["rustc", "+1.98.0", "-vV"], "node": ["node", "--version"],
                     "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"],
                     "os": ["sw_vers"], "arch": ["uname", "-m"]}.items():
    identities[key] = subprocess.check_output(command, text=True).strip()
workers = {}
for lane in ["baseline", "candidate"]:
    workers[lane] = {str(p.relative_to(log / (lane + "-build"))): sha(p)
                     for p in (log / (lane + "-build")).rglob("*") if p.is_file()}
raw = {str(p.relative_to(log)): sha(p) for p in log.glob("*/*.raw.json")}
write("provenance.json", {"baseline": baseline, "candidate": candidate, "workers": workers,
      "local_raw_root": str(log), "raw_sha256": raw, "identity": identities,
      "build_environment_observed": {k: os.environ.get(k) for k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX"]},
      "wasm_shim_scope": "Test-only checked Tensor host operation, not a public JS API or browser/WebGPU execution",
      "numerical_replay": False, "host_exclusivity": "unknown", "thermal_state": "unknown"})
torch = json.loads((log / "torch/stdout.log").read_text())
index = lambda cases: {(c["rows"], c["cols"], c["backward"]): c for c in cases}
comparisons = []
for report in torch["reports"]:
    if report["round"] == "warm":
        continue
    candidate_report = json.loads((log / "native" / ("candidate-" + report["round"] + ".json")).read_text())
    a, b = index(report["cases"]), index(candidate_report["cases"])
    assert a.keys() == b.keys()
    for key in sorted(a):
        comparisons.append({"condition": key, "round": report["round"],
                            "ratio": statistics.median(a[key]["elapsed_ns"]) / statistics.median(b[key]["elapsed_ns"])})
groups = {}
for backward in [False, True]:
    for size in ["all", "tiny", "non_tiny"]:
        values = [c["ratio"] for c in comparisons if c["condition"][2] == backward and
                  (size == "all" or ((c["condition"][0] * c["condition"][1] <= 64) == (size == "tiny")))]
        groups[f"{backward}/{size}"] = {"geomean": math.exp(statistics.mean(map(math.log, values))),
            "min": min(values), "max": max(values), "favorable": sum(v > 1 for v in values), "count": len(values)}
write("torch/comparison.json", {"comparisons": comparisons, "groups": groups,
      "boundary": "Torch/candidate ratio; Python and autograd entry costs differ, no fastest-Torch claim"})
print(json.dumps(groups, indent=2))
