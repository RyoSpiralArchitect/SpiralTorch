"""Publish compact evidence; full raw arrays and compiled workers remain local."""
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
def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
def write(name, data):
    (archive / name).write_text(json.dumps(data, indent=2) + "\n")
# Every command receipt, including failed preflights, stays visible.
for receipt in sorted(log.glob("*/receipt.json")):
    directory = receipt.parent
    for file in directory.iterdir():
        if file.is_file() and file.suffix in {".json", ".log", ".toml", ".txt", ".stderr"} and not file.name.endswith(".raw.json"):
            copy(file, archive / directory.name / file.name)
stages = json.loads((log / "verification/stages.json").read_text())
assert len(stages) == 14 and all(s["exit_code"] == 0 for s in stages)
copy(log / "verification/stages.json", archive / "verification/stages.json")
for stage in stages:
    for file in (log / "verification" / stage["name"]).iterdir():
        if file.is_file():
            copy(file, archive / "verification" / stage["name"] / file.name)
for directory in ["verification-before-lint-cleanup", "verification-before-test-lint-cleanup"]:
    for file in (log / directory).rglob("*"):
        if file.is_file() and file.suffix in {".json", ".log"}:
            copy(file, archive / file.relative_to(log))
for name in ["run.py", "check.py", "measure.py", "torch_encoding.py", "wasm.cjs", "execute.py", "build_candidate.py", "publish.py"]:
    copy(log / name, archive / "reproduction" / name)
for name in ["Cargo.toml", "Cargo.lock", "src/lib.rs", "src/main.rs"]:
    copy(log / "fixture" / name, archive / "reproduction/fixture" / name)
copy(Path.home() / "Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py",
     archive / "reproduction/python-client.py")
for name in ["PREFLIGHT.md", "wasm-atomic-wait-lineage.txt", "native-duplicate-tree.txt", "nerf_geometry.rs"]:
    copy(log / name, archive / name)
baseline = json.loads((log / "admitted-baseline-native-build/receipt.json").read_text())["source"]["commit"]
candidate = json.loads((log / "candidate-native-build/receipt.json").read_text())["source"]["commit"]
(archive / "runtime.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", baseline, candidate], cwd=root))
identities = {}
for key, command in {"rustc": ["rustc", "+1.98.0", "-vV"], "node": ["node", "--version"],
                     "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"], "os": ["sw_vers"],
                     "arch": ["uname", "-m"]}.items():
    identities[key] = subprocess.check_output(command, text=True).strip()
workers = {lane: {str(p.relative_to(log / (lane + "-build"))): sha(p)
    for p in (log / (lane + "-build")).rglob("*") if p.is_file()} for lane in ["baseline", "candidate"]}
write("provenance.json", {"baseline": baseline, "candidate": candidate, "workers": workers,
    "local_raw_root": str(log), "raw_sha256": {str(p.relative_to(log)): sha(p) for p in log.glob("*/*.raw.json")},
    "identity": identities, "numerical_replay": False, "host_exclusivity": "unknown", "thermal_state": "unknown",
    "fixture_target_boundary": "Private standalone fixture target, not the shared workspace target; stale failed-build copy excluded",
    "build_environment_observed": {k: os.environ.get(k) for k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX"]},
    "wasm_scope": "Actual st-vision/nerf through a Node test adapter, not a public browser or WebGPU API",
    "excluded_local_sha256": {str(p.relative_to(log)): sha(p)
        for directory in ["excluded-stale-build", "initial-native-build"]
        for p in (log / directory).rglob("*") if p.is_file()}})
torch = json.loads((log / "torch/stdout.log").read_text())
key = lambda c: (c["rows"], c["cols"], c["bands"], c["residual"])
comparisons = []
for report in torch["reports"]:
    if report["round"] == "warm":
        continue
    candidate_report = json.loads((log / "native" / ("candidate-" + report["round"] + ".json")).read_text())
    a, b = [{key(c): c for c in cases} for cases in [report["cases"], candidate_report["cases"]]]
    assert a.keys() == b.keys()
    for condition in sorted(a):
        comparisons.append({"condition": condition, "round": report["round"],
            "ratio": statistics.median(a[condition]["elapsed_ns"]) / statistics.median(b[condition]["elapsed_ns"])})
groups = {}
for size in ["all", "one_row", "multi_row"]:
    for bands in ["all", "zero", "nonzero"]:
        values = [c["ratio"] for c in comparisons
            if (size == "all" or ((c["condition"][0] == 1) == (size == "one_row")))
            and (bands == "all" or ((c["condition"][2] == 0) == (bands == "zero")))]
        groups[size + "/" + bands] = {"geomean": math.exp(statistics.mean(map(math.log, values))),
            "min": min(values), "max": max(values), "favorable": sum(v > 1 for v in values), "count": len(values)}
write("torch/comparison.json", {"comparisons": comparisons, "groups": groups,
    "boundary": "Torch/candidate ratio, vectorized eager CPU; Python/Rust dispatch differs; no fastest-Torch claim"})
print(json.dumps(groups, indent=2))
