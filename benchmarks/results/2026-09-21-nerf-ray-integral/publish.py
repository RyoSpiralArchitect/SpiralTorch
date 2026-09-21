"""Publish compact results and receipts; never copy raw tensor reports/binaries."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

root, log, output = map(Path, sys.argv[1:4])
output.mkdir(parents=True)
verification = log / "verification-v3"
measurement = log / "measurement-v3"
data = json.loads((measurement / "results.json").read_text())
assert len(data["results"]) == 18
assert data["commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))

for name in ["run.py", "check.py", "measure.py", "analyze.py", "test_analyze.py", "publish.py", "verify.py", "test_verify.py"]:
    copy(log / name, output / name)
copy(measurement / "results.json", output / "results.json")
copy(measurement / "preflight.json", output / "preflight.json")
copy(log / "baseline.py", output / "baseline.py")
for name in ["baseline", "baseline-v2", "clippy-v1", "measurement-v1", "measurement-v2"]:
    for src in sorted((log / name).iterdir()):
        if src.is_file() and src.name != "tests":
            copy(src, output / "excluded-and-baseline" / name / src.name)
stages = json.loads((verification / "stages.json").read_text())
assert len(stages) == 12 and all(s["exit_code"] == 0 for s in stages)
copy(verification / "stages.json", output / "validation/stages.json")
raw = []
for stage in stages:
    path = verification / stage["stage"]
    receipt = json.loads((path / "receipt.json").read_text())
    assert receipt["exit_code"] == 0 and receipt["source_unchanged"] is True
    assert receipt["source"]["commit"] == data["commit"]
    for name in ["receipt.json", "stderr.log"]:
        copy(path / name, output / "validation" / stage["stage"] / name)
    if stage["stage"].endswith("-preflight"):
        raw.append(path / "stdout.log")
    else:
        copy(path / "stdout.log", output / "validation" / stage["stage"] / "stdout.log")
for record in data["results"]:
    path = Path(record["raw_path"])
    assert sha(path) == record["raw_sha256"]
    raw.append(path)
    for name in ["receipt.json", "stderr.log"]:
        copy(path.parent / name, output / "measurement-receipts" / path.parent.name / name)
for path, digest in data["workers"].items():
    assert sha(Path(path)) == digest
    raw.append(Path(path))
write(output / "local-raw-manifest.json", {str(p): sha(p) for p in raw})
source = json.loads((verification / "vision-tests/receipt.json").read_text())["source"]
assert source["status"] == ""
assert all(sha(root / p) == digest for p, digest in source["files"].items())
write(output / "source.json", source)
conditions = []
for batch, samples, varying in itertools.product([1, 32, 256], [1, 8, 64], [False, True]):
    grouped = {}
    for worker in ["native", "wasm", "torch"]:
        rows = []
        for record in data["results"]:
            if record["worker"] != worker:
                continue
            matches = [c for c in record["cases"] if (c["batch"], c["samples"], c["varying"]) == (batch, samples, varying)]
            assert len(matches) == 1
            rows.append((record["block"], matches[0]))
        assert len(rows) == 6
        assert len({c["output_f32le_sha256"] for _, c in rows}) == 1
        grouped[worker] = dict(rows)
    ratios = {worker: [grouped["torch"][b]["median_ns"] / grouped[worker][b]["median_ns"] for b in range(6)]
              for worker in ["native", "wasm"]}
    conditions.append({"batch": batch, "samples": samples, "varying": varying,
        "median_ns": {worker: statistics.median(c["median_ns"] for c in rows.values()) for worker, rows in grouped.items()},
        "torch_over_runtime": {worker: {"geomean": gm(r), "min": min(r), "max": max(r)} for worker, r in ratios.items()}})
summary = {"measured_commit": data["commit"], "conditions": conditions,
    "records": 324, "preconditioning_records": 54,
    "intervals_per_record": 9, "iterations_per_interval": 4,
    "ratio_definition": "Within-block torch median / runtime median, then geometric mean over six balanced blocks; >1 favors Rust runtime",
    "aggregate": {worker: {"geomean": gm(c["torch_over_runtime"][worker]["geomean"] for c in conditions),
        "favorable_conditions": sum(c["torch_over_runtime"][worker]["geomean"] > 1 for c in conditions)} for worker in ["native", "wasm"]},
    "max_render_abs_vs_native": max(c["max_abs_vs_native_preflight"] for r in data["results"] for c in r["cases"]),
    "max_one_step_parameter_abs_vs_native": max(r["contracts"]["one_step_parameter_max_abs_vs_native"] for r in data["results"])}
write(output / "summary.json", summary)
print(json.dumps(summary, indent=2))
