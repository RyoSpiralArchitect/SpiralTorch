"""Frozen AB/BA positional-encoding grid, retaining all correctness failures."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time

root, backend = Path(sys.argv[1]), sys.argv[2]
assert backend in {"native", "wasm"}
out = root / backend
out.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
key = lambda c: (c["rows"], c["cols"], c["bands"], c["residual"])
expected = {(r, c, b, keep) for r in [1, 32, 1024] for c in [3, 6]
            for b in [0, 4, 10] for keep in [False, True]}
workers = {lane: root / (lane + "-build") / (
    "positional-geometry-fixture" if backend == "native" else
    "wasm/positional_geometry_fixture.js") for lane in ["baseline", "candidate"]}
inputs = dict(workers)
inputs["harness"] = Path(__file__)
if backend == "wasm":
    inputs["js"] = root / "wasm.cjs"
    inputs.update({lane + "_wasm": path.with_name("positional_geometry_fixture_bg.wasm")
                   for lane, path in workers.items()})
hashes = {name: sha(p) for name, p in inputs.items()}
env = dict(os.environ, RAYON_NUM_THREADS="4", SPIRAL_DETERMINISTIC="0",
           SPIRAL_DETERMINISTIC_REDUCTION="0")
for name in ["HOME", "SPIRALTORCH_AUTOTUNE", "SPIRALTORCH_AUTOTUNE_STORE"]:
    env.pop(name, None)
receipt = {"inputs": {k: str(v) for k, v in inputs.items()}, "sha256": hashes, "steps": [],
           "platform": platform.platform(), "machine": platform.machine(),
           "host_exclusivity": "unknown", "thermal_state": "unknown",
           "environment": {k: env[k] for k in ["RAYON_NUM_THREADS",
              "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION"]}}
reports = {}
for name, lane in [("warm-baseline", "baseline"), ("warm-candidate", "candidate"),
                   ("baseline-a", "baseline"), ("candidate-a", "candidate"),
                   ("candidate-b", "candidate"), ("baseline-b", "baseline")]:
    assert hashes == {k: sha(p) for k, p in inputs.items()}
    command = [str(workers[lane])] if backend == "native" else [
        "node", str(root / "wasm.cjs"), str(workers[lane])]
    start = time.monotonic()
    with (out / (name + ".raw.json")).open("xb") as stdout, (out / (name + ".stderr")).open("xb") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    receipt["steps"].append({"name": name, "lane": lane, "command": command,
        "exit_code": result.returncode, "seconds": time.monotonic() - start,
        "raw_sha256": sha(out / (name + ".raw.json"))})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()
    report = json.loads((out / (name + ".raw.json")).read_bytes())
    assert report["warmups"] == 3 and report["intervals"] == 15
    assert len(report["cases"]) == 36 and {key(c) for c in report["cases"]} == expected
    for case in report["cases"] + report.get("layouts", []):
        bits = case.pop("output_bits")
        case["output_elements"] = len(bits)
        case["output_sha256"] = hashlib.sha256(struct.pack("<" + "I" * len(bits), *bits)).hexdigest()
        assert len(bits) == case["rows"] * case["cols"] * (2 * case["bands"] + case["residual"])
    for case in report["cases"]:
        assert case["valid"] is True
        assert len(case["elapsed_ns"]) == 15
        assert all(math.isfinite(x) and x > 0 for x in case["elapsed_ns"])
    fields = report["contracts"]["fields"]
    assert len(fields) == 9 and {(f["layout"], f["seed_layout"]) for f in fields} == {
        (i, j) for i in range(3) for j in range(3)}
    if lane == "candidate":
        assert report["contracts"]["rope_history_independent"] is True
        assert all(all(f[k] is True for k in ["forward_equal", "gradient_equal", "update_equal",
                                            "input_gradient_zero"]) for f in fields)
    else:
        assert report["contracts"]["rope_history_independent"] is False
        assert any(f["gradient_equal"] is False for f in fields)
    if backend == "wasm":
        assert report["empty"] is True and report["signed_zero"] is True
        assert len(report["layouts"]) == 36
        assert {(c["rows"], c["cols"], c["bands"], c["residual"], c["layout"])
                for c in report["layouts"]} == {
            (r, c, b, k, l) for r, c in [(2, 3), (3, 6)] for b in [0, 4, 10]
            for k in [False, True] for l in range(3)}
        assert len(report["guards"]) == 3
        if lane == "candidate":
            assert all(c["valid"] is True for c in report["layouts"])
            assert all(c["correct"] is True for c in report["guards"])
        else:
            assert any(c["valid"] is False for c in report["layouts"])
            assert all(c["correct"] is False for c in report["guards"])
    (out / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    reports[name] = report
    print(backend, name, "validated", flush=True)
indexes = {n: {key(c): c for c in r["cases"]} for n, r in reports.items()}
comparisons = []
for condition in sorted(expected):
    for round_id in ["a", "b"]:
        a, b = [indexes[lane + "-" + round_id][condition] for lane in ["baseline", "candidate"]]
        comparisons.append({"condition": condition, "round": round_id,
            "ratio": statistics.median(a["elapsed_ns"]) / statistics.median(b["elapsed_ns"]),
            "bit_equal": a["output_sha256"] == b["output_sha256"]})
groups = {}
for size in ["all", "one_row", "multi_row"]:
    for bands in ["all", "zero", "nonzero"]:
        ratios = [r["ratio"] for r in comparisons
                  if (size == "all" or ((r["condition"][0] == 1) == (size == "one_row")))
                  and (bands == "all" or ((r["condition"][2] == 0) == (bands == "zero")))]
        groups[size + "/" + bands] = {"geomean": math.exp(statistics.mean(map(math.log, ratios))),
            "min": min(ratios), "max": max(ratios), "favorable": sum(v > 1 for v in ratios), "count": len(ratios)}
(out / "comparison.json").write_text(json.dumps({"comparisons": comparisons, "groups": groups}, indent=2) + "\n")
assert hashes == {k: sha(p) for k, p in inputs.items()}
print(json.dumps(groups, indent=2))
