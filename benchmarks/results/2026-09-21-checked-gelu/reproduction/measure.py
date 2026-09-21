"""Fixed AB/BA CPU GELU grid; retain raw values locally, publish hashes/timings."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

root = Path(sys.argv[1])
out = root / "gelu"
out.mkdir()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
workers = {lane: root / (lane + "-build") / "cpu_gelu" for lane in ["baseline", "candidate"]}
hashes = {lane: sha(path) for lane, path in workers.items()}
env = dict(os.environ, RAYON_NUM_THREADS="4", SPIRAL_DETERMINISTIC="0", SPIRAL_DETERMINISTIC_REDUCTION="0")
for key in ["HOME", "SPIRALTORCH_AUTOTUNE_STORE", "SPIRALTORCH_AUTOTUNE"]:
    env.pop(key, None)
expected = {(r, c, b) for r, c in [(1, 64), (8, 3072), (32, 3072), (64, 1024), (17, 195), (65, 97)] for b in [False, True]}
reports, receipts = {}, []
order = [("warm-baseline", "baseline"), ("warm-candidate", "candidate"),
         ("baseline-a", "baseline"), ("candidate-a", "candidate"),
         ("candidate-b", "candidate"), ("baseline-b", "baseline")]
for name, lane in order:
    assert sha(workers[lane]) == hashes[lane]
    start = time.monotonic()
    with (out / (name + ".raw.json")).open("xb") as stdout, (out / (name + ".stderr")).open("xb") as stderr:
        result = subprocess.run([str(workers[lane])], env=env, stdout=stdout, stderr=stderr)
    receipts.append({"name": name, "lane": lane, "exit_code": result.returncode,
                     "seconds": time.monotonic() - start, "raw_sha256": sha(out / (name + ".raw.json"))})
    (out / "receipt.json").write_text(json.dumps({"workers": hashes, "steps": receipts}, indent=2) + "\n")
    result.check_returncode()
    report = json.loads((out / (name + ".raw.json")).read_bytes())
    assert report["intervals"] == 15 and report["warmups"] == 3 and report["repetitions"] == 8
    assert len(report["cases"]) == len(expected)
    assert {(c["rows"], c["cols"], c["backward"]) for c in report["cases"]} == expected
    for case in report["cases"]:
        bits = case.pop("output_bits")
        assert len(bits) == case["rows"] * case["cols"]
        case["output_sha256"] = hashlib.sha256(struct.pack("<" + "I" * len(bits), *bits)).hexdigest()
        assert case["valid"] and len(case["elapsed_ns"]) == 15
        assert all(math.isfinite(x) and x > 0 for x in case["elapsed_ns"])
    reports[name] = report
    (out / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(name, "passed", flush=True)
indexes = {name: {(c["rows"], c["cols"], c["backward"]): c for c in report["cases"]} for name, report in reports.items()}
comparisons = []
for condition in sorted(expected):
    assert len({index[condition]["output_sha256"] for index in indexes.values()}) == 1
    for round_id in ["a", "b"]:
        a, b = [indexes[lane + "-" + round_id][condition] for lane in ["baseline", "candidate"]]
        comparisons.append({"condition": condition, "round": round_id,
                            "ratio": statistics.median(a["elapsed_ns"]) / statistics.median(b["elapsed_ns"]),
                            "baseline_allocations": a["allocation_calls"], "candidate_allocations": b["allocation_calls"],
                            "baseline_bytes": a["allocated_bytes"], "candidate_bytes": b["allocated_bytes"]})
groups = {}
for backward in [False, True]:
    values = [row["ratio"] for row in comparisons if row["condition"][2] == backward]
    groups[str(backward)] = {"geomean": math.exp(statistics.mean(map(math.log, values))),
                            "min": min(values), "max": max(values), "favorable": sum(x > 1 for x in values), "count": len(values)}
(out / "comparison.json").write_text(json.dumps({"comparisons": comparisons, "groups": groups}, indent=2) + "\n")
print(json.dumps(groups, indent=2))
