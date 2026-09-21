"""Fixed AB/BA grid; raw arrays stay local and every condition is retained."""
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
shapes = [(1, 1), (1, 8), (1, 32), (1, 64), (8, 3072), (32, 3072), (64, 1024), (17, 195), (65, 97)]
expected = {(r, c, d) for r, c in shapes for d in [1, 4, 16]}
workers = {lane: root / (lane + "-build") / ("cpu_gelu_chain" if backend == "native" else "wasm/host_forward_measurement_shim.js")
           for lane in ["baseline", "candidate"]}
inputs = dict(workers)
if backend == "wasm":
    inputs.update({lane + "_wasm": path.with_name("host_forward_measurement_shim_bg.wasm") for lane, path in workers.items()})
    inputs["harness"] = root / "wasm_chain.cjs"
hashes = {name: sha(p) for name, p in inputs.items()}
env = dict(os.environ, RAYON_NUM_THREADS="4", SPIRAL_DETERMINISTIC="0", SPIRAL_DETERMINISTIC_REDUCTION="0")
for key in ["HOME", "SPIRALTORCH_AUTOTUNE_STORE", "SPIRALTORCH_AUTOTUNE"]:
    env.pop(key, None)
receipt = {"inputs": {k: str(v) for k, v in inputs.items()}, "sha256": hashes, "steps": [],
           "platform": platform.platform(), "machine": platform.machine(), "host_exclusivity": "unknown",
           "thermal_state": "unknown", "environment": {k: env[k] for k in
               ["RAYON_NUM_THREADS", "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION"]}}
reports = {}
for name, lane in [("warm-baseline", "baseline"), ("warm-candidate", "candidate"),
                   ("baseline-a", "baseline"), ("candidate-a", "candidate"),
                   ("candidate-b", "candidate"), ("baseline-b", "baseline")]:
    assert hashes == {name: sha(p) for name, p in inputs.items()}
    command = [str(workers[lane])] if backend == "native" else ["node", str(root / "wasm_chain.cjs"), str(workers[lane])]
    start = time.monotonic()
    with (out / (name + ".raw.json")).open("xb") as stdout, (out / (name + ".stderr")).open("xb") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    receipt["steps"].append({"name": name, "lane": lane, "command": command, "exit_code": result.returncode,
                             "seconds": time.monotonic() - start, "raw_sha256": sha(out / (name + ".raw.json"))})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()
    report = json.loads((out / (name + ".raw.json")).read_bytes())
    assert report["warmups"] == 3 and report["intervals"] == 15
    assert len(report["cases"]) == len(expected)
    assert {(c["rows"], c["cols"], c["depth"]) for c in report["cases"]} == expected
    for case in [*report["cases"], *report.get("contracts", [])]:
        bits = case.pop("output_bits")
        case["output_elements"] = len(bits)
        case["output_sha256"] = hashlib.sha256(struct.pack("<" + "I" * len(bits), *bits)).hexdigest()
        assert case["valid"] is True
        if "gradient_bits" in case:
            gradient = case.pop("gradient_bits")
            assert len(gradient) == len(bits)
            case["gradient_sha256"] = hashlib.sha256(struct.pack("<" + "I" * len(gradient), *gradient)).hexdigest()
    for case in report["cases"]:
        assert case["output_elements"] == case["rows"] * case["cols"]
        assert len(case["elapsed_ns"]) == 15 and all(math.isfinite(x) and x > 0 for x in case["elapsed_ns"])
    if backend == "wasm":
        assert report["error_cases"] == 3 and report["empty_shapes"] == 2 and report["signed_zero"] is True
        assert len(report["contracts"]) == 24
        assert {(c["rows"], c["cols"], c["depth"], c["nested"], c["layout"]) for c in report["contracts"]} == {
            (r, c, d, n, l) for r, c in [(2, 6), (33, 195)] for d in [1, 4] for n in [False, True] for l in range(3)}
    (out / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    reports[name] = report
    print(backend, name, "passed", flush=True)
indexes = {name: {(c["rows"], c["cols"], c["depth"]): c for c in r["cases"]} for name, r in reports.items()}
if backend == "wasm":
    for shape in [(2, 6), (33, 195)]:
        for depth in [1, 4]:
            for field in ["output_sha256", "gradient_sha256"]:
                assert len({c[field] for r in reports.values() for c in r["contracts"]
                            if (c["rows"], c["cols"]) == shape and c["depth"] == depth}) == 1
comparisons = []
for condition in sorted(expected):
    assert len({index[condition]["output_sha256"] for index in indexes.values()}) == 1
    for round_id in ["a", "b"]:
        a, b = [indexes[lane + "-" + round_id][condition] for lane in ["baseline", "candidate"]]
        comparisons.append({"condition": condition, "round": round_id,
                            "ratio": statistics.median(a["elapsed_ns"]) / statistics.median(b["elapsed_ns"])})
groups = {}
for depth in [1, 4, 16]:
    for size in ["all", "tiny", "non_tiny"]:
        ratios = [r["ratio"] for r in comparisons if r["condition"][2] == depth
                  and (size == "all" or ((r["condition"][0] * r["condition"][1] <= 64) == (size == "tiny")))]
        groups[f"{depth}/{size}"] = {"geomean": math.exp(statistics.mean(map(math.log, ratios))),
            "min": min(ratios), "max": max(ratios), "favorable": sum(v > 1 for v in ratios), "count": len(ratios)}
(out / "comparison.json").write_text(json.dumps({"comparisons": comparisons, "groups": groups}, indent=2) + "\n")
assert hashes == {name: sha(p) for name, p in inputs.items()}
print(json.dumps(groups, indent=2))
