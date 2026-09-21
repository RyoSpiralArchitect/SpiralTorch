"""Untrimmed AB/BA measurements. Workers are immutable and models stay on CPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def key(case):
    return tuple(case.get(field) for field in ["operation", "rows", "inner", "cols", "backend", "invalidate"])

def compare(directory, count):
    reports = {name: json.loads((directory / (name + ".json")).read_text()) for name in
               ["baseline-a", "candidate-a", "candidate-b", "baseline-b", "warm-baseline", "warm-candidate"]}
    indexes = {}
    for name, report in reports.items():
        assert len(report["cases"]) == count
        index = {}
        for case in report["cases"]:
            assert key(case) not in index
            assert case["valid"] is True
            assert len(case["elapsed_ns"]) == 9
            assert all(math.isfinite(v) and v > 0 for v in case["elapsed_ns"])
            index[key(case)] = case
        indexes[name] = index
    conditions = set(indexes["baseline-a"])
    assert all(set(index) == conditions for index in indexes.values())
    comparisons = []
    for round_id in ["a", "b"]:
        for condition in sorted(conditions, key=str):
            baseline, candidate = (indexes[name + "-" + round_id][condition] for name in ["baseline", "candidate"])
            if "output_sha256" in baseline:
                assert baseline["output_sha256"] == candidate["output_sha256"]
            comparisons.append({"condition": condition, "round": round_id,
                                "ratio": statistics.median(baseline["elapsed_ns"]) / statistics.median(candidate["elapsed_ns"]),
                                "baseline": baseline, "candidate": candidate})
    groups = {}
    for row in comparisons:
        op, _, _, _, backend, invalidate = row["condition"]
        group = f"{op}/{backend}/{invalidate}"
        groups.setdefault(group, []).append(row["ratio"])
    return {"comparisons": comparisons, "groups": {
        name: {"geomean": math.exp(statistics.mean(math.log(v) for v in values)),
               "minimum": min(values), "maximum": max(values), "favorable": sum(v > 1 for v in values), "count": len(values)}
        for name, values in sorted(groups.items())}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--wasm-harness", type=Path)
    args = parser.parse_args()
    args.directory.mkdir()
    environment = dict(os.environ, RAYON_NUM_THREADS="4", SPIRAL_DETERMINISTIC="0", SPIRAL_DETERMINISTIC_REDUCTION="0")
    environment.pop("HOME", None)
    environment.pop("SPIRALTORCH_AUTOTUNE_STORE", None)
    inputs = {"baseline": args.baseline, "candidate": args.candidate}
    hashes = {name: digest(path) for name, path in inputs.items()}
    if args.wasm_harness:
        hashes.update({name + "_wasm": digest(path.with_name(path.stem + "_bg.wasm")) for name, path in inputs.items()})
        hashes["harness"] = digest(args.wasm_harness)
        assert hashes["baseline_wasm"] != hashes["candidate_wasm"]
    else:
        assert hashes["baseline"] != hashes["candidate"]
    receipt = {"inputs": {name: str(path) for name, path in inputs.items()}, "sha256": hashes,
               "environment": {name: environment[name] for name in ["RAYON_NUM_THREADS", "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION"]},
               "home_removed": True, "steps": [], "host_exclusivity": "unknown", "thermal_state": "unknown"}
    for name, revision in [("warm-baseline", "baseline"), ("warm-candidate", "candidate"),
                           ("baseline-a", "baseline"), ("candidate-a", "candidate"),
                           ("candidate-b", "candidate"), ("baseline-b", "baseline")]:
        command = [str(inputs[revision])]
        if args.wasm_harness:
            command = ["node", str(args.wasm_harness), *command]
        start = time.monotonic()
        print("START", name, flush=True)
        with (args.directory / (name + ".json")).open("x") as out, (args.directory / (name + ".stderr")).open("x") as err:
            result = subprocess.run(command, env=environment, stdout=out, stderr=err)
        receipt["steps"].append({"name": name, "command": command, "exit_code": result.returncode, "seconds": time.monotonic() - start})
        (args.directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result.check_returncode()
        print("END", name, flush=True)
    result = compare(args.directory, 12 if args.wasm_harness else 90)
    (args.directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["groups"], indent=2))

if __name__ == "__main__":
    main()
