"""Fixed AB/BA CPU dense comparison; retain all shapes, modes and timing intervals."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

SHAPES = [(1,31,8), (3,37,13), (4,5,16), (7,31,8), (8,31,8), (8,7,12),
          (17,37,29), (32,64,32), (96,128,96), (65,257,49), (129,129,97),
          (129,1025,33), (1,4096,1), (16,0,12), (0,31,8)]

def require(value, message):
    if not value:
        raise ValueError(message)

def identity(case):
    return tuple(case[name] for name in ["rows", "inner", "cols", "packed"])

def report(directory):
    comparisons = []
    for mode, threads, deterministic in [("serial", "1", "1"), ("parallel", "4", "0")]:
        documents = [json.loads((directory / f"{mode}-{name}.json").read_text())
                     for name in ["baseline-a", "candidate-a", "baseline-b", "candidate-b"]]
        mappings = []
        for doc in documents:
            require(doc["schema"] == "spiraltorch.cpu_dense_workspace.v1", "Unknown schema")
            require(doc["rayon_threads"] == threads and doc["deterministic"] == deterministic, "Wrong thread mode")
            require(len(doc["cases"]) == 30, "Incomplete case list")
            mapping = {identity(case): case for case in doc["cases"]}
            require(set(mapping) == {(*shape, p) for shape in SHAPES for p in [False, True]}, "Wrong or duplicate conditions")
            for case in mapping.values():
                require(case["bitwise_equal"] is True, "Sequential float32 reference mismatch")
                require(case["repetitions"] == 16, "Wrong repetitions")
                require(len(case["elapsed_ns"]) == 9 and all(math.isfinite(v) and v > 0 for v in case["elapsed_ns"]), "Invalid timing")
                require(all(isinstance(case[k], int) and case[k] >= 0 for k in ["allocation_calls", "allocated_bytes"]), "Invalid allocation counts")
            mappings.append(mapping)
        for key in mappings[0]:
            b1, c1, b2, c2 = [m[key] for m in mappings]
            comparisons.append({"mode": mode, "shape": list(key[:3]), "packed": key[3],
                "baseline_over_candidate": [statistics.median(b["elapsed_ns"]) / statistics.median(c["elapsed_ns"])
                                            for b, c in [(b1,c1), (b2,c2)]],
                "baseline_allocations": [[c["allocation_calls"], c["allocated_bytes"]] for c in [b1,b2]],
                "candidate_allocations": [[c["allocation_calls"], c["allocated_bytes"]] for c in [c1,c2]]})
    summaries = []
    for mode in ["serial", "parallel"]:
        for packed in [False, True]:
            ratios = [r for c in comparisons if c["mode"] == mode and c["packed"] == packed for r in c["baseline_over_candidate"]]
            summaries.append(dict(mode=mode, packed=packed, geometric_mean=statistics.geometric_mean(ratios),
                                  minimum=min(ratios), maximum=max(ratios), favorable=sum(r > 1 for r in ratios), count=len(ratios)))
    return dict(condition_runs=240, bitwise_passes=240, summaries=summaries, comparisons=comparisons,
                boundary="recorded CPU-only forward microbenchmark, not an autograd or GPU speed claim")

def measure(directory, baseline, candidate):
    directory.mkdir()
    receipt = {"steps": [], "binaries": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [baseline, candidate]}}
    for mode, threads, deterministic in [("serial", "1", "1"), ("parallel", "4", "0")]:
        env = dict(os.environ, RAYON_NUM_THREADS=threads, SPIRAL_DETERMINISTIC=deterministic,
                   SPIRAL_DETERMINISTIC_REDUCTION=deterministic, SPIRAL_DETERMINISTIC_SEED="17")
        env.pop("HOME", None)
        env.pop("SPIRALTORCH_AUTOTUNE_STORE", None)
        for name in ["baseline-a", "candidate-a", "candidate-b", "baseline-b"]:
            command = [str(baseline if name.startswith("baseline") else candidate)]
            with (directory / f"{mode}-{name}.json").open("x") as out, (directory / f"{mode}-{name}.stderr").open("x") as err:
                result = subprocess.run(command, env=env, stdout=out, stderr=err)
            receipt["steps"].append({"command": command, "mode": mode, "name": name, "exit_code": result.returncode})
            (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            result.check_returncode()
            print(mode, name, "complete", flush=True)
    (directory / "comparison.json").write_text(json.dumps(report(directory), indent=2) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["measure", "report"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.mode == "measure":
        measure(args.directory, args.baseline.resolve(), args.candidate.resolve())
    else:
        print(json.dumps(report(args.directory), indent=2))
