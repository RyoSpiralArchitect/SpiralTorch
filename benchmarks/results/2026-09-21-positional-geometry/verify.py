"""Offline byte integrity and recorded-contract checks, NOT numerical replay."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ORDER = ["warm-baseline", "warm-candidate", "baseline-a", "candidate-a", "candidate-b", "baseline-b"]
STAGES = ["format", "vision-clippy", "contracts", "changed-crates-clippy", "vision-wasm",
          "wasm-build", "bindgen", "wasm-gelu", "wasm-autograd", "python-build",
          "python-gelu", "python-autograd", "wgpu-linear", "wgpu-resident"]
EXPECTED = {(r, c, b, keep) for r in [1, 32, 1024] for c in [3, 6]
            for b in [0, 4, 10] for keep in [False, True]}
def require(condition, message):
    if not condition:
        raise ValueError(message)
def read(root, name):
    return json.loads((root / name).read_text())
def key(case):
    return (case["rows"], case["cols"], case["bands"], case["residual"])
def groups(comparisons):
    result = {}
    for size in ["all", "one_row", "multi_row"]:
        for bands in ["all", "zero", "nonzero"]:
            values = [c["ratio"] for c in comparisons
                if (size == "all" or ((c["condition"][0] == 1) == (size == "one_row")))
                and (bands == "all" or ((c["condition"][2] == 0) == (bands == "zero")))]
            result[size + "/" + bands] = {"geomean": math.exp(statistics.mean(map(math.log, values))),
                "min": min(values), "max": max(values), "favorable": sum(v > 1 for v in values), "count": len(values)}
    return result
def valid_cases(report):
    require(report["warmups"] == 3 and report["intervals"] == 15, "protocol changed")
    cases = report["cases"]
    require(len(cases) == 36 and {key(c) for c in cases} == EXPECTED, "missing/duplicate conditions")
    for c in cases:
        require(c["valid"] is True, "invalid output record")
        require(len(c["elapsed_ns"]) == 15 and all(math.isfinite(x) and x > 0 for x in c["elapsed_ns"]), "invalid timing")
    return {key(c): c for c in cases}
def verify(root):
    manifest = read(root, "manifest.json")
    require(manifest["numerical_replay"] is False, "unsupported replay claim")
    files = manifest["files"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    require(actual == set(files), "file inventory mismatch")
    for name, info in files.items():
        data = (root / name).read_bytes()
        require(len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"], "hash mismatch: " + name)
    p = read(root, "provenance.json")
    require(p["baseline"] == "a98d81c1513b42ea02b57691d02a3bfceb400b8f", "unexpected baseline")
    require(p["candidate"] == "6899536b46bea49d70d901330451bfbe275cf446", "unexpected candidate")
    require(p["numerical_replay"] is False, "unsupported replay claim")
    sources = {}
    fixtures = []
    for lane, prefix in [("baseline", "admitted-baseline"), ("candidate", "candidate")]:
        for kind in ["native-build", "wasm-build", "bindgen"]:
            receipt = read(root, prefix + "-" + kind + "/receipt.json")
            require(receipt["exit_code"] == 0 and receipt["source_unchanged"] is True, "failed build")
            require(receipt["source"]["commit"] == p[lane] and receipt["source"]["status"] == "", "unclean build source")
            if kind == "native-build":
                sources[lane] = receipt["source"]["files"]
            require(receipt["source"]["files"] == sources[lane], "native/WASM source mismatch")
            fixtures.append(receipt["source"]["fixture_files"])
    require(all(f == fixtures[0] for f in fixtures), "fixture source mismatch")
    for name, digest in fixtures[0].items():
        require(hashlib.sha256((root / "reproduction/fixture" / name).read_bytes()).hexdigest()
                == digest, "published fixture differs from built fixture")
    require(read(root, "verification/stages.json") == [{"name": n, "exit_code": 0} for n in STAGES], "incomplete verification")
    for name in STAGES:
        receipt = read(root, "verification/" + name + "/receipt.json")
        require(receipt["exit_code"] == 0 and receipt["source_unchanged"] is True, "failed check")
        require(receipt["source"]["commit"] == p["candidate"] and receipt["source"]["files"] == sources["candidate"], "check source mismatch")
    gpu = read(root, "verification/wgpu-resident/receipt.json")["command"]
    require("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1" in gpu
            and "resident_graph_forward" in gpu, "strict resident GPU check not opted in")
    for failed in ["baseline-native-build", "baseline-wasm-build", "baseline-contract",
                   "frozen-baseline-native-build", "seeded-training-preflight", "positive-density-preflight",
                   "runtime-contracts", "runtime-contracts-corrected"]:
        require(read(root, failed + "/receipt.json")["exit_code"] != 0, "negative attempt lost: " + failed)
    native = {}
    for backend in ["native", "wasm"]:
        receipt = read(root, backend + "/receipt.json")
        require(receipt["sha256"]["harness"] == hashlib.sha256(
            (root / "reproduction/measure.py").read_bytes()).hexdigest(), "measurement harness differs")
        if backend == "wasm":
            require(receipt["sha256"]["js"] == hashlib.sha256(
                (root / "reproduction/wasm.cjs").read_bytes()).hexdigest(), "WASM harness differs")
        require([s["name"] for s in receipt["steps"]] == ORDER, "incomplete AB/BA cohort")
        indexes = {}
        for step in receipt["steps"]:
            name, lane = step["name"], step["lane"]
            require(step["exit_code"] == 0 and step["raw_sha256"] == p["raw_sha256"][backend + "/" + name + ".raw.json"], "worker/raw mismatch")
            report = read(root, backend + "/" + name + ".json")
            indexes[name] = valid_cases(report)
            for c in report["cases"]:
                require(c["output_elements"] == c["rows"] * c["cols"] * (2*c["bands"] + c["residual"]), "output volume mismatch")
            fields = report["contracts"]["fields"]
            require(len(fields) == 9 and {(f["layout"], f["seed_layout"]) for f in fields} == {
                (i,j) for i in range(3) for j in range(3)}, "field layout pairs missing")
            require(report["contracts"]["rope_history_independent"] is (lane == "candidate"), "cache negative control changed")
            if lane == "candidate":
                require(all(all(f[k] is True for k in ["forward_equal", "gradient_equal", "update_equal", "input_gradient_zero"]) for f in fields),
                        "field contract failed")
            else:
                require(any(f["gradient_equal"] is False for f in fields), "field negative control missing")
            if backend == "wasm":
                layouts = report["layouts"]
                require(len(layouts) == 36 and {(key(c), c["layout"]) for c in layouts} == {
                    ((r,c,b,k),l) for r,c in [(2,3),(3,6)] for b in [0,4,10] for k in [False,True] for l in range(3)},
                    "missing WASM layout contract")
                require(report["empty"] is True and report["signed_zero"] is True, "empty/signed-zero failed")
                require({g["label"] for g in report["guards"]} == {"nerf_input", "nerf_phase", "nerf_frequency_count"}
                        and len(report["guards"]) == 3, "guards missing")
                if lane == "candidate":
                    require(all(c["valid"] is True for c in layouts) and all(c["correct"] is True for c in report["guards"]), "WASM contract failed")
                else:
                    require(any(c["valid"] is False for c in layouts) and all(c["correct"] is False for c in report["guards"]), "WASM negative control missing")
        for lane in ["baseline", "candidate"]:
            worker = "positional-geometry-fixture" if backend == "native" else "wasm/positional_geometry_fixture.js"
            require(receipt["sha256"][lane] == p["workers"][lane][worker], "worker identity mismatch")
            if backend == "wasm":
                require(receipt["sha256"][lane + "_wasm"] == p["workers"][lane]["wasm/positional_geometry_fixture_bg.wasm"], "WASM identity mismatch")
        comparisons = []
        for condition in sorted(EXPECTED):
            for round_id in ["a", "b"]:
                a,b = [indexes[lane+"-"+round_id][condition] for lane in ["baseline", "candidate"]]
                comparisons.append({"condition": list(condition), "round": round_id,
                    "ratio": statistics.median(a["elapsed_ns"]) / statistics.median(b["elapsed_ns"]),
                    "bit_equal": a["output_sha256"] == b["output_sha256"]})
        comparison = read(root, backend + "/comparison.json")
        require(comparison["comparisons"] == comparisons and comparison["groups"] == groups(comparisons), "summary mismatch")
        if backend == "native":
            native = indexes
    torch = read(root, "torch/stdout.log")
    require(torch["threads"] == 4 and torch["interop_threads"] == 1, "Torch threads differ")
    require([r["round"] for r in torch["reports"]] == ["warm", "a", "b"], "Torch cohort incomplete")
    receipt = read(root, "torch/receipt.json")
    require(receipt["exit_code"] == 0 and receipt["source_unchanged"] is True
            and receipt["source"]["files"] == sources["candidate"], "Torch source identity differs")
    comparisons = []
    for report in torch["reports"]:
        index = valid_cases(dict(report, warmups=torch["warmups"], intervals=torch["intervals"]))
        if report["round"] == "warm":
            continue
        for condition, c in sorted(index.items()):
            comparisons.append({"condition": list(condition), "round": report["round"],
                "ratio": statistics.median(c["elapsed_ns"]) / statistics.median(native["candidate-"+report["round"]][condition]["elapsed_ns"])})
    comparison = read(root, "torch/comparison.json")
    require(comparison["comparisons"] == comparisons and comparison["groups"] == groups(comparisons), "Torch summary mismatch")
    return {"ok": True, "files": len(files), "measured_conditions": 360, "preconditioning_conditions": 180,
            "numerical_replay": False, "raw_arrays": "local only; recorded hashes do not constitute replay"}

if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent), indent=2))
