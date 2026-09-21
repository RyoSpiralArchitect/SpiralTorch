"""Offline integrity and contract checks, not a numerical/performance rerun."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ORDER = ["warm-baseline", "warm-candidate", "baseline-a", "candidate-a", "candidate-b", "baseline-b"]
SHAPES = [(1, 1), (1, 8), (1, 32), (1, 64), (8, 3072), (32, 3072), (64, 1024), (17, 195), (65, 97)]
STAGES = ["format", "tensor", "nn", "clippy", "wasm-clippy", "wasm-build", "bindgen",
          "wasm-gelu", "wasm-autograd", "python-build", "python-gelu", "python-autograd", "wgpu-layout"]
def require(condition, message):
    if not condition:
        raise ValueError(message)
def read(root, name):
    return json.loads((root / name).read_text())
def key(case):
    return (case["rows"], case["cols"], case["backward"])
def valid_case(case, intervals):
    require(case["valid"] is True, "invalid numerical case")
    times = case["elapsed_ns"]
    require(len(times) == intervals and all(math.isfinite(x) and x > 0 for x in times), "invalid timing")
def groups(rows):
    result = {}
    for backward in sorted({r["condition"][2] for r in rows}):
        for size in ["all", "tiny", "non_tiny"]:
            values = [r["ratio"] for r in rows if r["condition"][2] == backward and
                      (size == "all" or ((r["condition"][0] * r["condition"][1] <= 64) == (size == "tiny")))]
            result[f"{backward}/{size}"] = {"geomean": math.exp(statistics.mean(map(math.log, values))),
                "min": min(values), "max": max(values), "favorable": sum(x > 1 for x in values), "count": len(values)}
    return result
def verify(root):
    manifest = read(root, "manifest.json")["files"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    require(actual == set(manifest), "file inventory mismatch")
    for name, info in manifest.items():
        data = (root / name).read_bytes()
        require(len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"], "hash mismatch: " + name)
    provenance = read(root, "provenance.json")
    require(provenance["baseline"] == "5c57b2326525b0b8529724e5e0c9ba4542c8d3bb", "wrong baseline")
    require(provenance["candidate"] == "6bf69c8656e31f63d3021b44044f67a0b2e4dc2b", "wrong candidate")
    require(provenance["numerical_replay"] is False, "unsupported replay claim")
    for lane in ["baseline", "candidate"]:
        for kind in ["native-build", "wasm-build", "bindgen"]:
            receipt = read(root, lane + "-" + kind + "/receipt.json")
            require(receipt["exit_code"] == 0 and receipt["source_unchanged"] is True, "failed build")
            require(receipt["source"]["commit"] == provenance[lane], "build source mismatch")
        a = read(root, lane + "-native-build/receipt.json")["source"]["files"]
        b = read(root, lane + "-wasm-build/receipt.json")["source"]["files"]
        require(a == b, "native/WASM source mismatch")
    baseline_test = read(root, "baseline-contract/receipt.json")
    require(baseline_test["exit_code"] == 0 and baseline_test["source_unchanged"] is True
            and baseline_test["source"]["commit"] == provenance["baseline"], "baseline contract failed")
    require(read(root, "verification/stages.json") == [{"name": n, "exit_code": 0} for n in STAGES], "incomplete validation")
    candidate_source = read(root, "candidate-native-build/receipt.json")["source"]["files"]
    for stage in STAGES:
        receipt = read(root, "verification/" + stage + "/receipt.json")
        require(receipt["exit_code"] == 0 and receipt["source_unchanged"] is True, "failed validation")
        require(receipt["source"]["commit"] == provenance["candidate"] and
                receipt["source"]["files"] == candidate_source, "validation source mismatch")
    require("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1" in
            read(root, "verification/wgpu-layout/receipt.json")["command"], "GPU test was not opted in")
    native = {}
    for backend in ["native", "wasm"]:
        expected = {(r, c, b) for r, c in SHAPES for b in ([False, True] if backend == "native" else [False])}
        receipt = read(root, backend + "/receipt.json")
        require([s["name"] for s in receipt["steps"]] == ORDER, "incomplete cohort")
        indexes, contracts = {}, []
        for step in receipt["steps"]:
            name = step["name"]
            require(step["exit_code"] == 0, "failed worker")
            require(step["raw_sha256"] == provenance["raw_sha256"][backend + "/" + name + ".raw.json"], "raw identity mismatch")
            report = read(root, backend + "/" + name + ".json")
            require(report["warmups"] == 3 and report["intervals"] == 15, "protocol mismatch")
            require(len(report["cases"]) == len(expected) and {key(c) for c in report["cases"]} == expected, "condition mismatch")
            for case in report["cases"]:
                valid_case(case, 15)
                require(case["output_elements"] == case["rows"] * case["cols"], "output shape mismatch")
            indexes[name] = {key(c): c for c in report["cases"]}
            if backend == "wasm":
                require(report["errors_checked"] == 11 and report["empty_shapes_checked"] == 2, "missing WASM guards")
                require(len(report["contracts"]) == 6 and {(c["layout"], c["outliers"]) for c in report["contracts"]} ==
                        {(l, o) for l in range(3) for o in [False, True]}, "missing layout contract")
                for c in report["contracts"]:
                    require(c["valid"] is True and c["output_elements"] == (12294 if c["outliers"] else 12288), "invalid wide contract")
                contracts.extend(report["contracts"])
        for lane in ["baseline", "candidate"]:
            worker = "cpu_gelu" if backend == "native" else "wasm/gelu_measurement_shim.js"
            require(receipt["sha256"][lane] == provenance["workers"][lane][worker], "worker identity mismatch")
            if backend == "wasm":
                require(receipt["sha256"][lane + "_wasm"] ==
                        provenance["workers"][lane]["wasm/gelu_measurement_shim_bg.wasm"], "WASM identity mismatch")
        for condition in expected:
            require(len({i[condition]["output_sha256"] for i in indexes.values()}) == 1, "output bits differ")
        for outliers in [False, True]:
            if backend == "wasm":
                require(len({c["output_sha256"] for c in contracts if c["outliers"] == outliers}) == 1, "wide output bits differ")
        comparisons = []
        for condition in sorted(expected):
            for round_id in ["a", "b"]:
                a, b = [indexes[lane + "-" + round_id][condition] for lane in ["baseline", "candidate"]]
                comparisons.append({"condition": list(condition), "round": round_id,
                                    "ratio": statistics.median(a["elapsed_ns"]) / statistics.median(b["elapsed_ns"])})
        comparison = read(root, backend + "/comparison.json")
        require(comparison["comparisons"] == comparisons and comparison["groups"] == groups(comparisons), "comparison mismatch")
        if backend == "native":
            native = indexes
    torch = read(root, "torch/stdout.log")
    torch_receipt = read(root, "torch/receipt.json")
    require(torch_receipt["exit_code"] == 0 and torch_receipt["source_unchanged"] is True
            and torch_receipt["source"]["commit"] == provenance["candidate"], "Torch validation identity mismatch")
    require(torch["threads"] == 4 and torch["interop_threads"] == 1, "Torch thread mismatch")
    require([r["round"] for r in torch["reports"]] == ["warm", "a", "b"], "incomplete Torch cohort")
    comparisons = []
    for report in torch["reports"]:
        require(len(report["cases"]) == 18 and {key(c) for c in report["cases"]} == set(native["candidate-a"]), "Torch shapes differ")
        for case in report["cases"]:
            valid_case(case, 15)
        if report["round"] == "warm":
            continue
        for k, c in sorted({key(c): c for c in report["cases"]}.items()):
            comparisons.append({"condition": list(k), "round": report["round"], "ratio":
                statistics.median(c["elapsed_ns"]) / statistics.median(native["candidate-" + report["round"]][k]["elapsed_ns"])})
    comparison = read(root, "torch/comparison.json")
    require(comparison["comparisons"] == comparisons and comparison["groups"] == groups(comparisons), "Torch comparison mismatch")
    sys.path.insert(0, str(root / "reproduction"))
    from nn_measure import compare
    require(read(root, "nn/comparison.json") == json.loads(json.dumps(compare(root / "nn", 90))), "NN comparison mismatch")
    receipt = read(root, "nn/receipt.json")
    require([s["name"] for s in receipt["steps"]] == ORDER and all(s["exit_code"] == 0 for s in receipt["steps"]), "incomplete NN cohort")
    for lane in ["baseline", "candidate"]:
        require(receipt["sha256"][lane] == provenance["workers"][lane]["cpu_nn_layout"], "NN worker identity mismatch")
    return {"ok": True, "files": len(manifest), "measured_conditions": 504, "preconditioning_conditions": 252,
            "numerical_replay": False, "raw_arrays": "local only; hashes, not embedded payloads"}

if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent), indent=2))
