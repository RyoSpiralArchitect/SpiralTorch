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
    return (case["rows"], case["cols"], case["depth"])
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
    require(provenance["baseline"] == "b6dcfb45e8d55fe408b1891a627b9d47e8bca268", "wrong baseline")
    require(provenance["candidate"] == "d17fdcdbee71a351f5521fa4e626e9948399b01e", "wrong candidate")
    require(provenance["baseline_contract"] == "afa83a24d06e4ccf2497ef1102b800935bfec64a", "wrong corrected baseline")
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
    require(baseline_test["exit_code"] != 0 and baseline_test["source_unchanged"] is True
            and baseline_test["source"]["commit"] == provenance["baseline"], "failed baseline attempt lost")
    corrected = read(root, "baseline-contract-corrected/receipt.json")
    require(corrected["exit_code"] == 0 and corrected["source_unchanged"] is True
            and corrected["source"]["commit"] == provenance["baseline_contract"], "corrected baseline contract failed")
    baseline_source = read(root, "baseline-native-build/receipt.json")["source"]["files"]
    for name, digest in baseline_source.items():
        if name != "crates/st-nn/tests/gelu_layout_contract.rs":
            require(corrected["source"]["files"][name] == digest, "baseline runtime changed during test correction")
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
        expected = {(r, c, d) for r, c in SHAPES for d in [1, 4, 16]}
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
                require(report["error_cases"] == 3 and report["empty_shapes"] == 2 and report["signed_zero"] is True, "missing WASM guards")
                require(len(report["contracts"]) == 24 and
                        {(c["rows"], c["cols"], c["depth"], c["nested"], c["layout"]) for c in report["contracts"]} ==
                        {(r, c, d, n, l) for r, c in [(2, 6), (33, 195)] for d in [1, 4] for n in [False, True] for l in range(3)},
                        "missing layout contract")
                for c in report["contracts"]:
                    require(c["valid"] is True and c["output_elements"] == c["rows"] * c["cols"], "invalid layout contract")
                contracts.extend(report["contracts"])
        for lane in ["baseline", "candidate"]:
            worker = "cpu_gelu_chain" if backend == "native" else "wasm/host_forward_measurement_shim.js"
            require(receipt["sha256"][lane] == provenance["workers"][lane][worker], "worker identity mismatch")
            if backend == "wasm":
                require(receipt["sha256"][lane + "_wasm"] ==
                        provenance["workers"][lane]["wasm/host_forward_measurement_shim_bg.wasm"], "WASM identity mismatch")
        for condition in expected:
            require(len({i[condition]["output_sha256"] for i in indexes.values()}) == 1, "output bits differ")
        if backend == "wasm":
            for r, c in [(2, 6), (33, 195)]:
                for depth in [1, 4]:
                    for field in ["output_sha256", "gradient_sha256"]:
                        require(len({v[field] for v in contracts if key(v) == (r, c, depth)}) == 1, "layout output/gradient bits differ")
        else:
            for name, index in indexes.items():
                for (rows, cols, depth), c in index.items():
                    count = depth if name.startswith("baseline") or name == "warm-baseline" else 1
                    require(c["allocation_calls"] == 3 * count and c["allocated_bytes"] == (rows * cols * 4 + 80) * count,
                            "allocation observation mismatch on recorded host")
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
        require(len(report["cases"]) == 27 and {key(c) for c in report["cases"]} == set(native["candidate-a"]), "Torch shapes differ")
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
    return {"ok": True, "files": len(manifest), "measured_conditions": 630, "preconditioning_conditions": 315,
            "numerical_replay": False, "raw_arrays": "local only; hashes, not embedded payloads"}

if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent), indent=2))
