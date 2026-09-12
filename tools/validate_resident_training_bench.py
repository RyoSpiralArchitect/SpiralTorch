#!/usr/bin/env python3
"""Read-only validation and compact summaries of frozen training measurements.

No GPU work, retries, sample selection, or imported evidence as instructions.
The recorded native torch captures are the independent numerical reference;
this validator does not claim to re-execute them or attest a browser device.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_resident_training_vs_torch as bench
import validate_resident_training_vs_torch as reference

CADENCES = ("immediate", "deferred")


def require(value, message):
    if not value:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def manifest_binding(manifest, source):
    binding = bench.audit.validate_source_binding(
        dict(schema=bench.audit.BUILD_IDENTITY_SCHEMA, manifest=manifest), source)
    require(binding["valid"], "build manifest does not match the frozen source")
    return binding


def summarize_case(row, config, lanes, *, learner=False, seed_fusion=False):
    require(row["config"] == config and row["fixture"]["config"] == config, "recipe differs")
    samples = row["samples"]
    require(len(samples) == 20, "expected two warmups and eight retained blocks per cadence")
    for index, sample in enumerate(samples):
        optimizers = sample.get("learner_optimizers", {lane: None for lane in lanes})
        require(set(optimizers) == set(lanes) and all(optimizers[lane] == config.get("learner_optimizer") for lane in lanes),
                "per-interval learner optimizer differs")
        selected = sample.get("fused_learner_seeds", {lane: False for lane in lanes})
        require(set(selected) == set(lanes) and all(selected[lane] is (seed_fusion and lane == "candidate") for lane in lanes),
                "per-interval seed fusion differs")
        cadence, block = CADENCES[index // 10], index % 10
        rotation = (block + config["seed"]) % len(lanes)
        order = lanes[rotation:] + lanes[:rotation]
        require(sample["cadence"] == cadence and type(sample["block"]) is int and
                sample["block"] == block and sample["warmup"] is (block < 2) and
                sample["order"] == list(order), "sample order/warmup differs")
        require(set(sample["times_ms"]) == set(lanes) and
                all(positive(x) for x in sample["times_ms"].values()), "invalid elapsed time")
    result = {}
    for cadence in CADENCES:
        statistics_by_lane = {}
        for lane in lanes:
            values = [s["times_ms"][lane] for s in samples if s["cadence"] == cadence and not s["warmup"]]
            statistics_by_lane[lane] = dict(median_ms=statistics.median(values), min_ms=min(values),
                                             max_ms=max(values), retained=len(values))
            capture = row["captures"][cadence + "_" + lane]
            bench.validate_sample(capture, cadence, config["steps"], learner=learner, lane=lane, seed_fusion=seed_fusion and lane=="candidate", learner_optimizer=config.get("learner_optimizer"))
            bench.close_values(capture["losses"], capture["losses"])
            require(positive(capture["initial_loss"]), "invalid initial loss")
            if lane != "torch":
                fingerprint = row["fingerprints"][lane]
                require(isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint)
                        and capture["state_sha256"] == fingerprint, "state fingerprint differs")
            if "summary" in row:
                require(row["summary"][cadence][lane] == statistics.median(values), "published median differs")
        result[cadence] = dict(lanes=statistics_by_lane, baseline_over_candidate=
            statistics_by_lane["baseline"]["median_ms"] / statistics_by_lane["candidate"]["median_ms"])
        if "torch" in lanes:
            result[cadence]["torch_over_candidate"] = (statistics_by_lane["torch"]["median_ms"] /
                                                       statistics_by_lane["candidate"]["median_ms"])
    return result


def validate_progress(path, cases, seed_fusion=False):
    expected = []
    for row in cases:
        for sample in row["samples"]:
            for lane in sample["order"]:
                expected.append((row, sample, lane))
    started = None
    completed = 0
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["stage"] not in ("sample_started", "sample_finished"):
                continue
            require(completed < len(expected), "extra browser sample")
            row, sample, lane = expected[completed]
            identity = dict(config=row["config"], cadence=sample["cadence"], block=sample["block"], lane=lane)
            require(all(event[k] == v for k, v in identity.items()), "browser progress order differs")
            if event["stage"] == "sample_started":
                require(started is None, "overlapping browser samples")
                started = identity
            else:
                require(event.get("learner_optimizer") == row["config"].get("learner_optimizer"), "browser interval optimizer differs")
                require(event.get("fused_learner_seeds", False) is (seed_fusion and lane=="candidate"), "browser interval seed fusion differs")
                require(started == identity and event["elapsed_ms"] == sample["times_ms"][lane] and
                        event["state_sha256"] == row["fingerprints"][lane] and
                        type(event["setup_ms"]) in (int, float) and math.isfinite(event["setup_ms"]) and
                        event["setup_ms"] >= 0, "browser progress receipt differs")
                started = None
                completed += 1
    require(started is None and completed == len(expected), "incomplete browser progress")
    return completed


def run(args, result):
    paths = [args.native, args.browser, args.browser_progress]
    identities = [bench.audit.file_identity(path) for path in paths]
    result["inputs"] = identities
    with args.native.open() as stream:
        native = json.load(stream)
    with args.browser.open() as stream:
        browser = json.load(stream)
    for value, schema in ((native, "spiraltorch.resident_training_comparison.v1"),
                          (browser, "spiraltorch.resident_training_browser_comparison.v1")):
        require(value["status"] == "passed" and value["schema"] == schema and len(value["cases"]) == 9,
                "expected complete successful reports")
    require(not browser["page_errors"], "browser page errors")
    workload = native.get("workload", "dense")
    require(workload in ("dense", "graph", "learner") and browser.get("workload", "dense") == workload,
            "workload differs")
    learner = workload == "learner"
    seed_fusion = native.get("learner_seed_fusion", False)
    require(type(seed_fusion) is bool and browser.get("learner_seed_fusion", False) is seed_fusion and (not seed_fusion or learner),
            "seed fusion selection differs")
    result["learner_seed_fusion"] = seed_fusion
    graph = workload in ("graph", "learner")
    result["workload"] = workload
    matrix = native.get("matrix", "standard")
    require(browser.get("matrix", "standard") == matrix, "workload matrix differs")
    optimizer = native.get("learner_optimizer")
    require(browser.get("learner_optimizer") == optimizer and (optimizer is None or learner), "learner optimizer differs")
    configs = bench.recipes(graph, matrix, optimizer)
    result["learner_optimizer"] = optimizer
    result["matrix"] = matrix
    fusion = native.get("pointwise_fusion", False)
    require(type(fusion) is bool and browser.get("pointwise_fusion", False) is fusion and
            (not fusion or (graph and not learner)), "pointwise fusion selection differs")
    result["pointwise_fusion"] = fusion
    sources = {lane: bench.source_for(getattr(args, lane + "_source")) for lane in ("baseline", "candidate")}
    result["native_products"] = native["native_products"]
    result["source_bindings"] = {lane: manifest_binding(native["native_products"][lane]["identity"]["manifest"], source)
                                 for lane, source in sources.items()}
    page = bench.audit.git_bytes("show", args.browser_harness_source + ":bindings/st-wasm/tests/resident_training_bench.html")
    require(hashlib.sha256(page).hexdigest() == browser["asset_sha256"]["/"], "browser page source differs")
    result["browser_harness_source"] = bench.source_for(args.browser_harness_source)
    result["browser_assets"] = browser["asset_sha256"]
    result["browser_version"] = browser["browser_version"]
    result["browser_adapter_probe"] = browser.get("adapter_probe")
    result["device_admission"] = native["device_admission"]
    result["torch"] = dict(version=native["torch"], device=native["torch_device"])
    for config, n, b in zip(configs, native["cases"], browser["cases"]):
        row = dict(config=config, native=summarize_case(n, config, ("baseline", "candidate", "torch"), learner=learner, seed_fusion=seed_fusion),
                   browser=summarize_case(b, config, ("baseline", "candidate"), learner=learner, seed_fusion=seed_fusion), max_abs_errors={})
        result["cases"].append(row)
        for key in ("config", "plan_json", "input", "target", "learning_rate", "kernel", "accumulation"):
            require(n["fixture"][key] == b["fixture"][key], "native/browser fixture differs: " + key)
        for manifest in (n["fixture"]["build_manifest"], b["fixture"]["build_manifest"]):
            manifest_binding(manifest, sources["baseline"])
        manifest_binding(b["candidate_build_manifest"], sources["candidate"])
        if fusion:
            for measured in (n, b):
                bench.graph_reference.match_fused_fixture(measured["fixture"], measured["candidate_fixture"])
                manifest_binding(measured["candidate_fixture"]["build_manifest"], sources["candidate"])
            require(n["candidate_fixture"]["plan_json"] == b["candidate_fixture"]["plan_json"],
                    "native/browser fused plans differ")
        if seed_fusion:
            for measured in (n, b):
                bench.match_seed_fixture(measured["fixture"], measured["candidate_fixture"])
                manifest_binding(measured["candidate_fixture"]["build_manifest"], sources["candidate"])
        require(n["fixture"]["adapter"]["device_type"] != "Cpu" and
                b["fixture"]["adapter"]["backend"] == "BrowserWebGpu", "incorrect recorded backend")
        fixed = n["captures"]["immediate_torch"]
        for origin, value in (("native", n), ("browser", b)):
            for key, capture in value["captures"].items():
                bench.close_values(capture["losses"], fixed["losses"])
                error = (bench.learner_reference.compare(capture["state"], fixed["state"]) if learner else
                         bench.graph_reference.compare(capture["state"], fixed["state"]) if graph else
                         max(reference.compare(capture["state"], fixed["state"]).values()))
                row["max_abs_errors"][origin + "_" + key] = error
        row["losses"] = fixed["losses"]
    result["browser_intervals_revalidated"] = validate_progress(args.browser_progress, browser["cases"], seed_fusion)
    require(identities == [bench.audit.file_identity(path) for path in paths], "evidence changed during validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "browser", "browser-progress", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("baseline-source", "candidate-source", "browser-harness-source"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = dict(schema="spiraltorch.resident_training_comparison.validation.v1", status="error", cases=[],
        boundary="read-only full captured-state comparison to recorded independent torch reference; every timing reaggregated, all browser interval fingerprints checked; fingerprints are consistency receipts, not independent state rehashes or device attestation; no selective retries or universal speed claim")
    with args.output.open("x") as output:
        try:
            run(args, result)
            result["status"] = "passed"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(status=result["status"], error=result.get("error"), cases=len(result["cases"]))))
    raise SystemExit(result["status"] != "passed")


if __name__ == "__main__":
    main()
