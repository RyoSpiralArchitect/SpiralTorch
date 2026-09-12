#!/usr/bin/env python3
"""Admit fixed completed-read intervals without dropping slow samples."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import admit_native, close, digest, f32_bytes

PROTOCOL = dict(matrices=4, warmup=2, samples=9, forwards=256)
ROUTES = ["baseline_module_d2h", "candidate_module_d2h", "baseline_scalar_d2h", "candidate_scalar_d2h"]


def number(value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or value < 0 or positive and value == 0):
        raise ValueError("invalid numeric measurement")
    return value


def validate(document, native, *, terminal_capture=False, same_terminal_api=False):
    if same_terminal_api and not terminal_capture:
        raise ValueError("same terminal API requires terminal capture")
    schema = "spiraltorch.module_terminal_intervals.v1" if terminal_capture else "spiraltorch.module_completed_intervals.v1"
    fixture = "nn-module-terminal-intervals" if terminal_capture else "nn-module-intervals"
    if same_terminal_api:
        fixture = "nn-module-terminal-matched-intervals"
    baseline_api = "forwardSnapshot" if same_terminal_api else "forward_then_snapshot"
    if terminal_capture and document.get("module_apis") != dict(baseline=baseline_api, candidate="forwardSnapshot"):
        raise ValueError("terminal API boundaries differ")
    ordinary = dict(baseline="forward_then_snapshot", candidate="forward_then_snapshot")
    if not terminal_capture and document.get("module_apis", ordinary) != ordinary:
        raise ValueError("ordinary API boundaries differ")
    if (document.get("schema") != schema
            or document.get("status") != "passed" or document.get("page_errors") != []
            or document.get("fixture_request") != fixture
            or document.get("protocol") != PROTOCOL or document.get("routes") != ROUTES
            or len(native.get("cases", [])) != 9 or len(document.get("cases", [])) != 36
            or any(type(document["protocol"][k]) is not int for k in PROTOCOL)
            or not isinstance(document.get("console_messages"), list)
            or any(message.get("type") == "error" for message in document["console_messages"])
            or type(document.get("cross_origin_isolated")) is not bool
            or not document.get("browser_version") or not document.get("user_agent")):
        raise ValueError("completed interval contract incomplete")
    clock = document["clock"]
    deltas = clock["positive_deltas_ms"]
    if (not 1 <= len(deltas) <= 256 or type(clock["equal_reads"]) is not int
            or not 0 <= clock["equal_reads"] <= 250000 - len(deltas)):
        raise ValueError("invalid clock probe")
    for delta in deltas:
        number(delta, positive=True)
    resolution = number(clock["minimum_positive_delta_ms"], positive=True)
    if resolution != min(deltas):
        raise ValueError("clock resolution differs")
    rows = []
    minimum_interval = math.inf
    total_forwards = comparisons = 0
    max_error = 0.
    for matrix in range(PROTOCOL["matrices"]):
        frozen_cases = native["cases"] if matrix % 2 == 0 else list(reversed(native["cases"]))
        for index, frozen in enumerate(frozen_cases):
            case = document["cases"][matrix * 9 + index]
            setup = ["baseline", "candidate"] if matrix % 2 == 0 else ["candidate", "baseline"]
            if (type(case.get("matrix")) is not int or case["matrix"] != matrix or case.get("setup_order") != setup
                    or any(case.get(k) != frozen[k] for k in ("seed", "shape", "depth"))
                    or any(set(case.get(k, {})) != set(ROUTES)
                           for k in ("last_outputs", "warmup_completed_reads", "warmup_max_abs_error"))
                    or any(set(case.get(k, {})) != {"baseline", "candidate"}
                           for k in ("adapters", "cache", "scalar_dispatches"))
                    or len(case.get("samples", [])) != PROTOCOL["samples"] * len(ROUTES)):
                raise ValueError("case identity/order/coverage differs")
            expected_forwards = (PROTOCOL["warmup"] + PROTOCOL["samples"]) * PROTOCOL["forwards"]
            for version in ("baseline", "candidate"):
                if (case["adapters"][version].get("backend") != "BrowserWebGpu"
                        or case["adapters"][version].get("device_type") in (None, "Cpu")
                        or case["cache"][version] != dict(compilations="1", cache_hits=str(expected_forwards),
                                                          submitted_forwards=str(expected_forwards + 1))
                        or case["scalar_dispatches"][version] != str(expected_forwards + 1)):
                    raise ValueError("device/cache/submission contract differs")
            tolerance = max(2e-5 + 2e-4*abs(value) for value in frozen["reference"])
            def error(value):
                if number(value) > tolerance:
                    raise ValueError("reported output error exceeds tolerance")
                return value
            for route in ROUTES:
                count = case["warmup_completed_reads"][route]
                if type(count) is not int or count != PROTOCOL["warmup"] * PROTOCOL["forwards"]:
                    raise ValueError("warmup reads differ")
                max_error = max(max_error, error(case["warmup_max_abs_error"][route]))
                f32_bytes(case["last_outputs"][route])
                max_error = max(max_error, close(case["last_outputs"][route], frozen["reference"]))
            timings = {route: [] for route in ROUTES}
            for block in range(PROTOCOL["samples"]):
                offset = (matrix + frozen["seed"] + block + PROTOCOL["warmup"]) % len(ROUTES)
                order = ROUTES[offset:] + ROUTES[:offset]
                for position, route in enumerate(order):
                    sample = case["samples"][block * len(ROUTES) + position]
                    if (type(sample.get("block")) is not int or sample["block"] != block
                            or sample.get("route") != route or sample.get("order") != order
                            or any(type(sample.get(k)) is not int or sample[k] != PROTOCOL["forwards"]
                                   for k in ("forwards", "completed_reads"))
                            or sample.get("submitted_forwards") != str(PROTOCOL["forwards"])):
                        raise ValueError("sample order or completed-read count differs")
                    elapsed = number(sample["elapsed_ms"], positive=True)
                    max_error = max(max_error, error(sample["max_abs_error"]))
                    minimum_interval = min(minimum_interval, elapsed)
                    timings[route].append(elapsed)
            summary = {route: dict(median_ms_per_forward=statistics.median(values)/PROTOCOL["forwards"],
                                   min_interval_ms=min(values), max_interval_ms=max(values), total_ms=sum(values))
                       for route, values in timings.items()}
            rows.append(dict(matrix=matrix, shape=case["shape"], seed=case["seed"], depth=case["depth"],
                summary=summary, candidate_over_baseline={family:
                    summary[f"candidate_{family}_d2h"]["median_ms_per_forward"] /
                    summary[f"baseline_{family}_d2h"]["median_ms_per_forward"] for family in ("module", "scalar")}))
            total_forwards += expected_forwards * len(ROUTES)
            comparisons += expected_forwards * len(ROUTES) * len(frozen["reference"])
    aggregates = []
    for shape in dict.fromkeys(tuple(row["shape"]) for row in rows):
        selected = [row for row in rows if tuple(row["shape"]) == shape]
        families = {}
        for family in ("module", "scalar"):
            ratios = [row["candidate_over_baseline"][family] for row in selected]
            families[family] = dict(median_of_case_median_ratios=statistics.median(ratios),
                min_ratio=min(ratios), max_ratio=max(ratios), regressions=sum(value > 1 for value in ratios),
                total_cases=len(ratios), pooled_total_ratio=
                sum(row["summary"][f"candidate_{family}_d2h"]["total_ms"] for row in selected) /
                sum(row["summary"][f"baseline_{family}_d2h"]["total_ms"] for row in selected))
        aggregates.append(dict(shape=list(shape), **families))
    return dict(cases=rows, aggregates=aggregates, checked_forwards_including_warmup=total_forwards,
        checked_values_including_warmup=comparisons, max_abs_error=max_error,
        minimum_interval_ms=minimum_interval, clock_minimum_positive_delta_ms=resolution,
        minimum_interval_over_observed_clock_delta=minimum_interval/resolution)


def product_assets(receipt_path, prefix):
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("status") != "passed" or not receipt.get("source", {}).get("commit"):
        raise ValueError("runtime was not verified")
    assets = {}
    for name, sha in receipt["products"].items():
        if digest(receipt_path.parent / name) != sha:
            raise ValueError("frozen product changed: " + name)
        if name.startswith("client/") and name.endswith((".js", ".wasm")):
            assets[prefix + name[len("client/"):]] = sha
    if prefix + "spiraltorch_wasm_bg.wasm" not in assets or prefix + "spiraltorch_wasm.js" not in assets:
        raise ValueError("frozen client products missing")
    return receipt, assets


def admit_case_stream(document, browser_path):
    stream = document.get("interval_state_artifacts", {})
    path = browser_path.with_name(browser_path.name + ".cases.jsonl")
    if (stream.get("path") != path.name or stream.get("rows") != 36
            or stream.get("bytes") != path.stat().st_size or stream.get("sha256") != digest(path)
            or [json.loads(line) for line in path.read_bytes().splitlines()] != document["cases"]):
        raise ValueError("streamed interval evidence differs")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "browser", "baseline-receipt", "candidate-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--terminal-capture", action="store_true")
    parser.add_argument("--same-terminal-api", action="store_true")
    args = parser.parse_args()
    source_paths = [args.fixture, args.browser, args.baseline_receipt, args.candidate_receipt, Path(__file__),
                    Path(__file__).with_name("bench_graph_forward_paths.py")]
    root = Path(__file__).resolve().parents[1]
    page = root / "bindings/st-wasm/tests/module_resident_intervals.html"
    helper = page.with_suffix(".mjs")
    source_paths += [page, helper, root / "tools/test_resident_browser.cjs"]
    report = dict(schema="spiraltorch.module_completed_intervals_validation.v1", status="error",
        boundary="Primary endpoint is median of 12 case-median candidate/baseline ratios per shape; all intervals retained. Pooled totals and explicit-dispatch controls remain separate. The clock probe describes observed granularity, not an uncertainty bound. This longer-interval completed-read workload does not replace the earlier single-call evidence or prove fastest-Torch/browser hardware performance.")
    with args.output.open("x") as output:
        try:
            hashes = {str(path.resolve()): digest(path) for path in source_paths}
            native, browser = (json.loads(path.read_bytes()) for path in (args.fixture, args.browser))
            admit_native(native)
            source_paths.append(admit_case_stream(browser, args.browser))
            hashes[str(source_paths[-1].resolve())] = digest(source_paths[-1])
            baseline, assets = product_assets(args.baseline_receipt, "/baseline/")
            candidate, candidate_assets = product_assets(args.candidate_receipt, "/module/")
            assets.update(candidate_assets)
            assets.update({"/fixture.json": digest(args.fixture), "/": digest(page),
                           "/module_resident_intervals.mjs": digest(helper)})
            if (browser.get("asset_sha256") != assets or browser.get("page_sha256") != digest(page)
                    or browser.get("wasm_sha256") != assets["/module/spiraltorch_wasm_bg.wasm"]):
                raise ValueError("served artifact identities differ")
            report.update(validate(browser, native, terminal_capture=args.terminal_capture,
                                   same_terminal_api=args.same_terminal_api))
            if args.terminal_capture:
                report.update(schema="spiraltorch.module_terminal_intervals_validation.v1", module_apis=browser["module_apis"])
                report["boundary"] += (" Both versions use forwardSnapshot; frozen runtime sources determine scheduling."
                                       if args.same_terminal_api else
                                       " Candidate forwardSnapshot versus baseline forward then snapshot.")
                report["boundary"] += " Host wrapper costs remain included; not isolated GPU queue cost."
            product_assets(args.baseline_receipt, "/baseline/")
            product_assets(args.candidate_receipt, "/module/")
            if hashes != {str(path.resolve()): digest(path) for path in source_paths}:
                raise ValueError("validation sources changed")
            report.update(status="passed", sources=hashes, baseline_source=baseline["source"],
                          candidate_source=candidate["source"], protocol=PROTOCOL)
        except BaseException as exc:
            report["error"] = repr(exc)
            raise
        finally:
            json.dump(report, output, indent=2, allow_nan=False); output.write("\n")


if __name__ == "__main__":
    main()
