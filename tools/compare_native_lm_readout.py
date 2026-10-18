"""Validate all matched readout cases before publishing loss errors and timings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

LOSS_ATOL = 2e-4
ENGINES = {"native-before": "native", "native-after": "native", "torch-full": "torch-full", "torch-selected": "torch-selected"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(directory):
    reports = {name: json.loads((directory/name/"report.json").read_text()) for name in ENGINES}
    reference = reports["torch-full"]
    common = ("schema", "source_sha256", "script_sha256", "vocabulary", "label_smoothing", "platform", "python", "thread_env", "boundary", "order")
    expected_ids = [f"rows{rows}-selected{active}" for rows in (16, 64) for active in (1, 4, rows)]
    for name, report in reports.items():
        if report["schema"] != "spiraltorch.native_lm_readout_bench.v1" or report["engine"] != ENGINES[name]:
            raise ValueError("wrong benchmark schema/engine")
        if any(report[key] != reference[key] for key in common):
            raise ValueError("unmatched benchmark conditions: " + name)
        if [c["id"] for c in report["cases"]] != expected_ids:
            raise ValueError("missing, duplicate or reordered cases")
        if name.startswith("native-") and report["runtime"].get("optional_ml_imports_forbidden") is not True:
            raise ValueError("native import guard not confirmed")
        if name.startswith("torch-"):
            runtime = report["runtime"]
            if runtime.get("device") != "cpu":
                raise ValueError("reference must be CPU")
            for key in ("torch", "transformers", "torch_threads"):
                if key not in runtime or runtime[key] != reference["runtime"].get(key):
                    raise ValueError("unmatched reference runtime: " + key)
            if any(not isinstance(runtime[key], str) or not runtime[key] for key in ("torch", "transformers")):
                raise ValueError("missing reference version string")
            if type(runtime["torch_threads"]) is not int or runtime["torch_threads"] != 1:
                raise ValueError("reference must report one intra-op thread")
        for case in report["cases"]:
            samples = case["timings_ms"]
            if len(samples) != 5 or any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError("invalid timing samples")
            if case["median_ms"] != statistics.median(samples) or not math.isfinite(case["loss"]):
                raise ValueError("invalid loss or reported median")
            if not math.isfinite(case["first_call_ms"]) or case["first_call_ms"] <= 0:
                raise ValueError("invalid first-call time")
            if (len(case["ids"]) != case["context_rows"] + 1 or len(case["labels"]) != len(case["ids"])
                    or sum(x != -100 for x in case["labels"][1:]) != case["selected_rows"]):
                raise ValueError("invalid selected/context row accounting")
    rows = []
    for index, expected in enumerate(reference["cases"]):
        inputs = {key: expected[key] for key in ("id", "ids", "labels", "context_rows", "selected_rows")}
        values = {}
        for name, report in reports.items():
            case = report["cases"][index]
            if any(case[key] != value for key, value in inputs.items()):
                raise ValueError("unmatched exact IDs/labels")
            error = abs(case["loss"] - expected["loss"])
            if error > LOSS_ATOL:
                raise ValueError(f"loss mismatch for {name}/{case['id']}: {error}")
            values[name] = {key: case[key] for key in ("loss", "first_call_ms", "median_ms", "timings_ms")}
            values[name]["abs_loss_error_vs_torch_full"] = error
        before, after = (values[name]["median_ms"] for name in ("native-before", "native-after"))
        rows.append({**inputs, "measurements": values, "native_before_over_after": before/after,
                     "torch_full_over_native_after": values["torch-full"]["median_ms"]/after,
                     "torch_selected_over_native_after": values["torch-selected"]["median_ms"]/after})
    return {"schema": "spiraltorch.native_lm_readout_comparison.v1", "conditions": {key: reference[key] for key in common},
            "raw_report_sha256": {name: digest(directory/name/"report.json") for name in ENGINES},
            "native_binary_sha256": {name: reports[name]["runtime"]["native_extension_sha256"] for name in ("native-before", "native-after")},
            "torch_runtime": reference["runtime"], "loss_atol": LOSS_ATOL, "cases": rows,
            "torch_runtimes": {name: reports[name]["runtime"] for name in ("torch-full", "torch-selected")},
            "limits": ["one serial process per engine, five warm samples per case; timing order/load can confound ratios",
                       "only no-geometry frozen-CPU loss evaluation is timed; no training/GPU/RSS/quality claim",
                       "torch-selected compacts after final norm; native compacts before it",
                       "all six cases and raw samples retained; loss gate does not imply speed superiority"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
