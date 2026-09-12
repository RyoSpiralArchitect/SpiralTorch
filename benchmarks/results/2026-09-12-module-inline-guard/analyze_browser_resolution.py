"""Supplementary raw-clock analysis; never replaces the primary paired ratios."""
import collections
import hashlib
import json
import lzma
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
summary_bytes = (ROOT / "summary.json").read_bytes()
summary = json.loads(summary_bytes)
manifest = {item["path"]: item for item in json.loads((ROOT / "manifest.json").read_bytes())}
paths = ["raw/verified-a/browser-module-matched.json.xz"] + [
    f"raw/replications/{i}-browser.json.xz" for i in (1, 2, 3)
]


def describe(values):
    return {
        "samples": len(values),
        "total_elapsed_ms": sum(values),
        "mean_elapsed_ms": statistics.mean(values),
        "median_elapsed_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "histogram_rounded_to_0_001_ms": dict(sorted(collections.Counter(
            round(value, 3) for value in values
        ).items())),
    }


trials, inputs = [], []
pooled = {version: [] for version in ("baseline", "candidate")}
for trial, path in enumerate(paths):
    packed = (ROOT / path).read_bytes()
    raw = lzma.decompress(packed)
    entry = manifest[path]
    assert sha(packed) == entry["compressed_sha256"]
    assert sha(raw) == entry["sha256"]
    document = json.loads(raw)
    assert document["status"] == "passed" and len(document["cases"]) == 9
    cases = [case for case in document["cases"] if case["shape"] == [2, 3, 7]]
    assert sorted(case["seed"] for case in cases) == [17, 29, 43]
    samples = {version: [
        sample["elapsed_ms"] for case in cases for sample in case["samples"]
        if sample["route"] == version + "_module_d2h"
    ] for version in pooled}
    assert all(len(values) == 27 for values in samples.values())
    for version in pooled:
        pooled[version].extend(samples[version])
    trials.append({
        "trial": "initial" if trial == 0 else str(trial),
        "versions": {version: describe(values) for version, values in samples.items()},
        "total_elapsed_candidate_over_baseline": sum(samples["candidate"]) / sum(samples["baseline"]),
    })
    inputs.append({"path": path, "sha256": sha(raw), "compressed_sha256": sha(packed)})

report = {
    "schema": "spiraltorch.inline_guard_browser_clock_diagnostic.v1",
    "source": summary["source"], "baseline_source": summary["baseline_source"],
    "analysis_script_sha256": sha(Path(__file__).read_bytes()),
    "primary_summary_sha256": sha(summary_bytes), "inputs": inputs,
    "shape": [2, 3, 7], "route": "module_d2h", "trials": trials,
    "primary_median_paired_ratio": next(group for group in summary["all_trials_groups"]
        if group["shape"] == [2, 3, 7])["browser_d2h_candidate_over_baseline"]["median"],
    "pooled": {version: describe(values) for version, values in pooled.items()},
    "total_elapsed_candidate_over_baseline": sum(pooled["candidate"]) / sum(pooled["baseline"]),
    "boundary": "Post-hoc diagnosis of the small-browser median discrepancy, not a replacement endpoint or evidence of a uniform speedup/slowdown. All 108 samples per version remain, without trimming or remeasurement. Histograms alone round values; calculations use unrounded elapsed times. Median paired ratios, pooled medians and total elapsed measure different things. Coarse clock bins and trial drift limit single-call inference.",
}
(ROOT / "browser_clock_diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"paired_ratio": report["primary_median_paired_ratio"],
    "total_ratio": report["total_elapsed_candidate_over_baseline"], "trials": len(trials)}))
