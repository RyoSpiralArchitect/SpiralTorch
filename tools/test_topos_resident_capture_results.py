"""Frozen profile arithmetic and receipt consistency, not a runtime witness."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

BUNDLE = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-07-topos-resident-capture"
SOURCE = "0bbc5a65f0dfc23d78add8d3f61f3d3a391d7125"
FILES = {"README.md", "profile.json.gz", "browser.json.gz", "verification.json"}


def load(name):
    raw = (BUNDLE / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


def test_hashes_and_runtime_identity():
    seen = set()
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name in FILES and name not in seen
        seen.add(name)
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest
    assert seen == FILES
    verify = load("verification.json")
    browser = load("browser.json.gz")
    profile = load("profile.json.gz")
    assert verify["source"] == profile["after_source"] == profile["probe_source"] == SOURCE
    hashes = verify["artifact_byte_sha256"]
    assert hashes["native"] == verify["runtime"]["native_sha256"]
    assert hashes["wasm"] == browser["wasm_sha256"]
    assert verify["source_sha256"]["bindings/st-wasm/tests/resident_topos_graph.html"] == browser["page_sha256"]
    for variant in ("before", "after"):
        assert hashes[variant + "_binary"] == profile["binaries_sha256"][variant]
    for values in (hashes, verify["source_sha256"], verify["local_log_sha256"]):
        assert values and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in values.values())
    for receipt in verify["local_browser_regressions"].values():
        assert receipt["status"] == "passed" and receipt["wasm_sha256"] == hashes["wasm"]
    assert verify["runtime"]["adapter"]["device_type"] != "Cpu"
    assert verify["validation"]["python_independent_torch_updates"] == 100
    assert verify["validation"]["python_rtol"] == 5e-4
    assert verify["validation"]["python_atol"] == 3e-5
    for error in verify["validation"]["python_max_abs_error"].values():
        assert math.isfinite(error) and 0 <= error < 2e-7
    assert verify["clippy"]["strict_exit"] != 0
    assert len(verify["clippy"]["unique_unchanged_files"]) == 7


def validate_sample_timestamps(sample):
    period = sample["timestamp_period_ns"]
    assert type(period) in (int, float) and math.isfinite(period) and period > 0
    passes = sample["passes"]
    assert passes

    def tick(value):
        assert isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value)
        result = int(value)
        assert 0 <= result < 2**64
        return result

    previous_end = None
    for timing in passes:
        start, end = tick(timing["start_tick"]), tick(timing["end_tick"])
        assert previous_end is None or start >= previous_end
        ticks = end - start
        assert ticks >= 0
        previous_end = end
        # Match Rust: subtract u64 ticks before converting to f64. Absolute
        # device clocks may be too large to represent exactly as floats.
        elapsed = float(ticks) * period
        assert math.isfinite(elapsed) and timing["elapsed_ns"] == elapsed
    span_ticks = tick(passes[-1]["end_tick"]) - tick(passes[0]["start_tick"])
    assert span_ticks >= 0
    span = float(span_ticks) * period
    assert math.isfinite(span) and sample["gpu_span_ns"] == span


def test_timestamp_contradictions_are_rejected():
    valid = {
        "timestamp_period_ns": .25,
        "gpu_span_ns": 1.75,
        "passes": [{"start_tick": str(2**60 + 3), "end_tick": str(2**60 + 10), "elapsed_ns": 1.75}],
    }
    validate_sample_timestamps(valid)
    for mutate in (
        lambda s: s.__setitem__("gpu_span_ns", 3.5),
        lambda s: s["passes"][0].__setitem__("elapsed_ns", 3.5),
        lambda s: s.__setitem__("timestamp_period_ns", .5),
        lambda s: s.__setitem__("timestamp_period_ns", 0.),
        lambda s: s["passes"][0].__setitem__("end_tick", str(2**60)),
        lambda s: s["passes"][0].__setitem__("start_tick", str(2**64)),
    ):
        bad = copy.deepcopy(valid)
        mutate(bad)
        try:
            validate_sample_timestamps(bad)
        except AssertionError:
            continue
        raise AssertionError("inconsistent timestamp receipt accepted")


def test_pass_order_and_overlap_are_rejected_without_rejecting_touching_intervals():
    for offset in (0, 2**60):
        for intervals, accepted in (
            ([(100, 110), (110, 110), (110, 120), (123, 130)], True),
            ([(100, 110), (0, 10), (120, 130)], False),
            ([(100, 110), (105, 115), (120, 130)], False),
            ([(100, 120), (110, 115), (116, 130)], False),
        ):
            sample = {
                "timestamp_period_ns": .25,
                "gpu_span_ns": .25 * (intervals[-1][1] - intervals[0][0]),
                "passes": [{"start_tick": str(offset + start), "end_tick": str(offset + end),
                            "elapsed_ns": .25 * (end - start)} for start, end in intervals],
            }
            try:
                validate_sample_timestamps(sample)
            except AssertionError:
                assert not accepted
            else:
                assert accepted, "reordered or overlapping pass timestamps accepted"


def test_all_profile_conditions_and_medians():
    profile = load("profile.json.gz")
    expected = [(r, c, it) for r, c in ((2, 3), (32, 256), (128, 1025)) for it in (1, 5, 64)]
    key = lambda case: (*case["shape"], case["iterations"])
    runs = profile["runs"]
    assert [run["variant"] for run in runs] == ["before", "after", "after", "before"]
    assert [run["case_order"] for run in runs] == ["forward", "reverse", "forward", "reverse"]
    assert len(profile["summary"]) == 9
    for index, run in enumerate(runs):
        assert run["index"] == index
        assert [key(case) for case in run["cases"]] == (expected if index % 2 == 0 else expected[::-1])
        for case in run["cases"]:
            assert case["warmups"] == 3 and case["learning_rate"] == 0
            assert (case["coupling"], case["saturation"], case["porosity"]) == (.2, 1., .3)
            assert len(case["samples"]) == 9
            assert re.fullmatch(r"[0-9a-f]{64}", case["state_bits_sha256"])
            for sample in case["samples"]:
                validate_sample_timestamps(sample)
                assert sample["accepted"] and sample["timing_complete"] and sample["instrumented"]
                assert sample["gpu_span_ns"] > 0 and sample["zero_intervals"] == 0
                for phase, total in sample["phase_totals_ns"].items():
                    assert math.isfinite(total) and total > 0
                    assert total == sum(p["elapsed_ns"] for p in sample["passes"] if p["phase"] == phase)
    for dimensions, summary in zip(expected, profile["summary"]):
        rows, cols, iterations = dimensions
        matched = [next(case for case in run["cases"] if key(case) == dimensions) for run in runs]
        assert len({case["state_bits_sha256"] for case in matched}) == 1
        assert summary["case"] == f"{rows}x{cols}/{iterations}" and summary["state_bits_equal"] is True
        for metric, values in summary["metrics"].items():
            medians = {}
            for variant in ("before", "after"):
                observations = [
                    (sample[metric] if metric == "gpu_span_ns" else sample["phase_totals_ns"][metric])
                    for run, case in zip(runs, matched) if run["variant"] == variant
                    for sample in case["samples"]
                ]
                assert len(observations) == 18
                medians[variant] = median(observations) / 1000
                assert values[variant + "_us"] == medians[variant]
            assert math.isclose(values["ratio"], medians["before"] / medians["after"], rel_tol=1e-14)


def test_browser_saved_learning_and_guards():
    browser = load("browser.json.gz")
    assert browser["status"] == "passed" and browser["checks"] == 4119
    assert browser["adapter_probe"]["is_fallback_adapter"] is False
    assert browser["adapter"]["backend"] == "BrowserWebGpu"
    assert browser["page_errors"] == browser["console_messages"] == []
    assert len(browser["guards"]) == 6 and all(v is True for v in browser["guards"].values())
    assert {(case["porosity"], case["policy"]) for case in browser["cases"]} == {
        (p, policy) for p in (0., .3) for policy in ("exact", "module_compatible")
    }
    for case in browser["cases"]:
        assert case["shape"] == [2, 2, 3] and len(case["trajectory"]) == 25
        previous = [.8, -.4, 1.1]
        for step, record in enumerate(case["trajectory"]):
            assert record["step"] == step and math.isfinite(record["loss"])
            for name, count in (("input", 12), ("target", 12), ("output", 12), ("dx", 12), ("dg", 3), ("gate", 3)):
                assert len(record[name]) == count and all(math.isfinite(v) for v in record[name])
            mse = sum((x - y) ** 2 for x, y in zip(record["output"], record["target"])) / 12
            assert abs(mse - record["loss"]) <= 2e-7 * max(1., abs(mse))
            for before, gradient, after in zip(previous, record["dg"], record["gate"]):
                assert abs(before - .03 * gradient - after) <= 2e-7 * max(1., abs(before))
            previous = record["gate"]


if __name__ == "__main__":
    test_hashes_and_runtime_identity()
    test_timestamp_contradictions_are_rejected()
    test_pass_order_and_overlap_are_rejected_without_rejecting_touching_intervals()
    test_all_profile_conditions_and_medians()
    test_browser_saved_learning_and_guards()
    print("Five saved Topos capture record checks passed (no GPU execution)")
