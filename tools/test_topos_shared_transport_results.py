"""Consistency of saved shared transport results, not a fresh runtime witness."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

BUNDLE = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-07-topos-shared-transport"
ROUTES = ["rust_list", "rust_buffer_recomputed", "rust_public", "rust_expanded_captured", "torch_reference"]
CONDITIONS = [(b, t, f, it) for b, t, f in [(2, 4, 3), (2, 32, 128), (2, 128, 768)] for it in (1, 5, 16)]


def load():
    return json.loads(gzip.decompress((BUNDLE / "measurements.json.gz").read_bytes()))


def verify_measurements(data):
    runs = data["runs"]
    assert [r["variant"] for r in runs] == ["before", "after", "after", "before"]
    assert len(data["summary"]) == len(CONDITIONS)
    for index, run in enumerate(runs):
        assert run["index"] == index
        assert run["case_order"] == ("reverse" if index % 2 else "forward")
        assert [(*c["shape"], c["config"]["iterations"]) for c in run["cases"]] == (CONDITIONS[::-1] if index % 2 else CONDITIONS)
        for case in run["cases"]:
            assert case["status"] == "measured" and case["device"] == "cpu" and case["dtype"] == "float32"
            assert case["seed"] == 239 and case["threads"] == 2 and case["warmup_per_route"] == 3
            assert case["native_profile_declared"] == "release" and case["capture_available"] is True
            assert case["all_inputs_require_grad"] is True
            assert case["config"]["coupling"] == .25 and case["config"]["saturation"] == 1.
            assert math.isclose(case["config"]["porosity"], .2, abs_tol=1e-8)
            assert case["public_gate_reduction"] == ("torch_f32" if run["variant"] == "before" else "rust_row_order_f64_to_f32")
            assert case["round_order"] == [ROUTES[i % 5:] + ROUTES[:i % 5] for i in range(15)]
            assert set(case["measurements_ms"]) == set(case["correctness"]) == set(ROUTES)
            for route in ROUTES:
                timings = case["measurements_ms"][route]
                assert len(timings) == 15 and all(type(t) in (float, int) and math.isfinite(t) and t > 0 for t in timings)
                assert median(timings) == case["median_ms"][route]
                for field in ("output", "input_gradient", "gate_gradient"):
                    check = case["correctness"][route][field]
                    assert re.fullmatch(r"[0-9a-f]{64}", check["sha256"])
                    assert math.isfinite(check["max_abs_error"]) and check["max_abs_error"] >= 0
                    if route != "torch_reference":
                        assert check["max_abs_error"] == 0
            assert case["public_vs_legacy"]["output"]["max_abs_error"] == 0
            assert case["public_vs_legacy"]["input_gradient"]["max_abs_error"] == 0
            assert 0 <= case["public_vs_legacy"]["gate_gradient"]["max_abs_error"] <= 1e-5
    for condition, summary in zip(CONDITIONS, data["summary"]):
        matched = [next(c for c in r["cases"] if (*c["shape"], c["config"]["iterations"]) == condition) for r in runs]
        assert summary["case"] == "x".join(map(str, condition[:3])) + "/" + str(condition[3])
        for key in ("input_sha256", "upstream_sha256", "benchmark_sha256", "transport_sha256", "torch", "machine", "config"):
            assert all(c[key] == matched[0][key] for c in matched)
        for route in ROUTES:
            for field in ("output", "input_gradient", "gate_gradient"):
                hashes = [c["correctness"][route][field]["sha256"] for c in matched]
                if route == "rust_public" and field == "gate_gradient":
                    assert hashes[0] == hashes[3] and hashes[1] == hashes[2]
                else:
                    assert len(set(hashes)) == 1
            for variant in ("before", "after"):
                times = [t for run, case in zip(runs, matched) if run["variant"] == variant for t in case["measurements_ms"][route]]
                assert len(times) == 30 and summary["ms"][route][variant] == median(times)
        ms = summary["ms"]
        for ratio, numerator in (("public_before_after_ratio", ms["rust_public"]["before"]),
                                 ("expanded_shared_ratio", ms["rust_expanded_captured"]["after"]),
                                 ("torch_shared_ratio", ms["torch_reference"]["after"])):
            assert summary[ratio] == numerator / ms["rust_public"]["after"]


def test_saved_matrix_and_rejection_controls():
    data = load()
    verify_measurements(data)
    for mutate in (
        lambda d: d["runs"].pop(),
        lambda d: d["runs"][0]["cases"].pop(),
        lambda d: d["summary"][0].__setitem__("public_before_after_ratio", 100.),
        lambda d: d["runs"][0]["cases"][0]["measurements_ms"]["rust_public"].__setitem__(0, -1.),
        lambda d: d["runs"][1]["cases"][0].__setitem__("public_gate_reduction", "torch_f32"),
    ):
        bad = copy.deepcopy(data)
        mutate(bad)
        try:
            verify_measurements(bad)
        except AssertionError:
            continue
        raise AssertionError("contradictory saved benchmark accepted")


def test_hashes_runtime_identity_and_wasm_receipts():
    seen = set()
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name in {"README.md", "measurements.json.gz", "verification.json"} and name not in seen
        seen.add(name)
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest
    assert len(seen) == 3
    data = load()
    verification = json.loads((BUNDLE / "verification.json").read_text())
    assert verification["production_source"] == data["production_source"] == "f52629934a408ee353aceeb3c0b8b461e93cb3b5"
    for run in data["runs"]:
        for case in run["cases"]:
            assert case["native_sha256"] == verification["artifact_sha256"]["native_" + run["variant"]]
            assert case["benchmark_sha256"] == verification["source_sha256"]["tools/benchmark_topos_learning.py"]
            if run["variant"] == "after":
                assert case["bridge_sha256"] == verification["source_sha256"]["bindings/st-py/spiraltorch/geometry_autograd.py"]
    assert data["pilots"]["excluded_from_summary"] is True
    for name in ("node", "browser_final"):
        receipt = data["wasm"][name]
        assert (receipt["status"], receipt["cases"], receipt["checks"]) == ("passed", 27, 469)
        assert receipt["learning"]["updates"] == 240 and receipt["learning"]["next_update_exact"] is True
        assert receipt["learning"]["matched_wide_sum_trajectory_exact"] is True
        assert 0 <= receipt["learning"]["final_loss"] < receipt["learning"]["initial_loss"] * .01
        assert receipt["wasm_sha256"] == verification["artifact_sha256"]["wasm_node" if name == "node" else "wasm_web"]
    assert data["wasm"]["node"]["learning"] == data["wasm"]["browser_final"]["learning"]
    assert data["wasm"]["browser_final"]["page_errors"] == data["wasm"]["browser_final"]["console_messages"] == []
    assert data["wasm"]["browser_final"]["page_sha256"] == verification["source_sha256"]["bindings/st-wasm/tests/topos_shared_transport.html"]


if __name__ == "__main__":
    test_saved_matrix_and_rejection_controls()
    test_hashes_runtime_identity_and_wasm_receipts()
    print("Saved shared-transport consistency checks passed (no fresh execution)")
