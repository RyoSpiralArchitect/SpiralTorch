"""Reconstruct the frozen NN ABBA study; not a fresh execution or speed gate."""

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


BUNDLE = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-07-topos-nn-shared-benchmark"
ROUTES = ("forward", "forward_backward")
FIELDS = ("input", "gate", "upstream", "output", "grad_input", "grad_gate")
CASES = [dict(id=f"{r}x{f}-k{k}", rows=r, features=f, iterations=k, coupling=.25)
         for r, f in ((8, 3), (64, 128), (256, 768)) for k in (1, 5, 16)]
PHASES = ["baseline", "candidate", "candidate", "baseline"]
REVISIONS = ["1ec25672cd2658d091aba31f26891eaa8dd1a8e3", "af7ef9aaa9adc466052611e8bdc950dbbf51dcfd"]
AGGREGATION = "median of per-process medians, two processes per arm per condition"
HARNESS = {
    "crates/st-nn/examples/topos_shared_module_probe.rs",
    "tools/benchmark_topos_shared_module_reference.py",
    "tools/benchmark_topos_module_reference.py",
    "tools/test_benchmark_topos_shared_module_reference.py",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_original(verification, name, raw):
    assert verification["private_files"][name] == dict(sha256=digest(raw), bytes=len(raw))


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    return type(value) is bool or finite(value)


def load():
    data = json.loads(gzip.decompress((BUNDLE / "measurements.json.gz").read_bytes()))
    verification = json.loads((BUNDLE / "verification.json").read_text())
    return data, verification


def verify(data, verification, *, revisions=REVISIONS, source_changes=None):
    assert data["schema"] == "spiraltorch.topos_nn_shared_benchmark.v1"
    assert verification["schema"] == "spiraltorch.topos_nn_shared_benchmark_verification.v1"
    plan = data["plan"]
    assert json.loads(data["plan_raw_json"]) == plan
    assert digest(data["plan_raw_json"].encode()) == verification["private_files"]["plan.json"]["sha256"]
    assert plan["schema"] == "spiraltorch.topos_nn_shared_benchmark_plan.v1"
    assert [plan[a + "_revision"] for a in ("baseline", "candidate")] == list(revisions)
    assert plan["cases"] == CASES and plan["phase_order"] == PHASES
    assert plan["case_order"] == ["forward", "reverse", "forward", "reverse"]
    assert plan["rounds"] == 20 and plan["warmup_per_route"] == 2
    assert plan["routes"] == list(ROUTES)
    assert set(plan["harness_sha256"]) == set(data["harness_sources"]) == HARNESS
    for path, source in data["harness_sources"].items():
        assert digest(source.encode()) == plan["harness_sha256"][path]
    assert set(data["builds"]) == {"baseline", "candidate"}
    for arm, build in data["builds"].items():
        # Original build JSON uses two-space indentation, ASCII strings and
        # integer sizes; retain its key order to reproduce the committed bytes.
        verify_original(verification, arm + "-build.json", (json.dumps(build, indent=2) + "\n").encode())
        assert build["source_revision"] == plan[arm + "_revision"]
        assert build["harness_sha256"] == plan["harness_sha256"]
        assert build["binary_sha256"] == verification["private_files"][arm + "-probe"]["sha256"]
        # Some original build records omit the duplicated size. Their complete
        # JSON bytes are still bound above; the file inventory retains its size.
        if "binary_bytes" in build:
            assert build["binary_bytes"] == verification["private_files"][arm + "-probe"]["bytes"]
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in build["source_sha256"].values())
    before = data["builds"]["baseline"]["source_sha256"]
    after = data["builds"]["candidate"]["source_sha256"]
    assert set(before) == set(after)
    if source_changes is None:
        source_changes = {"crates/st-core/src/dynamics/topos_resonator.rs", "crates/st-nn/src/layers/topos_resonator.rs"}
    assert {p for p in before if before[p] != after[p]} == set(source_changes)

    expected_jobs = []
    for phase, arm in enumerate(PHASES):
        for case in (CASES[::-1] if phase % 2 else CASES):
            expected_jobs.append(dict(phase=phase, arm=arm, **case, name=f"p{phase}-{arm}-{case['id']}"))
    assert [r["job"] for r in data["records"]] == expected_jobs
    assert verification["counts"] == dict(native_reports=36, torch_reports=36, timed_samples=2880)
    assert data["pilots"]["excluded_from_summary"] is True
    assert [r["arm"] for r in data["pilots"]["records"]] == ["baseline", "candidate"]
    for pilot in data["pilots"]["records"]:
        for kind, suffix in (("native", ".json"), ("torch", "-torch.json")):
            verify_original(verification, "pilot-" + pilot["arm"] + suffix, pilot[kind + "_raw_json"].encode())
    parsed = []
    for record in data["records"]:
        job = record["job"]
        native, torch = [json.loads(record[k + "_raw_json"]) for k in ("native", "torch")]
        for kind, suffix in (("native", ".json"), ("torch", "-torch.json")):
            verify_original(verification, job["name"] + suffix, record[kind + "_raw_json"].encode())
        shape = [job["rows"], job["features"]]
        for receipt in (native, torch):
            assert receipt["shape"] == shape and all(type(n) is int for n in receipt["shape"])
            assert receipt["dtype"] == "float32" and receipt["gate_layout"] == "shared_rows"
            assert receipt["gate_gradient_reduction"] == "sum_without_additional_mean"
            assert receipt["gradient_storage"] == "preallocated_zero_accumulator_for_reference_and_all_rounds"
            assert receipt["config"] == dict(iterations=job["iterations"], coupling=.25,
                                             porosity=0.20000000298023224, saturation=1.)
            assert type(receipt["config"]["iterations"]) is int
            assert type(receipt["warmup_per_route"]) is int and receipt["warmup_per_route"] == 2
            assert receipt["round_order"] == [[0, 1] if i % 2 == 0 else [1, 0] for i in range(20)]
            assert all(type(n) is int for order in receipt["round_order"] for n in order)
            assert set(receipt["measurements_ms"]) == set(ROUTES)
            for timings in receipt["measurements_ms"].values():
                assert len(timings) == 20 and all(finite(t) and t > 0 for t in timings)
        assert native["schema"] == "spiraltorch.topos_shared_nn_probe.v1"
        assert native["status"] == "measured" and native["backend"] == "cpu"
        assert native["vector_order"] == list(FIELDS)
        assert native["vector_shapes"] == [shape, [1, shape[1]], shape, shape, shape, [1, shape[1]]]
        assert len(native["vector_sha256"]) == 6
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in native["vector_sha256"])
        assert native["vectors_sha256"] == verification["private_files"][job["name"] + ".f32le"]["sha256"]
        assert verification["private_files"][job["name"] + ".f32le"]["bytes"] == (4 * math.prod(shape) + 2 * shape[1]) * 4
        for key in ("forward_audit", "backward_audit"):
            audit = native[key]
            assert finite_tree(audit)
            assert [audit["rows"], audit["features"]] == shape and audit["iterations"] == job["iterations"]
            assert 0 <= audit["max_formula_tolerance_ratio"] <= 1
        assert native["forward_audit"]["max_output_error"] == 0
        assert native["backward_audit"]["max_grad_input_error"] == native["backward_audit"]["max_grad_gate_error"] == 0
        assert torch["schema"] == "spiraltorch.topos_shared_nn_torch_reference.v1"
        assert torch["status"] == "passed" and torch["device"] == "cpu"
        assert type(torch["threads"]) is int and torch["threads"] == 1 and torch["torch"] == "2.12.1"
        assert torch["rtol"] == 5e-4 and torch["atol"] == 3e-5
        assert torch["vectors_sha256"] == native["vectors_sha256"]
        assert torch["receipt_sha256"] == digest(record["native_raw_json"].encode())
        assert torch["client_sha256"] == plan["harness_sha256"]["tools/benchmark_topos_shared_module_reference.py"]
        assert torch["reference_sha256"] == plan["harness_sha256"]["tools/benchmark_topos_module_reference.py"]
        for key, limit in (("max_abs_error", 3e-5), ("max_tolerance_ratio", 1.)):
            assert set(torch[key]) == set(FIELDS[3:])
            # The absolute-only bound is also satisfied by this frozen study.
            assert all(finite(v) and 0 <= v <= limit for v in torch[key].values())
        parsed.append((job, native, torch))

    summary = []
    for case in CASES:
        matched = [(j, n, t) for j, n, t in parsed if j["id"] == case["id"]]
        for key in ("vector_sha256", "vectors_sha256", "forward_audit", "backward_audit"):
            assert all(n[key] == matched[0][1][key] for _, n, _ in matched)
        row = dict(case)
        for kind, index in (("native", 1), ("torch", 2)):
            for arm in ("baseline", "candidate"):
                for route in ROUTES:
                    times = [median(r[index]["measurements_ms"][route]) for r in matched if r[0]["arm"] == arm]
                    assert len(times) == 2
                    row[f"{kind}_{arm}_{route}_run_medians_ms"] = times
                    row[f"{kind}_{arm}_{route}_ms"] = median(times)
        for route, label in (("forward", "forward"), ("forward_backward", "backward")):
            row[f"candidate_over_baseline_{label}"] = row[f"native_candidate_{route}_ms"] / row[f"native_baseline_{route}_ms"]
        row["native_candidate_over_torch_forward_backward"] = row["native_candidate_forward_backward_ms"] / row["torch_candidate_forward_backward_ms"]
        for key in ("max_abs_error", "max_tolerance_ratio"):
            row[key] = {field: max(t[key][field] for _, _, t in matched) for field in FIELDS[3:]}
        summary.append(row)
    assert data["summary"] == dict(aggregation=AGGREGATION, rows=summary)


def test_saved_records_and_hashes():
    expected = {"README.md", "measurements.json.gz", "verification.json"}
    seen = set()
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split("  ", 1)
        assert name in expected and name not in seen
        seen.add(name)
        assert digest((BUNDLE / name).read_bytes()) == sha
    assert seen == expected
    verify(*load())


def test_incomplete_or_misaggregated_matrix_is_rejected():
    data, verification = load()
    for mutate in (
        lambda d: d["records"].pop(),
        lambda d: d["records"].reverse(),
        lambda d: d["summary"]["rows"][0].__setitem__("native_candidate_forward_ms", .000001),
        lambda d: d["builds"]["candidate"].__setitem__("source_revision", REVISIONS[0]),
        lambda d: d["harness_sources"].__setitem__("tools/benchmark_topos_module_reference.py", "pass\n"),
        lambda d: d["pilots"].__setitem__("excluded_from_summary", False),
    ):
        bad = copy.deepcopy(data)
        mutate(bad)
        try:
            verify(bad, verification)
        except AssertionError:
            continue
        raise AssertionError("contradictory matrix accepted")


def test_self_consistent_bad_receipts_are_rejected():
    data, verification = load()
    for kind, key, value in (
        ("torch", "max_tolerance_ratio", dict.fromkeys(FIELDS[3:], 1.01)),
        ("torch", "max_abs_error", dict.fromkeys(FIELDS[3:], .001)),
        ("torch", "receipt_sha256", "0" * 64),
        ("torch", "reference_sha256", "0" * 64),
        ("torch", "measurements_ms", dict.fromkeys(ROUTES, [float("nan")] * 20)),
        ("native", "vector_sha256", ["0" * 64] * 6),
        ("native", "gradient_storage", "none"),
    ):
        bad, witness = copy.deepcopy(data), copy.deepcopy(verification)
        record = bad["records"][0]
        receipt = json.loads(record[kind + "_raw_json"])
        receipt[key] = value
        record[kind + "_raw_json"] = json.dumps(receipt)
        # Rebind outer hashes so validation must reject the semantic contradiction.
        if kind == "native":
            torch = json.loads(record["torch_raw_json"])
            torch["receipt_sha256"] = digest(record["native_raw_json"].encode())
            record["torch_raw_json"] = json.dumps(torch)
        for name, suffix in (("native", ".json"), ("torch", "-torch.json")):
            raw = record[name + "_raw_json"].encode()
            witness["private_files"][record["job"]["name"] + suffix] = dict(sha256=digest(raw), bytes=len(raw))
        try:
            verify(bad, witness)
        except AssertionError:
            continue
        raise AssertionError("contradictory receipt accepted")


def test_retained_build_and_pilot_changes_are_rejected():
    data, verification = load()
    for index, arm in enumerate(("baseline", "candidate")):
        mutations = [
            lambda d: d["builds"][arm]["source_sha256"].__setitem__("crates/st-core/src/dynamics/topos_resonator.rs", "0" * 64),
            lambda d: d["pilots"]["records"][index].__setitem__("native_raw_json", "{}"),
            lambda d: d["pilots"]["records"][index].__setitem__("torch_raw_json", "{}"),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(data)
            mutate(bad)
            try:
                verify(bad, verification)
            except AssertionError:
                continue
            raise AssertionError("retained original record identity drift accepted")
        for name in (arm + "-build.json", "pilot-" + arm + ".json", "pilot-" + arm + "-torch.json"):
            bad_witness = copy.deepcopy(verification)
            bad_witness["private_files"][name]["bytes"] += 1
            try:
                verify(data, bad_witness)
            except AssertionError:
                continue
            raise AssertionError("incorrect original byte length accepted")


if __name__ == "__main__":
    test_saved_records_and_hashes()
    test_incomplete_or_misaggregated_matrix_is_rejected()
    test_self_consistent_bad_receipts_are_rejected()
    test_retained_build_and_pilot_changes_are_rejected()
    print("Saved native NN benchmark checks passed (no fresh execution)")
