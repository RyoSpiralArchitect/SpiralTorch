"""Stdlib reconstruction of the frozen NN capture measurements, not a timing gate."""

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/results/2026-10-07-topos-nn-captured-backward"


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _load():
    return json.loads(gzip.decompress((BUNDLE / "measurements.json.gz").read_bytes()))


def test_nn_capture_hash_closure_and_frozen_clients():
    names = set()
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert name not in names and Path(name).name == name
        names.add(name)
        assert _hash((BUNDLE / name).read_bytes()) == digest
    assert names == {"README.md", "results.json", "measurements.json.gz", "verification.json"}
    archive = _load()
    plan = archive["accepted_plan"]
    assert plan["baseline"]["source_revision"] == "e8313540f6a955c965d4b08045b3496e6a72b591"
    assert plan["candidate"]["source_revision"] == "60eb80fbf0e8103ee348a50d65a4c744196c0f03"
    assert plan["baseline"]["client_sha256"] == plan["candidate"]["client_sha256"]
    assert set(archive["clients"]) == set(plan["candidate"]["client_sha256"])
    for name, source in archive["clients"].items():
        assert _hash(source.encode()) == plan["candidate"]["client_sha256"][name]


def _reconstruct(archive):
    plan = archive["accepted_plan"]
    records = archive["records"]
    assert len(records) == len(plan["jobs"]) == 32
    for job, record in zip(plan["jobs"], records):
        assert job == record["job"]
        native = json.loads(record["native_raw_json"])
        torch = json.loads(record["torch_raw_json"])
        assert native["schema"] == "spiraltorch.topos_nn_probe.v1"
        assert torch["schema"] == "spiraltorch.topos_nn_torch_reference.v1"
        assert native["status"] == "measured" and torch["status"] == "passed"
        assert native["backend"] == "cpu" and native["dtype"] == "float32"
        assert torch["receipt_sha256"] == _hash(record["native_raw_json"].encode())
        assert torch["vectors_sha256"] == native["vectors_sha256"]
        assert torch["client_sha256"] == plan["candidate"]["client_sha256"]["tools/benchmark_topos_module_reference.py"]
        assert native["shape"] == torch["shape"] == [job["rows"], job["features"]]
        assert native["config"] == torch["config"]
        assert native["config"]["iterations"] == job["iterations"]
        assert native["config"]["coupling"] == job["coupling"]
        assert native["gradient_storage"] == "preallocated_zero_accumulator_for_reference_and_all_rounds"
        for receipt in (native, torch):
            assert receipt["warmup_per_route"] == 2
            assert receipt["round_order"] == [[0, 1], [1, 0]] * 6
            assert set(receipt["measurements_ms"]) == {"forward", "forward_backward"}
            for times in receipt["measurements_ms"].values():
                assert len(times) == 12
                assert all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in times)
    rows = []
    for name, nrows, cols, iterations in [
        ("small-4", 32, 64, 4), ("large-4", 256, 768, 4),
        ("small-16", 32, 64, 16), ("large-16", 256, 768, 16),
    ]:
        selected = [r for r in records if (r["job"]["rows"], r["job"]["features"], r["job"]["iterations"]) == (nrows, cols, iterations)]
        assert len(selected) == 8
        assert [r["job"]["arm"] for r in selected] == ["baseline", "candidate", "candidate", "baseline", "candidate", "baseline", "baseline", "candidate"]
        assert [r["job"]["pair"] for r in selected] == [0, 0, 1, 1, 2, 2, 3, 3]
        native = [json.loads(r["native_raw_json"]) for r in selected]
        for key in ("vector_sha256", "forward_audit", "backward_audit"):
            assert all(r[key] == native[0][key] for r in native)
        row = {"id": name, "rows": nrows, "features": cols, "iterations": iterations}
        for arm in ("baseline", "candidate"):
            for key in ("forward", "forward_backward"):
                for prefix, field in (("", "native_raw_json"), ("torch_", "torch_raw_json")):
                    medians = [statistics.median(json.loads(r[field])["measurements_ms"][key]) for r in selected if r["job"]["arm"] == arm]
                    row[f"{prefix}{arm}_{key}_ms"] = statistics.median(medians)
        row["forward_ratio"] = row["candidate_forward_ms"] / row["baseline_forward_ms"]
        row["learning_ratio"] = row["candidate_forward_backward_ms"] / row["baseline_forward_backward_ms"]
        row["max_errors"] = {key: max(json.loads(r["torch_raw_json"])["max_abs_error"][key] for r in selected) for key in ("output", "grad_input", "grad_gate")}
        rows.append(row)
    return rows


def test_nn_capture_all_pairs_and_summary_reconstruct():
    results = json.loads((BUNDLE / "results.json").read_text())
    assert results["summary"] == _reconstruct(_load())
    assert results["native_reports"] == results["torch_reports"] == 32
    assert results["timing_samples"] == 1536


def test_nn_capture_missing_measurement_is_rejected():
    archive = _load()
    archive["records"].pop()
    try:
        _reconstruct(archive)
    except AssertionError:
        return
    raise AssertionError("missing process report was accepted")


def test_nn_capture_receipts_keep_failures_and_execution_scope():
    verification = json.loads((BUNDLE / "verification.json").read_text())
    assert verification["rust_tests"] == {"core": 1024, "nn_default": 762, "nn_wgpu_topos": 15}
    assert verification["native_wgpu_mixed_routes_executed"] is True
    assert verification["strict_nn_clippy"]["status"] == "failed"
    assert verification["strict_nn_clippy"]["diagnostics"] == 23
    assert verification["scoped_core_clippy"] == "passed"
    rejected = verification["rejected_probe"]
    assert rejected["status"] == "rejected" and rejected["mismatched_entries"] == 74
    assert rejected["all_signed_zero"] is True
    assert rejected["changes_to_mathematics_or_comparison_tolerance"] is False
    before, after = verification["wasm_before"], verification["wasm_after"]
    ignored = {"wasm_sha256", "wrapper_sha256"}
    assert {k: v for k, v in before.items() if k not in ignored} == {k: v for k, v in after.items() if k not in ignored}
    assert after["status"] == "passed" and len(after["cases"]) == 24
    assert after["guard_checks"] == 54 and after["learning"]["updates"] == 240
    checks = verification["benchmark_negative_checks"]
    assert len(checks) == 5 and all(c["exit_code"] != 0 and c["python_optimized"] for c in checks)
