"""Reconstruct row-VJP measurements and client checks, not a fresh execution."""

import copy
import gzip
import importlib.util
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/results/2026-10-08-topos-core-row-vjp"
SPEC = importlib.util.spec_from_file_location("nn_results", Path(__file__).with_name("test_topos_shared_nn_benchmark_results.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
REVISIONS = ["f9fc455dbf793af371e6682e4b63140288e6e1b3", "17a368b0711b01b1ae1da243322502f13e419d96"]
SCOPES = ["shared_state_validation", "core_shared_capture", "nn_shared_forward", "core_captured_vjp_audited"]


def load():
    return (
        json.loads(gzip.decompress((BUNDLE / "measurements.json.gz").read_bytes())),
        json.loads(gzip.decompress((BUNDLE / "diagnostics.json.gz").read_bytes())),
        json.loads((BUNDLE / "verification.json").read_text()),
    )


def verify(data, diagnostics, witness):
    BASE.verify(data, witness, revisions=REVISIONS, source_changes={"crates/st-core/src/dynamics/topos_resonator.rs"})
    assert data["plan"]["aggregation"] == BASE.AGGREGATION
    assert diagnostics["schema"] == "spiraltorch.topos_row_vjp_diagnostics.v1"
    plan = diagnostics["plan"]
    BASE.verify_original(witness, "phase-plan.json", diagnostics["plan_raw_json"].encode())
    assert json.loads(diagnostics["plan_raw_json"]) == plan
    assert plan["schema"] == "spiraltorch.topos_row_vjp_phase_plan.v1"
    assert [plan[a + "_revision"] for a in ("baseline", "candidate")] == REVISIONS
    assert plan["cases"] == BASE.CASES and plan["phase_order"] == BASE.PHASES
    assert plan["case_order"] == ["forward", "reverse", "forward", "reverse"]
    assert plan["rounds"] == 24 and plan["warmup_per_route"] == 2
    assert BASE.digest(diagnostics["probe_source"].encode()) == plan["harness_sha256"]
    for arm in ("baseline", "candidate"):
        assert data["builds"][arm]["phase_binary_sha256"] == witness["private_files"][arm + "-phase-probe"]["sha256"]
    assert [r["job"] for r in diagnostics["records"]] == [r["job"] for r in data["records"]]
    parsed = []
    for record in diagnostics["records"]:
        job = record["job"]
        BASE.verify_original(witness, "phase-" + job["name"] + ".json", record["raw_json"].encode())
        result = json.loads(record["raw_json"])
        assert result["schema"] == "spiraltorch.topos_shared_phase_probe.v1"
        assert result["status"] == "measured" and result["backend"] == "cpu" and result["dtype"] == "float32"
        assert result["shape"] == [job["rows"], job["features"]] and result["iterations"] == job["iterations"]
        assert result["coupling"] == .25 and result["porosity"] == 0.20000000298023224
        assert result["warmup_per_route"] == 2 and result["routes"] == SCOPES
        assert result["round_order"] == [[(i + j) % 4 for j in range(4)] for i in range(2, 26)]
        assert len(result["measurements_ms"]) == 4
        assert all(len(ts) == 24 and all(BASE.finite(t) and t >= 0 for t in ts) for ts in result["measurements_ms"])
        parsed.append((job, result))
    summary = []
    for case in BASE.CASES:
        runs = [(j, r) for j, r in parsed if j["id"] == case["id"]]
        scopes = []
        for i, scope in enumerate(SCOPES):
            row = dict(scope=scope)
            for arm in ("baseline", "candidate"):
                times = [median(r["measurements_ms"][i]) for j, r in runs if j["arm"] == arm]
                row[arm + "_run_medians_ms"] = times
                row[arm + "_ms"] = median(times)
            row["candidate_over_baseline"] = row["candidate_ms"] / row["baseline_ms"] if row["baseline_ms"] else None
            scopes.append(row)
        summary.append(dict(id=case["id"], scopes=scopes))
    assert diagnostics["summary"] == summary

    check = witness["learning"]
    previous = ROOT / "benchmarks/results/2026-10-07-topos-nn-shared-capture/learning.json.gz"
    assert BASE.digest(gzip.decompress(previous.read_bytes())) == check["receipt_sha256"]
    assert check["receipt_sha256"] == witness["private_files"]["learning.json"]["sha256"]
    assert check["status"] == "passed" and check["updates"] == 200 and check["source_backend"] == "cpu"
    assert check["rtol"] == 5e-4 and check["atol"] == 3e-5
    assert set(check["max_abs_error"]) == {"output", "grad_input", "grad_gate", "gate_after", "loss"}
    assert all(BASE.finite(v) and 0 <= v <= 3e-5 for v in check["max_abs_error"].values())
    BASE.verify_original(witness, "torch-learning.json", witness["learning_raw_json"].encode())
    assert json.loads(witness["learning_raw_json"]) == check

    clients = witness["clients"]
    for name, original, wasm in (
        ("node", "node.json", "wasm-node"),
        ("browser", "browser.json", "wasm-web"),
        ("legacy_node", "legacy-node.json", "wasm-node"),
    ):
        raw = clients[name + "_raw_json"]
        BASE.verify_original(witness, original, raw.encode())
        result = json.loads(raw)
        assert result["status"] == "passed"
        assert result["wasm_sha256"] == witness["private_files"][wasm + "/spiraltorch_wasm_bg.wasm"]["sha256"]
        assert result["learning"]["updates"] == 240 and result["learning"]["next_update_exact"] is True
        assert 0 <= result["learning"]["final_loss"] < 1e-10 < result["learning"]["initial_loss"]
        if name == "legacy_node":
            assert len(result["cases"]) == 24 and result["guard_checks"] == 54 and result["captured"] is True
            assert result["learning"]["legacy_trajectory_exact"] is True
        else:
            assert result["cases"] == 27 and result["checks"] == 469
            assert result["learning"]["matched_wide_sum_trajectory_exact"] is True
        if name == "browser":
            assert result["page_errors"] == [] and result["console_messages"] == []
            assert result["asset_sha256"]["/module/spiraltorch_wasm_bg.wasm"] == result["wasm_sha256"]
            wrapper = result["asset_sha256"]["/module/spiraltorch_wasm.js"]
        else:
            wrapper = result["wrapper_sha256"]
        assert wrapper == witness["private_files"][wasm + "/spiraltorch_wasm.js"]["sha256"]
    assert clients["python_native_sha256"] == witness["private_files"]["python/spiraltorch/spiraltorch.abi3.so"]["sha256"]
    assert clients["python_loaded_identity_verified"] is True
    assert witness["focused_tests"]["python"]["passed"] == 136
    assert witness["focused_tests"]["python"]["skipped"] == 0


def test_frozen_row_vjp_evidence():
    expected = {"README.md", "measurements.json.gz", "diagnostics.json.gz", "verification.json"}
    seen = set()
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split("  ", 1)
        assert name in expected and name not in seen
        seen.add(name)
        assert BASE.digest((BUNDLE / name).read_bytes()) == sha
    assert seen == expected
    verify(*load())


def test_incomplete_or_changed_evidence_is_rejected():
    original = load()
    for mutate in (
        lambda d, p, w: d["records"].pop(),
        lambda d, p, w: d["builds"]["candidate"].__setitem__("source_revision", REVISIONS[0]),
        lambda d, p, w: d["summary"]["rows"][0].__setitem__("candidate_over_baseline_backward", .1),
        lambda d, p, w: p["records"].reverse(),
        lambda d, p, w: p["summary"][0]["scopes"][3].__setitem__("candidate_ms", 0),
        lambda d, p, w: p.__setitem__("probe_source", "// different probe"),
        lambda d, p, w: w["clients"].__setitem__("browser_raw_json", "{}"),
        lambda d, p, w: w["clients"].__setitem__("python_native_sha256", "0" * 64),
        lambda d, p, w: w["learning"].__setitem__("receipt_sha256", "0" * 64),
    ):
        bad = copy.deepcopy(original)
        mutate(*bad)
        try:
            verify(*bad)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("contradictory row-VJP evidence accepted")


if __name__ == "__main__":
    test_frozen_row_vjp_evidence()
    test_incomplete_or_changed_evidence_is_rejected()
    print("Saved row-VJP checks passed (no fresh execution)")
