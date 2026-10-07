"""Reuse the frozen NN matrix checks with this study's pinned identities."""

import copy
import gzip
import importlib.util
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/results/2026-10-08-topos-nn-single-admission"
SPEC = importlib.util.spec_from_file_location("nn_results", Path(__file__).with_name("test_topos_shared_nn_benchmark_results.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
REVISIONS = ["5eefa75ba1627d72e44e7ac122ab91484d577f24", "66ad6eff886d6823fad206ee0b8b9b3a4ce245df"]


def verify(data, witness):
    BASE.verify(data, witness, revisions=REVISIONS, source_changes={"crates/st-nn/src/layers/topos_resonator.rs"})
    assert data["plan"]["aggregation"] == BASE.AGGREGATION
    check = witness["learning"]
    previous = ROOT / "benchmarks/results/2026-10-07-topos-nn-shared-capture/learning.json.gz"
    assert BASE.digest(gzip.decompress(previous.read_bytes())) == check["receipt_sha256"]
    assert check["receipt_sha256"] == witness["private_files"]["learning.json"]["sha256"]
    assert check["status"] == "passed" and check["updates"] == 200 and check["source_backend"] == "cpu"
    assert check["rtol"] == 5e-4 and check["atol"] == 3e-5
    assert set(check["max_abs_error"]) == {"output", "grad_input", "grad_gate", "gate_after", "loss"}
    assert all(BASE.finite(v) and 0 <= v <= 3e-5 for v in check["max_abs_error"].values())
    BASE.verify_original(witness, "learning-torch.json", witness["learning_raw_json"].encode())
    assert json.loads(witness["learning_raw_json"]) == check


def test_saved_matrix_and_learning():
    expected = {"README.md", "measurements.json.gz", "diagnostics.json.gz", "verification.json"}
    seen = set()
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split("  ", 1)
        assert name in expected and name not in seen
        seen.add(name)
        assert BASE.digest((BUNDLE / name).read_bytes()) == sha
    assert seen == expected
    data = json.loads(gzip.decompress((BUNDLE / "measurements.json.gz").read_bytes()))
    witness = json.loads((BUNDLE / "verification.json").read_text())
    verify(data, witness)
    for mutate in (
        lambda d: d["records"].pop(),
        lambda d: d["summary"]["rows"][1].__setitem__("candidate_over_baseline_backward", .5),
        lambda d: d["builds"]["candidate"]["source_sha256"].__setitem__("crates/st-nn/src/layers/topos_resonator.rs", "0" * 64),
        lambda d: d["pilots"]["records"][0].__setitem__("native_raw_json", "{}"),
        lambda d: d["plan"].pop("aggregation"),
    ):
        bad = copy.deepcopy(data)
        mutate(bad)
        try:
            verify(bad, witness)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("contradictory study accepted")


def test_diagnostic_scopes_are_complete_and_reconstructible():
    data = json.loads(gzip.decompress((BUNDLE / "diagnostics.json.gz").read_bytes()))
    witness = json.loads((BUNDLE / "verification.json").read_text())
    plan = data["plan"]
    assert plan["baseline_revision"] == "3dc8150f13f30a26e4614d37ebd0d2b44b06252d"
    assert plan["cases"] == BASE.CASES and plan["rounds"] == 24 and plan["replicates"] == 2
    assert plan["case_order"] == ["forward", "reverse"]
    assert BASE.digest(data["probe_source"].encode()) == plan["probe_sha256"]
    assert plan["probe_binary_sha256"] == witness["private_files"]["baseline-phase-probe"]["sha256"]
    records = data["records"]
    assert [(r["rep"], r["id"]) for r in records] == [
        (rep, c["id"]) for rep in range(2) for c in (BASE.CASES[::-1] if rep else BASE.CASES)]
    parsed = []
    for record in records:
        BASE.verify_original(witness, f"profile-{record['rep']}-{record['id']}.json", record["raw_json"].encode())
        result = json.loads(record["raw_json"])
        case = next(c for c in BASE.CASES if c["id"] == record["id"])
        assert result["shape"] == [case["rows"], case["features"]] and result["iterations"] == case["iterations"]
        assert result["status"] == "measured" and result["backend"] == "cpu" and result["dtype"] == "float32"
        assert result["coupling"] == .25 and result["porosity"] == 0.20000000298023224
        assert result["warmup_per_route"] == 2
        assert result["routes"] == ["shared_state_validation", "core_shared_capture", "nn_shared_forward", "core_captured_vjp_audited"]
        assert result["round_order"] == [[(i + j) % 4 for j in range(4)] for i in range(2, 26)]
        assert len(result["measurements_ms"]) == 4
        # Tiny validation-only samples can be below the clock resolution.
        # No speed ratio or subtraction is computed from these diagnostic scopes.
        assert all(len(ts) == 24 and all(BASE.finite(t) and t >= 0 for t in ts) for ts in result["measurements_ms"])
        parsed.append((record["id"], result))
    expected = []
    for case in BASE.CASES:
        runs = [r for name, r in parsed if name == case["id"]]
        scopes = []
        for index, route in enumerate(runs[0]["routes"]):
            medians = [median(r["measurements_ms"][index]) for r in runs]
            scopes.append(dict(route=route, run_medians=medians, ms=median(medians)))
        expected.append(dict(id=case["id"], medians_ms=scopes))
    assert data["summary"] == expected


if __name__ == "__main__":
    test_saved_matrix_and_learning()
    test_diagnostic_scopes_are_complete_and_reconstructible()
    print("Saved direct-CPU admission checks passed (no fresh execution)")
