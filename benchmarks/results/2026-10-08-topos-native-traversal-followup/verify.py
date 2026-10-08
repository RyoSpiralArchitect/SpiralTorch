"""Reuse the complete NN validator; separately verify follow-up-only claims."""

import copy
import gzip
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "nn_results", HERE.parents[2] / "tools/test_topos_shared_nn_benchmark_results.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
REVISIONS = ["71b7a05dff1d18db3d1b20dbf88f8c7389ab5703", "dcad72c1949e3d82d898b987f597a48c2ca3c517"]
SCHEMA = "spiraltorch.topos_native_traversal_followup_plan.v1"


def verify(data, witness, sensitivity):
    BASE.verify(data, witness, revisions=REVISIONS,
                source_changes={"crates/st-core/src/dynamics/topos_resonator.rs"}, plan_schema=SCHEMA)
    assert len(witness["private_files"]) == 201
    plan = data["plan"]
    BASE.verify_original(witness, "plan.json", data["plan_raw_json"].encode())
    assert BASE.digest(data["runner_source"].encode()) == plan["runner_sha256"]
    prior_name = "2026-10-08-topos-core-forward-traversal"
    prior = json.loads(gzip.decompress((HERE.parent / prior_name / "measurements.json.gz").read_bytes()))
    assert plan["prior_study"] == prior_name
    assert prior["browser_summary"]["promotion_screen_passed"] is False
    for arm in ("baseline", "candidate"):
        raw = data["original_build_raw_json"][arm]
        assert raw == prior["build_raw_json"][arm]
        BASE.verify_original(witness, arm + "-original-build.json", raw.encode())
        original, build = json.loads(raw), data["builds"][arm]
        assert build["provenance"] == "reused_exact_executable_not_fresh_build"
        assert build["original_build_study"] == prior_name
        assert build["original_build_sha256"] == BASE.digest(raw.encode())
        assert build["source_revision"] == original["source_revision"]
        assert build["source_sha256"] == original["source_sha256"]
        assert plan["binary_sha256"][arm] == build["binary_sha256"] == original["native"]["topos_shared_module_probe"]
        assert witness["private_files"][arm + "-native"] == witness["private_files"][arm + "-probe"]
    assert witness["source_restoration"] == dict(
        revision="35feb769b0af7951eb0999a395fd6e0fd37d4a55",
        tree="5f3f4b8a1a977820249f7f3897725a10bc9b6565", production_change=False)
    BASE.verify_original(witness, "decision.json", data["decision_raw_json"].encode())
    decision = json.loads(data["decision_raw_json"])
    assert decision == data["decision"]
    ratios = [r["candidate_over_baseline_backward"] for r in data["summary"]["rows"][-3:]]
    passed = sum(v <= .95 for v in ratios) >= 2 and all(v <= 1.05 for v in ratios)
    assert passed is True and decision["native_screen_passed"] is passed
    assert decision["large_forward_backward_ratios"] == ratios
    assert decision["production_change"] is False and decision["prior_wasm_screen"] == "failed_unchanged"
    assert sensitivity["schema"] == "spiraltorch.topos_native_traversal_process_sensitivity.v1"
    assert sensitivity["classification"] == "post_hoc_derived_from_existing_process_medians_not_new_measurements"
    assert sensitivity["formal_screen_unchanged"] is True
    expected = []
    for row in data["summary"]["rows"]:
        before, after = [row["native_" + arm + "_forward_backward_run_medians_ms"] for arm in ("baseline", "candidate")]
        expected.append(dict(id=row["id"], baseline_process_medians_ms=before, candidate_process_medians_ms=after,
                             all_candidate_over_baseline_process_pairs=[b / a for a in before for b in after],
                             minimum_ratio=min(after) / max(before), maximum_ratio=max(after) / min(before)))
    assert sensitivity["rows"] == expected
    assert all(row["maximum_ratio"] > .95 for row in expected[-3:])


def negative_controls(data, witness, sensitivity):
    for kind in ("missing_record", "provenance", "decision", "sensitivity"):
        bad, proof, analysis = copy.deepcopy(data), copy.deepcopy(witness), copy.deepcopy(sensitivity)
        if kind == "missing_record":
            bad["records"].pop()
        elif kind == "provenance":
            build = bad["builds"]["candidate"]
            build["provenance"] = "fresh_build"
            raw = (json.dumps(build, indent=2) + "\n").encode()
            proof["private_files"]["candidate-build.json"] = dict(sha256=BASE.digest(raw), bytes=len(raw))
        elif kind == "decision":
            bad["decision"]["native_screen_passed"] = False
            raw = json.dumps(bad["decision"])
            bad["decision_raw_json"] = raw
            proof["private_files"]["decision.json"] = dict(sha256=BASE.digest(raw.encode()), bytes=len(raw.encode()))
        else:
            analysis["rows"][-3]["maximum_ratio"] = .60
        try:
            verify(bad, proof, analysis)
        except AssertionError:
            continue
        raise AssertionError("contradictory evidence accepted: " + kind)


if __name__ == "__main__":
    expected = {"README.md", "measurements.json.gz", "verification.json", "process_sensitivity.json", "verify.py"}
    seen = set()
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        value, name = line.split("  ", 1)
        assert name in expected and name not in seen and BASE.digest((HERE / name).read_bytes()) == value
        seen.add(name)
    assert seen == expected
    data = json.loads(gzip.decompress((HERE / "measurements.json.gz").read_bytes()))
    witness = json.loads((HERE / "verification.json").read_text())
    sensitivity = json.loads((HERE / "process_sensitivity.json").read_text())
    verify(data, witness, sensitivity)
    negative_controls(data, witness, sensitivity)
    print("Native Topos follow-up verified: 72 reports, 2,880 timings; mixed, not adopted")
