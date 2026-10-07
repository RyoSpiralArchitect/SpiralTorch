"""Consistency of complete saved browser results, not fresh runtime evidence."""

import copy
import gzip
import importlib.util
import json
from pathlib import Path
import re
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/results/2026-10-08-topos-wasm-direct-snapshots"
SPEC = importlib.util.spec_from_file_location("nn_results", Path(__file__).with_name("test_topos_shared_nn_benchmark_results.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
REVISIONS = ["370eb3fb59483f59c940a737d94437a798d82e15", "6b3bcd5cc68bf342a1294b8ee489dad5d0c6442f"]
CASES = [{k: v for k, v in c.items() if k != "coupling"} for c in BASE.CASES]
ROUTES = ["snapshots", "learning_step"]
HARNESS = {"bindings/st-wasm/tests/topos_snapshot_benchmark.mjs", "bindings/st-wasm/tests/topos_snapshot_benchmark.html", "tools/test_resident_browser.cjs"}


def load():
    return (json.loads(gzip.decompress((BUNDLE / "measurements.json.gz").read_bytes())),
            json.loads((BUNDLE / "verification.json").read_text()))


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def verify(data, witness):
    assert data["schema"] == "spiraltorch.topos_snapshot_archive.v1"
    assert witness["schema"] == "spiraltorch.topos_snapshot_verification.v1"
    plan = data["plan"]
    BASE.verify_original(witness, "plan.json", data["plan_raw_json"].encode())
    assert json.loads(data["plan_raw_json"]) == plan
    assert [plan[a + "_revision"] for a in ("baseline", "candidate")] == REVISIONS
    assert plan["schema"] == "spiraltorch.topos_snapshot_plan.v1"
    assert plan["cases"] == CASES and plan["phase_order"] == BASE.PHASES
    assert plan["case_order"] == ["forward", "reverse", "forward", "reverse"]
    assert plan["routes"] == ROUTES and plan["rounds"] == 20 and plan["warmup_blocks_per_route"] == 2
    assert plan["learning_updates_per_case"] == 88 and plan["aggregation"] == BASE.AGGREGATION
    assert set(data["harness_sources"]) == set(plan["harness_sha256"]) == HARNESS
    for path, source in data["harness_sources"].items():
        assert BASE.digest(source.encode()) == plan["harness_sha256"][path]
    assert set(data["builds"]) == {"baseline", "candidate"}
    for i, arm in enumerate(("baseline", "candidate")):
        build = data["builds"][arm]
        BASE.verify_original(witness, arm + "-build.json", (json.dumps(build, indent=2) + "\n").encode())
        assert build["source_revision"] == REVISIONS[i]
        for target, suffix in (("nodejs", "node"), ("web", "web")):
            for field, file in (("wasm", "spiraltorch_wasm_bg.wasm"), ("wrapper", "spiraltorch_wasm.js")):
                assert build["packages"][target][field + "_sha256"] == witness["private_files"][f"{arm}-{suffix}/{file}"]["sha256"]
    before, after = [data["builds"][a]["source_sha256"] for a in ("baseline", "candidate")]
    assert set(before) == set(after) and all(sha(v) for v in [*before.values(), *after.values()])
    assert {p for p in before if before[p] != after[p]} == {"bindings/st-wasm/src/topos_resonator.rs"}
    assert data["builds"]["candidate"]["build_checkout_revision"] == REVISIONS[0]
    assert data["builds"]["candidate"]["source_commit_created_after_build"] is True
    types = data["public_topos_types"]
    assert types["baseline"] == types["candidate"]
    assert BASE.digest(types["candidate"].encode()) == data["builds"]["candidate"]["public_topos_types_sha256"]

    assert len(data["records"]) == 4
    parsed, zero_samples = [], 0
    for phase, record in enumerate(data["records"]):
        arm, order = BASE.PHASES[phase], plan["case_order"][phase]
        assert [record["phase"], record["arm"], record["case_order"]] == [phase, arm, order]
        BASE.verify_original(witness, f"p{phase}-{arm}.json", record["raw_json"].encode())
        result = json.loads(record["raw_json"])
        assert result["schema"] == "spiraltorch.topos_snapshot_benchmark.v1" and result["status"] == "passed"
        assert result["backend"] == "rust_f32_wasm" and result["case_order"] == order
        assert result["rounds"] == 20 and result["warmup_blocks_per_route"] == 2 and result["routes"] == ROUTES
        assert result["page_errors"] == [] and result["console_messages"] == []
        assert result["browser_version"] == witness["runtime"]["browser"]
        assert result["wasm_sha256"] == data["builds"][arm]["packages"]["web"]["wasm_sha256"]
        for path, digest in result["asset_sha256"].items():
            if path.startswith("/module/"):
                assert digest == witness["private_files"][arm + "-web/" + path.removeprefix("/module/")]["sha256"]
            else:
                source = "bindings/st-wasm/tests/" + ("topos_snapshot_benchmark.html" if path == "/" else path.removeprefix("/"))
                assert digest == plan["harness_sha256"][source]
        assert [c["id"] for c in result["cases"]] == [c["id"] for c in (CASES[::-1] if phase % 2 else CASES)]
        for case in result["cases"]:
            expected = next(c for c in CASES if c["id"] == case["id"])
            assert {k: case[k] for k in expected} == expected
            assert [case["coupling"], case["porosity"], case["saturation"]] == [.25, .2, 1]
            assert case["learning_rate"] == .1 * case["features"]
            assert case["repeats_per_sample"] == dict(snapshots=32 if case["rows"] * case["features"] < 8192 else 8, learning_step=4)
            assert case["round_order"] == [[0, 1] if i % 2 == 0 else [1, 0] for i in range(20)]
            assert set(case["measurements_ms"]) == set(ROUTES)
            for timings in case["measurements_ms"].values():
                assert len(timings) == 20 and all(BASE.finite(t) and t >= 0 for t in timings)
                zero_samples += timings.count(0)
            assert all(sha(case[k]) for k in ("input_sha256", "initial_gate_sha256", "target_sha256"))
            assert len(case["snapshot_sha256"]) == 3 and all(sha(v) for v in case["snapshot_sha256"])
            learning = case["learning"]
            assert learning["updates"] == 88 and len(learning["losses"]) == 88 and len(learning["blocks"]) == 22
            assert all(BASE.finite(v) and v >= 0 for v in learning["losses"])
            assert learning["losses"][0] == learning["initial_loss"]
            assert 0 <= learning["final_loss"] < learning["initial_loss"]
            assert sha(learning["final_gate_sha256"]) and sha(learning["final_output_sha256"])
            for round, block in enumerate(learning["blocks"]):
                assert block["round"] == round and block["updates"] == 4 * (round + 1)
                assert len(block["sha256"]) == 4 and all(sha(v) for v in block["sha256"])
            parsed.append((arm, case))
    assert witness["counts"] == dict(browser_processes=4, conditions=9, timed_samples=1440, zero_clock_samples=zero_samples, learning_updates=3168)
    summary = []
    for case in CASES:
        matched = [(a, r) for a, r in parsed if r["id"] == case["id"]]
        for key in ("input_sha256", "initial_gate_sha256", "target_sha256", "snapshot_sha256", "learning"):
            assert all(r[key] == matched[0][1][key] for _, r in matched)
        row = dict(case)
        for route in ROUTES:
            for arm in ("baseline", "candidate"):
                times = [median(r["measurements_ms"][route]) for a, r in matched if a == arm]
                row[f"{arm}_{route}_run_medians_ms"] = times
                row[f"{arm}_{route}_ms"] = median(times)
            before, after = [row[f"{a}_{route}_ms"] for a in ("baseline", "candidate")]
            row["candidate_over_baseline_" + route] = after / before if before and after else None
        row.update(initial_loss=matched[0][1]["learning"]["initial_loss"], final_loss=matched[0][1]["learning"]["final_loss"])
        summary.append(row)
    assert data["summary"] == dict(aggregation=BASE.AGGREGATION, rows=summary)

    for arm in ("baseline", "candidate"):
        for host, target in (("node", "nodejs"), ("browser", "web")):
            raw = witness["contracts"][arm + "_" + host + "_raw_json"]
            BASE.verify_original(witness, f"{arm}-{host}-contract.json", raw.encode())
            result = json.loads(raw)
            assert result["status"] == "passed" and result["cases"] == 27 and result["checks"] == 469
            assert result["wasm_sha256"] == data["builds"][arm]["packages"][target]["wasm_sha256"]
            ownership = result["snapshot_ownership"]
            assert [ownership[k] for k in ("status", "cases", "checks", "memory_growths")] == ["passed", 6, 121, 12]
            assert result["learning"]["updates"] == 240 and result["learning"]["matched_wide_sum_trajectory_exact"] is True
            assert result["learning"]["next_update_exact"] is True
    BASE.verify_original(witness, "candidate-legacy.json", witness["legacy_raw_json"].encode())
    legacy = json.loads(witness["legacy_raw_json"])
    assert legacy["status"] == "passed" and len(legacy["cases"]) == 24 and legacy["guard_checks"] == 54
    assert legacy["capture_required"] is True and legacy["learning"]["updates"] == 240
    assert legacy["learning"]["legacy_trajectory_exact"] is True and legacy["learning"]["next_update_exact"] is True


def test_frozen_browser_snapshot_evidence():
    expected = {"README.md", "measurements.json.gz", "verification.json"}
    seen = set()
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name in expected and name not in seen
        seen.add(name)
        assert BASE.digest((BUNDLE / name).read_bytes()) == digest
    assert seen == expected
    verify(*load())


def test_semantic_contradictions_are_rejected():
    original, proof = load()
    for field, value in (("updates", 84), ("losses", [float("nan")] * 88), ("final_gate_sha256", "0" * 64)):
        bad, witness = copy.deepcopy(original), copy.deepcopy(proof)
        result = json.loads(bad["records"][0]["raw_json"])
        result["cases"][0]["learning"][field] = value
        raw = json.dumps(result).encode()
        bad["records"][0]["raw_json"] = raw.decode()
        witness["private_files"]["p0-baseline.json"] = dict(sha256=BASE.digest(raw), bytes=len(raw))
        try:
            verify(bad, witness)
        except AssertionError:
            continue
        raise AssertionError("self-consistent bad learning evidence accepted")
    for mutate in (
        lambda d: d["records"].pop(),
        lambda d: d["records"].reverse(),
        lambda d: d["summary"]["rows"][0].__setitem__("candidate_over_baseline_learning_step", 0),
        lambda d: d["public_topos_types"].__setitem__("candidate", "different"),
    ):
        bad = copy.deepcopy(original)
        mutate(bad)
        try:
            verify(bad, proof)
        except AssertionError:
            continue
        raise AssertionError("contradictory snapshot study accepted")


if __name__ == "__main__":
    test_frozen_browser_snapshot_evidence()
    test_semantic_contradictions_are_rejected()
    print("Saved WASM snapshot checks passed (no fresh execution)")
