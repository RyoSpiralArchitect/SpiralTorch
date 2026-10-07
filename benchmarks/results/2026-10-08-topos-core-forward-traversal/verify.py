"""Check this frozen screening record, not current runtime performance."""

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


HERE = Path(__file__).resolve().parent
BASELINE = "71b7a05dff1d18db3d1b20dbf88f8c7389ab5703"
CANDIDATE = "dcad72c1949e3d82d898b987f597a48c2ca3c517"
RESTORED = "35feb769b0af7951eb0999a395fd6e0fd37d4a55"
TREE = "5f3f4b8a1a977820249f7f3897725a10bc9b6565"
CASES = [dict(id=f"{r}x{f}-k{k}", rows=r, features=f, iterations=k)
         for r, f in ((8, 3), (64, 128), (256, 768)) for k in (1, 5, 16)]
PHASES = ["baseline", "candidate", "candidate", "baseline"]
ORDERS = ["forward", "reverse", "forward", "reverse"]
ROUTES = ["snapshots", "learning_step"]
NATIVE_ROUTES = ["shared_state_validation", "core_shared_capture",
                 "nn_shared_forward", "core_captured_vjp_audited"]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def original(witness, name, raw):
    assert witness["private_files"][name] == dict(sha256=digest(raw.encode()), bytes=len(raw.encode()))
    return json.loads(raw)


def timings(values, count):
    assert len(values) == count
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values)


def aggregate(observed, routes, native=False):
    result = {}
    for index, route in enumerate(routes):
        medians = [median(r["measurements_ms"][index if native else route]) for r in observed]
        before, after = median([medians[0], medians[3]]), median([medians[1], medians[2]])
        result[route] = dict(process_medians_ms=medians, baseline_ms=before, candidate_ms=after,
                             ratio=after / before if before and after else None)
    return result


def verify(data, witness):
    assert data["schema"] == "spiraltorch.topos_forward_traversal_archive.v1"
    assert witness["schema"] == "spiraltorch.topos_forward_traversal_verification.v1"
    assert witness["decision"] == "rejected"
    assert [witness[k] for k in ("baseline_revision", "candidate_revision", "revert_revision")] == [BASELINE, CANDIDATE, RESTORED]
    assert witness["baseline_tree"] == witness["revert_tree"] == TREE
    assert len(witness["private_files"]) == 139
    assert all(sha(v["sha256"]) and type(v["bytes"]) is int and v["bytes"] >= 0 for v in witness["private_files"].values())
    plan = original(witness, "plan.json", data["plan_raw_json"])
    assert plan["schema"] == "spiraltorch.topos_forward_traversal_plan.v1"
    assert [plan[k] for k in ("baseline_revision", "candidate_revision")] == [BASELINE, CANDIDATE]
    assert plan["cases"] == CASES and plan["phase_order"] == PHASES and plan["case_order"] == ORDERS
    assert [plan[k] for k in ("browser_rounds", "browser_warmup_blocks", "browser_learning_updates_per_case", "native_rounds", "native_warmup_per_route")] == [20, 2, 88, 24, 2]
    assert plan["promotion_gate"].startswith("Require browser learning-step median at least 5% shorter in at least two")
    for path, expected in plan["browser_harness"].items():
        assert digest(data["harness_sources"][path].encode()) == expected
    assert digest(data["harness_sources"]["crates/st-nn/examples/topos_shared_phase_probe.rs"].encode()) == plan["native_harness_sha256"]
    builds = {arm: original(witness, arm + "-build.json", data["build_raw_json"][arm]) for arm in ("baseline", "candidate")}
    for arm, revision in zip(("baseline", "candidate"), (BASELINE, CANDIDATE)):
        build = builds[arm]
        assert build["source_revision"] == revision
        for target, package in build["packages"].items():
            assert target in ("web", "nodejs") and package["directory"] == f"{arm}-{target}-client"
            for field, file in (("wasm", "spiraltorch_wasm_bg.wasm"), ("wrapper", "spiraltorch_wasm.js")):
                assert package[field + "_sha256"] == witness["private_files"][package["directory"] + "/" + file]["sha256"]
        for name, value in build["native"].items():
            assert value == witness["private_files"][arm + "-" + name]["sha256"]
        node = original(witness, arm + "-node-contract.json", data["node_contract_raw_json"][arm])
        assert [node[k] for k in ("status", "cases", "checks")] == ["passed", 27, 469]
        assert node["wasm_sha256"] == build["packages"]["nodejs"]["wasm_sha256"]
        assert node["wrapper_sha256"] == build["packages"]["nodejs"]["wrapper_sha256"]
        assert node["learning"]["updates"] == 240 and node["learning"]["next_update_exact"] is True
        assert node["learning"]["matched_wide_sum_trajectory_exact"] is True
        assert [node["snapshot_ownership"][k] for k in ("status", "cases", "checks", "memory_growths")] == ["passed", 6, 121, 12]
    before, after = [builds[a]["source_sha256"] for a in ("baseline", "candidate")]
    assert set(before) == set(after) and all(sha(v) for v in [*before.values(), *after.values()])
    assert {p for p in before if before[p] != after[p]} == {"crates/st-core/src/dynamics/topos_resonator.rs"}

    assert len(data["browser_records"]) == 4
    parsed, zero_samples = [], 0
    for phase, record in enumerate(data["browser_records"]):
        arm = PHASES[phase]
        assert [record[k] for k in ("phase", "arm", "case_order", "file")] == [phase, arm, ORDERS[phase], f"browser-phase-{phase}-{arm}.json"]
        report = original(witness, record["file"], record["raw_json"])
        assert [report[k] for k in ("schema", "status", "backend", "case_order")] == ["spiraltorch.topos_snapshot_benchmark.v1", "passed", "rust_f32_wasm", ORDERS[phase]]
        assert report["rounds"] == 20 and report["warmup_blocks_per_route"] == 2 and report["routes"] == ROUTES
        assert report["page_errors"] == report["console_messages"] == []
        assert report["browser_version"] == witness["runtime"]["browser"]
        assert report["wasm_sha256"] == builds[arm]["packages"]["web"]["wasm_sha256"]
        assert {"/", "/module/spiraltorch_wasm.js", "/module/spiraltorch_wasm_bg.wasm", "/topos_snapshot_benchmark.mjs"} <= set(report["asset_sha256"])
        for path, value in report["asset_sha256"].items():
            if path.startswith("/module/"):
                assert value == witness["private_files"][f"{arm}-web-client/" + path.removeprefix("/module/")]["sha256"]
            else:
                source = "bindings/st-wasm/tests/" + ("topos_snapshot_benchmark.html" if path == "/" else path.removeprefix("/"))
                assert value == plan["browser_harness"][source]
        assert [{k: c[k] for k in CASES[0]} for c in report["cases"]] == (CASES[::-1] if phase % 2 else CASES)
        for case in report["cases"]:
            assert [case[k] for k in ("coupling", "porosity", "saturation", "learning_rate")] == [.25, .2, 1, .1 * case["features"]]
            assert case["repeats_per_sample"] == dict(snapshots=32 if case["rows"] * case["features"] < 8192 else 8, learning_step=4)
            assert case["round_order"] == [[0, 1] if i % 2 == 0 else [1, 0] for i in range(20)]
            assert set(case["measurements_ms"]) == set(ROUTES)
            for values in case["measurements_ms"].values():
                timings(values, 20)
                zero_samples += values.count(0)
            assert all(sha(case[k]) for k in ("input_sha256", "initial_gate_sha256", "target_sha256"))
            assert len(case["snapshot_sha256"]) == 3 and all(sha(v) for v in case["snapshot_sha256"])
            learning = case["learning"]
            assert learning["updates"] == 88 and len(learning["blocks"]) == 22
            timings(learning["losses"], 88)
            assert learning["initial_loss"] == learning["losses"][0] and 0 <= learning["final_loss"] < learning["initial_loss"]
            assert sha(learning["final_gate_sha256"]) and sha(learning["final_output_sha256"])
            for round, block in enumerate(learning["blocks"]):
                assert block["round"] == round and block["updates"] == 4 * (round + 1)
                assert len(block["sha256"]) == 4 and all(sha(v) for v in block["sha256"])
        parsed.append(report)
    summary = []
    for case in CASES:
        observed = [next(c for c in r["cases"] if c["id"] == case["id"]) for r in parsed]
        for key in ("input_sha256", "initial_gate_sha256", "target_sha256", "snapshot_sha256", "learning"):
            assert all(r[key] == observed[0][key] for r in observed)
        summary.append(dict(id=case["id"], routes=aggregate(observed, ROUTES)))
    ratios = [c["routes"]["learning_step"]["ratio"] for c in summary[-3:]]
    passed = sum(v is not None and v <= .95 for v in ratios) >= 2 and all(v is not None and v <= 1.05 for v in ratios)
    assert passed is False
    assert data["browser_summary"] == dict(schema="spiraltorch.topos_forward_traversal_browser_summary.v1", cases=summary, timing_samples=1440, zero_samples=zero_samples, losses=3168, saved_states_exact=True, promotion_screen_passed=passed)

    assert len(data["native_records"]) == 36
    parsed = {}
    for index, record in enumerate(data["native_records"]):
        phase, position = divmod(index, 9)
        arm, case = PHASES[phase], (CASES[::-1] if phase % 2 else CASES)[position]
        assert [record[k] for k in ("phase", "arm", "id", "file")] == [phase, arm, case["id"], f"native-phase-{phase}-{arm}-{case['id']}.json"]
        result = original(witness, record["file"], record["raw_json"])
        assert [result[k] for k in ("schema", "status", "backend", "dtype")] == ["spiraltorch.topos_shared_phase_probe.v1", "measured", "cpu", "float32"]
        assert result["shape"] == [case["rows"], case["features"]] and result["iterations"] == case["iterations"]
        assert result["coupling"] == .25 and result["porosity"] == .20000000298023224
        assert result["routes"] == NATIVE_ROUTES and result["warmup_per_route"] == 2
        assert result["round_order"] == [[(r + o) % 4 for o in range(4)] for r in range(2, 26)]
        assert len(result["measurements_ms"]) == 4
        for values in result["measurements_ms"]:
            timings(values, 24)
        parsed[phase, case["id"]] = result
    summary = [dict(id=c["id"], routes=aggregate([parsed[p, c["id"]] for p in range(4)], NATIVE_ROUTES, True)) for c in CASES]
    assert data["native_summary"]["cases"] == summary
    assert data["native_summary"]["records"] == 36 and data["native_summary"]["timing_samples"] == 3456
    assert witness["counts"] == dict(browser_processes=4, conditions=9, browser_samples=1440, browser_zero_samples=zero_samples, browser_learning_updates=3168, native_records=36, native_samples=3456, retained_originals=139)


def negative_controls(data, witness):
    for kind in ("learning", "native", "missing_phase", "false_promotion"):
        bad, proof = copy.deepcopy(data), copy.deepcopy(witness)
        if kind in ("learning", "native"):
            record = bad["browser_records" if kind == "learning" else "native_records"][0]
            result = json.loads(record["raw_json"])
            if kind == "learning":
                result["cases"][0]["learning"]["final_gate_sha256"] = "0" * 64
            else:
                result["measurements_ms"][0].pop()
            raw = json.dumps(result)
            record["raw_json"] = raw
            proof["private_files"][record["file"]] = dict(sha256=digest(raw.encode()), bytes=len(raw.encode()))
        elif kind == "missing_phase":
            bad["browser_records"].pop()
        else:
            bad["browser_summary"]["promotion_screen_passed"] = True
        try:
            verify(bad, proof)
        except AssertionError:
            continue
        raise AssertionError("contradictory evidence accepted: " + kind)


if __name__ == "__main__":
    expected = {"README.md", "measurements.json.gz", "verification.json", "verify.py"}
    seen = set()
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        value, name = line.split("  ", 1)
        assert name in expected and name not in seen and digest((HERE / name).read_bytes()) == value
        seen.add(name)
    assert seen == expected
    data = json.loads(gzip.decompress((HERE / "measurements.json.gz").read_bytes()))
    witness = json.loads((HERE / "verification.json").read_text())
    verify(data, witness)
    negative_controls(data, witness)
    print("Rejected Topos traversal screen verified: 1,440 browser + 3,456 native timings; no fresh execution")
