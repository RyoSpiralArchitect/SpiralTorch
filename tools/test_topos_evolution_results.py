"""Verify frozen Topos copy-elision results without timing or training."""

import gzip
import hashlib
import json
from pathlib import Path
import statistics

DATA = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-07-topos-in-place-evolution"
OWNED_DATA = DATA.with_name("2026-10-07-topos-owned-capture")
ROUTES = ("rust_list", "rust_buffer_recomputed", "rust_public", "torch_reference")


def read(name, directory=DATA):
    raw = (directory / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


def digest(raw):
    return hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()


def test_closed_numeric_archive_and_runtime_identity():
    for directory in (DATA, OWNED_DATA):
        check_closed_archive(directory)


def check_closed_archive(directory):
    entries = dict(line.split("  ", 1)[::-1] for line in (directory / "SHA256SUMS").read_text().splitlines())
    assert set(entries) == {p.name for p in directory.iterdir() if p.name != "SHA256SUMS"}
    for name, expected in entries.items():
        raw = (directory / name).read_bytes()
        assert digest(raw) == expected
        raw = gzip.decompress(raw) if name.endswith(".gz") else raw
        assert not any(marker in raw for marker in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))
    v, s = read("verification.json.gz", directory), read("summary.json", directory)
    before, after = v["runtime_before"], v["runtime_after"]
    assert len(before["files"]) == len(after["files"]) == 73
    assert before["files"].keys() == after["files"].keys()
    assert {n for n in before["files"] if before["files"][n] != after["files"][n]} == {"spiraltorch/spiraltorch.abi3.so"}
    assert s["source_revision"] == after["source_revision"]
    assert s["baseline_source_revision"] == before["source_revision"]


def test_every_raw_report_pair_gradient_and_statistic():
    for directory in (DATA, OWNED_DATA):
        check_paired_measurements(directory)


def check_paired_measurements(directory):
    m, v, s = (read(name, directory) for name in ("measurements.json.gz", "verification.json.gz", "summary.json"))
    assert digest(m["plan_json"]) == v["pair_plan_sha256"]
    plan = json.loads(m["plan_json"])
    assert plan["client_sha256"] == v["client_sha256"]["benchmark_topos_learning_py"]
    assert plan["baseline_manifest_sha256"] == v["runtime_before_manifest_sha256"]
    assert plan["source_revision"] == s["source_revision"]
    runs = {name: json.loads(raw) for name, raw in m["raw_reports"].items()}
    assert len(runs) == len(plan["runs"]) == s["run_count"] == 32
    assert set(runs) == {p["file"] for p in plan["runs"]} == set(v["raw_report_sha256"])
    assert s["timing_sample_count"] == 32 * 12 * len(ROUTES)
    for p in plan["runs"]:
        r = runs[p["file"]]
        assert digest(m["raw_reports"][p["file"]]) == v["raw_report_sha256"][p["file"]]
        assert r["status"] == "measured" and r["all_inputs_require_grad"] and r["capture_available"]
        assert r["shape"] == p["shape"] and r["config"]["iterations"] == p["iterations"]
        assert r["config"]["coupling"] == p["coupling"]
        assert (r["threads"], r["seed"], r["warmup_per_route"]) == (2, 239, 2)
        assert r["benchmark_sha256"] == plan["client_sha256"]
        manifest = v["runtime_" + p["phase"]]["files"]
        assert r["native_sha256"] == manifest["spiraltorch/spiraltorch.abi3.so"]
        assert r["bridge_sha256"] == manifest["spiraltorch/geometry_autograd.py"]
        assert r["transport_sha256"] == manifest["spiraltorch/_torch_transport.py"]
        assert set(r["measurements_ms"]) == set(ROUTES)
        for route in ROUTES:
            values = r["measurements_ms"][route]
            assert len(values) == 12 and min(values) > 0
            assert r["median_ms"][route] == statistics.median(values)
            for pos in range(4):
                assert sum(order[pos] == route for order in r["round_order"]) == 3
    assert len(s["rows"]) == 4
    assert {(r["iterations"], r["size"]) for r in s["rows"]} == {(i, n) for i in (4, 16) for n in ("small", "large")}
    for row in s["rows"]:
        selected = [p for p in plan["runs"] if (p["iterations"], p["size"]) == (row["iterations"], row["size"])]
        assert [p["phase"] for p in selected] == ["before", "after", "after", "before", "after", "before", "before", "after"]
        for pair in (1, 2, 3, 4):
            a, b = (runs[next(p["file"] for p in selected if p["pair"] == pair and p["phase"] == phase)] for phase in ("before", "after"))
            for key in ("shape", "config", "seed", "threads", "torch", "machine", "input_sha256", "upstream_sha256"):
                assert a[key] == b[key]
            for result in (a, b):
                for route in ROUTES[:-1]:
                    assert set(result["correctness"][route]) == {"output", "input_gradient", "gate_gradient"}
                    for field, value in result["correctness"][route].items():
                        assert value["sha256"] == a["correctness"]["rust_list"][field]["sha256"]
        for route in ROUTES:
            before = [runs[p["file"]]["median_ms"][route] for p in selected if p["phase"] == "before"]
            after = [runs[p["file"]]["median_ms"][route] for p in selected if p["phase"] == "after"]
            actual = row["routes"][route]
            assert actual["before_ms"] == statistics.median(before)
            assert actual["after_ms"] == statistics.median(after)
            assert actual["ratio"] == statistics.median(before) / statistics.median(after)
            assert actual["paired_ratios"] == [a / b for a, b in zip(before, after)]


def test_wasm_audit_learning_and_scope_receipts():
    v = read("verification.json.gz")
    a, b = v["wasm_before"], v["wasm_after"]
    assert a.keys() == b.keys()
    assert {k for k in a if a[k] != b[k]} == {"wasm_sha256"}
    assert len(a["cases"]) == 24
    assert all(len(c["captured_audit_sha256"]) == 64 for c in a["cases"])
    assert a["probe_sha256"] == v["client_sha256"]["audit_wasm_probe_mjs"]
    assert a["capture_required"] and a["captured"] and a["status"] == "passed"
    assert a["guard_checks"] == 50 and a["learning"]["updates"] == 240
    assert a["learning"]["legacy_trajectory_exact"] and a["learning"]["next_update_exact"]
    for phase in ("before", "after"):
        assert v["initial_wasm_" + phase]["wasm_sha256"] == v["wasm_" + phase]["wasm_sha256"]
    assert v["validation"]["strict_wasm_clippy"]["exit_code"] == 0
    assert not any(v["validation"][k] for k in ("pretrained_training", "heldout_rescoring", "cleanup"))
    assert len(v["initial_attempts"]) == 2


def test_owned_capture_wasm_aliasing_abi_and_scope_receipts():
    v = read("verification.json.gz", OWNED_DATA)
    a, b = v["wasm_before"], v["wasm_after"]
    assert a.keys() == b.keys()
    assert {k for k in a if a[k] != b[k]} == {"wasm_sha256", "wrapper_sha256"}
    assert len(a["cases"]) == 24
    assert all(len(c["captured_audit_sha256"]) == 64 for c in a["cases"])
    assert a["probe_sha256"] == v["client_sha256"]["ownership_wasm_probe_mjs"]
    assert a["capture_required"] and a["captured"] and a["status"] == "passed"
    assert a["guard_checks"] == 54 and a["learning"]["updates"] == 240
    assert a["learning"]["legacy_trajectory_exact"] and a["learning"]["next_update_exact"]
    assert [c["mode"] for c in a["ownership_cases"]] == ["alias", "overlap"]
    assert all(len(value) == 64 for c in a["ownership_cases"] for key, value in c.items() if key != "mode")
    for phase in ("before", "after"):
        for key in ("wasm_sha256", "wrapper_sha256"):
            assert v["initial_wasm_" + phase][key] == v["wasm_" + phase][key]
    abi = v["wasm_abi"]
    assert abi["typescript_line_multiset_identical"]
    assert set(abi["topos_classes"]) == {"ToposResonatorKernel", "ToposResonatorLearningBatch"}
    for result in abi["topos_classes"].values():
        assert result["wrapper_identical"] and result["typescript_identical"]
        assert len(result["wrapper_sha256"]) == len(result["typescript_sha256"]) == 64
    assert v["validation"]["strict_wasm_clippy"]["exit_code"] == 0
    assert v["validation"]["rust_core_full"] == 1022
    assert v["validation"]["python_geometry"] == 1038
    assert not any(v["validation"][k] for k in ("pretrained_training", "heldout_rescoring", "cleanup"))
    assert len(v["initial_attempts"]) == 1
