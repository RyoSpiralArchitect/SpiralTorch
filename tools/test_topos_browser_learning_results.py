"""Check frozen browser/Node Topos receipts without replaying learning."""

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-07-topos-browser-captured-learning"


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def test_closed_archive_and_host_identity():
    entries = dict(line.split("  ", 1)[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    assert set(entries) == {p.name for p in DATA.iterdir() if p.name != "SHA256SUMS"}
    for name, expected in entries.items():
        raw = (DATA / name).read_bytes()
        assert digest(raw) == expected
        assert not any(marker in raw for marker in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))
    r = json.loads((DATA / "results.json").read_text())
    fixture = json.loads(r["raw_fixture_json"])
    assert fixture["native_sha256"] == r["native_sha256"]
    assert r["browser"]["host"] == "Codex In-app Browser"
    assert r["browser"]["viewport"] == [1280, 720]
    assert r["browser"]["status"] == "passed" and r["browser"]["console"] == []
    raw_browser = r["browser"]["raw_report_json"]
    assert digest(raw_browser) == r["browser"]["raw_report_sha256"]
    assert json.loads(raw_browser) == r["browser"]["result"]
    for key in ("node_web", "node_commonjs"):
        node = r[key]
        assert node["result"] == r["browser"]["result"]
        assert node["fixture_sha256"] == digest(r["raw_fixture_json"])
        assert node["shared_contract_sha256"] == r["source_files_sha256"]["bindings/st-wasm/tests/topos_resonator_learning.mjs"]
        assert node["probe_sha256"] == r["source_files_sha256"]["tools/probe_topos_browser_learning.mjs"]
    assert r["node_web"]["wasm_sha256"] == r["served_assets_sha256"]["module/spiraltorch_wasm_bg.wasm"]
    assert r["node_web"]["wrapper_sha256"] == r["served_assets_sha256"]["module/spiraltorch_wasm.js"]
    assert any(name.startswith("module/snippets/") for name in r["served_assets_sha256"])


def test_learning_failures_and_scope_are_preserved():
    r = json.loads((DATA / "results.json").read_text())
    result = r["browser"]["result"]
    assert result["execution_backend"] == "rust_f32_wasm"
    assert result["learning_updates"] == 100 and len(result["learning_losses"]) == 101
    assert result["learning_losses"][-1] < result["learning_losses"][0] * 0.1
    assert all(result[key] for key in ("capture_matches_stateless_exactly", "learning_trajectory_exact",
                                     "saved_gate_next_update_exact", "snapshot_independent_after_kernel_free"))
    assert all(result[key] == 0 for key in ("forward_max_error", "input_vjp_max_error", "gate_vjp_max_error"))
    assert result["guard_checks"] == 70
    assert r["negative_checks"]["wrong_native_reference"]["exit_code"] == 1
    assert r["negative_checks"]["wrong_native_reference"]["status"] == "error"
    assert r["negative_checks"]["missing_http_fixture"]["http_status"] == 404
    assert r["negative_checks"]["missing_http_fixture"]["status"] == "error"
    assert r["browser"]["reload_after_fixture_restoration"] == "passed"
    assert len(r["initial_observations"]) == 2
    assert not any(r[key] for key in ("pretrained_training", "heldout_rescoring", "cleanup"))


def test_reference_guard_followup_preserves_original_evidence():
    original = json.loads((DATA / "results.json").read_text())
    raw = (DATA.parent / "2026-10-07-topos-browser-reference-guards.json").read_bytes()
    assert not any(marker in raw for marker in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))
    followup = json.loads(raw)
    assert followup["before"] == {"exit_code": 0, "status": "passed", "serialized_input_vjp_max_error": None}
    assert followup["after"]["exit_code"] == 1 and followup["after"]["status"] == "error"
    assert followup["after"]["browser_status"] == "error"
    assert followup["rejected_references"]["web"] == followup["rejected_references"]["commonjs"] == 27
    assert followup["browser_result_bytes_unchanged"] and followup["node_valid_result_fields_unchanged"]
    assert followup["browser_recovery_status"] == "passed"
    assert followup["frozen_browser_result_sha256"] == original["browser"]["raw_report_sha256"]
