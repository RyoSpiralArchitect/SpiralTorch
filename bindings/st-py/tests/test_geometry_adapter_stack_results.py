"""Validate public receipts, not private tensor execution or language quality."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "benchmarks/results/2026-10-07-geometry-adapter-stack"


def read(name):
    return json.loads((DATA / name).read_bytes())


def test_closed_numeric_archive_and_hashes():
    hashes = dict(line.split("  ", 1)[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    assert set(hashes) == {p.name for p in DATA.iterdir() if p.name != "SHA256SUMS"}
    assert len(hashes) == 4
    for name, digest in hashes.items():
        raw = (DATA / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        assert not any(value in raw for value in (b"/Users/", b"/home/", b"sk-proj-", b"-----BEGIN PRIVATE"))


def test_real_mixed_updates_and_continuation_are_complete():
    report = read("report.json")
    assert report["status"] == "bitwise_exact"
    assert report["reference_records"] == report["public_records"]
    assert report["resumed_records"] == report["reference_records"][2:]
    assert sum(len(report[name]) for name in ("reference_records", "public_records", "resumed_records")) == 7
    assert report["auxiliary_updates"] == 7 and report["unique_trajectory_updates"] == 3
    assert report["parameter_count"] == 12294 and report["backends"] == ["rust_f32_cpu"] * 4
    assert [r["adapter"] for r in report["config"]["placements"]] == [
        "WaveGateAdapter", "ToposResonatorAdapter", "EllipticAnchoredResidualAdapter",
        "FractionalAngleGainHistoryAdapter"]
    gradients = report["public_records"][2]["gradients"]
    assert len(gradients) == 12 and all(row["nonzero"] > 0 for row in gradients.values())
    assert report["base_unchanged"] and not report["heldout_scoring"] and not report["timing_evidence"]


def test_runtime_receipts_and_initial_setup_failure_are_preserved():
    report, validation, manifest = read("report.json"), read("validation.json"), read("runtime-sha256.json")
    assert len(manifest["files"]) == 72
    assert hashlib.sha256((DATA / "runtime-sha256.json").read_bytes()).hexdigest() == report["runtime_manifest_sha256"]
    assert manifest["files"]["spiraltorch/geometry_adapters.py"] == report["source_sha256"]["geometry_adapters.py"]
    assert manifest["files"]["spiraltorch/spiraltorch.abi3.so"] == report["native_sha256"]
    assert validation["native_reused_from_source"] == "381a8a10bc7cb6744696fa7e561b69066e9bd584"
    assert validation["source_revision"] == manifest["source_revision"] == "daf5d8068e90572f6b1c526f42a6824884fa6b30"
    assert validation["failure_preserved"]["before_training"]
    assert validation["private_states_and_logs_retained"] and validation["no_cleanup_performed"]
    assert validation["client_sha256"]["hf_geometry_adapter_stack.py"] == report["example_sha256"]
