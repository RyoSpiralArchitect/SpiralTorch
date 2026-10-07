"""Published numeric integrity, not an independent GPU execution witness."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/results/2026-10-07-topos-resident-kernel"
CHECK_PATH = Path(__file__).with_name("check_topos_shared_gate_learning.py")
SPEC = importlib.util.spec_from_file_location("resident_topos_check", CHECK_PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_resident_results_hashes_and_complete_trajectory_contract():
    sums = (BUNDLE / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 4
    names = set()
    for entry in sums:
        digest, name = entry.split("  ", 1)
        assert name in {"README.md", "native-learning.json.gz", "torch-check.json", "verification.json"}
        assert name not in names
        names.add(name)
        assert sha((BUNDLE / name).read_bytes()) == digest
    raw = gzip.decompress((BUNDLE / "native-learning.json.gz").read_bytes())
    receipt = json.loads(raw)
    CHECK.validate_receipt(receipt)
    report = json.loads((BUNDLE / "torch-check.json").read_text())
    verification = json.loads((BUNDLE / "verification.json").read_text())
    assert sha(raw) == verification["native_raw_sha256"] == report["receipt_sha256"]
    assert sha((BUNDLE / "native-learning.json.gz").read_bytes()) == verification["native_gzip_sha256"]
    assert sha((BUNDLE / "torch-check.json").read_bytes()) == verification["torch_check_sha256"]
    assert report["source_backend"] == receipt["backend"] == "wgpu"
    assert report["updates"] == 200
    assert verification["learning"]["max_abs_error"] == report["max_abs_error"]
    for error in report["max_abs_error"].values():
        assert CHECK.finite_number(error) and 0 <= error < 2e-7
    assert report["rtol"] == 5e-4 and report["atol"] == 3e-5
    assert report["client_sha256"] == verification["source_sha256"]["tools/check_topos_shared_gate_learning.py"]
    assert verification["source_revision"] == "20686ce788052fef3767fd64549a32e23552d53b"
    assert verification["validation"]["strict_native_clippy"]["status"] == "failed_existing_unknown_lints"
    assert verification["reproducibility"]["identical_rerun_bytes"] is True
    assert verification["reproducibility"]["repeated_native_raw_sha256"] == sha(raw)


def test_recorded_resident_updates_follow_explicit_sgd_without_reset():
    # Check the saved transition independently of Torch and without accepting
    # receipt labels as device-execution proof. Allow f32 arithmetic rounding.
    receipt = json.loads(gzip.decompress((BUNDLE / "native-learning.json.gz").read_bytes()))
    rate = receipt["learning_rate"]
    for case in receipt["cases"]:
        previous = case["initial_gate"]
        for record in case["records"]:
            for before, gradient, after in zip(previous, record["grad_gate"], record["gate_after"]):
                assert abs((before - rate * gradient) - after) <= 2e-7 * max(1., abs(before))
            previous = record["gate_after"]
