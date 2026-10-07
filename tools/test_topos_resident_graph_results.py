"""Frozen numeric integrity; neither a GPU witness nor a quality benchmark."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct


BUNDLE = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-07-topos-resident-graph"
SOURCE = "d4c61e03a664f0dc15fc9c02eebca55374db8008"
FILES = {"README.md", "browser.json.gz", "python-check.json", "verification.json"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_public_hashes_source_and_comparison_summary():
    entries = (BUNDLE / "SHA256SUMS").read_text().splitlines()
    assert len(entries) == len(FILES)
    seen = set()
    for entry in entries:
        digest, name = entry.split("  ", 1)
        assert name in FILES and name not in seen
        seen.add(name)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert sha((BUNDLE / name).read_bytes()) == digest
    assert seen == FILES
    verification = json.loads((BUNDLE / "verification.json").read_text())
    report = json.loads((BUNDLE / "python-check.json").read_text())
    raw = gzip.decompress((BUNDLE / "browser.json.gz").read_bytes())
    browser = json.loads(raw)
    assert verification["source_revision"] == report["source_revision"] == SOURCE
    assert verification["browser_raw_sha256"] == sha(raw)
    assert verification["browser_gzip_sha256"] == sha((BUNDLE / "browser.json.gz").read_bytes())
    assert verification["python_check_sha256"] == sha((BUNDLE / "python-check.json").read_bytes())
    assert report["updates"] == 100 and report["rtol"] == 5e-4 and report["atol"] == 3e-5
    assert set(report["max_abs_error"]) == {"output", "dx", "dg", "gate", "loss"}
    for error in report["max_abs_error"].values():
        assert finite(error) and 0 <= error < 2e-7
    assert report["runtime"]["native_sha256"] == verification["native_sha256"]
    assert report["runtime"]["adapter"]["backend"] == "Metal"
    assert report["runtime"]["adapter"]["device_type"] != "Cpu"
    for path, digest in report["source_sha256"].items():
        assert verification["source_sha256"][path] == digest
    assert len(report["source_sha256"]) == 2
    assert verification["wasm_sha256"] == browser["wasm_sha256"]
    assert verification["source_sha256"]["bindings/st-wasm/tests/resident_topos_graph.html"] == browser["page_sha256"]
    for collection in (verification["source_sha256"], verification["local_log_sha256"], browser["asset_sha256"]):
        assert collection
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in collection.values())
    assert verification["validation"]["nn_default"]["passed"] == 781
    assert verification["validation"]["nn_resident"]["passed"] == 53
    assert verification["validation"]["nn_resident"]["updates"] == 300
    assert verification["validation"]["python_regression"] == {"passed": 45, "subtests_passed": 32}
    assert verification["strict_nn_clippy"]["status"] == "failed_unchanged_files"
    assert verification["strict_nn_clippy"]["diagnostics"] == 39
    assert len(verification["strict_nn_clippy"]["unchanged_files"]) == 22
    assert verification["review"]["reproduced_findings"] == 2
    assert verification["review"]["followup_actionable_findings"] == 0


def test_complete_finite_trajectories_follow_explicit_sgd_and_mean_mse():
    browser = json.loads(gzip.decompress((BUNDLE / "browser.json.gz").read_bytes()))
    assert browser["schema"] == "spiraltorch.topos_resident_graph_browser.v1"
    assert browser["status"] == "passed" and browser["checks"] == 4119
    assert browser["page_errors"] == browser["console_messages"] == []
    assert browser["adapter_probe"]["is_fallback_adapter"] is False
    assert browser["adapter"]["backend"] == "BrowserWebGpu"
    assert browser["adapter"]["device_type"] != "Cpu"
    assert browser["fixture_request"] == "topos-resident-graph"
    assert set(browser["guards"]) == {
        "residual_drive_overflow", "update_atomicity", "valid_retry",
        "version_downgrade", "admission", "owning_handoff",
    }
    assert all(value is True for value in browser["guards"].values())
    assert len(browser["cases"]) == 4
    assert {(case["porosity"], case["policy"]) for case in browser["cases"]} == {
        (porosity, policy) for porosity in (0., .3) for policy in ("exact", "module_compatible")
    }
    for case in browser["cases"]:
        assert case["shape"] == [2, 2, 3] and len(case["trajectory"]) == 25
        previous = [f32(value) for value in (.8, -.4, 1.1)]
        for step, record in enumerate(case["trajectory"]):
            assert record["step"] == step
            for name, length in (("input", 12), ("target", 12), ("output", 12), ("dx", 12), ("dg", 3), ("gate", 3)):
                assert len(record[name]) == length and all(finite(value) for value in record[name])
            assert finite(record["loss"]) and record["loss"] >= 0
            for i, value in enumerate(record["input"]):
                assert value == f32(((i * 17 + step * 7) % 31) / 7. - 2.)
                assert record["target"][i] == f32(value * f32(.13))
            expected_loss = sum((value - target) ** 2 for value, target in zip(record["output"], record["target"])) / 12
            assert abs(expected_loss - record["loss"]) <= 2e-7 * max(1., abs(expected_loss))
            # Independent saved-state check, allowing normal f32 rounding.
            for before, gradient, after in zip(previous, record["dg"], record["gate"]):
                assert abs(before - .03 * gradient - after) <= 2e-7 * max(1., abs(before))
            previous = record["gate"]
