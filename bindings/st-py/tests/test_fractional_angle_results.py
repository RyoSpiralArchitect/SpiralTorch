"""Check public receipt consistency, not independent private tensor replay."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "benchmarks/results/2026-10-05-fractional-angle-chart"


def read(name):
    return json.loads((RESULT / name).read_bytes())


def test_angle_preflight_publication_hashes_and_all_tensor_receipts():
    manifest = {}
    for line in (RESULT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name and name not in manifest
        assert not (RESULT / name).is_symlink()
        assert hashlib.sha256((RESULT / name).read_bytes()).hexdigest() == expected
        manifest[name] = expected
    assert set(manifest) == {p.name for p in RESULT.iterdir() if p.name != "SHA256SUMS"}
    validation, preflight = read("validation.json"), read("preflight.json")
    assert validation["status"] == preflight["status"] == "passed"
    assert validation["preflight_process_exit_code"] == 0
    assert validation["preflight_sha256"] == manifest["preflight.json"]
    assert validation["wasm_learning_sha256"] == manifest["wasm-learning.json"]
    assert preflight["auxiliary_training_updates"] == 48 and preflight["continuation_only_updates"] == 12
    assert preflight["native_sha256"] == validation["native_sha256"]
    assert preflight["preflight_source_sha256"] == validation["source_sha256"]["tools/preflight_fractional_angle_chart.py"]
    assert preflight["frozen_base_unchanged"] is True and preflight["heldout_losses_computed"] is False
    assert preflight["actual_shape"] == [2, 128, 768] and preflight["steps_per_arm"] == 8
    assert set(preflight["runs"]) == {"41", "43", "47"}
    peaks = {"gradients": 0., "parameters": 0., "adam": 0., "loss": 0.}
    for run in preflight["runs"].values():
        assert run["exact_within_arm_continuation"] is True
        assert len(set(run["initial_parameter_sha256"].values())) == 1
        assert set(run["records"]) == {"ordinary_short", "native_angle_short"}
        assert all(len(records) == 8 for records in run["records"].values())
        assert [row["step"] for row in run["comparisons"]] == list(range(1, 9))
        for row in run["comparisons"]:
            for key in ("gradients", "parameters", "adam"):
                assert row[key]["close"] is True and row[key]["max_abs_error"]
                for value in row[key]["max_abs_error"].values():
                    assert 0 <= value <= preflight["atol"]
                    peaks[key] = max(peaks[key], value)
            peaks["loss"] = max(peaks["loss"], row["loss_abs_difference"])
    assert peaks == validation["preflight"]["max_abs_error"]
    wasm = read("wasm-learning.json")
    assert wasm["updates"] == 1000 and wasm["final_loss"] < .01 * wasm["initial_loss"]
    assert wasm["nonzero_angle_steps"] == wasm["nonzero_gain_steps"] == 1000
    assert validation["preservation"]["total_files"] == 577
