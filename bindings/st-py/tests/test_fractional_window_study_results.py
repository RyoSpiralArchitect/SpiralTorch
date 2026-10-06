"""Public paired-study receipts, not independent private model execution."""

import builtins
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "benchmarks/results/2026-10-07-fractional-window-study"
ORIGINAL = ROOT / "benchmarks/results/2026-10-05-fractional-angle-study"


def raw(name, directory=DIRECTORY):
    data = (directory / name).read_bytes()
    return gzip.decompress(data) if name.endswith(".gz") else data


def read(name, directory=DIRECTORY):
    return json.loads(raw(name, directory))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("archived", [False, True])
def test_current_and_archived_summary_rebuild_identically_without_model_imports(monkeypatch, archived):
    original_import = builtins.__import__
    def no_models(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "spiraltorch"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_models)
    path = (DIRECTORY if archived else ROOT / "tools") / "summarize_wave_gate_long_horizon.py"
    spec = importlib.util.spec_from_file_location("public_window_summary", path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    sources = {key: raw(f"{key}.json" + ("" if key == "journal" else ".gz"))
               for key in ("plan", "results", "journal")}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in sources.items()}
    observed = summary.summarize(*(json.loads(sources[key]) for key in ("plan", "results", "journal")), hashes["results"])
    observed["input_sha256"] = hashes
    assert (json.dumps(observed, indent=2, allow_nan=False) + "\n").encode() == raw("summary.json")


def test_complete_recipe_state_and_runtime_are_bound_without_claiming_cross_window_parity():
    inventory = {}
    for line in (DIRECTORY / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name and name not in inventory
        assert not (DIRECTORY / name).is_symlink() and sha(DIRECTORY / name) == expected
        inventory[name] = expected
    assert set(inventory) == {p.name for p in DIRECTORY.iterdir() if p.name != "SHA256SUMS"}
    assert not any(p.suffix in {".pt", ".so", ".safetensors", ".log"} for p in DIRECTORY.iterdir())
    plan, result, journal = read("plan.json.gz"), read("results.json.gz"), read("journal.json")
    summary, verification, validation = (read(name) for name in
        ("summary.json", "checkpoint-verification.json", "validation.json"))
    config = plan["config"]
    assert config["schema"] == "spiraltorch.fractional_window_protocol.v1"
    assert config["normalization_policy"] == summary["normalization_policy"] == "full_declared_kernel_before_window"
    assert config["lag_windows"] == {"history_window_short": [1, 3], "history_window_full": None}
    assert config["kernel"]["kernel_len"] == 32
    assert result["status"] == journal["status"] == "completed"
    assert verification["status"] == "passed" and verification["summary_rebuilt_byte_identical"]
    assert validation["training_process_exit_code"] == validation["completed_resume_exit_code"] == 0
    assert validation["checkpoint_verification_exit_code"] == 0
    assert validation["completed_resume_rescored_endpoints"] is False
    assert validation["sealed_files_unchanged_after_completed_resume"] is True
    assert len(result["runs"]) == len(journal["runs"]) == verification["planned_runs_verified"] == 6
    assert summary["primary_updates"] == verification["primary_updates"] == 3072
    assert verification["continuation_only_updates"] == 12
    assert "paired_short_states" not in verification
    assert verification["frozen_files_verified"] == {"client": 9, "runtime": 71}
    assert verification["summary_artifact_sha256"] == sha(DIRECTORY / "summary.json")
    assert verification["summary_source_sha256"] == sha(DIRECTORY / "summarize_wave_gate_long_horizon.py")
    assert verification["verifier_sha256"] == sha(DIRECTORY / "verify_fractional_gain_study.py")
    assert verification["verifier_helper_sha256"] == sha(DIRECTORY / "verify_fractional_history_factorial.py")
    assert verification["manifest_sha256"] == {
        "client": sha(DIRECTORY / "client-sha256.json"), "runtime": sha(DIRECTORY / "frozen-runtime-sha256.json")}
    assert validation["checkpoint_verification_sha256"] == sha(DIRECTORY / "checkpoint-verification.json")
    assert set(summary["angle_trajectories"]) == set(summary["angular_order_trajectories"])
    assert len(summary["gain_trajectories"]) == len(summary["angle_trajectories"]) == 6
    for name, values in summary["paired_window_contrasts"].items():
        assert set(values) == {"full_minus_retained_short"}
        assert len(values["full_minus_retained_short"]["per_seed"]) == 3


def test_prior_full_replay_is_reported_separately_and_does_not_erase_old_failed_parity():
    reproduction = read("full-arm-reproduction.json")
    current, previous = read("results.json.gz"), read("results.json.gz", ORIGINAL)
    new_states, old_states = read("checkpoint-verification.json"), read("checkpoint-verification.json", ORIGINAL)
    current_rows, previous_rows = ({r["run_key"]: r for r in report["runs"]} for report in (current, previous))
    assert reproduction["current_results_sha256"] == hashlib.sha256(raw("results.json.gz")).hexdigest()
    assert reproduction["previous_results_sha256"] == hashlib.sha256(raw("results.json.gz", ORIGINAL)).hexdigest()
    assert reproduction["current_verification_sha256"] == sha(DIRECTORY / "checkpoint-verification.json")
    assert reproduction["previous_verification_sha256"] == sha(ORIGINAL / "checkpoint-verification.json")
    assert reproduction["baseline_equal"] == (current["baseline"] == previous["baseline"])
    assert reproduction["counts_as_additional_independent_seeds"] is False
    assert reproduction["old_short_parameter_parity_status"] == old_states["paired_short_state_status"] == "failed"
    assert set(reproduction["runs"]) == {"41", "43", "47"}
    flags = [reproduction["baseline_equal"]]
    for seed, row in reproduction["runs"].items():
        key, old_key = f"{seed}:history_window_full", f"{seed}:history_angle_full"
        left, right = current_rows[key], previous_rows[old_key]
        a, b = new_states["runs"][key], old_states["runs"][old_key]
        expected = {"records_equal": left["records"] == right["records"],
                    "development_equal": left["development"] == right["development"],
                    "endpoint_scores_equal": left["scores"] == right["scores"],
                    "initial_parameters_equal": left["initial_parameter_sha256"] == right["initial_parameter_sha256"],
                    "saved_parameter_receipts_equal": a["parameters"] == b["parameters"],
                    "saved_named_adam_receipts_equal": a["named_adam"] == b["named_adam"]}
        assert row["parity"] == expected
        flags.extend(expected.values())
    assert reproduction["status"] == ("exact_replay" if all(flags) else "mismatch")
