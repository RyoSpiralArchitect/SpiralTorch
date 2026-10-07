"""Reconstruct public numeric outcomes, not private checkpoint execution."""

import builtins
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(params=["gain", "angle"])
def publication(request):
    return ROOT / f"benchmarks/results/2026-10-05-fractional-{request.param}-study"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(directory, name):
    raw = (directory / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


@pytest.mark.parametrize("source_kind", ["current", "archived"])
def test_public_summary_rebuilds_without_model_libraries(publication, source_kind, monkeypatch):
    source = ((ROOT / "tools") if source_kind == "current" else publication) / "summarize_wave_gate_long_horizon.py"
    original = builtins.__import__
    def no_model_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "spiraltorch"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_model_import)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("published_gain_summary", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def no_platform_atan(_):
        raise AssertionError("platform atan must not enter serialized angular receipts")
    monkeypatch.setattr(module.math, "atan", no_platform_atan)
    inputs = {key: (gzip.decompress((publication / f"{key}.json.gz").read_bytes())
                    if key != "journal" else (publication / "journal.json").read_bytes())
              for key in ("plan", "results", "journal")}
    hashes = {key: hashlib.sha256(raw).hexdigest() for key, raw in inputs.items()}
    report = module.summarize(*(json.loads(inputs[key]) for key in ("plan", "results", "journal")), hashes["results"])
    report["input_sha256"] = hashes
    assert (json.dumps(report, indent=2, allow_nan=False) + "\n").encode() == (publication / "summary.json").read_bytes()


def test_complete_publication_has_bound_source_outputs_and_all_three_contrasts(publication):
    manifest = {}
    for line in (publication / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name and name not in manifest
        assert not (publication / name).is_symlink() and sha(publication / name) == expected
        manifest[name] = expected
    entries = {p.name for p in publication.iterdir() if p.name != "SHA256SUMS"
               and not (p.name == "__pycache__" and p.is_dir() and not p.is_symlink())}
    assert set(manifest) == entries
    assert not any(p.suffix in {".pt", ".so", ".safetensors", ".log"} for p in publication.iterdir())
    validation, summary = read(publication, "validation.json"), read(publication, "summary.json")
    verification, plan, journal = (read(publication, name) for name in
                                   ("checkpoint-verification.json", "plan.json.gz", "journal.json"))
    angular = plan["config"]["schema"] == "spiraltorch.fractional_angle_protocol.v1"
    assert validation["status"] == ("completed_saved_states_verified" if angular else "completed_and_verified")
    assert validation["training_process_exit_code"] == validation["completed_resume_exit_code"] == 0
    assert validation["completed_resume_rescored_endpoints"] is False
    assert validation["sealed_files_unchanged_after_completed_resume"] is True
    assert len(journal["runs"]) == verification["planned_runs_verified"] == summary["runs"] == 9
    assert validation["primary_updates"] == verification["primary_updates"] == summary["primary_updates"] == 4608
    assert validation["continuation_only_updates"] == verification["continuation_only_updates"] == 18
    assert verification["status"] == "passed" and verification["summary_rebuilt_byte_identical"] is True
    assert verification["frozen_files_verified"] == {"client": 8 if angular else 7, "runtime": 71}
    assert verification["summary_source_sha256"] == sha(publication / "summarize_wave_gate_long_horizon.py")
    assert verification["summary_artifact_sha256"] == sha(publication / "summary.json")
    assert verification["verifier_sha256"] == sha(publication / "verify_fractional_gain_study.py")
    assert verification["verifier_helper_sha256"] == sha(publication / "verify_fractional_history_factorial.py")
    assert validation["checkpoint_verification_sha256"] == sha(publication / "checkpoint-verification.json")
    assert verification["artifacts"] == summary["input_sha256"]
    assert verification["training_source_revision"] == plan["source_revision"] == validation["training_source_revision"]
    assert verification["study_id"] == summary["study_id"] == journal["study_id"] == plan["study_id"] == validation["study_id"]
    assert verification["runtime_build_source_revision"] == read(publication, "frozen-runtime-sha256.json")["source_revision"]
    client_manifest = "verification-client-sha256.json" if angular else "client-sha256.json"
    for name, filename in (("client", client_manifest), ("runtime", "frozen-runtime-sha256.json")):
        assert verification["manifest_sha256"][name] == sha(publication / filename)
    assert len(summary["gain_trajectories"]) == 9
    assert len(summary["order_trajectories"]) == (0 if angular else 6)
    assert len(summary["angle_trajectories"]) == (9 if angular else 3)
    for contrasts in summary["paired_gain_contrasts"].values():
        assert set(contrasts) == {"gl_short_minus_ordinary_short", "gl_full_minus_ordinary_short", "gl_full_minus_gl_short"}
        assert all(len(row["per_seed"]) == 3 for row in contrasts.values())
    for key, row in verification["runs"].items():
        shape_name = "history_angle" if angular or "ordinary" in key else "log_alpha"
        assert set(row["parameters"]) == set(row["named_adam"]) == {"gate", "local_gate", "log_gain", shape_name}
        assert row["final_coordinates"]["gain"] == summary["gain_trajectories"][key]["final_gain"]
        assert row["final_coordinates"]["effective_history_gate_l2"] == summary["gain_trajectories"][key]["final_effective_history_gate_l2"]
    if angular:
        assert "terminal_failure" not in journal and len(summary["angular_order_trajectories"]) == 9
        assert set(verification["paired_short_states"]) == {str(s) for s in plan["config"]["seeds"]}
        passed = True
        for seed, pair in verification["paired_short_states"].items():
            assert set(pair) == {"parameters", "adam"}
            left, right = (verification["runs"][f"{seed}:{arm}"] for arm in plan["config"]["arms"][:2])
            for field, receipt in pair.items():
                assert field in {"parameters", "adam"}
                assert receipt["rtol"] == 3e-6 and receipt["atol"] == 3e-7
                assert receipt["status"] == ("passed" if receipt["allclose"] else "failed")
                source = "parameters" if field == "parameters" else "named_adam"
                assert receipt["byte_equal"] == (left[source] == right[source])
                assert all(math.isfinite(value) and value >= 0 for value in receipt["max_abs_error"].values())
                passed = passed and receipt["allclose"]
        assert verification["paired_short_state_status"] == ("passed" if passed else "failed")
        assert validation["paired_short_state_status"] == verification["paired_short_state_status"]
        assert validation["all_declared_equivalence_checks_passed"] == passed
        assert validation["paired_tolerance"] == {"rtol": 3e-6, "atol": 3e-7, "relaxed": False}
        for field, name in (("parameters", "paired_short_parameter_checks"),
                            ("adam", "paired_short_adam_checks")):
            statuses = [pair[field]["status"] for pair in verification["paired_short_states"].values()]
            assert validation[name] == {status: statuses.count(status) for status in ("passed", "failed")}


def test_portable_angular_summary_preserves_originals_and_changes_only_derived_lower_distances():
    directory = ROOT / "benchmarks/results/2026-10-05-fractional-angle-study"
    original = read(directory, "original-summary.json")
    corrected = read(directory, "summary.json")
    previous = read(directory, "original-checkpoint-verification.json")
    current = read(directory, "checkpoint-verification.json")
    assert previous["summary_artifact_sha256"] == sha(directory / "original-summary.json")
    assert previous["summary_source_sha256"] == sha(directory / "original-summarize_wave_gate_long_horizon.py")
    assert previous["manifest_sha256"]["client"] == sha(directory / "client-sha256.json")
    assert previous["artifacts"] == current["artifacts"] == original["input_sha256"] == corrected["input_sha256"]
    assert previous["runs"] == current["runs"]
    assert previous["paired_short_states"] == current["paired_short_states"]
    old_client = read(directory, "client-sha256.json")
    new_client = read(directory, "verification-client-sha256.json")
    assert set(old_client) == set(new_client)
    assert {name for name in old_client if old_client[name] != new_client[name]} == {"summarize_wave_gate_long_horizon.py"}
    assert new_client["summarize_wave_gate_long_horizon.py"] == current["summary_source_sha256"]
    assert old_client["summarize_wave_gate_long_horizon.py"] == previous["summary_source_sha256"]
    key = "min_angle_distance_to_lower_boundary"
    assert len(original["angular_order_trajectories"]) == len(corrected["angular_order_trajectories"]) == 9
    for run, row in original["angular_order_trajectories"].items():
        old, new = row.pop(key), corrected["angular_order_trajectories"][run].pop(key)
        assert abs(old - new) <= math.ulp(0.4636476090008061)
    assert original == corrected


def test_angular_control_reproduction_matches_every_prior_endpoint_block():
    current_dir = ROOT / "benchmarks/results/2026-10-05-fractional-angle-study"
    prior_dir = ROOT / "benchmarks/results/2026-10-05-fractional-gain-study"
    current_raw, prior_raw = (gzip.decompress((directory / "results.json.gz").read_bytes())
                              for directory in (current_dir, prior_dir))
    current, prior = json.loads(current_raw), json.loads(prior_raw)
    report = read(current_dir, "control-reproduction.json")
    assert report["schema"] == "spiraltorch.angular_control_endpoint_reproduction.v1"
    assert report["current_results_sha256"] == hashlib.sha256(current_raw).hexdigest()
    assert report["prior_results_sha256"] == hashlib.sha256(prior_raw).hexdigest()
    assert report["baseline_equal"] == (current["baseline"] == prior["baseline"])
    current_runs, prior_runs = ({row["run_key"]: row for row in result["runs"]}
                               for result in (current, prior))
    plan = read(current_dir, "plan.json.gz")
    prior_plan = read(prior_dir, "plan.json.gz")
    assert set(report["runs"]) == {str(seed) for seed in plan["config"]["seeds"]}
    for seed, endpoints in report["runs"].items():
        left = current_runs[f"{seed}:{plan['config']['reference_arm']}"]["scores"]
        right = prior_runs[f"{seed}:{prior_plan['config']['reference_arm']}"]["scores"]
        assert set(endpoints) == set(left) == set(right)
        for name, receipt in endpoints.items():
            assert receipt == {
                "mean_equal": left[name]["mean"] == right[name]["mean"],
                "block_losses_equal": left[name]["block_losses"] == right[name]["block_losses"],
            }
