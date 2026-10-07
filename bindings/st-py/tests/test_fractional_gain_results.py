"""Reconstruct public numeric outcomes, not private checkpoint execution."""

import builtins
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "benchmarks/results/2026-10-05-fractional-gain-study"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    raw = (RESULT / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


@pytest.mark.parametrize("source", [ROOT / "tools/summarize_wave_gate_long_horizon.py",
                                    RESULT / "summarize_wave_gate_long_horizon.py"])
def test_public_summary_rebuilds_without_model_libraries(source, monkeypatch):
    original = builtins.__import__
    def no_model_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "spiraltorch"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_model_import)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("published_gain_summary", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = {key: (gzip.decompress((RESULT / f"{key}.json.gz").read_bytes())
                    if key != "journal" else (RESULT / "journal.json").read_bytes())
              for key in ("plan", "results", "journal")}
    hashes = {key: hashlib.sha256(raw).hexdigest() for key, raw in inputs.items()}
    report = module.summarize(*(json.loads(inputs[key]) for key in ("plan", "results", "journal")), hashes["results"])
    report["input_sha256"] = hashes
    assert (json.dumps(report, indent=2, allow_nan=False) + "\n").encode() == (RESULT / "summary.json").read_bytes()


def test_complete_publication_has_bound_source_outputs_and_all_three_contrasts():
    manifest = {}
    for line in (RESULT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name and name not in manifest
        assert not (RESULT / name).is_symlink() and sha(RESULT / name) == expected
        manifest[name] = expected
    entries = {p.name for p in RESULT.iterdir() if p.name != "SHA256SUMS"
               and not (p.name == "__pycache__" and p.is_dir() and not p.is_symlink())}
    assert set(manifest) == entries
    assert not any(p.suffix in {".pt", ".so", ".safetensors", ".log"} for p in RESULT.iterdir())
    validation, summary = read("validation.json"), read("summary.json")
    verification, plan, journal = read("checkpoint-verification.json"), read("plan.json.gz"), read("journal.json")
    assert validation["status"] == "completed_and_verified"
    assert validation["training_process_exit_code"] == validation["completed_resume_exit_code"] == 0
    assert validation["completed_resume_rescored_endpoints"] is False
    assert validation["sealed_files_unchanged_after_completed_resume"] is True
    assert len(journal["runs"]) == verification["planned_runs_verified"] == summary["runs"] == 9
    assert validation["primary_updates"] == verification["primary_updates"] == summary["primary_updates"] == 4608
    assert validation["continuation_only_updates"] == verification["continuation_only_updates"] == 18
    assert verification["status"] == "passed" and verification["summary_rebuilt_byte_identical"] is True
    assert verification["frozen_files_verified"] == {"client": 7, "runtime": 71}
    assert verification["summary_source_sha256"] == sha(RESULT / "summarize_wave_gate_long_horizon.py")
    assert verification["summary_artifact_sha256"] == sha(RESULT / "summary.json")
    assert verification["verifier_sha256"] == sha(RESULT / "verify_fractional_gain_study.py")
    assert verification["verifier_helper_sha256"] == sha(RESULT / "verify_fractional_history_factorial.py")
    assert validation["checkpoint_verification_sha256"] == sha(RESULT / "checkpoint-verification.json")
    assert verification["artifacts"] == summary["input_sha256"]
    assert verification["training_source_revision"] == plan["source_revision"] == validation["training_source_revision"]
    assert verification["study_id"] == summary["study_id"] == journal["study_id"] == plan["study_id"] == validation["study_id"]
    assert verification["runtime_build_source_revision"] == read("frozen-runtime-sha256.json")["source_revision"]
    for name, filename in (("client", "client-sha256.json"), ("runtime", "frozen-runtime-sha256.json")):
        assert verification["manifest_sha256"][name] == sha(RESULT / filename)
    assert len(summary["gain_trajectories"]) == 9
    assert len(summary["order_trajectories"]) == 6 and len(summary["angle_trajectories"]) == 3
    for contrasts in summary["paired_gain_contrasts"].values():
        assert set(contrasts) == {"gl_short_minus_ordinary_short", "gl_full_minus_ordinary_short", "gl_full_minus_gl_short"}
        assert all(len(row["per_seed"]) == 3 for row in contrasts.values())
    for key, row in verification["runs"].items():
        shape_name = "history_angle" if "ordinary" in key else "log_alpha"
        assert set(row["parameters"]) == set(row["named_adam"]) == {"gate", "local_gate", "log_gain", shape_name}
        assert row["final_coordinates"]["gain"] == summary["gain_trajectories"][key]["final_gain"]
        assert row["final_coordinates"]["effective_history_gate_l2"] == summary["gain_trajectories"][key]["final_effective_history_gate_l2"]
