"""Check published numeric evidence, without treating it as private-state replay."""

import builtins
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "benchmarks/results/2026-10-05-fractional-history-factorial"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    raw = (RESULT / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


@pytest.mark.parametrize("source", [ROOT / "tools/summarize_wave_gate_long_horizon.py",
                                    RESULT / "summarize_wave_gate_long_horizon.py"])
def test_public_summary_rebuilds_without_torch(source, monkeypatch):
    original = builtins.__import__

    def no_model_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "spiraltorch"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_import)
    spec = importlib.util.spec_from_file_location("published_history_factorial_summary", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = {key: (gzip.decompress((RESULT / f"{key}.json.gz").read_bytes())
                    if key != "journal" else (RESULT / "journal.json").read_bytes())
              for key in ("plan", "results", "journal")}
    hashes = {key: hashlib.sha256(raw).hexdigest() for key, raw in inputs.items()}
    report = module.summarize(*(json.loads(inputs[key]) for key in ("plan", "results", "journal")), hashes["results"])
    report["input_sha256"] = hashes
    assert (json.dumps(report, indent=2, allow_nan=False) + "\n").encode() == (RESULT / "summary.json").read_bytes()
    assert "same_math_state_parity" not in report


def test_complete_publication_has_consistent_hashes_and_separate_replay_criterion():
    manifest = {}
    for line in (RESULT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name and name not in manifest
        assert not (RESULT / name).is_symlink() and sha(RESULT / name) == expected
        manifest[name] = expected
    assert set(manifest) == {p.name for p in RESULT.iterdir() if p.name != "SHA256SUMS"}
    assert not any(p.suffix in {".pt", ".so", ".safetensors", ".log"} for p in RESULT.iterdir())
    validation, summary = read("validation.json"), read("summary.json")
    verification, plan, journal = read("checkpoint-verification.json"), read("plan.json.gz"), read("journal.json")
    assert validation["status"] == "completed_and_verified"
    assert validation["training_process_exit_code"] == validation["completed_resume_exit_code"] == 0
    assert validation["completed_resume_rescored_endpoints"] is False
    assert validation["sealed_files_unchanged_after_completed_resume"] is True
    assert len(journal["runs"]) == verification["planned_runs_verified"] == summary["runs"] == 12
    assert validation["primary_updates"] == verification["primary_updates"] == summary["primary_updates"] == 6144
    assert validation["continuation_only_updates"] == verification["continuation_only_updates"] == 24
    assert verification["status"] == "passed" and verification["summary_rebuilt_byte_identical"] is True
    assert verification["frozen_files_verified"] == {"client": 5, "runtime": 71}
    assert verification["summary_source_sha256"] == sha(RESULT / "summarize_wave_gate_long_horizon.py")
    assert verification["summary_artifact_sha256"] == sha(RESULT / "summary.json")
    assert verification["verifier_sha256"] == sha(RESULT / "verify_fractional_history_factorial.py")
    assert validation["checkpoint_verification_sha256"] == sha(RESULT / "checkpoint-verification.json")
    assert verification["artifacts"] == summary["input_sha256"]
    assert verification["training_source_revision"] == plan["source_revision"] == validation["training_source_revision"]
    assert verification["study_id"] == summary["study_id"] == journal["study_id"] == plan["study_id"] == validation["study_id"]
    replay = verification["historical_raw_full"]
    assert replay["counts_as_additional_independent_seeds"] is False
    assert replay["status"] == ("exact_replay" if all(all(r["parity"].values()) for r in replay["runs"].values()) else "mismatch")
    assert set(replay["runs"]) == {"41", "43", "47"}
    for label, contrasts in summary["paired_factorial_contrasts"].items():
        assert len(contrasts) == 5
        assert all(len(row["per_seed"]) == 3 for row in contrasts.values())
    for key, row in verification["runs"].items():
        description = row["filter_description"]
        assert description["backend"] == "rust_f32_cpu"
        assert 0 <= description["lag_3_plus_energy_fraction"] <= 1
        if ":history_l2_" in key:
            assert description["coefficient_l2"] == pytest.approx(5**.5, rel=2e-7)
        if key.endswith("_short"):
            assert description["lag_3_plus_energy_fraction"] == 0
