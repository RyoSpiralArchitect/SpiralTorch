"""Replay public numeric evidence without loading private weights or importing Torch."""

import builtins
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "benchmarks/results/2026-10-05-fractional-two-lag-study"


def read(name):
    return json.loads((RESULT / name).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_receipt_summary_is_reproducible_but_not_private_state_verification(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def no_model_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "spiraltorch"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_import)
    source = ROOT / "tools/summarize_wave_gate_long_horizon.py"
    spec = importlib.util.spec_from_file_location("two_lag_public_summary", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [str(source), "--plan", str(RESULT / "plan.json.gz"),
                                    "--results", str(RESULT / "results.json.gz"),
                                    "--journal", str(RESULT / "journal.json"),
                                    "--output", str(output)])
    module.main()
    assert output.read_bytes() == (RESULT / "summary-receipts.json").read_bytes()
    receipt_only, locally_verified = json.loads(output.read_bytes()), read("summary.json")
    assert receipt_only["same_math_parity_status"] == "unverified"
    assert locally_verified["same_math_parity_status"] == "passed"
    for field in ("comparisons", "paired_fractional_contrasts", "order_trajectories",
                  "same_math_receipt_parity", "input_sha256"):
        assert receipt_only[field] == locally_verified[field]
    # A published receipt cannot substitute for actually reading private checkpoints.
    assert all(row["status"] == "unverified" and row["saved_gates_equal"] is None
               and row["saved_named_adam_equal"] is None
               for row in receipt_only["same_math_state_parity"].values())


def test_publication_binds_every_numeric_file_and_completion_claim():
    entries = {}
    for line in (RESULT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert Path(name).name == name and name not in entries
        assert not (RESULT / name).is_symlink()
        assert sha(RESULT / name) == digest
        entries[name] = digest
    assert set(entries) == {p.name for p in RESULT.iterdir() if p.name != "SHA256SUMS"}
    assert not any(p.suffix in {".pt", ".so", ".safetensors", ".log"} for p in RESULT.iterdir())
    plan_raw = gzip.decompress((RESULT / "plan.json.gz").read_bytes())
    result_raw = gzip.decompress((RESULT / "results.json.gz").read_bytes())
    plan, result = json.loads(plan_raw), json.loads(result_raw)
    journal, summary = read("journal.json"), read("summary.json")
    validation, verification = read("validation.json"), read("checkpoint-verification.json")
    assert result["status"] == journal["status"] == "completed"
    assert len(result["runs"]) == len(journal["runs"]) == summary["runs"] == 12
    assert validation["status"] == "completed_and_verified"
    assert validation["training_process_exit_code"] == validation["completed_resume_exit_code"] == 0
    assert validation["completed_resume_rescored_endpoints"] is False
    assert validation["sealed_files_unchanged_after_completed_resume"] is True
    assert validation["primary_updates"] == verification["primary_updates"] == summary["primary_updates"] == 6144
    assert validation["continuation_only_updates"] == verification["extra_continuation_updates"] == 24
    assert verification["status"] == "passed" and verification["planned_runs_verified"] == 12
    assert verification["frozen_client_files_verified"] == 6
    assert verification["frozen_package_files_verified"] == 70
    assert summary["same_math_parity_status"] == "passed"
    assert summary["input_sha256"]["results"] == journal["results_sha256"]
    for name, raw in (("plan", plan_raw), ("results", result_raw),
                      ("journal", (RESULT / "journal.json").read_bytes())):
        digest = hashlib.sha256(raw).hexdigest()
        assert summary["input_sha256"][name] == verification["artifacts"][f"{name}.json"] == digest
    assert len({plan["study_id"], result["study_id"], journal["study_id"], summary["study_id"],
                validation["study_id"], verification["study_id"]}) == 1
    rows = {r["run_key"]: r for r in result["runs"]}
    for seed in plan["config"]["seeds"]:
        state = summary["same_math_state_parity"][str(seed)]
        assert state["status"] == "passed" and state["saved_gates_equal"] and state["saved_named_adam_equal"]
        left, right = (state["states"][arm] for arm in ("lag2", "history_fixed_two"))
        assert left["gates"] == right["gates"] and left["named_adam"] == right["named_adam"]
        assert left["adam_group_sha256"] == right["adam_group_sha256"]
        for arm in ("lag2", "history_fixed_two"):
            key = f"{seed}:{arm}"
            assert state["states"][arm]["checkpoint_sha256"] == rows[key]["checkpoint"]["sha256"]
        assert verification["same_math_parity"][str(seed)]["saved_gates_equal"]
        assert verification["same_math_parity"][str(seed)]["saved_named_adam_equal"]
    analysis = read("analysis-manifest.json")
    assert analysis["training_source_revision"] == plan["source_revision"] == validation["training_source_revision"]
    assert analysis["analysis_source_revision"] == verification["analysis_source_revision"]
    assert analysis["summary_sha256"] == verification["summary_sha256"]


def test_repeated_one_initialization_is_labeled_as_replay_not_extra_seeds():
    replay = read("one-replay-verification.json")
    assert replay["status"] in {"exact_replay", "mismatch"}
    assert replay["same_model_data_schedule_and_recipe"] and replay["sealed_inputs_unchanged"]
    assert replay["current_study_id"] != replay["prior_study_id"]
    assert set(replay["runs"]) == {"41", "43", "47"}
    equal = all(all(row["parity"].values()) for row in replay["runs"].values())
    assert (replay["status"] == "exact_replay") is equal
    assert read("summary.json")["runs"] == 12
