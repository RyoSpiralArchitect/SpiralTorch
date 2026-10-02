from __future__ import annotations

import importlib.util
import copy
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("run_hf_repetition_schedule_study.py")
SPEC = importlib.util.spec_from_file_location("schedule_study", MODULE_PATH)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_schedule_study_rotates_arms_without_dropping_coordinates():
    spec = study.read_json(study.DEFAULT_SPEC)
    rows = list(study.coordinates(spec))
    assert len(rows) == spec["acceptance"]["required_runs"] == 9
    assert len({(seed, arm["name"]) for seed, arm in rows}) == 9
    assert [rows[i][1]["name"] for i in (0, 3, 6)] == [
        "ordinary_ft",
        "periodic_constant",
        "periodic_decay",
    ]
    assert len(rows) * len(spec["milestones"]) == 36


def test_training_keeps_full_horizon_and_only_schedule_differs_between_treatments():
    spec = study.read_json(study.DEFAULT_SPEC)
    commands = [
        study.train_args(spec, Path("model"), Path("corpus"), Path("run"), 151, arm)
        for arm in spec["arms"]
    ]
    for command in commands:
        assert command[command.index("--max-steps") + 1] == "256"
        assert command[command.index("--max-eval-blocks") + 1] == "0"
        assert "--training-use-cpu" in command
        assert command.count("--causal-lm-mask-first-label") == 1
        assert "--resume-from-checkpoint" not in command
    assert commands[2][: len(commands[1])] == commands[1]
    assert commands[2][len(commands[1]) :] == [
        "--zspace-repetition-unlikelihood-decay-start-update",
        "0",
        "--zspace-repetition-unlikelihood-decay-end-update",
        "256",
        "--zspace-repetition-unlikelihood-decay-final-scale",
        "0",
    ]


def test_v2_requires_alignment_and_excludes_invalidated_seeds():
    spec = study.read_json(study.DEFAULT_SPEC)
    study.validate_spec_alignment(spec)
    assert not set(spec["seeds"]) & {7, 109, 113, 127, 137, 139, 149}
    prior = study.read_json(
        study.REPO / "docs/benchmarks/hf_repetition_schedule_256step_prespec_20261002.json"
    )
    assert spec["acceptance"] == prior["acceptance"]
    assert spec["milestones"] == prior["milestones"]
    with pytest.raises(ValueError, match="aligned v2"):
        study.validate_spec_alignment(prior)
    spec["training_args"].remove("--causal-lm-mask-first-label")
    with pytest.raises(ValueError, match="aligned v2"):
        study.validate_spec_alignment(spec)


def test_generation_binds_study_and_never_adds_inference_intervention():
    spec = study.read_json(study.DEFAULT_SPEC)
    command = study.generation_command(
        spec, Path("model"), Path("run"), 137, 256, "sha256:abc"
    )
    assert "run/checkpoint-256" in command
    assert "--baseline-only" in command and "--no-do-sample" in command
    assert (
        command[command.index("--generation-evidence-protocol-id") + 1] == "sha256:abc"
    )


def test_clients_preload_the_tested_wheel_before_source_examples():
    command = study.client_command(Path("example.py"), ["--train"])
    assert command[1:3] == ["-I", "-c"]
    assert command[3].index("import spiraltorch") < command[3].index("runpy.run_path")
    assert command[4:] == ["example.py", "--train"]


def test_outputs_are_exclusive_and_source_drift_is_rejected(tmp_path):
    path = tmp_path / "sealed.json"
    study.write_new(path, {"a": 1})
    before = study.digest(path)
    with pytest.raises(FileExistsError):
        study.write_new(path, {"a": 2})
    assert study.digest(path) == before
    study.verify_sources({str(path): before})
    with pytest.raises(ValueError, match="source/input changed"):
        study.verify_sources({str(path): "0" * 64})


def test_uniform_label_gate_rejects_masked_or_partial_rows():
    torch = pytest.importorskip("torch")

    def collate(rows):
        return {"labels": torch.tensor([row["labels"] for row in rows])}

    study.require_uniform_labels([{"labels": [-100, 2, 3, 4]}], collate, 3)
    for labels in ([-100, 2, -100, 4], [-100, 2, 3]):
        with pytest.raises(ValueError):
            study.require_uniform_labels([{"labels": labels}], collate, 3)
    with pytest.raises(ValueError, match="unpredicted first label"):
        study.require_uniform_labels([{"labels": [1, 2, 3, 4]}], collate, 3)
    with pytest.raises(ValueError, match="empty"):
        study.require_uniform_labels([], collate, 3)


def test_missing_or_partial_generations_cannot_be_marked_complete():
    spec = study.read_json(study.DEFAULT_SPEC)
    with pytest.raises(ValueError, match="did not complete"):
        study.validate_generation({"status": "partial", "runs": []}, spec, "sha256:abc")


def test_failed_training_card_cannot_be_marked_complete(tmp_path):
    with pytest.raises(ValueError, match="failure"):
        study.validate_training({"failure_stage": "train"}, {}, None, {}, tmp_path)


def test_training_validation_checks_horizon_partition_and_exact_policy(tmp_path):
    spec = study.read_json(study.DEFAULT_SPEC)
    for step in spec["milestones"]:
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        study.write_new(checkpoint / "trainer_state.json", {"global_step": step})
        (checkpoint / "adapter_model.safetensors").touch()
    card = {
        "model_saved": True,
        "adapter_saved": True,
        "tokenized_dataset_identity": {"observed_identity_id": "tokens"},
        "training_recipe_identity": {
            "identity_payload": {
                "trainer_contract": {
                    "data_collator": {
                        "class": "spiraltorch.HfRepetitionUnlikelihoodCollator",
                        "base_class": "spiraltorch.HfCausalLabelAlignmentCollator",
                        "mlm": False,
                        "causal_label_alignment": study.LABEL_ALIGNMENT,
                    }
                },
                "training_arguments": {
                    "use_cpu": True,
                    "world_size": 1,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 16,
                }
            }
        },
        "trainer_trace_lineage": {"ready": True, "trace_last_global_step": 256},
        "eval_before_train": {"status": "ok", "eval_loss": 5.0},
        "eval_after_train": {"status": "ok", "eval_loss": 4.8},
        "zspace_repetition_unlikelihood_receipt": {
            "periodic_candidate_count": 1,
            "mean_weighted_auxiliary_loss": 0.01,
            "last_objective_control": {
                "completed_update_slots": 255,
                "policy": {
                    "base_strength": 0.1,
                    "config": {
                        "normalization": "active_positions",
                        "schedule": {
                            "kind": "linear_decay",
                            "start_update": 0,
                            "end_update": 256,
                            "final_scale": 0.0,
                        },
                    },
                },
            },
        },
    }
    arm = spec["arms"][2]
    study.validate_training(card, spec, "tokens", arm, tmp_path)
    changed = copy.deepcopy(card)
    del changed["training_recipe_identity"]["identity_payload"]["trainer_contract"]
    with pytest.raises(ValueError, match="shared causal label alignment"):
        study.validate_training(changed, spec, "tokens", arm, tmp_path)
    changed = copy.deepcopy(card)
    changed["training_recipe_identity"]["identity_payload"]["trainer_contract"][
        "data_collator"
    ]["class"] = "spiraltorch.HfCausalLabelAlignmentCollator"
    study.validate_training(changed, spec, "tokens", spec["arms"][0], tmp_path)
    changed = copy.deepcopy(card)
    changed["trainer_trace_lineage"]["trace_last_global_step"] = 64
    with pytest.raises(ValueError, match="horizon"):
        study.validate_training(changed, spec, "tokens", arm, tmp_path)
    changed = copy.deepcopy(card)
    changed["training_recipe_identity"]["identity_payload"]["training_arguments"][
        "use_cpu"
    ] = False
    with pytest.raises(ValueError, match="partition"):
        study.validate_training(changed, spec, "tokens", arm, tmp_path)
    changed = copy.deepcopy(card)
    changed["zspace_repetition_unlikelihood_receipt"]["last_objective_control"][
        "policy"
    ]["config"]["schedule"]["end_update"] = 64
    with pytest.raises(ValueError, match="policy"):
        study.validate_training(changed, spec, "tokens", arm, tmp_path)
