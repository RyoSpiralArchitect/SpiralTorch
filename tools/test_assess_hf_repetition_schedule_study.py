from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).with_name("assess_hf_repetition_schedule_study.py")
LOADER = importlib.util.spec_from_file_location("assess_schedule", PATH)
assessor = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(assessor)


def fixture():
    spec = assessor.study.read_json(assessor.study.DEFAULT_SPEC)
    training = [
        {"seed": seed, "arm": arm["name"], "eval_before": 5.0, "eval_after": 4.8}
        for seed in spec["seeds"]
        for arm in spec["arms"]
    ]
    generations = [
        {
            "seed": seed,
            "arm": arm["name"],
            "step": step,
            "loop_score": {
                "ordinary_ft": 0.7,
                "periodic_constant": 0.8,
                "periodic_decay": 0.5,
            }[arm["name"]],
        }
        for seed in spec["seeds"]
        for arm in spec["arms"]
        for step in spec["milestones"]
    ]
    return spec, training, generations


def test_complete_paired_fixture_passes_bounded_gate():
    result = assessor.paired_assessment(*fixture())
    assert result["decision"] == "passed_bounded_gate"
    assert all(result["gates"].values())


def test_early_wins_do_not_replace_negative_final_endpoint():
    spec, training, generations = fixture()
    for row in generations:
        if row["arm"] == "periodic_decay" and row["step"] == 256:
            row["loop_score"] = 1.1
    result = assessor.paired_assessment(spec, training, generations)
    assert result["decision"] == "ready_but_negative"
    assert result["gates"]["decay_vs_constant"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_incomplete_coordinates_cannot_produce_a_verdict(mutation):
    spec, training, generations = fixture()
    if mutation == "missing":
        generations.pop()
    elif mutation == "duplicate":
        generations.append(copy.deepcopy(generations[0]))
    else:
        generations[0]["seed"] = 7
    with pytest.raises(ValueError, match="coordinates"):
        assessor.paired_assessment(spec, training, generations)


def test_nonfinite_loss_and_initial_parity_are_rejected():
    spec, training, generations = fixture()
    training[0]["eval_after"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        assessor.paired_assessment(spec, training, generations)
    training[0]["eval_after"] = 4.8
    training[0]["eval_before"] = 4.9
    with pytest.raises(ValueError, match="parity"):
        assessor.paired_assessment(spec, training, generations)


def test_loop_improvement_cannot_override_ce_harm():
    spec, training, generations = fixture()
    for row in training:
        if row["arm"] == "periodic_decay":
            row["eval_after"] = 4.9
    result = assessor.paired_assessment(spec, training, generations)
    assert result["gates"]["decay_vs_constant"] is True
    assert result["gates"]["heldout_ce_safety"] is False
    assert result["decision"] == "ready_but_negative"


def test_hash_mismatch_rejected_before_json_is_consumed(tmp_path):
    path = tmp_path / "artifact.json"
    assessor.study.write_new(path, {"status": "complete"})
    with pytest.raises(ValueError, match="hash mismatch"):
        assessor.checked_json(path, "0" * 64)
    assert assessor.checked_json(path, assessor.study.digest(path)) == {
        "status": "complete"
    }


def test_study_without_completion_record_is_not_assessed(tmp_path):
    # No successful status can be inferred from a sealed plan alone.
    assessor.study.write_new(tmp_path / "sealed-plan.json", {})
    with pytest.raises(FileNotFoundError):
        assessor.assess(tmp_path, assessor.study.DEFAULT_SPEC)
