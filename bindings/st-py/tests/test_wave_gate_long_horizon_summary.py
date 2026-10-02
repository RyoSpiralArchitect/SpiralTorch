import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def summary_module():
    source = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "summarize_wave_gate_long_horizon.py"
    )
    spec = importlib.util.spec_from_file_location("wave_gate_summary_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    arms, seeds = ["tangent", "wave_gate_radius4"], [41, 43]
    plan = {
        "study_id": "fixture",
        "config": {"arms": arms, "seeds": seeds, "steps": 2},
        "data": {"evaluation_block_hashes": {"tail": ["a", "b"]}},
        "batch_schedules": {str(seed): [[0, 1], [2, 3], [4, 5]] for seed in seeds},
    }
    result = {
        "study_id": "fixture",
        "status": "completed",
        "runs": [],
        "baseline": {"tail": {"mean": 3.0, "block_losses": [2.0, 4.0]}},
    }
    journal = {
        "study_id": "fixture",
        "status": "completed",
        "results_sha256": "sealed",
        "runs": {},
    }
    for seed in seeds:
        for arm in arms:
            key = f"{seed}:{arm}"
            loss = 2.0 if arm == "tangent" else (1.8 if seed == 41 else 2.4)
            checkpoint = {"filename": f"checkpoint-{key}.pt", "sha256": key}
            journal["runs"][key] = {
                "status": "completed",
                "cursor": 2,
                "checkpoint": checkpoint,
                "frozen_base_unchanged": True,
                "resume_next_update_equal": True,
            }
            result["runs"].append(
                {
                    "run_key": key,
                    "checkpoint": checkpoint,
                    "resume_next_update_equal": True,
                    "scores": {
                        "tail": {"mean": loss, "block_losses": [loss - 0.5, loss + 0.5]}
                    },
                    "records": [
                        {"step": i + 1, "batch_indices": [i * 2, i * 2 + 1]}
                        for i in range(2)
                    ],
                }
            )
    return plan, result, journal


def test_paired_summary_keeps_losing_seeds_and_baseline_separate(summary_module):
    plan, result, journal = fixture()
    before = copy.deepcopy((plan, result, journal))
    report = summary_module.summarize(plan, result, journal, "sealed")
    arm = report["comparisons"]["tail"]["arms"]["wave_gate_radius4"]
    assert arm["mean_ce"] == pytest.approx(2.1)
    assert arm["mean_delta_vs_baseline"] == pytest.approx(-0.9)
    assert arm["mean_delta_vs_tangent"] == pytest.approx(0.1)
    assert arm["seeds_better_than_tangent"] == 1
    assert [row["delta_vs_tangent"] for row in arm["per_seed"]] == pytest.approx(
        [-0.2, 0.4]
    )
    assert report["primary_updates"] == 8
    assert (plan, result, journal) == before


def test_protocol_specific_notes_do_not_invent_radius_or_seed_claims(summary_module):
    plan, result, journal = fixture()
    plan["config"]["summary_schema"] = "spiraltorch.elliptic_nonlinear_summary.v1"
    plan["config"]["comparison_notes"] = [
        "Paired projection initialization and data order.",
        "Equal parameter counts, unequal feature rank.",
    ]
    summary = summary_module.summarize(plan, result, journal, "sealed")
    assert summary["schema"] == "spiraltorch.elliptic_nonlinear_summary.v1"
    assert all(
        note in summary["interpretation"] for note in plan["config"]["comparison_notes"]
    )
    assert not any(
        "radius 4" in note or "not the frozen model" in note
        for note in summary["interpretation"]
    )


def factorial_fixture():
    plan, result, journal = fixture()
    arms = ["tangent", "elliptic", "causal_tangent", "causal_elliptic"]
    plan["config"].update(
        arms=arms,
        features=8,
        schema="spiraltorch.elliptic_causal_protocol.v1",
        comparison_notes=["Paired factorial fixture."],
    )
    template = result["runs"][0]
    journal_template = next(iter(journal["runs"].values()))
    result["runs"], journal["runs"] = [], {}
    losses = {41: [2.0, 2.5, 1.75, 1.5], 43: [3.0, 2.75, 2.5, 2.75]}
    for seed, values in losses.items():
        for arm, loss in zip(arms, values):
            key = f"{seed}:{arm}"
            row = copy.deepcopy(template)
            row.update(
                run_key=key,
                initial_parameter_sha256=f"{seed:064x}",
                parameter_count=90,
            )
            row["scores"]["tail"] = {
                "mean": loss,
                "block_losses": [loss - 0.5, loss + 0.5],
            }
            result["runs"].append(row)
            journal["runs"][key] = copy.deepcopy(journal_template)
    return plan, result, journal


def test_factorial_contrasts_pair_seeds_and_preserve_opposing_effects(summary_module):
    plan, result, journal = factorial_fixture()
    summary = summary_module.summarize(plan, result, journal, "sealed")
    contrasts = summary["paired_factorial_contrasts"]["tail"]
    expected = {
        "geometry_pointwise": [0.5, -0.25],
        "geometry_causal": [-0.25, 0.25],
        "mixing_tangent": [-0.25, -0.5],
        "mixing_elliptic": [-1.0, 0.0],
        "interaction": [-0.75, 0.5],
    }
    for name, values in expected.items():
        observed = contrasts[name]
        assert [row["ce_difference"] for row in observed["per_seed"]] == values
        assert observed["mean_ce_difference"] == sum(values) / 2
        assert observed["negative_seeds"] == sum(value < 0 for value in values)
    assert contrasts["interaction"]["paired_sample_sd"] == pytest.approx(0.883883476)


@pytest.mark.parametrize("corruption", ["hash", "missing_hash", "parameters", "arms"])
def test_factorial_rejects_unpaired_or_incomplete_arms(summary_module, corruption):
    plan, result, journal = factorial_fixture()
    if corruption == "hash":
        result["runs"][0]["initial_parameter_sha256"] = "a" * 64
    elif corruption == "missing_hash":
        del result["runs"][0]["initial_parameter_sha256"]
    elif corruption == "parameters":
        result["runs"][0]["parameter_count"] += 1
    else:
        plan["config"]["arms"].remove("elliptic")
        result["runs"] = [
            row for row in result["runs"] if not row["run_key"].endswith(":elliptic")
        ]
        journal["runs"] = {
            key: value
            for key, value in journal["runs"].items()
            if not key.endswith(":elliptic")
        }
    with pytest.raises(ValueError):
        summary_module.summarize(plan, result, journal, "sealed")


@pytest.mark.parametrize(
    "corruption",
    [
        "running",
        "hash",
        "duplicate",
        "endpoint",
        "schedule",
        "mean",
        "nan",
        "blocks",
        "sets",
    ],
)
def test_invalid_or_partial_evidence_cannot_be_summarized(summary_module, corruption):
    plan, result, journal = fixture()
    if corruption == "running":
        result["status"] = "evaluating"
    elif corruption == "hash":
        journal["results_sha256"] = "different"
    elif corruption == "duplicate":
        result["runs"].append(result["runs"][0])
    elif corruption == "endpoint":
        journal["runs"]["41:tangent"]["cursor"] = 1
    elif corruption == "schedule":
        result["runs"][0]["records"][0]["batch_indices"] = [5, 6]
    elif corruption == "mean":
        result["baseline"]["tail"]["mean"] = 1.0
    elif corruption == "nan":
        result["baseline"]["tail"]["block_losses"][0] = float("nan")
    elif corruption == "blocks":
        result["baseline"]["tail"]["block_losses"].pop()
    elif corruption == "sets":
        result["runs"][0]["scores"]["another"] = result["runs"][0]["scores"].pop("tail")
    with pytest.raises(ValueError):
        summary_module.summarize(plan, result, journal, "sealed")


@pytest.mark.parametrize("compressed", [False, True])
def test_cli_binds_input_bytes_and_never_overwrites(
    summary_module, tmp_path, monkeypatch, compressed
):
    plan, result, journal = fixture()
    raw_result = json.dumps(result).encode()
    journal["results_sha256"] = hashlib.sha256(raw_result).hexdigest()
    files = {
        "plan": json.dumps(plan).encode(),
        "results": raw_result,
        "journal": json.dumps(journal).encode(),
    }
    arguments = ["summarize"]
    for name, raw in files.items():
        suffix = ".json.gz" if compressed and name == "results" else ".json"
        path = tmp_path / f"{name}{suffix}"
        path.write_bytes(gzip.compress(raw, mtime=0) if suffix.endswith(".gz") else raw)
        arguments.extend([f"--{name}", str(path)])
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", arguments + ["--output", str(output)])
    summary_module.main()
    summary_bytes = output.read_bytes()
    assert json.loads(summary_bytes)["input_sha256"] == {
        name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
    }
    with pytest.raises(FileExistsError):
        summary_module.main()
    assert output.read_bytes() == summary_bytes
    result_path = tmp_path / ("results.json.gz" if compressed else "results.json")
    changed = raw_result + b"\n"
    result_path.write_bytes(gzip.compress(changed, mtime=0) if compressed else changed)
    other = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(sys, "argv", arguments + ["--output", str(other)])
    with pytest.raises(ValueError, match="hash"):
        summary_module.main()
    assert not other.exists()
