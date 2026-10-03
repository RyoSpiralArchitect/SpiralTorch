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


def test_gate_serialization_is_independent_of_libm_and_decimal_context(summary_module, monkeypatch):
    from decimal import ROUND_DOWN, localcontext

    def forbidden(_):
        raise AssertionError("platform tanh must not enter the summary")

    monkeypatch.setattr(summary_module.math, "tanh", forbidden)
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        assert summary_module.canonical_gate(-0.4858208894729614) == -0.450893236365
        assert summary_module.canonical_gate(-0.46712788939476013) == -0.435875785093
        assert summary_module.canonical_gate(0.047157686203718185) == 0.047122760106
        for raw in (0.0, -0.0, -1e-20):
            assert json.dumps(summary_module.canonical_gate(raw)) == "0.0"
        for raw in (20.0, 1e308):
            assert summary_module.canonical_gate(raw) == 1.0
            assert summary_module.canonical_gate(-raw) == -1.0
    for raw in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            summary_module.canonical_gate(raw)


@pytest.mark.parametrize("study", ["elliptic-gated-study", "elliptic-anchored-study", "elliptic-chart-step-study", "fractional-memory-study", "fractional-history-study"])
def test_committed_summary_rebuilds_byte_for_byte(summary_module, monkeypatch, tmp_path, study):
    directory = Path(__file__).resolve().parents[3] / f"benchmarks/results/2026-10-03-{study}"
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "summarize", "--plan", str(directory / "plan.json.gz"),
        "--results", str(directory / "results.json.gz"),
        "--journal", str(directory / "journal.json"), "--output", str(output),
    ])
    summary_module.main()
    assert output.read_bytes() == (directory / "summary.json").read_bytes()


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


def gated_fixture():
    plan, result, journal = factorial_fixture()
    plan["config"]["schema"] = "spiraltorch.elliptic_gated_protocol.v1"
    plan["config"]["arms"] = [a.replace("causal_", "gated_") for a in plan["config"]["arms"]]
    journal["runs"] = {k.replace("causal_", "gated_"): v for k, v in journal["runs"].items()}
    for row in result["runs"]:
        row["run_key"] = row["run_key"].replace("causal_", "gated_")
        if "gated_" in row["run_key"]:
            row["parameter_count"] += 1
            row["initial_projection_sha256"] = row["initial_parameter_sha256"]
            row["initial_parameter_sha256"] = "a" * 64
            row["final_raw_mix"] = -0.2
            for i, record in enumerate(row["records"]):
                record.update(raw_mix_before_update=-0.1*i, raw_mix_after_update=-0.1*(i+1), raw_mix_gradient=0.3)
    return plan, result, journal


def test_gated_summary_pairs_projections_and_keeps_signed_trajectory(summary_module):
    plan, result, journal = gated_fixture()
    report = summary_module.summarize(plan, result, journal, "sealed")
    assert report["paired_factorial_contrasts"]["tail"]["geometry_gated"]["mean_ce_difference"] == 0
    gate = report["gate_trajectories"]["41:gated_elliptic"]
    assert gate["final_raw_mix"] == -0.2 and gate["final_mix"] < 0
    assert gate["min_raw_mix"] == -0.2 and gate["nonzero_gradient_steps"] == 2


def anchored_fixture():
    plan, result, journal = gated_fixture()
    rename = {"tangent": "anchored_tangent", "elliptic": "anchored_elliptic"}
    plan["config"].update(
        schema="spiraltorch.elliptic_anchored_protocol.v1",
        reference_arm="anchored_tangent",
        arms=[rename.get(arm, arm) for arm in plan["config"]["arms"]],
    )
    for row in result["runs"]:
        key = row["run_key"]
        seed, arm = key.split(":")
        row["run_key"] = f"{seed}:{rename.get(arm, arm)}"
        if row["run_key"] != key:
            journal["runs"][row["run_key"]] = journal["runs"].pop(key)
            row.update(
                parameter_count=91, initial_projection_sha256=row["initial_parameter_sha256"],
                initial_parameter_sha256="a" * 64, final_raw_mix=-0.2,
            )
            for i, record in enumerate(row["records"]):
                record.update(raw_mix_before_update=-0.1*i, raw_mix_after_update=-0.1*(i+1), raw_mix_gradient=0.3)
    return plan, result, journal


def test_anchored_factorial_reports_real_reference_and_all_four_gates(summary_module):
    plan, result, journal = anchored_fixture()
    before = copy.deepcopy((plan, result, journal))
    report = summary_module.summarize(plan, result, journal, "sealed")
    assert report["reference_arm"] == "anchored_tangent"
    assert set(report["gate_trajectories"]) == set(journal["runs"])
    row = report["comparisons"]["tail"]["arms"]["anchored_elliptic"]
    assert "mean_delta_vs_tangent" not in row
    assert row["mean_delta_vs_anchored_tangent"] == 0.125
    assert row["seeds_better_than_anchored_tangent"] == 1
    expected = {
        "geometry_anchored": [0.5, -0.25],
        "geometry_gated": [-0.25, 0.25],
        "anchor_tangent": [0.25, 0.5],
        "anchor_elliptic": [1.0, 0.0],
        "interaction": [0.75, -0.5],
    }
    for name, values in expected.items():
        observed = report["paired_factorial_contrasts"]["tail"][name]
        assert [r["ce_difference"] for r in observed["per_seed"]] == values
        assert observed["mean_ce_difference"] == sum(values) / 2
        assert observed["negative_seeds"] == sum(value < 0 for value in values)
    assert (plan, result, journal) == before


def chart_fixture():
    plan, result, journal = anchored_fixture()
    rename = dict(zip(plan["config"]["arms"], ["adam_tangent", "adam_elliptic", "chart_tangent", "chart_elliptic"]))
    plan["config"].update(schema="spiraltorch.elliptic_chart_step_protocol.v1", arms=list(rename.values()), reference_arm="adam_tangent", relative_damping=0.1)
    for row in result["runs"]:
        old = row["run_key"]
        seed, arm = old.split(":")
        row["run_key"] = f"{seed}:{rename[arm]}"
        journal["runs"][row["run_key"]] = journal["runs"].pop(old)
        for record in row["records"]:
            enabled = rename[arm].startswith("chart_")
            record["optimizer_step"] = {"enabled": enabled}
            if enabled:
                record["optimizer_step"].update(metric=[1., 0., 0., 2.], damped_condition=(2. + .15) / (1. + .15), proposal_l2=.1, step_l2=.1, applied_step_l2=.100000001, cosine=.9, gradient_dot_proposal=-.01, gradient_dot_applied_step=-.01)
    return plan, result, journal


def test_chart_factorial_keeps_reference_interaction_and_step_receipts(summary_module):
    plan, result, journal = chart_fixture()
    report = summary_module.summarize(plan, result, journal, "sealed")
    contrasts = report["paired_factorial_contrasts"]["tail"]
    expected = {"chart_elliptic": [-1., 0.], "chart_tangent": [-.25, -.5], "geometry_adam": [.5, -.25], "geometry_chart": [-.25, .25], "interaction": [-.75, .5]}
    for name, values in expected.items():
        assert [r["ce_difference"] for r in contrasts[name]["per_seed"]] == values
    assert report["reference_arm"] == "adam_tangent"
    assert report["chart_step_trajectories"]["41:chart_elliptic"]["nonzero_proposals"] == 2


@pytest.mark.parametrize("corruption", ["pairing", "count", "enabled", "missing", "nan", "norm", "applied_norm", "vanished_tiny_step", "cosine", "metric", "indefinite_metric", "condition", "damping"])
def test_chart_summary_rejects_inconsistent_receipts(summary_module, corruption):
    plan, result, journal = chart_fixture()
    row = next(r for r in result["runs"] if r["run_key"].endswith(":chart_elliptic"))
    receipt = row["records"][0]["optimizer_step"]
    if corruption == "pairing":
        row["initial_parameter_sha256"] = "b" * 64
    elif corruption == "count":
        row["parameter_count"] -= 1
    elif corruption == "enabled":
        receipt["enabled"] = False
    elif corruption == "missing":
        del row["records"][0]["optimizer_step"]
    elif corruption == "nan":
        receipt["gradient_dot_applied_step"] = float("nan")
    elif corruption == "norm":
        receipt["step_l2"] = .2
    elif corruption == "applied_norm":
        receipt["applied_step_l2"] = .2
    elif corruption == "vanished_tiny_step":
        receipt.update(proposal_l2=1e-30, step_l2=1e-30, applied_step_l2=0.)
    elif corruption == "cosine":
        receipt["cosine"] = -1
    elif corruption == "indefinite_metric":
        receipt["metric"] = [1., 2., 2., 1.]
    elif corruption == "condition":
        receipt["damped_condition"] = 1.8
    elif corruption == "damping":
        plan["config"]["relative_damping"] = 0.2
    else:
        receipt["metric"] = [0., 0., 0., 0.]
    with pytest.raises(ValueError):
        summary_module.summarize(plan, result, journal, "sealed")


@pytest.mark.parametrize("scale", [1e-300, 1., 1e300, 1e308])
@pytest.mark.parametrize("cross,condition", [(0., 1.), (1., 21.), (-1., 21.), (1. + 1e-15, 21.)])
def test_chart_metric_scale_and_rank_one_roundoff(summary_module, scale, cross, condition):
    plan, result, journal = chart_fixture()
    receipt = result["runs"][2]["records"][0]["optimizer_step"]
    assert receipt["enabled"]
    receipt.update(metric=[scale, cross * scale, cross * scale, scale], damped_condition=condition)
    summary_module.summarize(plan, result, journal, "sealed")


@pytest.mark.parametrize("metric", [[1., 1. + 1e-8, 1. + 1e-8, 1.], [1e-300, 1e300, 1e300, 1e-300]])
def test_chart_metric_rejects_negative_determinant_and_overflow(summary_module, metric):
    plan, result, journal = chart_fixture()
    result["runs"][2]["records"][0]["optimizer_step"]["metric"] = metric
    with pytest.raises(ValueError, match="positive semidefinite"):
        summary_module.summarize(plan, result, journal, "sealed")


@pytest.mark.parametrize("damping", [None, True, "0.1", 0., 2., float("nan"), float("inf")])
def test_chart_summary_rejects_invalid_damping(summary_module, damping):
    plan, result, journal = chart_fixture()
    plan["config"]["relative_damping"] = damping
    with pytest.raises(ValueError, match="relative damping"):
        summary_module.summarize(plan, result, journal, "sealed")


@pytest.mark.parametrize("corruption", ["projection", "parameters", "hash", "reference", "missing_reference", "gate", "endpoint", "nan"])
def test_anchored_summary_rejects_unpaired_or_mislabeled_evidence(summary_module, corruption):
    plan, result, journal = anchored_fixture()
    row = result["runs"][0]
    if corruption == "projection":
        row["initial_projection_sha256"] = "b" * 64
    elif corruption == "parameters":
        row["parameter_count"] -= 1
    elif corruption == "hash":
        row["initial_parameter_sha256"] = "b" * 64
    elif corruption == "reference":
        plan["config"]["reference_arm"] = "gated_tangent"
    elif corruption == "missing_reference":
        del plan["config"]["reference_arm"]
    elif corruption == "gate":
        del row["records"][0]["raw_mix_gradient"]
    elif corruption == "endpoint":
        row["final_raw_mix"] = 0.3
    else:
        row["records"][0]["raw_mix_before_update"] = float("nan")
    with pytest.raises(ValueError):
        summary_module.summarize(plan, result, journal, "sealed")


@pytest.mark.parametrize("corruption", ["projection", "gate_hash", "parameters", "missing", "endpoint", "discontinuous", "nan"])
def test_gated_summary_rejects_invalid_pairing_or_trajectory(summary_module, corruption):
    plan, result, journal = gated_fixture()
    row = next(r for r in result["runs"] if "gated_" in r["run_key"])
    if corruption == "projection":
        row["initial_projection_sha256"] = "b" * 64
    elif corruption == "gate_hash":
        row["initial_parameter_sha256"] = "b" * 64
    elif corruption == "parameters":
        row["parameter_count"] -= 1
    elif corruption == "missing":
        del row["records"][0]["raw_mix_gradient"]
    elif corruption == "endpoint":
        row["final_raw_mix"] = 0.2
    elif corruption == "discontinuous":
        row["records"][1]["raw_mix_before_update"] = 0.2
    else:
        row["records"][0]["raw_mix_gradient"] = float("nan")
    with pytest.raises(ValueError):
        summary_module.summarize(plan, result, journal, "sealed")


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
