from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import spiraltorch as st


REPO = Path(__file__).resolve().parents[3]
PRESPEC_PATH = REPO / (
    "docs/benchmarks/hf_periodic_gpt2_pride_full_corpus_256step_prespec_20260823.json"
)
REPORT_PATH = REPO / (
    "docs/benchmarks/hf_periodic_gpt2_pride_full_corpus_256step_20260823.json"
)
EVIDENCE_BUNDLE_PATH = REPO / (
    "docs/benchmarks/"
    "hf_periodic_gpt2_pride_full_corpus_256step_generation_evidence_20260823.json"
)
PROTOCOL_ID = "sha256:1270acc1a14a0783ea1b168f0f8b33df84e82296b22fcc36c392d1501c5a8a9c"
GENERATION_METRIC_PROTOCOL_ID = (
    "sha256:a2396fa18c4bd3fe00b6214a139ed37a42cc266291513269dd5911a2ad9c787b"
)
STUDY_ID = "sha256:3e1846f0418cc7f644ce092fa67a4af2ea00fe9a80fe13c89080c47155826d43"
EVIDENCE_BUNDLE_ID = (
    "sha256:4f0d21baa623f294cd6c42605904d1024411fd4a43bb7b2ba06d14bcf8d5e06a"
)
SEEDS = (109, 113, 127)
ARMS = ("causal_lm_baseline", "model_topk_history", "model_topk_periodic")
MILESTONES = (64, 128, 192, 256)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_id(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_gpt2_pride_long_horizon_protocol_was_sealed_before_training() -> None:
    prespec = _load(PRESPEC_PATH)
    design = prespec["scientific_design"]

    assert prespec["schema"] == "spiraltorch.hf_periodic_long_horizon_protocol.v1"
    assert prespec["status"] == "sealed_before_training"
    assert prespec["design_id"] == _sha256_id(design)
    assert prespec["protocol_id"] == PROTOCOL_ID
    assert prespec["design_prespec_commit"] == (
        "3e5873347e34ccfc4a89b59be8926a8f7f5cdc7f"
    )
    assert prespec["preflight_commitment"]["status"] == "sealed_before_training"
    assert design["training"]["seeds"] == list(SEEDS)
    assert design["training"]["optimizer_update_count"] == 256
    assert design["generation"]["milestones"] == list(MILESTONES)
    assert design["acceptance"]["required_generation_reports"] == 36


def test_gpt2_pride_generation_bundle_roundtrips_through_rust() -> None:
    bundle = _load(EVIDENCE_BUNDLE_PATH)
    report = _load(REPORT_PATH)
    committed = dict(bundle)
    bundle_id = committed.pop("bundle_id")
    entries = bundle["entries"]
    milestone_rows = report["results"]["generation_milestones"]
    milestones_by_coordinate = {
        (row["seed"], row["arm"], row["step"]): row for row in milestone_rows
    }

    assert bundle["schema"] == "spiraltorch.hf_generation_evidence_bundle.v1"
    assert bundle["status"] == "ready"
    assert bundle["protocol_id"] == PROTOCOL_ID
    assert bundle_id == _sha256_id(committed)
    assert bundle_id == EVIDENCE_BUNDLE_ID
    assert bundle["report_count"] == 36
    assert bundle["sample_count"] == 432
    assert len(entries) == 36
    assert len(milestones_by_coordinate) == 36
    assert {(entry["seed"], entry["arm"], entry["step"]) for entry in entries} == {
        (seed, arm, step) for seed in SEEDS for arm in ARMS for step in MILESTONES
    }

    for entry in entries:
        coordinate = (entry["seed"], entry["arm"], entry["step"])
        milestone = milestones_by_coordinate[coordinate]
        evidence = entry["evidence"]
        request = evidence["request"]
        aggregate = evidence["aggregate"]
        ngrams = {row["order"]: row for row in aggregate["ngrams"]}
        assert st.validate_zspace_generation_evidence(evidence) == evidence
        assert evidence["status"] == "ready"
        assert evidence["evidence_validated"] is True
        assert aggregate["sample_count"] == 12
        assert aggregate["empty_sample_count"] == 0
        assert request["protocol_id"] == GENERATION_METRIC_PROTOCOL_ID
        assert request["prompt_set_id"] == bundle["prompt_set_id"]
        assert {sample["seed"] for sample in request["samples"]} == {entry["seed"]}
        assert evidence["evidence_id"] == milestone["evidence_id"]
        assert request["model_artifact_id"] == milestone["model_artifact_id"]
        assert request["decoding_config_id"] == milestone["decoding_config_id"]
        assert milestone["generated_continuation_set_id"] == _sha256_id(
            {
                "schema": "spiraltorch.hf_generated_continuation_set.v1",
                "prompt_set_id": request["prompt_set_id"],
                "samples": request["samples"],
            }
        )
        assert aggregate["sample_count"] == milestone["sample_count"]
        assert aggregate["empty_sample_count"] == milestone["empty_sample_count"]
        assert (
            aggregate["sample_mean_loop_score"] == milestone["sample_mean_loop_score"]
        )
        assert (
            aggregate["periodic_loop_sample_ratio"]
            == milestone["periodic_loop_sample_ratio"]
        )
        assert (
            aggregate["periodic_suffix_repeated_token_ratio"]
            == milestone["periodic_suffix_repeated_token_ratio"]
        )
        assert ngrams[3]["repetition_ratio"] == milestone["trigram_repetition_ratio"]
        assert ngrams[2]["distinct_ratio"] == milestone["bigram_distinct_ratio"]
        assert ngrams[4]["distinct_ratio"] == milestone["fourgram_distinct_ratio"]

    assert report["provenance"]["generation_evidence_bundle_id"] == bundle_id
    assert report["provenance"]["generation_evidence_metric_protocol_id"] == (
        GENERATION_METRIC_PROTOCOL_ID
    )
    assert (
        report["provenance"]["generation_evidence_inner_protocol_bound_to_study"]
        is False
    )
    assert report["identity_gates"]["generation_evidence_roundtrip_count"] == 36


def test_gpt2_pride_report_recomputes_the_frozen_decision() -> None:
    prespec = _load(PRESPEC_PATH)
    report = _load(REPORT_PATH)
    bundle = _load(EVIDENCE_BUNDLE_PATH)
    training_rows = report["results"]["training_runs"]
    milestone_rows = report["results"]["generation_milestones"]
    acceptance = report["acceptance"]

    assert report["schema"] == "spiraltorch.hf_periodic_long_horizon_benchmark.v1"
    assert report["study_id"] == STUDY_ID
    assert report["status"] == "ready_but_negative"
    assert report["decision"] == "ready_but_negative_prespecified_gate"
    assert report["protocol"]["protocol_id"] == PROTOCOL_ID
    assert report["protocol"]["design_id"] == prespec["design_id"]
    assert report["study_spec"] == prespec["scientific_design"]
    assert len(training_rows) == 9
    assert len(milestone_rows) == 36
    assert all(row["optimizer_horizon"] == 256 for row in training_rows)
    assert all(row["sample_count"] == 12 for row in milestone_rows)
    assert all(row["empty_sample_count"] == 0 for row in milestone_rows)

    preflight = prespec["preflight_commitment"]
    preflight_by_seed = {row["seed"]: row for row in preflight["by_seed"]}
    assert {(row["seed"], row["arm"]) for row in training_rows} == {
        (seed, arm) for seed in SEEDS for arm in ARMS
    }
    for row in training_rows:
        seed = row["seed"]
        arm = row["arm"]
        seed_preflight = preflight_by_seed[seed]
        arm_preflight = seed_preflight["arms"][arm]
        assert row["training_identities"] == {
            "training_input_id": preflight["common"]["training_input_id"],
            "dataset_materialization_id": seed_preflight["dataset_materialization_id"],
            "tokenized_dataset_id": seed_preflight["tokenized_dataset_id"],
            "model_runtime_id": preflight["common"]["model_runtime_id"],
            "execution_id": preflight["common"]["execution_id"],
            "training_recipe_id": arm_preflight["training_recipe_id"],
            "finetune_replay_id": arm_preflight["finetune_replay_id"],
        }

    final_artifacts = {
        (row["seed"], row["arm"]): row["model_artifact_id"]
        for row in milestone_rows
        if row["step"] == 256
    }
    assert final_artifacts == {
        (row["seed"], row["arm"]): row["final_adapter_id"] for row in training_rows
    }

    evidence_ids = {entry["evidence"]["evidence_id"] for entry in bundle["entries"]}
    assert {row["evidence_id"] for row in milestone_rows} == evidence_ids

    final_curve = next(
        row
        for row in report["results"]["generation_learning_curve"]
        if row["step"] == 256
    )
    replication = final_curve["periodic_vs_history"]
    absolute = final_curve["periodic_vs_baseline"]
    replication_pass = bool(
        replication["mean_periodic_minus_control"] < 0.0
        and replication["periodic_seed_win_count"] >= 2
    )
    absolute_pass = bool(
        absolute["mean_periodic_minus_control"] <= 0.0
        and absolute["periodic_seed_non_loss_count"] >= 2
    )
    assert acceptance["periodic_vs_history_replication_passed"] is replication_pass
    assert acceptance["periodic_vs_baseline_absolute_control_passed"] is absolute_pass
    assert math.isclose(
        replication["mean_periodic_minus_control"],
        0.06261107482062123,
        abs_tol=1e-15,
    )
    assert replication["periodic_seed_win_count"] == 0
    assert math.isclose(
        absolute["mean_periodic_minus_control"],
        0.03820993270706315,
        abs_tol=1e-15,
    )
    assert absolute["periodic_seed_non_loss_count"] == 2

    by_run = {(row["seed"], row["arm"]): row for row in training_rows}
    baseline_deltas = [
        by_run[(seed, "causal_lm_baseline")]["loss"]["eval_after_train"]
        - by_run[(seed, "causal_lm_baseline")]["loss"]["eval_before_train"]
        for seed in SEEDS
    ]
    sanity = report["results"]["training_sanity"]
    assert math.isclose(
        sanity["mean_after_minus_before"],
        sum(baseline_deltas) / len(baseline_deltas),
        abs_tol=1e-15,
    )
    sanity_pass = all(delta < 0.0 for delta in baseline_deltas) and (
        sum(baseline_deltas) / len(baseline_deltas) <= -0.05
    )
    assert acceptance["training_sanity_passed"] is sanity_pass

    safety_pass = True
    for arm in ("model_topk_history", "model_topk_periodic"):
        deltas = [
            by_run[(seed, arm)]["loss"]["eval_after_train"]
            - by_run[(seed, "causal_lm_baseline")]["loss"]["eval_after_train"]
            for seed in SEEDS
        ]
        row = report["results"]["held_out_ce_safety"][arm]
        assert math.isclose(
            row["mean_treatment_minus_baseline"],
            sum(deltas) / len(deltas),
            abs_tol=1e-15,
        )
        assert math.isclose(
            row["maximum_treatment_minus_baseline"],
            max(deltas),
            abs_tol=1e-15,
        )
        safety_pass = safety_pass and sum(deltas) / len(deltas) <= 0.02
        safety_pass = safety_pass and max(deltas) <= 0.05
    assert acceptance["held_out_ce_safety_passed"] is safety_pass
    periodic_safety = report["results"]["held_out_ce_safety"]["model_topk_periodic"]
    assert math.isclose(
        periodic_safety["mean_treatment_minus_baseline"],
        0.03418461481730143,
        abs_tol=1e-15,
    )
    assert periodic_safety["passed"] is False

    expected_all = all(
        (
            acceptance["readiness_passed"],
            acceptance["identity_passed"],
            acceptance["complete_optimizer_horizon_passed"],
            acceptance["before_train_comparability_passed"],
            acceptance["training_sanity_passed"],
            acceptance["mechanism_passed"],
            acceptance["held_out_ce_safety_passed"],
            replication_pass,
            absolute_pass,
        )
    )
    assert acceptance["all_prespecified_gates_passed"] is expected_all
    assert report["status"] == ("passed" if expected_all else "ready_but_negative")
    assert report["decision"] == (
        "passed_prespecified_gate"
        if expected_all
        else "ready_but_negative_prespecified_gate"
    )
    assert report["claims"]["efficacy_claim_ready"] is expected_all
    assert report["diagnosis"]["failed_gates"] == [
        "held_out_ce_safety",
        "periodic_vs_history_replication",
        "periodic_vs_baseline_absolute_control",
    ]
    crossover = report["diagnosis"]["milestone_crossover"]
    assert crossover["early_all_seed_win_steps"] == [64, 128]
    assert crossover["first_all_seed_loss_step"] == 192
    effects = crossover["periodic_history_mean_effects"]
    assert [row["periodic_seed_win_count"] for row in effects] == [3, 3, 0, 0]
    assert math.isclose(
        effects[0]["mean_periodic_minus_history"],
        -0.17566476400812903,
        abs_tol=1e-15,
    )
    objective_budget = report["diagnosis"]["posthoc_objective_budget_diagnostic"]
    assert math.isclose(
        objective_budget["periodic_to_history_weighted_auxiliary_loss_ratio"],
        1.9436070876627936,
        abs_tol=1e-15,
    )
    assert report["next_prespecified_experiment"]["classification"] == (
        "posthoc_recommendation_not_part_of_this_gate"
    )
