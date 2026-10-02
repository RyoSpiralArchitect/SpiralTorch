#!/usr/bin/env python3
"""Assess all frozen schedule-study endpoints; never select a best checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import statistics

_loader = importlib.util.spec_from_file_location(
    "schedule_study", Path(__file__).with_name("run_hf_repetition_schedule_study.py")
)
study = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(study)


def exact_coordinates(rows, expected, keys):
    coordinates = [tuple(row[key] for key in keys) for row in rows]
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise ValueError("missing, duplicate or unexpected study coordinates")


def paired_assessment(spec, training, generations):
    seeds = spec["seeds"]
    arms = [arm["name"] for arm in spec["arms"]]
    exact_coordinates(training, {(s, a) for s in seeds for a in arms}, ("seed", "arm"))
    exact_coordinates(
        generations,
        {(s, a, step) for s in seeds for a in arms for step in spec["milestones"]},
        ("seed", "arm", "step"),
    )
    horizon = spec["acceptance"]["final_update_slot"]
    by_run = {(row["seed"], row["arm"]): row for row in training}
    by_generation = {(row["seed"], row["arm"], row["step"]): row for row in generations}
    paired = []
    for seed in seeds:
        base = by_run[seed, "ordinary_ft"]
        decay = by_run[seed, "periodic_decay"]
        before = [by_run[seed, arm]["eval_before"] for arm in arms]
        loop = {arm: by_generation[seed, arm, horizon]["loop_score"] for arm in arms}
        values = [
            *before,
            *loop.values(),
            *(by_run[seed, arm]["eval_after"] for arm in arms),
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("nonfinite study metric")
        if (
            max(before) - min(before)
            > spec["acceptance"]["before_train_eval_parity_tolerance"]
        ):
            raise ValueError("initial held-out loss parity failed")
        paired.append(
            {
                "seed": seed,
                "baseline_ce_delta": base["eval_after"] - base["eval_before"],
                "decay_minus_baseline_ce": decay["eval_after"] - base["eval_after"],
                "decay_minus_constant_loop": loop["periodic_decay"]
                - loop["periodic_constant"],
                "decay_minus_baseline_loop": loop["periodic_decay"]
                - loop["ordinary_ft"],
            }
        )
    means = {
        key: statistics.mean(row[key] for row in paired)
        for key in paired[0]
        if key != "seed"
    }
    limits = spec["acceptance"]
    gates = {
        "baseline_learning": (
            all(row["baseline_ce_delta"] < 0 for row in paired)
            and means["baseline_ce_delta"] <= limits["mean_baseline_ce_delta_at_most"]
        ),
        "decay_vs_constant": (
            means["decay_minus_constant_loop"]
            < limits["decay_minus_constant_mean_final_loop_score_less_than"]
            and sum(row["decay_minus_constant_loop"] < 0 for row in paired)
            >= limits["decay_vs_constant_minimum_seed_wins"]
        ),
        "decay_vs_ordinary_ft": (
            means["decay_minus_baseline_loop"]
            <= limits["decay_minus_baseline_mean_final_loop_score_at_most"]
            and sum(row["decay_minus_baseline_loop"] <= 0 for row in paired)
            >= limits["decay_vs_baseline_minimum_seed_nonlosses"]
        ),
        "heldout_ce_safety": (
            means["decay_minus_baseline_ce"]
            <= limits["decay_minus_baseline_mean_ce_at_most"]
            and max(row["decay_minus_baseline_ce"] for row in paired)
            <= limits["decay_minus_baseline_maximum_ce_at_most"]
        ),
    }
    return {
        "decision": "passed_bounded_gate"
        if all(gates.values())
        else "ready_but_negative",
        "gates": gates,
        "paired_final_effects": paired,
        "means": means,
        "boundary": "Fixed final endpoint only. No statistical significance, cross-book generalization, or general language-quality superiority is established.",
    }


def checked_json(path, expected):
    if study.digest(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path.name}")
    return study.read_json(path)


def generation_entry(report, spec, protocol, seed, arm, step, checkpoint, tokenizer):
    import spiraltorch as st

    study.validate_generation(report, spec, protocol)
    row = report["runs"][0]
    evidence = row["generation_evidence"]
    request = evidence["request"]
    if (
        request["model_artifact_id"]
        != st.hf_adapter_fingerprint(checkpoint)["adapter_id"]
    ):
        raise ValueError("generation evidence refers to different adapter weights")
    prompts = study.read_json(study.REPO / spec["generation"]["prompt_path"])
    expected_prompts = {row["label"]: row["text"] for row in prompts["prompts"]}
    if request["prompt_set_id"] != prompts["prompt_set_id"]:
        raise ValueError("generation prompt identity differs from frozen set")
    if (
        report["seed"] != seed
        or report["do_sample"] is not False
        or report["max_new_tokens"] != spec["generation"]["max_new_tokens"]
        or row["kind"] != "baseline"
    ):
        raise ValueError("generation setting differs from prespecified decoding")
    samples = {sample["prompt_id"]: sample for sample in request["samples"]}
    texts = []
    for item in row["generations"]:
        generated = item["generation"]
        sample = samples[item["prompt_id"]]
        if sample["seed"] != seed:
            raise ValueError("generation evidence has a different seed")
        prompt = expected_prompts[item["prompt_label"]]
        if prompt != generated["prompt"] or generated["generation_control"] is not None:
            raise ValueError("unexpected prompt or inference intervention")
        tokens = sample["continuation_token_ids"]
        text = tokenizer.decode(
            tokenizer(prompt)["input_ids"] + tokens, skip_special_tokens=True
        )
        continuation = text[len(prompt) :] if text.startswith(prompt) else text
        if (
            generated["generated_text"] != text
            or generated["generated_continuation_text"] != continuation
            or generated["generated_continuation_sha256"]
            != hashlib.sha256(continuation.encode()).hexdigest()
            or generated["new_token_count"] != len(tokens)
        ):
            raise ValueError(
                "published text disagrees with committed continuation tokens"
            )
        texts.append(
            {
                "prompt_label": item["prompt_label"],
                "prompt": prompt,
                "prompt_id": item["prompt_id"],
                "text": continuation,
                "token_count": len(tokens),
            }
        )
    exact_coordinates(
        texts, {(label,) for label in expected_prompts}, ("prompt_label",)
    )
    aggregate = evidence["aggregate"]
    return {
        "seed": seed,
        "arm": arm,
        "step": step,
        "loop_score": aggregate["sample_mean_loop_score"],
        "mean_generated_tokens": aggregate["total_token_count"]
        / aggregate["sample_count"],
        "minimum_generated_tokens": aggregate["minimum_token_count"],
        "maximum_generated_tokens": aggregate["maximum_token_count"],
        "evidence": evidence,
        "texts": texts,
    }


def assess(root, spec_path):
    spec = study.read_json(spec_path)
    protocol = "sha256:" + study.digest(spec_path)
    seal = study.read_json(root / "sealed-plan.json")
    completed = study.read_json(root / "completed.json")
    if (
        seal["protocol_id"] != protocol
        or seal["spec"] != spec
        or completed["protocol_id"] != protocol
        or completed["status"] != "execution_complete_pending_matched_assessment"
    ):
        raise ValueError("study is incomplete or protocol differs")
    expected = {(seed, arm["name"]) for seed, arm in study.coordinates(spec)}
    exact_coordinates(completed["runs"], expected, ("seed", "arm"))
    study.verify_sources(seal["source_hashes"])
    from transformers import AutoTokenizer

    training, generations = [], []
    identities = {}
    decoding = {}
    for verified in completed["runs"]:
        seed, arm = verified["seed"], verified["arm"]
        directory = root / f"seed-{seed}-{arm}"
        if study.read_json(directory / "verified.json") != verified:
            raise ValueError("per-run verification stamp differs")
        card = checked_json(directory / "run-card.json", verified["run_card_sha256"])
        study.validate_training(
            card,
            spec,
            seal["preflight"][str(seed)]["identity_id"],
            {"name": arm},
            directory,
        )
        runtime = card["model_runtime_identity_after_model"]["observed_identity_id"]
        if runtime != identities.setdefault(seed, runtime):
            raise ValueError("paired runtime identity differs")
        receipt = card.get("zspace_repetition_unlikelihood_receipt")
        training.append(
            {
                "seed": seed,
                "arm": arm,
                "run_card_sha256": verified["run_card_sha256"],
                "eval_before": card["eval_before_train"]["eval_loss"],
                "eval_after": card["eval_after_train"]["eval_loss"],
                "dataset_id": seal["preflight"][str(seed)]["identity_id"],
                "training_recipe_id": card["training_recipe_identity"][
                    "observed_identity_id"
                ],
                "objective_receipt": receipt,
            }
        )
        tokenizer = AutoTokenizer.from_pretrained(
            card["tokenizer_name"], local_files_only=True
        )
        for step in spec["milestones"]:
            report = checked_json(
                directory / f"generation-{step}.json",
                verified["generation_sha256"][str(step)],
            )
            entry = generation_entry(
                report,
                spec,
                protocol,
                seed,
                arm,
                step,
                directory / f"checkpoint-{step}",
                tokenizer,
            )
            request = entry["evidence"]["request"]
            if request["decoding_config_id"] != decoding.setdefault(
                seed, request["decoding_config_id"]
            ):
                raise ValueError("paired decoding identity differs")
            entry["generation_report_sha256"] = verified["generation_sha256"][str(step)]
            generations.append(entry)
    return {
        "schema": "spiraltorch.hf_repetition_schedule_assessment.v1",
        "protocol_id": protocol,
        "source_commit": seal["source_commit"],
        "sealed_plan_sha256": study.digest(root / "sealed-plan.json"),
        "completion_record_sha256": study.digest(root / "completed.json"),
        "versions": seal["versions"],
        "spec": spec,
        "training": training,
        "generations": generations,
        "assessment": paired_assessment(spec, training, generations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--spec", type=Path, default=study.DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.study, args.spec)
    study.write_new(args.output, result)
    print(result["assessment"]["decision"])


if __name__ == "__main__":
    main()
