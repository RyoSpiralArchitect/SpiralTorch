#!/usr/bin/env python3
"""Execute the sealed CPU study through existing HF training/generation clients.

This is orchestration, not another implementation of objective or metric math.
Outputs are exclusive: failed/partial attempts require explicit investigation,
not automatic restart, skipped coordinates or a shorter training horizon.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "bindings/st-py/examples"
DEFAULT_SPEC = (
    REPO / "docs/benchmarks/hf_repetition_schedule_aligned_256step_prespec_20261002.json"
)
LABEL_ALIGNMENT = {
    "rule": "mask_unpredicted_first_label",
    "ignore_index": -100,
    "base_class": "transformers.DataCollatorForLanguageModeling",
}


def validate_spec_alignment(spec):
    if (
        spec.get("schema") != "spiraltorch.hf_repetition_schedule_study.v2"
        or spec.get("causal_label_alignment") != LABEL_ALIGNMENT
        or "--causal-lm-mask-first-label" not in spec["training_args"]
    ):
        raise ValueError("matched study requires the aligned v2 protocol")


def client_command(script, arguments):
    # Examples prepend the source tree. Import the tested wheel first so an
    # unrelated in-tree native build cannot silently replace the study runtime.
    bootstrap = (
        "import runpy,sys; import spiraltorch; "
        "script=sys.argv.pop(1); runpy.run_path(script,run_name='__main__')"
    )
    return [sys.executable, "-I", "-c", bootstrap, str(script), *arguments]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def coordinates(spec):
    arms = spec["arms"]
    for index, seed in enumerate(spec["seeds"]):
        for arm in arms[index:] + arms[:index]:
            yield seed, arm


def train_args(spec, model, corpus, directory, seed, arm):
    return [
        "--model-name",
        str(model),
        "--tokenizer-name",
        str(model),
        "--train-file",
        str(corpus),
        "--output-dir",
        str(directory),
        "--run-card",
        str(directory / "run-card.json"),
        "--seed",
        str(seed),
        *spec["training_args"],
        *arm["args"],
    ]


def generation_command(spec, model, directory, seed, step, protocol_id):
    return client_command(
        EXAMPLES / "hf_zspace_generation_control_sweep.py",
        [
            "--model-name",
            str(directory / f"checkpoint-{step}"),
            "--model-artifact-kind",
            "peft-adapter",
            "--tokenizer-name",
            str(model),
            "--prompt-set",
            str(REPO / spec["generation"]["prompt_path"]),
            "--baseline-only",
            "--no-do-sample",
            "--seed",
            str(seed),
            "--max-new-tokens",
            str(spec["generation"]["max_new_tokens"]),
            "--generation-evidence-protocol-id",
            protocol_id,
            "--out",
            str(directory / f"generation-{step}.json"),
        ],
    )


def require_uniform_labels(dataset, collator, shifted_count):
    if dataset is None or not len(dataset):
        raise ValueError("empty train/eval dataset")
    for index, row in enumerate(dataset):
        labels = collator([row])["labels"]
        if labels.shape != (1, shifted_count + 1):
            raise ValueError(f"row {index} is not fixed length")
        if int((labels[:, 1:] != -100).sum()) != shifted_count:
            raise ValueError(f"row {index} has unequal causal label count")
        if int((labels != -100).sum()) != shifted_count:
            raise ValueError(f"row {index} includes an unpredicted first label")


def dataset_preflight(spec, model, corpus, output):
    import datasets
    import spiraltorch as st
    import torch
    import transformers

    validate_spec_alignment(spec)
    if torch.get_default_device().type != "cpu":
        raise ValueError("disable automatic device patches before Python startup")
    sys.path.insert(0, str(EXAMPLES))
    bridge = importlib.import_module("hf_gpt2_finetune_bridge")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    collator = st.HfCausalLabelAlignmentCollator(
        transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    result = {}
    for seed in spec["seeds"]:
        args = bridge.parse_args(
            train_args(
                spec,
                model,
                corpus,
                output / f"preflight-{seed}",
                seed,
                spec["arms"][0],
            )
        )
        raw_train, raw_eval, _ = bridge._load_raw_datasets(datasets, args)
        train = bridge._tokenize_dataset(raw_train, tokenizer, args)
        evaluation = bridge._tokenize_dataset(raw_eval, tokenizer, args)
        for split in (train, evaluation):
            require_uniform_labels(split, collator, args.block_size - 1)
        identity = bridge._tokenized_dataset_identity_report(args, train, evaluation)
        if identity["status"] != "ready":
            raise ValueError("tokenized dataset identity is not ready")
        result[str(seed)] = {
            "identity_id": identity["observed_identity_id"],
            "train_rows": len(train),
            "eval_rows": len(evaluation),
            "shifted_labels_per_row": args.block_size - 1,
            "trainer_counted_labels_per_row": args.block_size - 1,
            "causal_label_alignment": LABEL_ALIGNMENT,
        }
    return result


def validate_training(card, spec, expected_dataset, arm, directory):
    if card.get("failure_stage") or card.get("failure_error"):
        raise ValueError("training run card reports failure")
    if card.get("adapter_saved") is not True or card.get("model_saved") is not True:
        raise ValueError("training did not save its adapter")
    if card["tokenized_dataset_identity"]["observed_identity_id"] != expected_dataset:
        raise ValueError("tokenized dataset changed after preflight")
    identity = card["training_recipe_identity"]["identity_payload"]
    collator = identity.get("trainer_contract", {}).get("data_collator", {})
    expected_class = (
        "spiraltorch.HfCausalLabelAlignmentCollator"
        if arm["name"] == "ordinary_ft"
        else "spiraltorch.HfRepetitionUnlikelihoodCollator"
    )
    if (
        collator.get("causal_label_alignment") != LABEL_ALIGNMENT
        or collator.get("class") != expected_class
        or collator.get("base_class") != "spiraltorch.HfCausalLabelAlignmentCollator"
        or collator.get("mlm") is not False
    ):
        raise ValueError("training omitted the shared causal label alignment")
    training = identity["training_arguments"]
    if (
        training["use_cpu"] is not True
        or training["world_size"] != 1
        or training["per_device_train_batch_size"] != 1
        or training["gradient_accumulation_steps"] != 16
    ):
        raise ValueError("device or batch partition changed")
    horizon = spec["acceptance"]["final_update_slot"]
    lineage = card["trainer_trace_lineage"]
    if lineage["ready"] is not True or lineage["trace_last_global_step"] != horizon:
        raise ValueError("training horizon/trace incomplete")
    for key in ("eval_before_train", "eval_after_train"):
        if card[key]["status"] != "ok" or not math.isfinite(card[key]["eval_loss"]):
            raise ValueError("missing or nonfinite held-out loss")
    for step in spec["milestones"]:
        checkpoint = directory / f"checkpoint-{step}"
        if read_json(checkpoint / "trainer_state.json")["global_step"] != step:
            raise ValueError("checkpoint step mismatch")
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise ValueError("missing checkpoint adapter weights")
    if arm["name"] != "ordinary_ft":
        receipt = card["zspace_repetition_unlikelihood_receipt"]
        control = receipt["last_objective_control"]
        if receipt["periodic_candidate_count"] <= 0:
            raise ValueError("treatment did not exercise its mechanism")
        if control["completed_update_slots"] != horizon - 1:
            raise ValueError("objective clock did not reach final slot")
        expected_schedule = (
            {"kind": "constant"}
            if arm["name"] == "periodic_constant"
            else {
                "kind": "linear_decay",
                "start_update": 0,
                "end_update": horizon,
                "final_scale": 0.0,
            }
        )
        policy = control["policy"]
        if policy["base_strength"] != 0.1 or policy["config"] != {
            "normalization": "active_positions",
            "schedule": expected_schedule,
        }:
            raise ValueError("unexpected objective policy")
        if not math.isfinite(receipt["mean_weighted_auxiliary_loss"]):
            raise ValueError("nonfinite auxiliary objective")


def validate_generation(report, spec, protocol_id):
    import spiraltorch as st

    if report["status"] != "complete" or len(report["runs"]) != 1:
        raise ValueError("generation did not complete the baseline-only condition")
    evidence = report["runs"][0]["generation_evidence"]
    st.validate_zspace_generation_evidence(evidence)
    if evidence["request"]["protocol_id"] != protocol_id:
        raise ValueError("generation protocol changed")
    aggregate = evidence["aggregate"]
    if aggregate["sample_count"] != spec["generation"]["prompt_count"]:
        raise ValueError("incomplete prompt set")
    if aggregate["empty_sample_count"] != 0:
        raise ValueError("empty generation sample")


def fingerprint_sources(model, corpus, spec_path):
    import spiraltorch as st

    paths = [Path(__file__), spec_path, corpus, Path(st._rs.__file__)]
    paths.extend((REPO / "bindings/st-py").rglob("*.py"))
    paths.extend(Path(st.__file__).parent.rglob("*.py"))
    paths.extend(path for path in model.iterdir() if path.is_file())
    return {str(path.resolve()): digest(path) for path in sorted(set(paths))}


def verify_sources(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"source/input changed during the study: {path}")


def execute(command, log_path, output):
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise RuntimeError("less than 8 GiB free; preserve partial outputs")
    print(f"START {log_path.name}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    print(f"DONE {log_path.name}", flush=True)


def run(args):
    spec = read_json(args.spec)
    validate_spec_alignment(spec)
    if digest(args.corpus) != spec["corpus_sha256"]:
        raise ValueError("wrong corpus bytes")
    if args.model.name != spec["model_revision"]:
        raise ValueError("expected pinned local model snapshot directory")
    if (
        digest(REPO / spec["generation"]["prompt_path"])
        != spec["generation"]["prompt_sha256"]
    ):
        raise ValueError("prompt set changed")
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = fingerprint_sources(args.model, args.corpus, args.spec)
    preflight = dataset_preflight(spec, args.model, args.corpus, args.output)
    protocol_id = "sha256:" + digest(args.spec)
    commands = []
    for seed, arm in coordinates(spec):
        directory = args.output / f"seed-{seed}-{arm['name']}"
        command = client_command(
            EXAMPLES / "hf_finetune_bridge.py",
            train_args(spec, args.model, args.corpus, directory, seed, arm),
        )
        command += [
            "--expected-tokenized-dataset-id",
            preflight[str(seed)]["identity_id"],
        ]
        commands.append(
            {"seed": seed, "arm": arm, "directory": str(directory), "command": command}
        )
    seal = {
        "schema": "spiraltorch.hf_repetition_schedule_execution.v1",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol_id,
        "spec": spec,
        "preflight": preflight,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_hashes": hashes,
        "commands": commands,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "datasets",
                "spiraltorch",
            )
        },
    }
    write_new(args.output / "sealed-plan.json", seal)
    print(f"SEALED {protocol_id}; no training yet", flush=True)
    if args.preflight_only:
        return
    results = []
    before_by_seed = {}
    runtime_by_seed = {}
    for item in commands:
        verify_sources(hashes)
        seed, arm = item["seed"], item["arm"]
        directory = Path(item["directory"])
        directory.mkdir()
        execute(
            item["command"], args.output / f"{directory.name}-train.log", args.output
        )
        card_path = directory / "run-card.json"
        card = read_json(card_path)
        validate_training(
            card, spec, preflight[str(seed)]["identity_id"], arm, directory
        )
        before = card["eval_before_train"]["eval_loss"]
        reference = before_by_seed.setdefault(seed, before)
        if (
            abs(before - reference)
            > spec["acceptance"]["before_train_eval_parity_tolerance"]
        ):
            raise ValueError("paired initial held-out losses differ")
        runtime = card["model_runtime_identity_after_model"]["observed_identity_id"]
        if runtime != runtime_by_seed.setdefault(seed, runtime):
            raise ValueError("paired model/tokenizer runtime identities differ")
        generation_hashes = {}
        for step in spec["milestones"]:
            verify_sources(hashes)
            command = generation_command(
                spec, args.model, directory, seed, step, protocol_id
            )
            execute(
                command,
                args.output / f"{directory.name}-generation-{step}.log",
                args.output,
            )
            path = directory / f"generation-{step}.json"
            validate_generation(read_json(path), spec, protocol_id)
            generation_hashes[str(step)] = digest(path)
        result = {
            "seed": seed,
            "arm": arm["name"],
            "run_card_sha256": digest(card_path),
            "generation_sha256": generation_hashes,
        }
        write_new(directory / "verified.json", result)
        results.append(result)
    write_new(
        args.output / "completed.json",
        {
            "protocol_id": protocol_id,
            "runs": results,
            "status": "execution_complete_pending_matched_assessment",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("spec", "model", "corpus", "output"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONHASHSEED": "0",
            "SPIRALTON_MAGIC": "0",
            "SPIRALTON_TORCH": "0",
            "SPIRALTON_MODEL_PATCHES": "0",
            "SPIRALTON_NUMPY": "0",
        }
    )
    run(args)


if __name__ == "__main__":
    main()
