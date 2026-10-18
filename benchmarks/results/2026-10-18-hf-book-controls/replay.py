#!/usr/bin/env python3
"""Run the declared book controls through the public HF Trainer example only.

All outputs are private until explicitly selected for publication. The child
process group is bounded; a failed case stops the study without retry or tuning.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXAMPLE = ROOT / "bindings/st-py/examples/hf_geometry_finetune.py"
HELPER = HERE.parent / "2026-10-18-hf-affine-control/replay.py"
OWNER = ROOT / "tools/owned_process.py"
PREFLIGHT = runpy.run_path(str(HERE / "preflight.py"))
READ = runpy.run_path(str(HERE / "prepare.py"))["read_bounded"]


def identity(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(path):
    return identity(path.read_bytes())


def load_pinned(path, pin, budget=1_048_576):
    raw = READ(path, budget)
    if identity(raw) != pin:
        raise ValueError("input does not match its independently pinned identity")
    return json.loads(raw)


def cases_for(plan):
    names = [a["name"] for a in plan["arms"]]
    if names != ["ordinary", "wave_gate", "wave_gate_radius1", "wave_gate_radius4"]:
        raise ValueError("this runner implements the four declared controls only")
    if plan["seeds"] != [41, 43, 47]:
        raise ValueError("this runner implements the three declared seeds only")
    defaults = {"gradient_accumulation_steps": 1, "optimizer": "adamw_torch",
                "scheduler": "linear", "warmup_steps": 0, "weight_decay": 0,
                "generation_max_new_tokens": 24, "generation_sampling": False}
    if any(plan[key] != value for key, value in defaults.items()):
        raise ValueError("plan differs from the public example's fixed defaults")
    model_defaults = {"base_frozen": True, "dtype": "float32", "device": "mps",
                      "attention": "eager", "kv_cache": False}
    if any(plan["model"][key] != value for key, value in model_defaults.items()):
        raise ValueError("plan differs from the public example's runtime policy")
    return [{"arm": arm, "seed": seed} for seed in plan["seeds"] for arm in names]


def command_for(plan, preflight, selected, model_dir, artifacts, output, case):
    command = [sys.executable, "-I", "-B", str(EXAMPLE), "--model-dir", str(model_dir),
               "--device", plan["model"]["device"], "--prompt", plan["generation_prompt"],
               "--expected-dataset-id", preflight["tokenized_dataset_identity"],
               "--geometry-artifact", str(artifacts / f"{case['arm']}-initial.json"),
               "--geometry-artifact-id", preflight["artifacts"][case["arm"]]["artifact_id"],
               "--seed", str(case["seed"]), "--output", str(output)]
    for key in ("context_length", "max_steps", "save_steps", "batch_size", "learning_rate", "max_grad_norm"):
        command.extend(("--" + key.replace("_", "-"), str(plan[key])))
    for split, option in (("train", "--train-text"), ("development", "--eval-text")):
        for path in selected[split]:
            command.extend((option, str(path)))
    return command


def validate_result(result, plan, preflight, case):
    expected = {"schema": "spiraltorch.hf_geometry_example.v1", "device": plan["model"]["device"],
                "runtime_identity": preflight["runtime_identity"],
                "tokenized_dataset_identity": preflight["tokenized_dataset_identity"],
                "initial_artifact_id": preflight["artifacts"][case["arm"]]["artifact_id"],
                "initial_layout_id": preflight["artifacts"][case["arm"]]["layout_id"],
                "seed": case["seed"], "step": plan["max_steps"], "max_steps": plan["max_steps"],
                "context_length": plan["context_length"], "batch_size": plan["batch_size"],
                "learning_rate": plan["learning_rate"], "max_grad_norm": plan["max_grad_norm"],
                "base_unchanged": True, "restored_logits_exact": True,
                "train_window_plan": preflight["window_plans"]["train"],
                "eval_window_plan": preflight["window_plans"]["development"],
                "run_config": {"attention": "eager", "dtype": "float32", "cache": False,
                               "optimizer": "adamw_torch", "data_seed": case["seed"]}}
    if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in expected.items()):
        raise ValueError("run did not execute its complete declared contract")
    for key in ("bare_eval_loss", "initial_geometry_eval_loss", "trained_geometry_eval_loss"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]):
            raise ValueError("evaluation loss must be finite")
    if result["bare_eval_loss"] != result["initial_geometry_eval_loss"]:
        raise ValueError("identity-start loss differs from the frozen base")


def state_metrics(initial, final):
    before, after = initial["placements"], final["placements"]
    if len(before) != len(after):
        raise ValueError("adapter placements changed")
    metrics = []
    for a, b in zip(before, after):
        if a["path"] != b["path"] or a["recipe"] != b["recipe"]:
            raise ValueError("adapter path or recipe changed")
        old = {t["name"]: t for t in a["tensors"]}
        new = {t["name"]: t for t in b["tensors"]}
        if old.keys() != new.keys():
            raise ValueError("adapter tensor layout changed")
        deltas, count = [], 0
        for name, left in old.items():
            right = new[name]
            if ({k: v for k, v in left.items() if k != "values"}
                    != {k: v for k, v in right.items() if k != "values"}
                    or len(left["values"]) != len(right["values"])):
                raise ValueError("adapter tensor metadata changed")
            if left["role"] == "parameter" and left["requires_grad"]:
                count += len(left["values"])
                deltas.extend(y - x for x, y in zip(left["values"], right["values"]))
            elif left != right:
                raise ValueError("fixed adapter tensor changed")
        if not deltas or not all(math.isfinite(v) for v in deltas) or not any(deltas):
            raise ValueError("every intended site must have finite learned parameter changes")
        metrics.append({"path": a["path"], "trainable_values": count,
                        "parameter_delta_l2": math.sqrt(math.fsum(v * v for v in deltas)),
                        "parameter_delta_max_abs": max(abs(v) for v in deltas)})
    return metrics


def window_summary(plan):
    return {**{k: v for k, v in plan.items() if k != "windows"}, "windows": len(plan["windows"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "plan.json")
    parser.add_argument("--expected-plan-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-manifest-id", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("study output must be a new private directory")

    def check_inputs():
        return PREFLIGHT["inputs"](args.plan, args.expected_plan_id, args.data, args.expected_manifest_id)
    plan, manifest, selected, _, _ = check_inputs()
    cases = cases_for(plan)
    preflight_path = args.preflight / "preflight.json"
    preflight = load_pinned(preflight_path, args.expected_preflight_id)
    required = {"schema": "spiraltorch.hf_book_preflight.v2", "plan_id": args.expected_plan_id,
                "preparation_manifest_id": args.expected_manifest_id,
                "source_id": manifest["raw_source_id"], "runtime_identity": plan["model"]["runtime_identity"],
                "identity_start_exact": True, "initial_parameter_gradients_exact_within_pairs": True,
                "gradient_pairs": plan["paired_contrasts"], "reserve_opened": False,
                "model_loaded": False, "optimizer_updates": 0}
    if any(type(preflight.get(k)) is not type(v) or preflight[k] != v for k, v in required.items()):
        raise ValueError("preflight is not the successful matching pre-training declaration")
    if any(digest(ROOT / name) != pin for name, pin in preflight["sources"].items()):
        raise ValueError("preflight implementation changed; qualify it before training")

    import spiraltorch as st
    import torch
    import transformers
    import accelerate
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": transformers.__version__, "accelerate": accelerate.__version__}
    if versions != preflight["versions"]:
        raise ValueError("installed runtime changed since preflight")
    initial_states = {}
    for case in cases[:len(plan["arms"])]:
        arm = case["arm"]
        artifact = st.GeometryAdapterArtifact.from_json(
            READ(args.preflight / f"{arm}-initial.json", 1_048_576).decode("utf-8"),
            preflight["artifacts"][arm]["artifact_id"], preflight["runtime_identity"])
        initial_states[arm] = json.loads(artifact.state_json())
    package = Path(st.__file__).parent
    package_files = sorted([*package.glob("*.py"), *package.glob("*.so")])
    package_ids = {p.name: digest(p) for p in package_files}
    source_files = [Path(__file__), HERE / "preflight.py", HERE / "prepare.py", EXAMPLE, HELPER, OWNER]
    source_ids = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "spiraltorch.hf_book_execution.v1", "status": "declared_before_training",
               "plan_id": args.expected_plan_id, "preparation_manifest_id": args.expected_manifest_id,
               "preflight_id": args.expected_preflight_id, "versions": versions, "sources": source_ids,
               "installed_package_ids": package_ids, "case_timeout_seconds": 3600,
               "cases": cases, "reserve_opened": False,
               "window_plans": {k: window_summary(v) for k, v in preflight["window_plans"].items()}}
    (args.output / "execution.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    run_owned = runpy.run_path(str(OWNER))["run"]
    execute_cases = runpy.run_path(str(HELPER))["execute_cases"]

    def unchanged():
        check_inputs()
        load_pinned(preflight_path, args.expected_preflight_id)
        if (any(digest(ROOT / name) != pin for name, pin in source_ids.items())
                or any(digest(package / name) != pin for name, pin in package_ids.items())):
            raise ValueError("study code or installed runtime changed during execution")

    def execute(case):
        unchanged()
        destination = args.output / f"{case['arm']}-seed-{case['seed']}"
        command = command_for(plan, preflight, selected, args.model_dir, args.preflight, destination, case)
        log_path = destination.with_suffix(".log")
        print(json.dumps(dict(case, status="running")), flush=True)
        with log_path.open("w") as log:
            run_owned(command, timeout=3600, stdout=log, cwd=ROOT).check_returncode()
        result_path = destination / "result.json"
        result = json.loads(READ(result_path, 1_048_576))
        validate_result(result, plan, preflight, case)
        final_path = destination / "geometry-export/spiraltorch-geometry.json"
        artifact = st.GeometryAdapterArtifact.from_json(
            READ(final_path, 1_048_576).decode("utf-8"), result["artifact_id"], preflight["runtime_identity"])
        metrics = state_metrics(initial_states[case["arm"]], json.loads(artifact.state_json()))
        if sum(m["trainable_values"] for m in metrics) != plan["trainable_values_per_arm"]:
            raise ValueError("final trainable parameter budget differs")
        unchanged()
        record = {k: result[k] for k in (
            "runtime_identity", "tokenized_dataset_identity", "initial_artifact_id", "initial_layout_id",
            "artifact_id", "checkpoint_artifact_id", "seed", "step", "base_parameter_hash",
            "base_unchanged", "restored_logits_exact", "bare_eval_loss", "initial_geometry_eval_loss",
            "trained_geometry_eval_loss")}
        record.update(arm=case["arm"], learned_sites=metrics, private_result_id=digest(result_path),
                      private_artifact_id=digest(final_path), private_log_id=digest(log_path))
        print(json.dumps({k: record[k] for k in ("arm", "seed", "step", "trained_geometry_eval_loss")}), flush=True)
        return record

    def validate(records):
        unchanged()
        if (len({r["base_parameter_hash"] for r in records}) != 1
                or len({r["bare_eval_loss"] for r in records}) != 1):
            raise ValueError("matched runs did not share the same frozen base result")

    records = execute_cases(args.output / "run-manifest.json", cases, execute, validate)
    pairs = []
    for seed in plan["seeds"]:
        by_arm = {r["arm"]: r for r in records if r["seed"] == seed}
        for left, right in plan["paired_contrasts"]:
            pairs.append({"seed": seed, "contrast": f"{right}_minus_{left}",
                          "development_loss_difference": by_arm[right]["trained_geometry_eval_loss"]
                          - by_arm[left]["trained_geometry_eval_loss"]})
    summary = {"schema": "spiraltorch.hf_book_comparison.v1", "status": "twelve_runs_completed",
               "execution_id": digest(args.output / "execution.json"), "plan_id": args.expected_plan_id,
               "runs": records, "paired_differences": pairs, "reserve_opened": False,
               "claims": {"general_quality_advantage": False, "gpu_speed": False,
                          "promoted_default": False, "pretraining_unseen": False}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
