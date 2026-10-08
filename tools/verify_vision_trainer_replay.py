#!/usr/bin/env python3
"""Recheck retained real-image trainer runs without importing ML runtimes."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


def verify_recipe_contract(runner, recipe, seed, contract):
    """Bind public execution conditions to the retained worker contract."""
    require = runner.require
    data, config = contract["data"], contract["config"]
    require(contract["environment"]["torch_device"] == recipe["torch_device"],
            "recipe torch_device differs from raw contract")
    require(config["model_seed"] == config["shuffle_seed"] == contract["augmentation_seed"] == str(seed),
            "recipe seed differs from raw contract")
    require(config["batch_size"] == recipe["batch_size"], "recipe batch_size differs from raw contract")
    for split in ("train", "test"):
        per_class = recipe[f"{split}_per_class"]
        ids = data[f"{split}_indices"]
        require(type(per_class) is int and per_class > 0
                and data[f"{split}_per_class"] == per_class
                and len(ids) == len(set(ids)) == per_class * 10,
                f"recipe {split}_per_class differs from raw contract")
    require(type(recipe["horizontal_flip"]) is bool
            and contract["horizontal_flip"] is recipe["horizontal_flip"]
            and data["augmentation"] == ("rust_horizontal_flip_0.5" if recipe["horizontal_flip"] else "none"),
            "recipe augmentation differs from raw contract")
    schedule = recipe["schedule"]
    require(schedule in ("constant", "cosine") and contract["schedule"] == schedule,
            "recipe schedule differs from raw contract")
    total = recipe["epochs"] * recipe["train_per_class"] * 10 // recipe["batch_size"]
    expected = (dict(kind="constant", rate=recipe["rate"]) if schedule == "constant" else
                dict(kind="warmup_cosine", state=dict(base_lr=recipe["rate"], min_lr=recipe["rate"] / 10,
                     warmup_steps=min(10, total), total_steps=total, step=0)))
    require(config["learning_rate"] == expected, "recipe learning rate differs from raw contract")
    control = contract.get("intervention")
    scale, feedback = recipe.get("control_scale"), recipe.get("optimizer_feedback", False)
    require(type(feedback) is bool, "invalid feedback recipe")
    width = recipe.get("feedback_window_observations", 1)
    require(type(width) is int and 1 <= width < 2 ** 53, "invalid feedback window recipe")
    if feedback:
        require(config.get("optimizer_feedback", {}).get("loss_window_observations", 1) == width,
                "recipe feedback window differs from raw contract")
    else:
        require(width == 1, "feedback window requires feedback")
    if scale is None:
        require(control is None and not feedback and "optimizer_feedback" not in config,
                "unexpected optimizer intervention")
    else:
        require(0 < scale < 1 and control is not None and control["requested_scale"] == scale
                and control["feedback_enabled"] is feedback, "recipe intervention differs from raw contract")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--source-ref", help="check measured source hashes against this Git revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = Path(__file__).with_name("run_vision_trainer_matched_learning.py")
    spec = importlib.util.spec_from_file_location("trainer_replay", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    directory = args.run_directory
    summary = json.loads((directory / "summary.json").read_text())
    runner.require(summary["schema"] == "spiraltorch.vision.trainer_matched_learning.v1"
                   and summary["status"] == "passed", "run did not complete successfully")
    recipe, runs = summary["recipe"], summary["runs"]
    runner.require([r["seed"] for r in runs] == recipe["seeds"]
                   and len(set(recipe["seeds"])) == len(recipe["seeds"]), "seed coverage differs")
    hashes = runner.source_hashes()
    if args.source_ref:
        hashes = {name: runner.sha(subprocess.check_output(
            ["git", "show", f"{args.source_ref}:tools/{name}"], cwd=source.parent.parent)) for name in hashes}
    total = recipe["epochs"] * recipe["train_per_class"] * 10 // recipe["batch_size"]
    reports = []
    for run in runs:
        root = directory / f"seed-{run['seed']}"
        phases = {}
        for phase in ("control", "prefix", "resume"):
            file = root / f"{phase}.json"
            runner.require(runner.receipt(file) == run["phase_receipts"][phase], "phase record fixity differs")
            phases[phase] = json.loads(file.read_text())
        control = phases["control"]
        runner.require(control["contract"] == run["contract"] and run["contract"]["source_sha256"] == hashes,
                       "source/runtime/input contract differs")
        verify_recipe_contract(runner, recipe, run["seed"], control["contract"])
        for key in ("admission", "initial_evaluation", "epochs", "final_parameter_comparison", "reference_checkpoint"):
            runner.require(control[key] == run[key], f"published {key} differs from raw control")
        runner.require(run["checkpoints"] == {phase: p["checkpoints"] for phase, p in phases.items()},
                       "checkpoint manifest differs")
        replay = runner.verify_replay(root, phases, total, recipe["restart_at"],
                                      run["contract"]["data"]["train_indices"], recipe["batch_size"])
        runner.require(replay == run["replay"], "computed replay differs")
        runner.require(len(run["epochs"]) == recipe["epochs"], "epoch coverage differs")
        flips = sum(sum(row["flips"]) for row in control["records"])
        samples = total * recipe["batch_size"]
        rates = {row["rate_bits"] for row in control["records"]}
        runner.require(0 < flips < samples if recipe["horizontal_flip"] else flips == 0,
                       "augmentation was absent/unexpected")
        if control["contract"].get("intervention") is None:
            runner.require(len(rates) > 1 if recipe["schedule"] == "cosine"
                           else rates == {runner.bits(recipe["rate"])}, "rate control was absent/unexpected")
        evaluations = [run["initial_evaluation"], *[epoch["evaluation"] for epoch in run["epochs"]]]
        reports.append(dict(seed=run["seed"], replay=replay, flipped_images=flips, observed_images=samples,
            unique_learning_rates=len(rates),
            all_evaluation_accuracies_equal=all(e["spiraltorch"]["accuracy"] == e["torch"]["accuracy"] for e in evaluations),
            max_evaluation_ce_difference=max(abs(e["spiraltorch"]["loss"] - e["torch"]["loss"]) for e in evaluations)))
    result = dict(schema="spiraltorch.vision.trainer_realdata_verification.v1", status="passed",
                  scope="saved records/checkpoint/reference-weight verification, not a new training run",
                  summary=runner.receipt(directory / "summary.json"), runs=reports)
    if args.output:
        runner.write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
