#!/usr/bin/env python3
"""Recheck retained real-image trainer runs without importing ML runtimes."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


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
