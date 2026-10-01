#!/usr/bin/env python3
"""Matched real-image loss-gate ablation; not a geometric optimizer or speed test.

All learning and policy transitions use the existing Rust-owned trainer. The
fourth arm is a retrospective constant-rate mechanism control, not a deployable
policy: its integrated rate matches the realized feedback arm up to f32 rounding.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("trainer_matched", HERE / "run_vision_trainer_matched_learning.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
require = runner.require
ARMS = ("baseline", "fixed_proposal", "loss_feedback", "dose_matched")
BOUNDARY = ("Prescribed rate proposal plus existing Rust loss gate on a shared small ConvNeXt. "
            "Retrospective integrated-rate control; no geometric-update, full producer, speed or untouched-test claim.")


def dose_rate(records):
    require(records and all(row["accepted"] is True for row in records), "dose requires accepted updates")
    rates = [runner.unbits(row["rate_bits"]) for row in records]
    require(all(math.isfinite(rate) and rate > 0 for rate in rates), "invalid observed dose")
    return runner.unbits(runner.bits(math.fsum(rates) / len(rates)))


def rate_summary(records):
    rates = [runner.unbits(row["rate_bits"]) for row in records]
    return dict(updates=len(rates), total=math.fsum(rates), mean=math.fsum(rates) / len(rates),
                minimum=min(rates), maximum=max(rates), unique=len(set(row["rate_bits"] for row in records)))


def verify_arm(directory, source_ref=None):
    command = [sys.executable, "-I", str(HERE / "verify_vision_trainer_replay.py"), str(directory)]
    if source_ref:
        command.extend(("--source-ref", source_ref))
    checked = subprocess.run(command, capture_output=True, text=True)
    require(checked.returncode == 0, f"arm replay verification failed: {checked.stderr[-4000:]}")
    return json.loads(checked.stdout)


def compare_seed(directory, recipe, seed):
    summaries, runs, controls, initials = {}, {}, {}, {}
    for arm in ARMS:
        root = directory / arm
        summary = json.loads((root / "summary.json").read_text())
        require(summary["status"] == "passed" and len(summary["runs"]) == 1, "arm incomplete")
        run = summary["runs"][0]
        require(run["seed"] == seed and summary["recipe"]["seeds"] == [seed], "arm seed differs")
        raw = root / f"seed-{seed}"
        control = json.loads((raw / "control.json").read_text())
        require(run["contract"] == control["contract"], "arm contract differs")
        for key in ("epochs", "initial_evaluation"):
            require(run[key] == control[key], "arm scores differ from raw record")
        require(run["replay"]["all_batch_records_exact"] and run["replay"]["all_bound_checkpoints_exact"],
                "arm restart incomplete")
        initials[arm] = json.loads(runner.read_checkpoint(raw, control["checkpoints"]["initial"]))
        summaries[arm], runs[arm], controls[arm] = summary, run, control
    target_rate = dose_rate(controls["loss_feedback"]["records"])
    baseline = runs["baseline"]["contract"]
    expected_common = {key: recipe[key] for key in (
        "torch_device", "train_per_class", "test_per_class", "batch_size", "epochs", "restart_at")}
    rows = {}
    batch_keys = ("revision", "epoch", "accepted", "sample_ids", "flips", "input_sha256")
    baseline_batches = [{key: row[key] for key in batch_keys} for row in controls["baseline"]["records"]]
    for arm in ARMS:
        contract, arm_recipe = runs[arm]["contract"], summaries[arm]["recipe"]
        rate = target_rate if arm == "dose_matched" else recipe["rate"]
        scale = recipe["control_scale"] if arm in ("fixed_proposal", "loss_feedback") else None
        require({key: arm_recipe[key] for key in expected_common} == expected_common
                and arm_recipe["rate"] == rate and arm_recipe["control_scale"] == scale
                and arm_recipe["optimizer_feedback"] is (arm == "loss_feedback")
                and arm_recipe["schedule"] == "constant" and not arm_recipe["horizontal_flip"],
                "arm recipe differs from declared ablation")
        for key in ("data", "dataset_sha256", "adapter", "source_sha256", "native_binary_sha256", "environment"):
            require(contract[key] == baseline[key], f"unmatched arm {key}")
        require(initials[arm]["model"] == initials["baseline"]["model"], "initial model differs")
        require(initials[arm]["input"] == initials["baseline"]["input"], "initial input state differs")
        require(runs[arm]["initial_evaluation"] == runs["baseline"]["initial_evaluation"], "initial scores differ")
        records = controls[arm]["records"]
        require([{key: row[key] for key in batch_keys} for row in records] == baseline_batches,
                "arm batches/acceptance differ")
        rates = rate_summary(records)
        evaluations = [runs[arm]["initial_evaluation"], *[epoch["evaluation"] for epoch in runs[arm]["epochs"]]]
        rows[arm] = dict(rates=rates, epochs=runs[arm]["epochs"],
                         replay=runs[arm]["replay"],
                         all_evaluation_accuracies_equal=all(e["spiraltorch"]["accuracy"] == e["torch"]["accuracy"]
                                                           for e in evaluations),
                         max_evaluation_ce_difference=max(abs(e["spiraltorch"]["loss"] - e["torch"]["loss"])
                                                          for e in evaluations),
                         summary=runner.receipt(directory / arm / "summary.json"))
    feedback_records = controls["loss_feedback"]["records"]
    feedback_states = [row["optimizer_feedback"]["state"] for row in feedback_records]
    difference = rows["dose_matched"]["rates"]["total"] - rows["loss_feedback"]["rates"]["total"]
    require(abs(difference) <= rows["loss_feedback"]["rates"]["total"] * 2 ** -23,
            "integrated rates do not match within f32 rounding")
    require(all(row["rate_bits"] == runner.bits(target_rate) for row in controls["dose_matched"]["records"]),
            "dose-matched control is not constant")
    return dict(seed=seed, arms=rows, identical_initial_model_and_inputs=True,
                identical_batch_trajectories=True, dose_matched_rate=target_rate,
                integrated_rate_difference=difference,
                feedback=dict(open_observations=sum(s["gate"] > 0 for s in feedback_states),
                              halted_observations=sum(s["halted"] for s in feedback_states),
                              maximum_gate=max(s["gate"] for s in feedback_states),
                              changed_rate_updates=sum(a["rate_bits"] != b["rate_bits"] for a, b in
                                  zip(feedback_records, controls["baseline"]["records"], strict=True))))


def run_arm(args, directory, seed, arm, rate):
    command = [sys.executable, "-I", str(HERE / "run_vision_trainer_matched_learning.py"),
               "--data-root", str(args.data_root.resolve()), "--output", str(directory), "--seeds", str(seed)]
    for key in ("torch_device", "train_per_class", "test_per_class", "batch_size", "epochs", "restart_at"):
        command.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    command.extend(("--rate", repr(rate)))
    if arm in ("fixed_proposal", "loss_feedback"):
        command.extend(("--control-scale", repr(args.control_scale)))
    if arm == "loss_feedback":
        command.append("--optimizer-feedback")
    subprocess.run(command, check=True)
    runner.write_json(directory / "verification.json", verify_arm(directory))


def verify_saved(args):
    directory = args.verify
    saved = json.loads((directory / "summary.json").read_text())
    require(saved["schema"] == "spiraltorch.vision.feedback_ablation.v1" and saved["status"] == "passed",
            "ablation incomplete")
    recipe = saved["recipe"]
    require(recipe["seeds"] and len(set(recipe["seeds"])) == len(recipe["seeds"]), "invalid seed coverage")
    runs = []
    for seed in recipe["seeds"]:
        root = directory / f"seed-{seed}"
        for arm in ARMS:
            recomputed = verify_arm(root / arm, args.source_ref)
            require(recomputed == json.loads((root / arm / "verification.json").read_text()),
                    "saved arm verification differs")
        runs.append(compare_seed(root, recipe, seed))
    require(runs == saved["runs"], "published ablation differs from retained evidence")
    result = dict(schema="spiraltorch.vision.feedback_ablation_verification.v1", status="passed",
                  boundary=BOUNDARY, summary=runner.receipt(directory / "summary.json"), seeds=recipe["seeds"])
    runner.write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--test-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rate", type=float, default=0.01)
    parser.add_argument("--control-scale", type=float, default=0.5)
    parser.add_argument("--restart-at", type=int, default=37)
    parser.add_argument("--verify", type=Path, help="recheck a retained run without ML imports; output is a new JSON file")
    parser.add_argument("--source-ref", help="measured runner revision for --verify")
    args = parser.parse_args()
    if args.verify:
        return verify_saved(args)
    if args.data_root is None or not 0 < args.control_scale < 1:
        parser.error("data-root and a proposal scale strictly between zero and one are required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = {key: value for key, value in vars(args).items() if key not in ("data_root", "output", "verify", "source_ref")}
    result = dict(schema="spiraltorch.vision.feedback_ablation.v1", status="error", boundary=BOUNDARY,
                  recipe=recipe, orchestrator_sha256=runner.sha(Path(__file__).read_bytes()), runs=[])
    try:
        for seed in args.seeds:
            directory = args.output / f"seed-{seed}"
            directory.mkdir()
            for arm in ARMS:
                rate = args.rate
                if arm == "dose_matched":
                    feedback = json.loads((directory / "loss_feedback" / f"seed-{seed}" / "control.json").read_text())
                    rate = dose_rate(feedback["records"])
                run_arm(args, directory / arm, seed, arm, rate)
            result["runs"].append(compare_seed(directory, recipe, seed))
            print(json.dumps(dict(seed=seed, completed_arms=list(ARMS), feedback=result["runs"][-1]["feedback"])), flush=True)
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    runner.write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status=result["status"], completed_seeds=len(result["runs"]), error=result.get("error"))), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
