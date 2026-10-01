#!/usr/bin/env python3
"""Recheck saved frozen-model inputs/metrics and replay the canonical Rust gate.

This does not rerun model inference or infer unrecorded parameter behavior.
Use the probe command for a new frozen-model inference measurement.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stationarity", HERE / "probe_vision_feedback_stationarity.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
runner, require = probe.runner, probe.require


def verify_rows(rows, originals, train_ids, per_epoch):
    require(len(rows) == len(originals) and rows, "frozen record coverage differs")
    for index, (row, original) in enumerate(zip(rows, originals, strict=True), 1):
        require(row["step"] == index and row["sample_ids"] == original["sample_ids"]
                and row["input_sha256"] == original["input_sha256"], "frozen input trajectory differs")
        actual, reference = row["loss"], row["torch_loss"]
        require(math.isfinite(actual) and math.isfinite(reference) and actual > 0 and reference > 0,
                "invalid frozen CE reference")
        require(runner.bits(actual) == row["loss_bits"], "frozen loss bits differ")
        absolute = abs(actual - reference)
        scaled = absolute / (1 + abs(reference))
        require(scaled < 2e-4 and row["parity"] == dict(max_abs_error=absolute,
                max_scaled_error=scaled, values=1), "frozen CE parity differs")
    return probe.epoch_summaries(rows, train_ids, per_epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--probe-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import spiraltorch as st

    saved = json.loads((args.directory / "summary.json").read_text())
    require(saved["status"] == "passed" and saved["schema"] == "spiraltorch.vision.feedback_stationarity.v1"
            and saved["boundary"] == probe.BOUNDARY, "probe incomplete or boundary differs")
    source = subprocess.check_output(["git", "show", f"{args.probe_ref}:tools/probe_vision_feedback_stationarity.py"], cwd=HERE.parent)
    require(saved["probe_sha256"] == runner.sha(source) == runner.sha((HERE / "probe_vision_feedback_stationarity.py").read_bytes()),
            "probe source differs")
    require(saved["source_ablation"] == runner.receipt(args.ablation / "summary.json"), "source ablation differs")
    ablation = json.loads((args.ablation / "summary.json").read_text())
    require(saved["recipe"] == ablation["recipe"], "source recipe differs")
    recipe = saved["recipe"]
    expected = [(seed, checkpoint) for seed in recipe["seeds"] for checkpoint in ("initial", "final")]
    require([(case["seed"], case["checkpoint"]) for case in saved["cases"]] == expected, "probe case coverage differs")
    binary = {p.name: runner.sha(p.read_bytes()) for p in Path(st.__file__).parent.glob("*.so")}
    require(binary and binary == saved["native_binary_sha256"], "native gate binary differs")
    controls = {}
    for seed in recipe["seeds"]:
        root = args.ablation / f"seed-{seed}" / "baseline"
        verified = probe.ablation.verify_arm(root, saved["source_ref"])
        require(verified == json.loads((args.directory / f"seed-{seed}-source-verification.json").read_text()),
                "retained source verification differs")
        controls[seed] = json.loads((root / f"seed-{seed}" / "control.json").read_text())
    cases = []
    for case in saved["cases"]:
        seed, checkpoint = case["seed"], case["checkpoint"]
        raw = json.loads(runner.read_checkpoint(args.directory, case["raw"]))
        require((raw["seed"], raw["checkpoint"]) == (seed, checkpoint)
                and raw["model_unchanged"] is True and case["model_unchanged"] is True
                and case["parameter_updates"] == 0, "frozen case identity differs")
        control = controls[seed]
        require(case["source_checkpoint"] == control["checkpoints"][checkpoint], "source model checkpoint differs")
        per_epoch = recipe["train_per_class"] * 10 // recipe["batch_size"]
        epochs = verify_rows(raw["records"], control["records"], control["contract"]["data"]["train_indices"], per_epoch)
        require(epochs == case["epochs"], "epoch metrics differ")
        require(case["frozen_epoch_mean_spread"] == max(e["mean_loss"] for e in epochs) - min(e["mean_loss"] for e in epochs)
                and case["max_ce_scaled_error"] == max(row["parity"]["max_scaled_error"] for row in raw["records"]),
                "published loss comparison differs")
        shadows = {name: probe.shadow_replay(st, values, recipe["control_scale"]) for name, values in
            probe.stationary_sequences([row["loss"] for row in raw["records"]], per_epoch).items()}
        require(shadows == raw["shadows"] and {name: value["summary"] for name, value in shadows.items()} == case["shadows"],
                "Rust policy replay differs")
        cases.append(dict(seed=seed, checkpoint=checkpoint, records=len(raw["records"]),
                          shadow_streams=len(shadows), gate_states_exact=True, retained_inputs_exact=True))
    result = dict(schema="spiraltorch.vision.feedback_stationarity_verification.v1", status="passed",
        scope="Retained input/metric verification and native Rust gate replay, not a new inference measurement.",
        summary=runner.receipt(args.directory / "summary.json"), probe_ref=args.probe_ref, cases=cases)
    runner.write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
