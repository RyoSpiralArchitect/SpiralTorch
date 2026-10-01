#!/usr/bin/env python3
"""Replay retained frozen-model losses through default and windowed Rust gates.

This compares a new binary with the recorded default trajectories; it does not
relax the original probe's binary-bound verifier or rerun model inference.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_probe", HERE / "probe_vision_feedback_stationarity.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
runner, require = probe.runner, probe.require
BOUNDARY = ("Retained frozen losses replayed through a new Rust binary, plus synthetic regression latency. "
            "No new inference, parameter updates, learning-quality or throughput claim.")


def replay(st, losses, scale, width, restart_at=37):
    require(0 < restart_at < len(losses), "restart must lie inside the observation sequence")
    initial = st.zspace_optimizer_feedback_init({"loss_window_observations": width})
    config, state = initial["config"], initial["state"]
    records, resumed = [], None

    def advance(previous, step, loss):
        control = st.zspace_optimizer_feedback_control(config=config, state=previous,
            target_step=step, proposed_learning_rate_scale=scale)
        observed = st.zspace_optimizer_feedback_observe(config=config, state=control["state_after"],
            observation={"step": step, "loss": loss})
        return dict(step=step, loss=loss, applied_scale=control["applied_learning_rate_scale"],
                    action=observed["action"], relative_loss_delta=observed["relative_loss_delta"],
                    state_after=observed["state_after"])

    for step, loss in enumerate(losses, 1):
        row = advance(state, step, loss)
        if resumed is not None:
            checked = advance(resumed, step, loss)
            require(checked == row, "partial-window restart changed a transition")
            resumed = checked["state_after"]
        state = row["state_after"]
        if step == restart_at:
            saved = json.loads(json.dumps(dict(config=config, state=state), allow_nan=False))
            resumed = st.zspace_optimizer_feedback_restore(**saved)["state"]
            require(resumed == state, "partial-window state changed on restore")
        records.append(row)
    return dict(config=config, records=records, summary=probe.summarize_shadow(records))


def checked_source(directory, verification):
    saved = json.loads((directory / "summary.json").read_text())
    checked = json.loads(verification.read_text())
    require(saved["status"] == "passed" and saved["schema"] == "spiraltorch.vision.feedback_stationarity.v1",
            "frozen probe incomplete")
    require(checked["status"] == "passed"
            and checked["schema"] == "spiraltorch.vision.feedback_stationarity_verification.v1"
            and checked["summary"] == runner.receipt(directory / "summary.json"), "source verification differs")
    recipe = saved["recipe"]
    require(recipe["seeds"] and len(set(recipe["seeds"])) == len(recipe["seeds"]), "invalid source seeds")
    expected = [(seed, phase) for seed in recipe["seeds"] for phase in ("initial", "final")]
    require([(case["seed"], case["checkpoint"]) for case in saved["cases"]] == expected,
            "frozen case coverage differs")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stationarity", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import spiraltorch as st

    saved = checked_source(args.stationarity, args.verification)
    recipe = saved["recipe"]
    samples, batch = recipe["train_per_class"] * 10, recipe["batch_size"]
    require(samples > 0 and batch > 0 and samples % batch == 0, "source has partial batches")
    width = samples // batch
    require(width > 1, "source needs more than one batch per pass")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="spiraltorch.vision.feedback_window_comparison.v1", status="error", boundary=BOUNDARY,
        source=runner.receipt(args.stationarity / "summary.json"),
        source_verification=runner.receipt(args.verification),
        source_binary_sha256=saved["native_binary_sha256"],
        native_binary_sha256={p.name: runner.sha(p.read_bytes()) for p in Path(st.__file__).parent.glob("*.so")},
        implementation_ref=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True).strip(),
        comparator_sha256=runner.sha(Path(__file__).read_bytes()),
        recipe=dict(window_observations=width, derivation="one complete recorded input pass", restart_at=37),
        cases=[], latency=[])
    try:
        require(result["native_binary_sha256"], "native binary missing")
        for case in saved["cases"]:
            raw = json.loads(runner.read_checkpoint(args.stationarity, case["raw"]))
            require(raw["seed"] == case["seed"] and raw["checkpoint"] == case["checkpoint"]
                    and raw["model_unchanged"] is True and case["parameter_updates"] == 0,
                    "frozen case identity differs")
            rows = raw["records"]
            require(len(rows) == width * recipe["epochs"]
                    and all(row["step"] == i and len(row["sample_ids"]) == batch
                            and runner.bits(row["loss"]) == row["loss_bits"] for i, row in enumerate(rows, 1)),
                    "source observation coverage differs")
            ids = [sample for row in rows[:width] for sample in row["sample_ids"]]
            require(len(set(ids)) == samples, "source pass repeats samples")
            require(probe.epoch_summaries(rows, ids, width) == case["epochs"], "source pass means differ")
            outputs = {}
            for name, losses in probe.stationary_sequences([row["loss"] for row in rows], width).items():
                legacy = replay(st, losses, recipe["control_scale"], 1)
                require(legacy == raw["shadows"][name], "new binary changed the recorded default policy")
                outputs[name] = dict(default=legacy, windowed=replay(st, losses, recipe["control_scale"], width))
            file = args.output / case["raw"]["file"]
            runner.write_json(file, outputs)
            result["cases"].append(dict(seed=case["seed"], checkpoint=case["checkpoint"], source_raw=case["raw"],
                default_trajectory_exact=True, all_partial_restarts_exact=True, raw=runner.receipt(file),
                comparisons={name: {arm: stream["summary"] for arm, stream in arms.items()}
                             for name, arms in outputs.items()}))
        for offset in (0, width // 2, width - 1):
            losses = [4.] * width + [2.] * (width + offset) + [8.] * (width * 2)
            onset = width * 2 + offset + 1
            latency = dict(regression_step=onset, offset_inside_window=offset, arms={})
            for name, count in (("default", 1), ("windowed", width)):
                stream = replay(st, losses, recipe["control_scale"], count)
                require(stream["records"][onset - 2]["state_after"]["gate"] > 0,
                        "synthetic regression did not start from an open gate")
                first = next((row["step"] for row in stream["records"][onset - 1:]
                              if row["state_after"]["halted"]), None)
                latency["arms"][name] = dict(first_halted_step=first,
                    additional_observations=None if first is None else first - onset,
                    summary=stream["summary"])
            result["latency"].append(latency)
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    runner.write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
