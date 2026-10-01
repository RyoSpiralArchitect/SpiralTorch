#!/usr/bin/env python3
"""Probe the Rust loss gate with fixed real-image models, never applying control.

Reuses verified initial/final checkpoints and recorded input orders. Every model
parameter remains frozen. Shadow policy responses are not counterfactual training
results or evidence that a different optimizer would improve quality.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("feedback_ablation", HERE / "run_vision_feedback_ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)
runner = ablation.runner
require = runner.require
BOUNDARY = ("Frozen-model inference and unapplied Rust policy replay on recorded training batches. "
            "No parameter updates, counterfactual learning benefit or throughput claim.")


def summarize_shadow(records):
    require(records, "empty shadow replay")
    for step, row in enumerate(records, 1):
        require(row["step"] == step and math.isfinite(row["loss"]), "invalid shadow observation")
        state = row["state_after"]
        require(state["control_step"] == state["observation_count"] == step
                and state["last_observation_step"] == step and state["last_loss"] == row["loss"],
                "shadow core clocks/loss differ")
        require(math.isfinite(state["gate"]) and 0 <= state["gate"] <= 1, "invalid shadow gate")
        require(math.isfinite(row["applied_scale"]) and 0.1 <= row["applied_scale"] <= 1.5,
                "invalid shadow scale")
    return dict(observations=len(records), actions=dict(sorted(Counter(r["action"] for r in records).items())),
                halted_observations=sum(r["state_after"]["halted"] for r in records),
                nonidentity_controls=sum(r["applied_scale"] != 1. for r in records),
                maximum_gate=max(r["state_after"]["gate"] for r in records),
                adjacent_loss_increases=sum(b["loss"] > a["loss"] for a, b in zip(records, records[1:])))


def shadow_replay(st, losses, scale):
    initial = st.zspace_optimizer_feedback_init({})
    config, state = initial["config"], initial["state"]
    records = []
    for step, loss in enumerate(losses, 1):
        control = st.zspace_optimizer_feedback_control(config=config, state=state,
            target_step=step, proposed_learning_rate_scale=scale)
        observed = st.zspace_optimizer_feedback_observe(config=config, state=control["state_after"],
            observation={"step": step, "loss": loss})
        state = observed["state_after"]
        records.append(dict(step=step, loss=loss, applied_scale=control["applied_learning_rate_scale"],
                            action=observed["action"], relative_loss_delta=observed["relative_loss_delta"],
                            state_after=state))
    return dict(config=config, records=records, summary=summarize_shadow(records))


def stationary_sequences(losses, per_epoch):
    require(losses and per_epoch > 0 and len(losses) % per_epoch == 0, "partial/empty frozen epoch")
    require(all(math.isfinite(value) and value > 0 for value in losses), "invalid frozen CE")
    mean = math.fsum(losses[:per_epoch]) / per_epoch
    return {"recorded_order": losses, "reversed_order": list(reversed(losses)),
            "constant_loss": [mean] * len(losses)}


def epoch_summaries(rows, train_ids, per_epoch):
    require(rows and len(rows) % per_epoch == 0, "partial frozen epoch")
    result = []
    for start in range(0, len(rows), per_epoch):
        epoch = rows[start:start + per_epoch]
        require(sorted(i for row in epoch for i in row["sample_ids"]) == sorted(train_ids),
                "frozen epoch does not cover the same samples")
        losses = [row["loss"] for row in epoch]
        require(all(math.isfinite(loss) and loss > 0 for loss in losses), "invalid frozen loss")
        result.append(dict(epoch=start // per_epoch, batches=per_epoch,
            mean_loss=math.fsum(losses) / per_epoch, minimum_loss=min(losses), maximum_loss=max(losses)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--source-ref", required=True, help="revision of the retained ablation")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = json.loads((args.ablation / "summary.json").read_text())
    require(saved["status"] == "passed" and saved["schema"] == "spiraltorch.vision.feedback_ablation.v1",
            "ablation did not complete")
    recipe = saved["recipe"]
    require(recipe["seeds"] and len(set(recipe["seeds"])) == len(recipe["seeds"]), "invalid seed coverage")
    result = dict(schema="spiraltorch.vision.feedback_stationarity.v1", status="error", boundary=BOUNDARY,
        source_ref=args.source_ref, source_ablation=runner.receipt(args.ablation / "summary.json"),
        probe_sha256=runner.sha(Path(__file__).read_bytes()), recipe=recipe, cases=[])
    try:
        runner.load_runtime()
        shared, st, np, torch = runner.shared, runner.st, runner.np, runner.torch
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        if recipe["torch_device"] == "mps":
            require(torch.backends.mps.is_available() and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
                    "MPS requires hardware and disabled fallback")
        data, evidence = shared.dataset(argparse.Namespace(**recipe, data_root=args.data_root, download=False))
        ids = evidence["train_indices"]
        lookup = {sample_id: index for index, sample_id in enumerate(ids)}
        per_epoch = len(ids) // recipe["batch_size"]
        device = st.WgpuTensorDevice.create()
        require(device.adapter_info()["device_type"] != "Cpu", "real GPU required")
        result["adapter"] = device.adapter_info()
        native = {p.name: runner.sha(p.read_bytes()) for p in Path(st.__file__).parent.glob("*.so")}
        require(native, "native wheel binary missing")
        result["native_binary_sha256"] = native
        reference_type = shared.load_reference()
        for seed in recipe["seeds"]:
            root = args.ablation / f"seed-{seed}" / "baseline"
            verified = ablation.verify_arm(root, args.source_ref)
            runner.write_json(args.output / f"seed-{seed}-source-verification.json", verified)
            raw = root / f"seed-{seed}"
            control = json.loads((raw / "control.json").read_text())
            require(control["contract"]["data"] == evidence, "retained dataset differs")
            require(control["contract"]["native_binary_sha256"] == native, "native binary differs from source run")
            require(control["contract"]["source_sha256"] == runner.source_hashes(), "input/reference helper source differs")
            require(len(control["records"]) == recipe["epochs"] * per_epoch, "incomplete input trajectory")
            for checkpoint in ("initial", "final"):
                payload = runner.read_checkpoint(raw, control["checkpoints"][checkpoint]).decode("utf-8")
                model = runner.model_payload(payload)
                owner = st.ResidentConvNeXtClassifier.from_checkpoint_json(device, model)
                before = owner.checkpoint_snapshot().read_json()
                require(json.loads(before) == json.loads(model), "restored model differs")
                reference = reference_type(model, recipe["torch_device"]).eval()
                pipeline = st.TransformPipeline(seed=seed)
                pipeline.add_normalize([0.5] * 3, [0.25] * 3)
                pipeline.enable_wgpu()
                rows = []
                with torch.no_grad():
                    for index, recorded in enumerate(control["records"]):
                        require(not any(recorded["flips"]), "stationarity probe excludes augmentation")
                        indices = [lookup[sample_id] for sample_id in recorded["sample_ids"]]
                        images, normalized = shared.inputs(device, pipeline, data[0][0][indices])
                        actual = shared.read(images)
                        require(shared.digest(actual.astype("<f4")) == recorded["input_sha256"], "recorded input differs")
                        shared.compare(actual, normalized.reshape(-1), "normalized probe input")
                        target = device.upload([len(indices), 1], data[0][1][indices].astype(np.float32).tolist())
                        prediction = owner.forward(images).prediction_tensor()
                        objective = st.nn.CrossEntropyWithLogits().evaluate_resident(prediction, target)
                        loss = float(shared.read(objective.loss_tensor())[0])
                        torch_loss = float(shared.F.cross_entropy(
                            reference(torch.tensor(normalized, device=recipe["torch_device"])),
                            torch.tensor(data[0][1][indices], device=recipe["torch_device"])).item())
                        parity = shared.compare([loss], [torch_loss], "frozen-model CE")
                        rows.append(dict(step=index + 1, sample_ids=recorded["sample_ids"],
                            input_sha256=recorded["input_sha256"], loss=loss, loss_bits=runner.bits(loss),
                            torch_loss=torch_loss, parity=parity))
                after = owner.checkpoint_snapshot().read_json()
                require(before == after, "frozen model changed")
                epochs = epoch_summaries(rows, ids, per_epoch)
                shadows = {name: shadow_replay(st, sequence, recipe["control_scale"])
                    for name, sequence in stationary_sequences([row["loss"] for row in rows], per_epoch).items()}
                raw_path = args.output / f"seed-{seed}-{checkpoint}.json"
                runner.write_json(raw_path, dict(seed=seed, checkpoint=checkpoint, records=rows,
                    model_sha256=runner.sha(before.encode()), model_unchanged=True, shadows=shadows))
                case = dict(seed=seed, checkpoint=checkpoint, parameter_updates=0, model_unchanged=True,
                    source_checkpoint=control["checkpoints"][checkpoint], epochs=epochs,
                    frozen_epoch_mean_spread=max(e["mean_loss"] for e in epochs) - min(e["mean_loss"] for e in epochs),
                    max_ce_scaled_error=max(row["parity"]["max_scaled_error"] for row in rows),
                    shadows={name: value["summary"] for name, value in shadows.items()}, raw=runner.receipt(raw_path))
                result["cases"].append(case)
                print(json.dumps(case), flush=True)
                del owner, reference
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    runner.write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status=result["status"], completed_cases=len(result["cases"]), error=result.get("error"))), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
