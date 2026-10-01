#!/usr/bin/env python3
"""Offline evidence checks for the fixed two-stage CIFAR-10 comparison recipe.

This validates recorded coverage, not the truth of self-reported GPU execution.
Optional local artifacts check checkpoint bytes and recorded source provenance.
No dataset, Torch, device, or network access is needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


CONFIG = dict(input_channels=3, input_hw=[32, 32], stage_dims=[8, 16],
              stage_depths=[1, 1], patch_size=[4, 4], epsilon=0.001, curvature=-1.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def parameter_sizes():
    sizes = {"convnext.stem::weight": 384, "convnext.stem::bias": 8}
    for stage, channels in enumerate((8, 16)):
        prefix = f"convnext.stage{stage}.block0."
        for role, size in (("dw::weight", channels * 49), ("dw::bias", channels),
                           ("ln_gamma", channels), ("ln_beta", channels),
                           ("fc1::weight", channels * channels * 4), ("fc1::bias", channels * 4),
                           ("fc2::weight", channels * channels * 4), ("fc2::bias", channels)):
            sizes[prefix + role] = size
        if stage == 0:
            sizes.update({"convnext.stage0.downsample::weight": 512,
                          "convnext.stage0.downsample::bias": 16})
    sizes.update({"convnext.final_norm_gamma": 256, "convnext.final_norm_beta": 256,
                  "convnext.classifier::weight": 160, "convnext.classifier::bias": 10})
    return sizes


def check_error(error, size):
    require(type(error["values"]) is int and error["values"] == size, "comparison element count differs")
    for key in ("max_abs_error", "max_scaled_error"):
        require(finite(error[key]) and error[key] >= 0, "invalid recorded error")
    require(error["max_scaled_error"] < 2e-4, "admission tolerance exceeded")


def check_evaluation(evaluation, count):
    for arm in ("spiraltorch", "torch"):
        metric = evaluation[arm]
        require(metric["examples"] == count, "evaluation count differs")
        require(finite(metric["loss"]) and metric["loss"] >= 0, "invalid evaluation loss")
        require(finite(metric["accuracy"]) and 0 <= metric["accuracy"] <= 1, "invalid accuracy")
    return {key: abs(evaluation["spiraltorch"][key] - evaluation["torch"][key])
            for key in ("accuracy", "loss")}


def verify(report, checkpoint_dir=None, source_ref=None, native_binary=None):
    require(report["schema"] == "spiraltorch.vision.matched_learning.v1", "unknown report schema")
    require(report["status"] == "passed" and report["config"] == CONFIG, "incomplete run or different recipe")
    seeds, epochs, batch = report["requested_seeds"], report["requested_epochs"], report["batch_size"]
    require(seeds and len(seeds) == len(set(seeds)), "empty or duplicate seeds")
    require(all(type(seed) is int and 0 <= seed < 2 ** 64 for seed in seeds), "invalid seed")
    require(type(epochs) is int and epochs >= 0 and type(batch) is int and batch > 0, "invalid loop dimensions")
    require(batch == 16 and report["rate"] == 0.01 and report["torch_threads"] == 1,
            "fixed batch, learning rate or thread count differs")
    require(report["torch_device"] in ("cpu", "mps"), "unrecorded reference device")
    phase = "learning" if epochs else "admission_only"
    require(report.get("phase", phase) == phase, "incorrect phase")
    require([r["seed"] for r in report["runs"]] == seeds, "missing or reordered seeds")
    data = report["data"]
    require(data["name"] == "CIFAR-10" and data["official_archive_md5"] == "c58f30108f718f92721af3b95e74349a",
            "fixed dataset differs")
    require(data["normalization"] == dict(input_divisor=255, mean=[0.5] * 3, std=[0.25] * 3)
            and data["augmentation"] == "none", "fixed preprocessing differs")
    counts = {}
    for split, limit in (("train", 50000), ("test", 10000)):
        count = data[f"{split}_per_class"] * 10
        indices = data[f"{split}_indices"]
        require(count > 0 and count % batch == 0, "invalid sample count")
        require(len(indices) == count and len(set(indices)) == count, "missing or repeated samples")
        require(all(type(i) is int and 0 <= i < limit for i in indices), "sample outside split")
        counts[split] = count
    sizes = parameter_sizes()
    summaries = []
    checked_checkpoints = 0
    for run in report["runs"]:
        require(run["status"] == "passed" and len(run["epochs"]) == epochs, "incomplete seed")
        admission = run["admission"]
        require(admission["accepted_revision"] == 1, "admission update clock differs")
        for name, size in (("normalize", batch * 3072), ("input_gradient", batch * 3072),
                           ("logits", batch * 10), ("next_logits", batch * 10), ("loss", 1)):
            check_error(admission[name], size)
        for key in ("parameter_gradients", "updated_weights"):
            rows = admission[key]
            require([r["name"] for r in rows] == list(sizes), f"{key}: parameter coverage differs")
            for row in rows:
                check_error(row, sizes[row["name"]])
        deltas = [check_evaluation(run["initial_evaluation"], counts["test"])]
        for index, epoch in enumerate(run["epochs"], 1):
            require(epoch["epoch"] == index, "missing or repeated epoch")
            require(epoch["accepted_updates"] == index * counts["train"] // batch, "update count differs")
            require(len(epoch["order_sha256"]) == 64, "missing permutation digest")
            for arm in ("spiraltorch", "torch"):
                require(finite(epoch["train_loss"][arm]) and epoch["train_loss"][arm] >= 0, "invalid train loss")
            deltas.append(check_evaluation(epoch["evaluation"], counts["test"]))
        if checkpoint_dir is not None:
            for when in ("initial", "final"):
                receipt = run[f"{when}_checkpoint"]
                require(receipt["file"] == f"seed-{run['seed']}-{when}.json", "unexpected checkpoint filename")
                payload = (checkpoint_dir / receipt["file"]).read_bytes()
                require(len(payload) == receipt["bytes"] and sha256(payload) == receipt["sha256"]
                        == run[f"{when}_checkpoint_sha256"], "checkpoint fixity mismatch")
                state = json.loads(payload)
                require(state["schema"] == "spiraltorch.convnext.classifier_plain_sgd_checkpoint.v1"
                        and state["classes"] == 10 and state["backbone"]["config"] == CONFIG,
                        "checkpoint architecture differs")
                require(state["backbone"]["batch"] == batch
                        and state["backbone"]["attempted_updates"] == (0 if when == "initial" else epochs * counts["train"] // batch),
                        "checkpoint batch or update clock differs")
                parameters = state["backbone"]["parameters"] + state["head"]
                require([p["name"] for p in parameters] == list(sizes), "checkpoint roles differ")
                for parameter in parameters:
                    values = parameter["values"]
                    require(len(values) == sizes[parameter["name"]]
                            and math.prod(parameter["shape"]) == len(values) and all(map(finite, values)),
                            "checkpoint values incomplete or non-finite")
                checked_checkpoints += 1
        summaries.append(dict(seed=run["seed"], epochs=epochs, accepted_updates=epochs * counts["train"] // batch,
                              max_evaluation_delta={k: max(d[k] for d in deltas) for k in ("accuracy", "loss")}))
    if source_ref is not None:
        require(set(report["source_sha256"]) == {"run_vision_matched_learning.py", "vision_convnext_torch_reference.py"},
                "source provenance incomplete")
        root = Path(__file__).resolve().parents[1]
        for filename, digest in report["source_sha256"].items():
            payload = subprocess.check_output(["git", "show", f"{source_ref}:tools/{filename}"], cwd=root)
            require(sha256(payload) == digest, f"source fixity mismatch: {filename}")
    if native_binary is not None:
        require(report["native_binary_sha256"] == {native_binary.name: sha256(native_binary.read_bytes())},
                "native binary fixity mismatch")
    return dict(status="verified", phase=phase, sample_counts=counts, parameter_tensors=len(sizes),
                parameter_values=sum(sizes.values()), runs=summaries, checked_checkpoint_files=checked_checkpoints,
                checked_source_ref=source_ref, checked_native_binary=native_binary is not None,
                boundary="record coverage/fixity only; no execution replay, throughput, resume or policy-effect proof")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--source-ref")
    parser.add_argument("--native-binary", type=Path)
    args = parser.parse_args()
    payload = args.report.read_bytes()
    result = verify(json.loads(payload), args.checkpoint_dir, args.source_ref, args.native_binary)
    print(json.dumps(dict(report_sha256=sha256(payload), **result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
