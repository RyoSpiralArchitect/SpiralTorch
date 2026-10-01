#!/usr/bin/env python3
"""Matched CIFAR-10 classifier admission and learning through public clients.

No performance claim: admission maps every gradient, and learning explicitly
observes each update. Data download is opt-in; reports never overwrite.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from torchvision.datasets import CIFAR10
import spiraltorch as st


def load_reference():
    path = Path(__file__).with_name("vision_convnext_torch_reference.py")
    spec = importlib.util.spec_from_file_location("vision_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConvNeXtReference


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_checkpoint(directory, name, payload):
    path = directory / name
    with path.open("x") as out:
        out.write(payload)
    return dict(file=name, sha256=hashlib.sha256(payload.encode()).hexdigest(), bytes=len(payload.encode()))


def compare(actual, expected, label, tolerance=2e-4):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{label}: shape, empty or finite-domain failure")
    absolute = float(np.abs(a - b).max())
    scaled = float((np.abs(a - b) / (1 + np.abs(b))).max())
    if scaled >= tolerance:
        raise ValueError(f"{label}: scaled error {scaled} >= {tolerance}")
    return dict(max_abs_error=absolute, max_scaled_error=scaled, values=int(a.size))


def read(tensor):
    return np.asarray(tensor.snapshot().read_values(), dtype=np.float32)


def matched_parameters(names, tensors, parameters):
    if not names or len(set(names)) != len(names) or not (len(names) == len(tensors) == len(parameters)):
        raise ValueError("parameter roles/count differ or are empty/duplicated")
    return zip(names, tensors, parameters)


def balanced_indices(targets, per_class, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)
    selected = []
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        if per_class <= 0 or per_class > len(candidates):
            raise ValueError("per-class sample count outside dataset")
        selected.extend(rng.permutation(candidates)[:per_class])
    return rng.permutation(np.asarray(selected, dtype=np.int64))


def dataset(args):
    train = CIFAR10(args.data_root, train=True, download=args.download)
    test = CIFAR10(args.data_root, train=False, download=False)
    train_ids = balanced_indices(train.targets, args.train_per_class, 20261001)
    test_ids = balanced_indices(test.targets, args.test_per_class, 20261002)
    if len(train_ids) % args.batch_size or len(test_ids) % args.batch_size:
        raise ValueError("fixed batch requires exact train/evaluation divisibility; no silent tail drop")
    def subset(source, indices):
        pixels = np.ascontiguousarray(source.data[indices].transpose(0, 3, 1, 2))
        labels = np.asarray(source.targets, dtype=np.int64)[indices]
        return pixels, labels
    data = subset(train, train_ids), subset(test, test_ids)
    evidence = dict(name="CIFAR-10", source="https://www.cs.toronto.edu/~kriz/cifar.html",
                    official_archive_md5=CIFAR10.tgz_md5, train_indices=train_ids.tolist(), test_indices=test_ids.tolist(),
                    train_pixels_sha256=digest(data[0][0]), train_labels_sha256=digest(data[0][1].astype("<i8")),
                    test_pixels_sha256=digest(data[1][0]), test_labels_sha256=digest(data[1][1].astype("<i8")),
                    train_per_class=args.train_per_class, test_per_class=args.test_per_class,
                    normalization=dict(input_divisor=255, mean=[0.5] * 3, std=[0.25] * 3), augmentation="none")
    return data, evidence


def inputs(device, pipeline, pixels):
    raw = pixels.astype(np.float32) / np.float32(255)
    images = [st.ImageTensor(3, 32, 32, image.reshape(-1).tolist()) for image in raw]
    resident = pipeline.apply_resident_batch(images, device)
    normalized = (raw - np.float32(0.5)) / np.float32(0.25)
    return resident, normalized


def admission(owner, reference, device, pipeline, pixels, labels, rate):
    x, normalized = inputs(device, pipeline, pixels)
    checks = dict(normalize=compare(read(x), normalized.reshape(-1), "normalize"))
    forward = owner.forward(x)
    target = device.upload([len(labels), 1], labels.astype(np.float32).tolist())
    loss = st.nn.CrossEntropyWithLogits().evaluate_resident(forward.prediction_tensor(), target)
    gradients = owner.backward(forward, loss.prediction_gradient_tensor())
    tx = torch.tensor(normalized, device=next(reference.parameters()).device, requires_grad=True)
    ty = torch.tensor(labels, device=tx.device)
    prediction = reference(tx)
    objective = F.cross_entropy(prediction, ty)
    objective.backward()
    checks["logits"] = compare(read(forward.prediction_tensor()), prediction.detach().cpu().numpy().reshape(-1), "logits")
    checks["loss"] = compare(read(loss.loss_tensor()), [objective.item()], "loss")
    checks["input_gradient"] = compare(read(gradients.input_gradient_tensor()), tx.grad.cpu().numpy().reshape(-1), "input gradient")
    observed = gradients.parameter_gradient_tensors()
    if owner.parameter_names() != reference.names or len(observed) != len(reference.values):
        raise ValueError("parameter roles/order/count differ")
    checks["parameter_gradients"] = []
    for name, actual, parameter in matched_parameters(reference.names, observed, reference.values):
        if parameter.grad is None:
            raise ValueError(f"unused reference parameter: {name}")
        checks["parameter_gradients"].append(dict(name=name, **compare(read(actual), parameter.grad.cpu().numpy().reshape(-1), name)))
    revision = owner.sgd(gradients, rate).read()
    optimizer = torch.optim.SGD(reference.parameters(), lr=rate, foreach=False, fused=False)
    optimizer.step()
    checks["updated_weights"] = []
    for name, actual, parameter in matched_parameters(reference.names, owner.parameter_tensors(), reference.values):
        checks["updated_weights"].append(dict(name=name, **compare(read(actual), parameter.detach().cpu().numpy().reshape(-1), name)))
    checks["next_logits"] = compare(read(owner.forward(x).prediction_tensor()), reference(tx.detach()).detach().cpu().numpy().reshape(-1), "next logits")
    checks["accepted_revision"] = revision
    return checks


def evaluate(owner, reference, device, pipeline, data, batch_size):
    pixels, labels = data
    records = {name: dict(loss_sum=0., correct=0, examples=0) for name in ("spiraltorch", "torch")}
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x, normalized = inputs(device, pipeline, pixels[start:start + batch_size])
            target = torch.tensor(labels[start:start + batch_size])
            st_logits = torch.from_numpy(read(owner.forward(x).prediction_tensor()).reshape(batch_size, 10))
            torch_logits = reference(torch.tensor(normalized, device=next(reference.parameters()).device)).cpu()
            for name, logits in (("spiraltorch", st_logits), ("torch", torch_logits)):
                if not torch.isfinite(logits).all():
                    raise ValueError("non-finite evaluation logits")
                records[name]["loss_sum"] += F.cross_entropy(logits, target, reduction="sum").item()
                records[name]["correct"] += int((logits.argmax(1) == target).sum())
                records[name]["examples"] += len(target)
    return {name: dict(loss=r["loss_sum"] / r["examples"], accuracy=r["correct"] / r["examples"], examples=r["examples"])
            for name, r in records.items()}


def run_seed(args, seed, data, config, reference_type, device, report, artifacts):
    kind = st.ResidentConvNeXtClassifier
    initial = kind.create(device, json.dumps(config), 10, args.batch_size, seed).checkpoint_snapshot().read_json()
    report.update(seed=seed, initial_checkpoint_sha256=hashlib.sha256(initial.encode()).hexdigest(), epochs=[])
    report["initial_checkpoint"] = save_checkpoint(artifacts, f"seed-{seed}-initial.json", initial)
    pipeline = st.TransformPipeline(seed=seed)
    pipeline.add_normalize([0.5] * 3, [0.25] * 3)
    pipeline.enable_wgpu()
    owner = kind.from_checkpoint_json(device, initial)
    reference = reference_type(initial, args.torch_device)
    report["admission"] = admission(owner, reference, device, pipeline, data[0][0][:args.batch_size],
                                    data[0][1][:args.batch_size], args.rate)
    # Admission is a disposable update, never an extra training step on either arm.
    owner = kind.from_checkpoint_json(device, initial)
    reference = reference_type(initial, args.torch_device)
    report["initial_evaluation"] = evaluate(owner, reference, device, pipeline, data[1], args.batch_size)
    optimizer = torch.optim.SGD(reference.parameters(), lr=args.rate, foreach=False, fused=False)
    rng = np.random.default_rng(seed)
    for epoch in range(args.epochs):
        order = rng.permutation(len(data[0][1]))
        train_loss = {"spiraltorch": 0., "torch": 0.}
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x, normalized = inputs(device, pipeline, data[0][0][indices])
            labels = data[0][1][indices]
            target = device.upload([args.batch_size, 1], labels.astype(np.float32).tolist())
            forward = owner.forward(x)
            loss = st.nn.CrossEntropyWithLogits().evaluate_resident(forward.prediction_tensor(), target)
            gradients = owner.backward(forward, loss.prediction_gradient_tensor())
            accepted = owner.sgd(gradients, args.rate).read()
            if accepted != epoch * (len(order) // args.batch_size) + start // args.batch_size + 1:
                raise ValueError("unexpected accepted update clock")
            train_loss["spiraltorch"] += float(read(loss.loss_tensor())[0]) * len(indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = reference(torch.tensor(normalized, device=args.torch_device))
            objective = F.cross_entropy(prediction, torch.tensor(labels, device=args.torch_device))
            objective.backward()
            if not torch.isfinite(objective) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in reference.parameters()):
                raise ValueError("non-finite or absent Torch gradient")
            optimizer.step()
            train_loss["torch"] += objective.item() * len(indices)
        record = dict(epoch=epoch + 1, order_sha256=digest(order.astype("<i8")), accepted_updates=owner.attempted_updates,
                      train_loss={k: v / len(order) for k, v in train_loss.items()},
                      evaluation=evaluate(owner, reference, device, pipeline, data[1], args.batch_size))
        report["epochs"].append(record)
        print(json.dumps(dict(seed=seed, **record)), flush=True)
    final = owner.checkpoint_snapshot().read_json()
    report["final_checkpoint"] = save_checkpoint(artifacts, f"seed-{seed}-final.json", final)
    report["final_checkpoint_sha256"] = report["final_checkpoint"]["sha256"]
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--torch-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--test-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.batch_size <= 0 or args.epochs < 0 or args.rate <= 0 or not math.isfinite(args.rate)
            or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2 ** 64 for s in args.seeds)):
        parser.error("invalid batch, epoch, rate or duplicate seeds")
    report = dict(schema="spiraltorch.vision.matched_learning.v1", status="error", runs=[],
                  phase="learning" if args.epochs else "admission_only",
                  boundary="bounded real-image learning; explicit per-step readbacks; no throughput, memory or full-training-resume claim",
                  torch=torch.__version__, torchvision=torchvision.__version__, spiraltorch=st.__version__,
                  torch_device=args.torch_device, torch_threads=1, rate=args.rate, batch_size=args.batch_size,
                  requested_epochs=args.epochs, requested_seeds=args.seeds,
                  python=sys.version.split()[0], numpy=np.__version__,
                  startup_customization_present="sitecustomize" in sys.modules)
    with args.output.open("x") as out:
        try:
            artifacts = args.output.with_suffix(".checkpoints")
            artifacts.mkdir(exist_ok=False)
            report["checkpoint_directory"] = artifacts.name
            report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (Path(__file__), Path(__file__).with_name("vision_convnext_torch_reference.py"))}
            native_files = [*Path(st.__file__).parent.glob("*.so"), *Path(st.__file__).parent.glob("*.pyd")]
            if not native_files:
                raise RuntimeError("native binding binary not found for provenance")
            report["native_binary_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native_files}
            torch.set_num_threads(1)
            torch.set_float32_matmul_precision("highest")
            torch.use_deterministic_algorithms(True)
            if args.torch_device == "mps" and (not torch.backends.mps.is_available() or os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0"):
                raise RuntimeError("MPS requires an available adapter and explicit disabled fallback")
            data, report["data"] = dataset(args)
            device = st.WgpuTensorDevice.create()
            report["rust_runtime_adapter"] = device.adapter_info()
            if report["rust_runtime_adapter"].get("device_type") in (None, "Cpu"):
                raise RuntimeError("real WGPU adapter required")
            config = json.loads(st.ResidentConvNeXtClassifier.default_config_json())
            config.update(input_channels=3, input_hw=[32, 32], stage_dims=[8, 16], stage_depths=[1, 1], patch_size=[4, 4], epsilon=1e-3)
            report["config"] = config
            reference_type = load_reference()
            for seed in args.seeds:
                record = dict(status="running")
                report["runs"].append(record)
                try:
                    run_seed(args, seed, data, config, reference_type, device, record, artifacts)
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                    raise
            report["status"] = "passed"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(status=report["status"], runs=len(report["runs"]), error=report.get("error"))), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
