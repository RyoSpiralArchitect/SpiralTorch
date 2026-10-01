#!/usr/bin/env python3
"""Real-image learning and fresh-process replay through the Rust-owned trainer.

The Torch model is independent; batch order, augmentation draws and learning
rates are supplied by Rust. Explicit per-step observations exclude speed claims.
Heavy runtime imports stay in workers so replay checks need only the stdlib.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def unbits(value):
    require(type(value) is int and 0 <= value < 2 ** 32, "invalid f32 bits")
    return struct.unpack("<f", struct.pack("<I", value))[0]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def receipt(path):
    raw = Path(path).read_bytes()
    return dict(file=Path(path).name, bytes=len(raw), sha256=sha(raw))


def read_checkpoint(directory, record):
    name = record["file"]
    require(isinstance(name, str) and Path(name).name == name, "invalid checkpoint filename")
    raw = (directory / name).read_bytes()
    require(len(raw) == record["bytes"] and sha(raw) == record["sha256"], "checkpoint fixity mismatch")
    return raw


def verify_replay(directory, phases, total, split, train_ids, batch_size):
    require(0 < split < total and batch_size > 0, "invalid replay split/batch")
    require(train_ids and len(set(train_ids)) == len(train_ids), "empty/duplicate sample IDs")
    require(len(train_ids) % batch_size == 0, "partial input epoch")
    per_epoch = len(train_ids) // batch_size
    require(total % per_epoch == 0, "partial training epoch")
    require(set(phases) == {"control", "prefix", "resume"}, "missing/extra replay phase")
    control, prefix, resumed = (phases[key] for key in ("control", "prefix", "resume"))
    require(len({p["pid"] for p in phases.values()}) == 3, "workers must be distinct processes")
    ranges = {"control": (0, total), "prefix": (0, split), "resume": (split, total)}
    for phase, run in phases.items():
        require(run["status"] == "passed" and run["phase"] == phase, "unsuccessful/wrong phase")
        require(run["contract"] == control["contract"], "runtime/input/configuration differs")
        start, end = ranges[phase]
        require(len(run["records"]) == end - start, "missing/extra attempted updates")
        for offset, row in enumerate(run["records"], start + 1):
            require(row["revision"] == offset and row["epoch"] == (offset - 1) // per_epoch,
                    "wrong attempted-update/epoch clock")
            require(row["accepted"] is True, "real-image update was not accepted")
            require(len(row["sample_ids"]) == batch_size and len(set(row["sample_ids"])) == batch_size,
                    "duplicate/partial batch")
            require(set(row["sample_ids"]).issubset(train_ids), "sample outside training subset")
            require(math.isfinite(unbits(row["rate_bits"])) and unbits(row["rate_bits"]) > 0,
                    "non-finite/non-positive rate")
            require(math.isfinite(unbits(row["loss_bits"])), "non-finite loss")
            require(len(row["input_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in row["input_sha256"]), "invalid input hash")
    for start in range(0, total, per_epoch):
        ids = [i for row in control["records"][start:start + per_epoch] for i in row["sample_ids"]]
        require(sorted(ids) == sorted(train_ids), "epoch does not cover each sample once")
    require(control["records"] == prefix["records"] + resumed["records"], "step trajectory differs")
    payloads = {phase: {key: read_checkpoint(directory, value)
                        for key, value in run["checkpoints"].items()}
                for phase, run in phases.items()}
    require(payloads["control"]["initial"] == payloads["prefix"]["initial"], "initial states differ")
    require(payloads["control"]["split"] == payloads["prefix"]["final"], "prefix state differs")
    require(payloads["prefix"]["final"] == payloads["resume"]["initial"], "restored state differs")
    require(payloads["control"]["final"] == payloads["resume"]["final"], "final state differs")
    trace = json.dumps(control["records"], sort_keys=True, separators=(",", ":")).encode()
    return dict(attempts=total, accepted=total, rejected=0, split=[split, total - split],
                distinct_processes=True, all_batch_records_exact=True, all_bound_checkpoints_exact=True,
                trajectory_sha256=sha(trace), final_checkpoint_sha256=sha(payloads["control"]["final"]))


def load_runtime():
    global shared, np, torch, st
    path = Path(__file__).with_name("run_vision_matched_learning.py")
    spec = importlib.util.spec_from_file_location("matched_vision", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    np, torch, st = shared.np, shared.torch, shared.st


def prepare_input(recipe, data, evidence):
    pixels, targets = data
    raw = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
    dataset = st.TensorVisionDataset("CIFAR10")
    identity = hashlib.sha256(b"spiraltorch.cifar_trainer_input.v1\0")
    for image, target, sample_id in zip(raw, targets, evidence["train_indices"], strict=True):
        label = str(sample_id)
        identity.update(image.astype("<f4").tobytes())
        identity.update(struct.pack("<f", target))
        identity.update(label.encode("ascii") + b"\0")
        dataset.push(st.ImageTensor(3, 32, 32, image.reshape(-1).tolist()),
                     target=st.Tensor(1, 1, [float(target)]), label=label)
    pipeline = st.TransformPipeline(seed=recipe["seed"])
    if recipe["horizontal_flip"]:
        pipeline.add_horizontal_flip(0.5)
    pipeline.add_normalize([0.5] * 3, [0.25] * 3)
    return dataset, pipeline, identity.hexdigest(), raw


def matched_input(observed, raw, allow_flip):
    normalized = (raw - np.float32(0.5)) / np.float32(0.25)
    observed = observed.reshape(normalized.shape)
    require(np.isfinite(observed).all(), "non-finite trainer input")
    flips = []
    for index, (actual, expected) in enumerate(zip(observed, normalized, strict=True)):
        candidates = [expected, expected[:, :, ::-1]] if allow_flip else [expected]
        errors = [float(np.max(np.abs(actual - x) / (1 + np.abs(x)))) for x in candidates]
        choice = min(range(len(errors)), key=errors.__getitem__)
        require(errors[choice] < 2e-4, "trainer image is neither normalized source nor permitted flip")
        normalized[index] = candidates[choice].copy()
        flips.append(bool(choice))
    shared.compare(observed, normalized, "trainer input")
    return normalized, flips


def trainer_config(recipe, total):
    config = json.loads(st.ResidentVisionTrainer.default_config_json())
    config.update(num_classes=10, batch_size=recipe["batch_size"], model_seed=str(recipe["seed"]),
                  shuffle_seed=str(recipe["seed"]), shuffle=True)
    config["model"].update(input_channels=3, input_hw=[32, 32], stage_dims=[8, 16],
                           stage_depths=[1, 1], patch_size=[4, 4], epsilon=0.001)
    config["learning_rate"] = (
        dict(kind="constant", rate=recipe["rate"]) if recipe["schedule"] == "constant" else
        dict(kind="warmup_cosine", state=dict(base_lr=recipe["rate"], min_lr=recipe["rate"] / 10,
                                               warmup_steps=min(10, total), total_steps=total, step=0)))
    return config


def source_hashes():
    return {name: sha(Path(__file__).with_name(name).read_bytes()) for name in (
        Path(__file__).name, "run_vision_matched_learning.py", "vision_convnext_torch_reference.py")}


def model_payload(payload):
    return json.dumps(json.loads(payload)["model"])


def worker(recipe, phase, directory, report):
    load_runtime()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if recipe["torch_device"] == "mps":
        require(torch.backends.mps.is_available() and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
                "MPS requires available hardware and explicit disabled fallback")
    args = argparse.Namespace(**recipe, download=False)
    data, evidence = shared.dataset(args)
    evidence["augmentation"] = "rust_horizontal_flip_0.5" if recipe["horizontal_flip"] else "none"
    total = recipe["epochs"] * len(data[0][1]) // recipe["batch_size"]
    split = recipe["restart_at"]
    config = trainer_config(recipe, total)
    dataset, pipeline, data_id, raw = prepare_input(recipe, data[0], evidence)
    device = st.WgpuTensorDevice.create()
    adapter = device.adapter_info()
    require(adapter.get("device_type") not in (None, "Cpu"), "real WGPU adapter required")
    binaries = list(Path(st.__file__).parent.glob("*.so")) + list(Path(st.__file__).parent.glob("*.pyd"))
    require(binaries, "native binding binary not found")
    report["contract"] = dict(config=config, data=evidence, dataset_sha256=data_id, schedule=recipe["schedule"],
        horizontal_flip=recipe["horizontal_flip"], augmentation_seed=str(recipe["seed"]), adapter=adapter,
        source_sha256=source_hashes(), native_binary_sha256={p.name: sha(p.read_bytes()) for p in binaries},
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
                         torchvision=shared.torchvision.__version__, spiraltorch=st.__version__,
                         torch_device=recipe["torch_device"], startup_customization_present="sitecustomize" in sys.modules))
    kind = st.ResidentVisionTrainer
    if phase == "resume":
        prefix = json.loads((directory / "prefix.json").read_text())
        payload = read_checkpoint(directory, prefix["checkpoints"]["final"]).decode("utf-8")
        trainer = kind.from_checkpoint_json(device, dataset, data_id, payload, pipeline)
    else:
        trainer = kind.create(device, dataset, data_id, json.dumps(config), pipeline)
    report["checkpoints"] = {}

    def capture(name):
        payload = trainer.checkpoint_snapshot().read_json()
        report["checkpoints"][name] = shared.save_checkpoint(directory, f"{phase}-{name}.json", payload)
        return payload

    initial = capture("initial")
    eval_pipeline = st.TransformPipeline(seed=recipe["seed"])
    eval_pipeline.add_normalize([0.5] * 3, [0.25] * 3)
    eval_pipeline.enable_wgpu()
    reference = optimizer = None
    report.update(records=[], epochs=[])
    if phase == "control":
        reference_type = shared.load_reference()
        payload = model_payload(initial)
        temporary = st.ResidentConvNeXtClassifier.from_checkpoint_json(device, payload)
        disposable_reference = reference_type(payload, recipe["torch_device"])
        report["admission"] = shared.admission(temporary, disposable_reference, device, eval_pipeline,
            data[0][0][:recipe["batch_size"]], data[0][1][:recipe["batch_size"]], recipe["rate"])
        del temporary, disposable_reference
        reference = reference_type(payload, recipe["torch_device"])
        optimizer = torch.optim.SGD(reference.parameters(), lr=recipe["rate"], foreach=False, fused=False)
        owner = st.ResidentConvNeXtClassifier.from_checkpoint_json(device, payload)
        report["initial_evaluation"] = shared.evaluate(owner, reference, device, eval_pipeline, data[1], recipe["batch_size"])
        del owner
    index_lookup = {str(sample_id): i for i, sample_id in enumerate(evidence["train_indices"])}
    per_epoch = len(data[0][1]) // recipe["batch_size"]
    start, end = (split, total) if phase == "resume" else (0, split if phase == "prefix" else total)
    train_loss = dict(spiraltorch=0., torch=0.)
    for offset in range(start, end):
        submitted = trainer.submit_next()
        labels = submitted.labels()
        require(len(labels) == recipe["batch_size"] and len(set(labels)) == len(labels), "invalid sample labels")
        indices = [index_lookup[label] for label in labels]
        actual = shared.read(submitted.images())
        normalized, flips = matched_input(actual, raw[indices], recipe["horizontal_flip"])
        outcome = trainer.settle()
        require(outcome.accepted and outcome.attempted_revision == offset + 1, "rejected/wrong training update")
        loss = float(shared.read(submitted.loss_tensor())[0])
        require(math.isfinite(loss), "non-finite training loss")
        report["records"].append(dict(revision=outcome.attempted_revision, epoch=submitted.epoch,
            accepted=outcome.accepted, sample_ids=[int(label) for label in labels], flips=flips,
            input_sha256=shared.digest(actual.astype("<f4")), rate_bits=bits(submitted.learning_rate), loss_bits=bits(loss)))
        train_loss["spiraltorch"] += loss * len(indices)
        if reference is not None:
            for group in optimizer.param_groups:
                group["lr"] = submitted.learning_rate
            optimizer.zero_grad(set_to_none=True)
            prediction = reference(torch.tensor(normalized, device=recipe["torch_device"]))
            objective = shared.F.cross_entropy(prediction, torch.tensor(data[0][1][indices], device=recipe["torch_device"]))
            objective.backward()
            require(torch.isfinite(objective).item()
                    and all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in reference.parameters()),
                    "non-finite/absent reference gradient")
            shared.compare([loss], [objective.item()], "training loss")
            optimizer.step()
            train_loss["torch"] += objective.item() * len(indices)
        if phase == "control" and offset + 1 == split:
            capture("split")
        if reference is not None and (offset + 1) % per_epoch == 0:
            snapshot = trainer.checkpoint_snapshot().read_json()
            owner = st.ResidentConvNeXtClassifier.from_checkpoint_json(device, model_payload(snapshot))
            evaluation = shared.evaluate(owner, reference, device, eval_pipeline, data[1], recipe["batch_size"])
            del owner
            epoch = dict(epoch=(offset + 1) // per_epoch, accepted_updates=offset + 1,
                         train_loss={k: v / len(raw) for k, v in train_loss.items()}, evaluation=evaluation)
            report["epochs"].append(epoch)
            print(json.dumps(dict(seed=recipe["seed"], phase=phase, **epoch)), flush=True)
            train_loss = dict(spiraltorch=0., torch=0.)
    final = capture("final")
    state = json.loads(final)
    require(state["trainer"]["accepted_updates"] == end and state["trainer"]["rejected_updates"] == 0,
            "settled trainer clocks differ")
    if reference is not None:
        parameters = state["model"]["backbone"]["parameters"] + state["model"]["head"]
        require([p["name"] for p in parameters] == reference.names, "final parameter roles differ")
        report["final_parameter_comparison"] = [dict(name=name, **shared.compare(
            p["values"], tensor.detach().cpu().numpy().reshape(-1), name))
            for name, p, tensor in shared.matched_parameters(reference.names, parameters, reference.values)]
    report["status"] = "passed"


def run_worker(recipe_path, phase):
    recipe_path = Path(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    result = dict(schema="spiraltorch.vision.trainer_realdata_phase.v1", status="error", phase=phase, pid=os.getpid())
    try:
        worker(recipe, phase, recipe_path.parent, result)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(recipe_path.parent / f"{phase}.json", result)
    print(json.dumps(dict(phase=phase, status=result["status"], error=result.get("error"))), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        return run_worker(sys.argv[2], sys.argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--test-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rate", type=float, default=0.01)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--restart-at", type=int, default=37)
    args = parser.parse_args()
    if (min(args.train_per_class, args.test_per_class, args.batch_size, args.epochs) <= 0
            or args.train_per_class * 10 % args.batch_size or args.test_per_class * 10 % args.batch_size
            or not math.isfinite(args.rate) or not 0 < args.rate <= 1
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2 ** 64 for s in args.seeds)):
        parser.error("invalid full-batch recipe, rate or seeds")
    total = args.epochs * args.train_per_class * 10 // args.batch_size
    if not 0 < args.restart_at < total or total >= 2 ** 32:
        parser.error("restart must be inside the finite schedule")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="spiraltorch.vision.trainer_matched_learning.v1", status="error", runs=[],
                  boundary="real-image trainer correctness/restart; no throughput, peak-memory or Z-space advantage claim",
                  recipe={k: v for k, v in vars(args).items() if k not in ("data_root", "output")})
    try:
        for seed in args.seeds:
            directory = args.output / f"seed-{seed}"
            directory.mkdir()
            recipe = dict(result["recipe"], seed=seed, data_root=str(args.data_root.resolve()))
            write_json(directory / "recipe.json", recipe)
            phases = {}
            for phase in ("control", "prefix", "resume"):
                subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--worker",
                                str((directory / "recipe.json").resolve()), phase], check=True)
                phases[phase] = json.loads((directory / f"{phase}.json").read_text())
            control = phases["control"]
            replay = verify_replay(directory, phases, total, args.restart_at,
                                   control["contract"]["data"]["train_indices"], args.batch_size)
            result["runs"].append(dict(seed=seed, replay=replay, contract=control["contract"],
                admission=control["admission"], initial_evaluation=control["initial_evaluation"],
                epochs=control["epochs"], final_parameter_comparison=control["final_parameter_comparison"],
                checkpoints={phase: run["checkpoints"] for phase, run in phases.items()},
                phase_receipts={phase: receipt(directory / f"{phase}.json") for phase in phases}))
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status=result["status"], completed_seeds=len(result["runs"]), error=result.get("error"))), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
