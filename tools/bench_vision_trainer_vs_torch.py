#!/usr/bin/env python3
"""Transfer-inclusive real-image trainer timings, separate from admission reads.

Each fresh worker resets both frameworks after warmup, then times the same
checkpoint's first batches. Rust owns feedback; its proposal is identity here,
so Torch follows the same constant rate without implementing a second policy.
Torch eager SGD synchronizes completion but does not provide Rust's numerical
rejection transaction. This is not torchvision ConvNeXt or a memory benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_runner():
    path = Path(__file__).with_name("run_vision_trainer_matched_learning.py")
    spec = importlib.util.spec_from_file_location("trainer_learning", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.load_runtime()
    return runner


def validate_recipe(recipe):
    for field in ("batch_size", "steps", "warmup", "train_per_class", "test_per_class"):
        require(type(recipe[field]) is int and recipe[field] > 0, f"invalid {field}")
    require(type(recipe["seed"]) is int and 0 <= recipe["seed"] < 2 ** 64, "invalid seed")
    require(recipe["mode"] in ("plain", "feedback"), "unknown feedback mode")
    require(type(recipe["profile"]) is bool, "invalid profile mode")
    require(sorted(recipe["framework_order"]) == ["spiraltorch", "torch"], "invalid framework order")
    require(math.isfinite(recipe["rate"]) and recipe["rate"] > 0, "invalid rate")
    batch = recipe["batch_size"]
    require(10 * recipe["train_per_class"] % batch == 0
            and 10 * recipe["test_per_class"] % batch == 0, "partial dataset batch")
    require(max(recipe["steps"], recipe["warmup"]) * batch <= 10 * recipe["train_per_class"],
            "interval must remain inside the initial input epoch")


def worker(recipe, directory, result):
    validate_recipe(recipe)
    r = load_runner()
    st, torch, np, shared = r.st, r.torch, r.np, r.shared
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    require(os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "0" and torch.backends.mps.is_available(),
            "available MPS and explicitly disabled fallback required")
    data, evidence = shared.dataset(argparse.Namespace(**recipe, download=False))
    setup = dict(recipe, schedule="constant", horizontal_flip=False,
                 optimizer_feedback=recipe["mode"] == "feedback")
    dataset, pipeline, data_id, raw = r.prepare_input(setup, data[0], evidence)
    config = r.trainer_config(setup, recipe["steps"])
    device = st.WgpuTensorDevice.create()
    require(device.adapter_info().get("device_type") not in (None, "Cpu"), "real WGPU adapter required")
    kind = st.ResidentVisionTrainer
    trainer = kind.create(device, dataset, data_id, json.dumps(config), pipeline)
    initial = trainer.checkpoint_snapshot().read_json()
    del trainer
    initial_state = json.loads(initial)
    order = initial_state["input"]["order"]
    require(sorted(order) == list(range(len(raw))) and initial_state["input"]["position"] == 0,
            "invalid initial Rust batch permutation")
    batches = [order[start:start + recipe["batch_size"]]
               for start in range(0, len(order), recipe["batch_size"])]
    payload = r.model_payload(initial)
    reference_type = shared.load_reference()
    rate = r.unbits(r.bits(recipe["rate"]))
    require(math.isfinite(rate) and rate > 0, "rate is outside positive f32")
    binary_files = sorted(Path(st.__file__).parent.glob("*.so"))
    require(binary_files, "native binary identity missing")
    source_names = [Path(__file__).name, "run_vision_trainer_matched_learning.py",
                    "run_vision_matched_learning.py", "vision_convnext_torch_reference.py"]
    result["contract"] = dict(config=config, data=evidence, dataset_sha256=data_id,
        initial_checkpoint_sha256=sha(initial.encode()), adapter=device.adapter_info(),
        native_binary_sha256={p.name: sha(p.read_bytes()) for p in binary_files},
        source_sha256={n: sha(Path(__file__).with_name(n).read_bytes()) for n in source_names},
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
                         torchvision=shared.torchvision.__version__, spiraltorch=st.__version__,
                         sitecustomize="sitecustomize" in sys.modules, torch_threads=1,
                         mps_fallback=False),
        boundaries=dict(timed="host batch selection, normalization/upload, forward, VJP, SGD, completion",
                        excluded="dataset decoding/construction, model creation, admission, warmup, checkpoints",
                        spiraltorch_acceptance="guarded_receipts", torch_acceptance="synchronized_only",
                        feedback="Rust identity proposal; default loss gate; Torch scalar observation only",
                        epoch_boundary=False, memory_claim=False, quality_claim=False))
    shared.save_checkpoint(directory, "initial.json", initial)
    first = batches[0]
    admission_pipeline = st.TransformPipeline(seed=recipe["seed"])
    admission_pipeline.add_normalize([0.5] * 3, [0.25] * 3)
    admission_pipeline.enable_wgpu()
    probe = st.ResidentConvNeXtClassifier.from_checkpoint_json(device, payload)
    reference = reference_type(payload, "mps")
    result["admission"] = shared.admission(probe, reference, device, admission_pipeline,
        data[0][0][first], data[0][1][first], rate)
    del probe, reference

    def native(count, profile):
        trainer = kind.from_checkpoint_json(device, dataset, data_id, initial, pipeline)
        # A mapped checkpoint fences setup before starting the timed interval.
        require(trainer.checkpoint_snapshot().read_json() == initial, "restore changed initial state")
        timings, trace = [], []
        begin = time.perf_counter_ns()
        for index in range(count):
            start = time.perf_counter_ns() if profile else 0
            submitted = trainer.submit_next()
            middle = time.perf_counter_ns() if profile else 0
            outcome = trainer.settle()
            if profile:
                timings.append([middle - start, time.perf_counter_ns() - middle])
            require(outcome.accepted and outcome.attempted_revision == index + 1, "rejected/wrong update")
            require(r.bits(submitted.learning_rate) == r.bits(rate), "identity feedback altered rate")
            trace.append(submitted.labels())
        elapsed = time.perf_counter_ns() - begin
        final = trainer.checkpoint_snapshot().read_json()
        state = json.loads(final)
        require(state["trainer"]["accepted_updates"] == count
                and state["trainer"]["rejected_updates"] == 0
                and state["input"]["position"] == count * recipe["batch_size"], "incorrect final clock")
        require(trace == [[str(evidence["train_indices"][i]) for i in ids] for ids in batches[:count]],
                "Rust batch identity differs from checkpoint permutation")
        loss = float(shared.read(submitted.loss_tensor())[0])
        return dict(elapsed_ns=elapsed, phases_ns=timings, last_loss=loss), final

    def eager(count, profile):
        reference = reference_type(payload, "mps")
        optimizer = torch.optim.SGD(reference.parameters(), lr=rate, foreach=False, fused=False)
        torch.mps.synchronize()
        timings = []
        begin = time.perf_counter_ns()
        for indices in batches[:count]:
            start = time.perf_counter_ns() if profile else 0
            x = (torch.tensor(raw[indices], device="mps") - 0.5) / 0.25
            y = torch.tensor(data[0][1][indices], device="mps")
            optimizer.zero_grad(set_to_none=True)
            objective = shared.F.cross_entropy(reference(x), y)
            objective.backward()
            optimizer.step()
            middle = time.perf_counter_ns() if profile else 0
            if recipe["mode"] == "feedback":
                require(math.isfinite(objective.item()), "invalid observed Torch loss")
            torch.mps.synchronize()
            if profile:
                timings.append([middle - start, time.perf_counter_ns() - middle])
        elapsed = time.perf_counter_ns() - begin
        parameters = [dict(name=n, shape=list(p.shape), values=p.detach().cpu().reshape(-1).tolist())
                      for n, p in zip(reference.names, reference.values, strict=True)]
        return dict(elapsed_ns=elapsed, phases_ns=timings, last_loss=objective.item()), parameters

    runners = dict(spiraltorch=native, torch=eager)
    result["measurements"] = {}
    finals = {}
    for framework in recipe["framework_order"]:
        runners[framework](recipe["warmup"], False)
        measured, final = runners[framework](recipe["steps"], recipe["profile"])
        measured["examples_per_second"] = recipe["steps"] * recipe["batch_size"] * 1e9 / measured["elapsed_ns"]
        result["measurements"][framework] = measured
        finals[framework] = final
    native_state = json.loads(finals["spiraltorch"])
    parameters = native_state["model"]["backbone"]["parameters"] + native_state["model"]["head"]
    require([p["name"] for p in parameters] == [p["name"] for p in finals["torch"]], "final parameter roles")
    result["final_comparison"] = []
    for a, b in zip(parameters, finals["torch"], strict=True):
        require(a["shape"] == b["shape"], "final parameter shape")
        result["final_comparison"].append(dict(name=a["name"], **shared.compare(a["values"], b["values"], a["name"])))
    result["last_loss_comparison"] = shared.compare(
        [result["measurements"]["spiraltorch"]["last_loss"]], [result["measurements"]["torch"]["last_loss"]], "last loss")
    result["final_checkpoint"] = shared.save_checkpoint(directory, "spiraltorch-final.json", finals["spiraltorch"])
    write_json(directory / "torch-final.json", finals["torch"])
    result["torch_final"] = r.receipt(directory / "torch-final.json")
    result["status"] = "passed"


def summarize(records):
    require(records and all(r["status"] == "passed" for r in records), "incomplete/failed measurement")
    groups = {}
    for record in records:
        recipe = record["recipe"]
        validate_recipe(recipe)
        require(all(recipe[key] == records[0]["recipe"][key] for key in ("profile", "steps", "warmup")),
                "mixed measurement boundaries")
        for measured in record["measurements"].values():
            elapsed = measured["elapsed_ns"]
            require(type(elapsed) is int and elapsed > 0, "invalid duration")
            require(len(measured["phases_ns"]) == (recipe["steps"] if recipe["profile"] else 0),
                    "profile leaked into throughput or is incomplete")
            require(measured["examples_per_second"] == recipe["steps"] * recipe["batch_size"] * 1e9 / elapsed,
                    "reported throughput differs from duration")
        key = (recipe["batch_size"], recipe["mode"], recipe["runtime"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (batch, mode, runtime), group in sorted(groups.items()):
        rows.append(dict(batch_size=batch, mode=mode, runtime=runtime, intervals=len(group),
            profile=group[0]["recipe"]["profile"],
            median_examples_per_second={name: statistics.median(r["measurements"][name]["examples_per_second"] for r in group)
                                        for name in ("spiraltorch", "torch")},
            max_parameter_scaled_error=max(v["max_scaled_error"] for r in group for v in r["final_comparison"])))
    paired = {}
    for r in records:
        c = r["recipe"]
        key = (c["seed"], c["batch_size"], c["mode"], c["repeat"])
        pair = paired.setdefault(key, {})
        require(c["runtime"] not in pair, "duplicate runtime measurement")
        pair[c["runtime"]] = r
    comparisons = []
    expected_runtimes = {r["recipe"]["runtime"] for r in records}
    require(expected_runtimes in ({"baseline"}, {"baseline", "candidate"}), "missing baseline/unknown runtime")
    for key, pair in paired.items():
        require(set(pair) == expected_runtimes, "missing paired measurement")
        if len(pair) == 1:
            continue
        require(set(pair) == {"baseline", "candidate"}, "unknown runtime")
        a, b = pair["baseline"], pair["candidate"]
        for field in ("config", "data", "dataset_sha256", "initial_checkpoint_sha256", "source_sha256", "environment", "adapter"):
            require(a["contract"][field] == b["contract"][field], f"A/B contract differs: {field}")
        require(a["final_checkpoint"]["sha256"] == b["final_checkpoint"]["sha256"], "A/B bound checkpoint differs")
        comparisons.append(dict(seed=key[0], batch_size=key[1], mode=key[2], repeat=key[3],
            exact_checkpoint=True, candidate_over_baseline=a["measurements"]["spiraltorch"]["elapsed_ns"]
            / b["measurements"]["spiraltorch"]["elapsed_ns"]))
    return dict(rows=rows, pairs=comparisons)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        path = Path(sys.argv[2])
        recipe = json.loads(path.read_text())
        result = dict(schema="spiraltorch.vision.training_timing.v1", status="error",
                      recipe={k: v for k, v in recipe.items() if k != "data_root"})
        try:
            worker(recipe, path.parent, result)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        write_json(path.parent / "result.json", result)
        print(json.dumps(dict(status=result["status"], error=result.get("error"))), flush=True)
        return 0 if result["status"] == "passed" else 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--modes", choices=("plain", "feedback"), nargs="+", default=["plain", "feedback"])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="diagnostic phase clocks, not an ordinary throughput interval")
    args = parser.parse_args()
    require(args.repeats > 0, "invalid repeats")
    for values in (args.seeds, args.batches, args.modes):
        require(values and len(set(values)) == len(values), "empty/duplicate cases")
    runtimes = dict(baseline=args.baseline_python)
    if args.candidate_python:
        runtimes["candidate"] = args.candidate_python
    require(all(path.is_file() for path in runtimes.values()), "missing interpreter")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in args.seeds:
        for batch in args.batches:
            for mode in args.modes:
                for repeat in range(args.repeats):
                    runtime_order = list(runtimes)
                    if repeat % 2:
                        runtime_order.reverse()
                    for runtime in runtime_order:
                        frameworks = ["spiraltorch", "torch"]
                        if (repeat + args.seeds.index(seed)) % 2:
                            frameworks.reverse()
                        recipe = dict(seed=seed, batch_size=batch, mode=mode, repeat=repeat, runtime=runtime,
                            data_root=str(args.data_root.resolve()), steps=args.steps, warmup=args.warmup,
                            rate=0.01, train_per_class=128, test_per_class=32, profile=args.profile,
                            framework_order=frameworks)
                        validate_recipe(recipe)
                        directory = args.output / f"seed-{seed}-batch-{batch}-{mode}-{repeat}-{runtime}"
                        directory.mkdir()
                        write_json(directory / "recipe.json", recipe)
                        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0")
                        with (directory / "worker.log").open("x") as log:
                            subprocess.run([str(runtimes[runtime]), "-I", str(Path(__file__).resolve()),
                                            "--worker", str(directory / "recipe.json")],
                                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                        record = json.loads((directory / "result.json").read_text())
                        records.append(record)
                        print(json.dumps(dict(case=directory.name, measurements=record["measurements"])), flush=True)
    requested = dict(seeds=args.seeds, batches=args.batches, modes=args.modes,
                     repeats=args.repeats, runtimes=list(runtimes), steps=args.steps,
                     warmup=args.warmup, profile=args.profile)
    write_json(args.output / "summary.json", dict(schema="spiraltorch.vision.training_timing_sweep.v1",
        status="passed", requested=requested, **summarize(records), records=records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
