#!/usr/bin/env python3
"""Offline auxiliary learning replay, not a quality or throughput benchmark.

Use a separately pinned native package in each process. Two tensor-valued
module outputs receive public Rust-backed adapters. The later native input
VJP must actually execute, not merely exist in the operator API.
"""

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random

import torch
import transformers
import spiraltorch as st
import spiraltorch.fractional_autograd as bridge
import spiraltorch.spiraltorch as native

import verify_fractional_history_factorial as common

SCHEMA = "spiraltorch.fractional_multi_adapter_replay.v1"
require, equal = common.require, common.equal


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def tensor_receipt(value):
    value = value.detach().cpu().contiguous()
    require(bool(torch.isfinite(value).all()), "nonfinite tensor")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "l2": float(value.double().norm()), "nonzero": int(torch.count_nonzero(value))}


def model_digest(model):
    return identity({name: tensor_receipt(value) for name, value in model.state_dict().items()})


class ObservedSnapshot:
    """Record real native calls without replacing the differential rule."""

    def __init__(self, snapshot, events, site):
        self.snapshot, self.events, self.site = snapshot, events, site

    def __getattr__(self, name):
        operation = getattr(self.snapshot, name)
        if not name.startswith("vjp"):
            return operation

        def invoke(direction):
            result = operation(direction)
            row = {"site": self.site, "method": name,
                   "upstream": tensor_receipt(torch.from_numpy(direction))}
            if name == "vjp_buffer":
                row["input_vjp"] = tensor_receipt(torch.frombuffer(memoryview(result[0]), dtype=torch.float32))
                row["alpha_vjp"], row["log_gain_vjp"] = result[1:]
            elif name == "vjp_parameters_buffer":
                row["alpha_vjp"], row["log_gain_vjp"] = result
            else:
                raise ValueError(f"unexpected native route: {name}")
            self.events.append(row)
            return result

        return invoke


class ObservedAdapter(st.FractionalAngleGainHistoryAdapter):
    def __init__(self, site, events, *, observe=True, **kwargs):
        super().__init__(**kwargs)
        self.site, self.events, self.observe = site, events, observe

    def _history(self, value, alpha):
        result = super()._history(value, alpha)
        require(getattr(result.grad_fn, "buffer_transport", False), "buffer transport required")
        if self.observe:
            result.grad_fn.snapshot = ObservedSnapshot(result.grad_fn.snapshot, self.events, self.site)
        return result


def validate_config(config):
    fields = {"schema", "model_snapshot", "corpus_sha256", "corpus_end_marker", "blocks",
              "features", "seed", "steps", "checkpoint_after", "batch_size", "block_size",
              "learning_rate", "strength", "threads", "initial_angle", "initial_gain", "kernel"}
    require(set(config) == fields and config["schema"] == SCHEMA, "recipe fields differ")
    for name in ("features", "steps", "checkpoint_after", "batch_size", "block_size", "threads"):
        require(type(config[name]) is int and config[name] > 0, f"invalid {name}")
    require(type(config["seed"]) is int and 0 <= config["seed"] < 2**32, "invalid seed")
    require(1 <= config["checkpoint_after"] < config["steps"] <= 16
            and config["block_size"] >= 2, "invalid auxiliary update budget")
    blocks = config["blocks"]
    require(isinstance(blocks, list) and len(blocks) == 2
            and all(isinstance(p, str) and p and all(p.split(".")) for p in blocks)
            and blocks[0] != blocks[1]
            and not any(a.startswith(b + ".") for a, b in (blocks, blocks[::-1])),
            "two distinct non-nested module paths required")
    for key in ("learning_rate", "strength", "initial_gain"):
        require(type(config[key]) in (float, int) and math.isfinite(config[key])
                and config[key] > 0, f"invalid {key}")
    require(type(config["initial_angle"]) in (float, int)
            and math.isfinite(config["initial_angle"]), "invalid angle")
    require(isinstance(config["kernel"], dict) and config["kernel"].get("kernel_len", 0) >= 3,
            "kernel must include short window")


def make_adapters(config, window, events, *, observe=True):
    validate_config(config)
    require(window in ("short", "full"), "unknown window")
    return torch.nn.ModuleDict({f"site{index}": ObservedAdapter(
        f"site{index}", events, observe=observe, features=config["features"],
        initial_angle=config["initial_angle"], initial_gain=config["initial_gain"],
        strength=config["strength"], lag_window=(1, 3) if window == "short" else None,
        **config["kernel"]) for index in range(2)})


@contextmanager
def attached(model, adapters, paths):
    # Resolve all paths before mutation; always restore even if forward fails.
    targets = []
    for path in paths:
        prefix, _, child = path.rpartition(".")
        parent = model.get_submodule(prefix) if prefix else model
        targets.append((parent, child, parent.get_submodule(child)))
    require(targets[0][2] is not targets[1][2], "aliased insertion modules")
    changed = []
    try:
        for (parent, child, original), adapter in zip(targets, adapters.values()):
            parent.add_module(child, torch.nn.Sequential(original, adapter))
            changed.append((parent, child, original))
        yield
    finally:
        for parent, child, original in reversed(changed):
            parent.add_module(child, original)


def make_optimizer(adapters, config):
    return torch.optim.Adam(adapters.parameters(), lr=config["learning_rate"], foreach=False, fused=False)


def batch_schedule(config, count):
    require(count >= config["batch_size"], "too few training blocks")
    rng, indices = random.Random(config["seed"]), []
    while len(indices) < config["steps"] * config["batch_size"]:
        epoch = list(range(count))
        rng.shuffle(epoch)
        indices.extend(epoch)
    size = config["batch_size"]
    return [indices[i * size:(i + 1) * size] for i in range(config["steps"])]


def checkpoint(adapters, optimizer, binding, runtime, cursor, records):
    return {"schema": SCHEMA, "binding": copy.deepcopy(binding), "runtime": runtime,
            "cursor": cursor, "records": copy.deepcopy(records),
            "parameter_names": list(dict(adapters.named_parameters())),
            "adapter": copy.deepcopy(adapters.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": torch.get_rng_state().clone()}


def restore(saved, adapters, optimizer, binding, allowed_native):
    require(saved["schema"] == SCHEMA and equal(saved["binding"], binding), "checkpoint binding differs")
    require(saved["runtime"]["native_sha256"] == allowed_native, "source runtime is not admitted")
    require(type(saved["cursor"]) is int and saved["cursor"] == binding["config"]["checkpoint_after"],
            "checkpoint cursor differs")
    require(len(saved["records"]) == saved["cursor"]
            and [r["step"] for r in saved["records"]] == list(range(1, saved["cursor"] + 1)),
            "checkpoint trajectory differs")
    require(saved["parameter_names"] == list(dict(adapters.named_parameters())), "parameter names differ")
    expected = adapters.state_dict()
    require(saved["adapter"].keys() == expected.keys(), "adapter fields differ")
    for name, current in expected.items():
        value = saved["adapter"][name]
        if isinstance(current, torch.Tensor):
            require(isinstance(value, torch.Tensor) and value.dtype == current.dtype
                    and value.shape == current.shape and bool(torch.isfinite(value).all()),
                    f"invalid parameter: {name}")
        else:
            require(equal(value, current), "adapter recipe differs")
    require(saved["optimizer"]["param_groups"][0]["params"] == optimizer.state_dict()["param_groups"][0]["params"],
            "Adam parameter order differs")
    common.named_adam(saved, adapters, optimizer, saved["cursor"])
    adapters.load_state_dict(saved["adapter"])
    require(equal(adapters.state_dict(), saved["adapter"]), "adapter roundtrip differs")
    for adapter in adapters.values():
        require(math.isfinite(adapter.alpha) and math.isfinite(adapter.gain), "invalid geometry domain")
    torch.set_rng_state(saved["rng"])


def learning_run(model, train, config, window, runtime, context, *, saved=None, allowed_native=None,
                 observe=True, on_checkpoint=None):
    events = []
    adapters = make_adapters(config, window, events, observe=observe)
    optimizer = make_optimizer(adapters, config)
    require(not model.training and not model.config.use_cache
            and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            "base must be frozen, eval, cache-free and gradient-free")
    require(train.device.type == "cpu" and train.dtype == torch.long and train.ndim == 2
            and train.shape[1] == config["block_size"], "invalid unpadded token blocks")
    schedule = batch_schedule(config, len(train))
    base_hash = model_digest(model)
    binding = {"config": config, "window": window, "base_sha256": base_hash,
               "tokens": tensor_receipt(train), "schedule": schedule,
               "model_class": type(model).__qualname__, "model_config": identity(model.config.to_dict()),
               "context": context, "torch": str(torch.__version__), "transformers": str(transformers.__version__)}
    torch.manual_seed(config["seed"])
    start, records = 0, []
    if saved is not None:
        restore(saved, adapters, optimizer, binding, allowed_native)
        start, records = saved["cursor"], copy.deepcopy(saved["records"])
    midpoint = None
    gradients = []
    with attached(model, adapters, config["blocks"]):
        for cursor in range(start, config["steps"]):
            events.clear()
            optimizer.zero_grad(set_to_none=True)
            loss = model(train[schedule[cursor]], labels=train[schedule[cursor]], use_cache=False).loss
            require(bool(torch.isfinite(loss)), "nonfinite learning loss")
            loss.backward()
            grad = {name: p.grad.detach().clone() for name, p in adapters.named_parameters() if p.grad is not None}
            require(len(grad) == len(list(adapters.parameters())), "missing adapter gradient")
            grad_receipts = {name: tensor_receipt(value) for name, value in grad.items()}
            if observe:
                require([(e["site"], e["method"]) for e in events] == [
                    ("site1", "vjp_buffer"), ("site0", "vjp_parameters_buffer")], "native gradient path differs")
                if cursor > 0:
                    require(events[0]["input_vjp"]["nonzero"] > 0, "later input VJP never became active")
                    require(all(grad_receipts[f"site{i}.{p}"]["nonzero"] > 0
                                for i in range(2) for p in ("gate", "local_gate", "history_angle", "log_gain")),
                            "geometry parameter gradient stayed inactive")
            optimizer.step()
            for adapter in adapters.values():
                require(math.isfinite(adapter.alpha) and math.isfinite(adapter.gain), "geometry left domain")
            row = {"step": cursor + 1, "batch_indices": schedule[cursor], "loss": tensor_receipt(loss),
                   "loss_value": float(loss.detach()), "gradients": grad_receipts,
                   "parameters": {n: tensor_receipt(p) for n, p in adapters.named_parameters()},
                   "native_calls": copy.deepcopy(events), "rng": tensor_receipt(torch.get_rng_state())}
            records.append(row)
            gradients.append(grad)
            print(json.dumps({"step": cursor + 1, "window": window, "loss": row["loss_value"]}), flush=True)
            if cursor + 1 == config["checkpoint_after"]:
                midpoint = checkpoint(adapters, optimizer, binding, runtime, cursor + 1, records)
                if on_checkpoint:
                    on_checkpoint(midpoint)
    require(model_digest(model) == base_hash and all(p.grad is None for p in model.parameters()),
            "frozen base changed")
    final = checkpoint(adapters, optimizer, binding, runtime, config["steps"], records)
    common.named_adam(final, adapters, optimizer, config["steps"])
    return {"final": final, "midpoint": midpoint, "executed_gradients": gradients,
            "start_cursor": start, "base_unchanged": True}


def write_json(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_runtime(root, manifest):
    require(isinstance(manifest, dict) and set(manifest) == {"source_revision", "files"}
            and isinstance(manifest["source_revision"], str)
            and len(manifest["source_revision"]) == 40
            and all(c in "0123456789abcdef" for c in manifest["source_revision"]),
            "invalid frozen runtime manifest")
    return common.verify_files(root, manifest["files"])


def save_state(path, value):
    with path.open("xb") as handle:
        torch.save(value, handle)
    return {"file": path.name, "sha256": common.digest(path)}


def compare_runs(before, after, resumed):
    left, right, continued = before["final"], after["final"], resumed["final"]
    require(before["start_cursor"] == after["start_cursor"] == 0, "fresh replay must start at zero")
    cursor = left["binding"]["config"]["checkpoint_after"]
    require(resumed["start_cursor"] == cursor, "resume must execute a suffix")
    require(equal(right["runtime"], continued["runtime"]), "continuation target runtime differs")
    require(all(r["base_unchanged"] for r in (before, after, resumed)), "base preservation missing")
    for run in (before, after, resumed):
        steps = run["final"]["binding"]["config"]["steps"]
        require(run["final"]["cursor"] == len(run["final"]["records"]) == steps
                and len(run["executed_gradients"]) == steps - run["start_cursor"], "executed update count differs")
        for row, gradients in zip(run["final"]["records"][run["start_cursor"]:], run["executed_gradients"]):
            require(equal(row["gradients"], {name: tensor_receipt(value) for name, value in gradients.items()}),
                    "gradient receipt differs")
    for name in ("binding", "cursor", "records", "parameter_names", "adapter", "optimizer", "rng"):
        require(equal(left[name], right[name]) and equal(left[name], continued[name]), f"replay mismatch: {name}")
    require(equal(before["executed_gradients"], after["executed_gradients"])
            and equal(before["executed_gradients"][cursor:], resumed["executed_gradients"]),
            "raw gradients differ")
    require(before["midpoint"] is not None and after["midpoint"] is not None, "midpoint missing")
    for name in ("adapter", "optimizer", "rng", "records"):
        require(equal(before["midpoint"][name], after["midpoint"][name]), f"midpoint differs: {name}")
    return {"status": "bitwise_exact", "binding_sha256": identity(left["binding"]),
            "updates_executed": sum(len(r["executed_gradients"]) for r in (before, after, resumed)),
            "old_native": left["runtime"], "new_native": right["runtime"],
            "records": left["records"], "base_unchanged": True,
            "scope": "Auxiliary CPU learning and state migration, not heldout quality or model throughput."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("config", "model-dir", "corpus", "package-root", "runtime-manifest", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--window", choices=("short", "full"), required=True)
    run.add_argument("--resume", type=Path)
    run.add_argument("--resume-sha256")
    run.add_argument("--allow-source-native-sha256")
    compare = sub.add_parser("compare")
    for name in ("before", "after", "resumed", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    if args.command == "compare":
        results = []
        receipts = {}
        for name in ("before", "after", "resumed"):
            directory = getattr(args, name)
            report = json.loads((directory / "report.json").read_bytes())
            require(report["status"] == "completed", "incomplete auxiliary run")
            path = directory / "state.pt"
            require(common.digest(path) == report["state"]["sha256"], "state artifact hash differs")
            results.append(torch.load(path, weights_only=True, map_location="cpu"))
            receipts[name] = {"report_sha256": common.digest(directory / "report.json"), **report["state"]}
        result = compare_runs(*results)
        result["artifacts"] = receipts
        write_json(args.output, result)
        print(json.dumps({"status": result["status"], "updates": result["updates_executed"]}), flush=True)
        return
    require(bool(args.resume) == bool(args.resume_sha256) == bool(args.allow_source_native_sha256),
            "resume needs explicit checkpoint hash and source native admission")
    config = json.loads(args.config.read_bytes())
    validate_config(config)
    require(Path(st.__file__).resolve().parent == (args.package_root / "spiraltorch").resolve(), "wrong package")
    manifest = json.loads(args.runtime_manifest.read_bytes())
    verify_runtime(args.package_root, manifest)
    runtime = {"native_sha256": common.digest(Path(native.__file__)),
               "bridge_sha256": common.digest(Path(bridge.__file__)),
               "build_source_revision": manifest["source_revision"],
               "manifest_sha256": common.digest(args.runtime_manifest)}
    require(args.model_dir.name == config["model_snapshot"], "model snapshot differs")
    require(common.digest(args.corpus) == config["corpus_sha256"], "corpus differs")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    text = args.corpus.read_text()
    require(text.count(config["corpus_end_marker"]) == 1, "corpus end marker differs")
    text = text[:text.index(config["corpus_end_marker"])]
    cut = text.rfind("\n\n", 0, int(len(text) * .9))
    require(cut > 0, "training split boundary missing")
    ids = tokenizer.encode(text[:cut], add_special_tokens=False)
    size = config["block_size"]
    train = torch.tensor(ids[:len(ids) // size * size], dtype=torch.long).reshape(-1, size)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    model.config.use_cache = False
    context = {"source_sha256": common.digest(Path(__file__)),
               "verifier_sha256": common.digest(Path(common.__file__)),
               "bridge_sha256": runtime["bridge_sha256"], "training_split_character_offset": cut}
    saved = None
    if args.resume:
        require(common.digest(args.resume) == args.resume_sha256, "checkpoint hash differs")
        saved = torch.load(args.resume, weights_only=True, map_location="cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = learning_run(model, train, config, args.window, runtime, context,
            saved=saved, allowed_native=args.allow_source_native_sha256,
            on_checkpoint=lambda state: save_state(args.output / "midpoint.pt", state))
        verify_runtime(args.package_root, manifest)
        receipt = save_state(args.output / "state.pt", result)
        report = {"schema": SCHEMA, "status": "completed", "window": args.window,
                  "runtime": runtime, "binding": result["final"]["binding"],
                  "start_cursor": result["start_cursor"], "updates_executed": len(result["executed_gradients"]),
                  "records": result["final"]["records"], "state": receipt,
                  "resume_source_sha256": args.resume_sha256, "base_unchanged": True,
                  "heldout_scoring": False, "timing_evidence": False}
        if result["midpoint"] is not None:
            report["midpoint"] = {"file": "midpoint.pt", "sha256": common.digest(args.output / "midpoint.pt")}
        write_json(args.output / "report.json", report)
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
