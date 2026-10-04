"""Offline training-only comparison of chart-aligned short filters.

Eight updates per arm/seed plus separate exact continuation checks. Never
score endpoints, save model weights or alter the preceding completed study.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
import transformers
import spiraltorch as st
import spiraltorch.spiraltorch as native

STEPS = 8
RTOL, ATOL = 3e-6, 3e-7


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right):
    assert left.keys() == right.keys()
    errors, close = {}, True
    for name, a in left.items():
        a, b = a.detach(), right[name].detach()
        assert a.shape == b.shape and a.dtype == b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all()
        errors[name] = float((a-b).abs().max())
        close = close and torch.allclose(a, b, rtol=RTOL, atol=ATOL)
    return {"close": bool(close), "max_abs_error": errors}


def adam_named(adapter, optimizer):
    group, = optimizer.state_dict()["param_groups"]
    states = optimizer.state_dict()["state"]
    return {f"{name}/{field}": value
            for name, identifier in zip(dict(adapter.named_parameters()), group["params"])
            for field, value in states[identifier].items()}


def main():
    if not __debug__:
        raise RuntimeError("preflight assertions must not be disabled")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("client-root", "package-root", "config", "model-dir", "corpus", "transfer-corpus",
                 "previous-study", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert Path(st.__file__).resolve().parent == (args.package_root / "spiraltorch").resolve()
    sys.path.insert(0, str(args.client_root))
    import hf_fractional_gain_study as client
    driver, pilot = client.study, client.study.pilot
    config = json.loads(args.config.read_bytes())
    client.validate_protocol(config)
    previous = json.loads((args.previous_study / "plan.json").read_bytes())
    journal = json.loads((args.previous_study / "journal.json").read_bytes())
    assert driver.completed_result(args.previous_study, journal, previous) is not None
    assert config == previous["config"] and args.model_dir.name == config["model_snapshot"]
    torch.set_num_threads(config["threads"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    train, _, _, data = driver.prepare_data(tokenizer, args.corpus.read_bytes(), args.transfer_corpus.read_bytes(), config)
    assert data == previous["data"]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    model.config.use_cache = False
    base_hash = pilot.model_digest(model)
    assert base_hash == previous["base_parameter_sha256"]
    parent_name, child = config["block"].rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original = parent.get_submodule(child)

    def factory(arm, seed):
        if arm == "ordinary_short":
            return client.adapter_for(client.ARMS[0], config, seed)
        return st.FractionalAngleGainHistoryAdapter(
            config["features"], initial_angle=config["initial_history_angle"],
            initial_gain=config["initial_gain"], strength=config["strength"],
            **{**config["kernel"], "kernel_len": config["short_kernel_len"]})

    runs, all_close = {}, True
    for seed in config["seeds"]:
        batches = pilot.schedule(seed, len(train), config["steps"]+1, config["batch_size"])
        assert batches == previous["batch_schedules"][str(seed)]
        adapters = {arm: factory(arm, seed) for arm in ("ordinary_short", "native_angle_short")}
        initial_hashes = {arm: pilot.model_digest(adapter) for arm, adapter in adapters.items()}
        assert len(set(initial_hashes.values())) == 1
        optimizers = {arm: driver.make_optimizer(adapter, arm, config, None) for arm, adapter in adapters.items()}
        captures = []
        hook = original.register_forward_hook(lambda _m, _args, output: captures.append(output.detach()))
        try:
            with torch.no_grad(): model(train[batches[0]])
        finally:
            hook.remove()
        assert len(captures) == 1 and list(captures[0].shape) == [config["batch_size"], config["block_size"], config["features"]]
        for adapter in adapters.values():
            assert torch.equal(adapter(captures[0]), captures[0])
            assert sum(p.numel() for p in adapter.parameters()) == 2*config["features"]+2
        records, comparisons = {arm: [] for arm in adapters}, []
        for index, indices in enumerate(batches[:STEPS]):
            gradients = {}
            try:
                for arm, adapter in adapters.items():
                    parent.add_module(child, torch.nn.Sequential(original, adapter))
                    records[arm].append(pilot.update(model, adapter, optimizers[arm], train[indices]))
                    gradients[arm] = {name: p.grad.detach().clone() for name, p in adapter.named_parameters()}
            finally:
                parent.add_module(child, original)
            left, right = adapters.values()
            comparison = {"step": index+1, "batch_indices": indices,
                          "gradients": compare(*gradients.values()),
                          "parameters": compare(dict(left.named_parameters()), dict(right.named_parameters())),
                          "adam": compare(*(adam_named(adapter, optimizers[arm]) for arm, adapter in adapters.items())),
                          "loss_abs_difference": abs(records["ordinary_short"][-1]["loss"]-records["native_angle_short"][-1]["loss"])}
            all_close = all_close and all(comparison[key]["close"] for key in ("gradients", "parameters", "adam"))
            comparisons.append(comparison)
        for arm, adapter in adapters.items():
            saved = copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizers[arm].state_dict())
            restored = factory(arm, seed)
            restored.load_state_dict(saved[0])
            resumed = driver.make_optimizer(restored, arm, config, None)
            resumed.load_state_dict(saved[1])
            try:
                parent.add_module(child, torch.nn.Sequential(original, adapter))
                reference = pilot.update(model, adapter, optimizers[arm], train[batches[STEPS]])
                parent.add_module(child, torch.nn.Sequential(original, restored))
                replay = pilot.update(model, restored, resumed, train[batches[STEPS]])
            finally:
                parent.add_module(child, original)
            assert reference == replay and pilot.equal_state(adapter.state_dict(), restored.state_dict())
            assert pilot.equal_state(optimizers[arm].state_dict(), resumed.state_dict())
        assert records["ordinary_short"][0]["loss"] == records["native_angle_short"][0]["loss"]
        runs[str(seed)] = {"initial_parameter_sha256": initial_hashes, "records": records,
                          "comparisons": comparisons, "exact_within_arm_continuation": True}
        print(f"seed {seed}: both short filters and exact continuation completed", flush=True)
    assert pilot.model_digest(model) == base_hash and all(p.grad is None for p in model.parameters())
    report = {"schema": "spiraltorch.fractional_angle_preflight.v1", "status": "passed" if all_close else "parity_failed",
              "steps_per_arm": STEPS, "seeds": config["seeds"], "rtol": RTOL, "atol": ATOL,
              "auxiliary_training_updates": 2*STEPS*len(runs), "continuation_only_updates": 4*len(runs),
              "native_sha256": sha(Path(native.__file__)), "asset_config_sha256": sha(args.config),
              "preflight_source_sha256": sha(Path(__file__)), "base_parameter_sha256": base_hash,
              "train_tokens_sha256": data["train_tokens_sha256"], "actual_shape": list(captures[0].shape),
              "source_sha256": {p.name: sha(p) for p in args.client_root.glob("*.py")},
              "frozen_base_unchanged": True, "heldout_losses_computed": False, "runs": runs,
              "scope": "Training-only chart alignment, not long-run quality, bitwise cross-arm parity, speed or endpoint evidence."}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if not all_close:
        raise SystemExit("chart-aligned tensor criterion failed; retained all observations")


if __name__ == "__main__":
    main()
