"""Offline training-only admission for a fixed gain/angle study, not quality scoring."""

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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError("preflight assertions must not be disabled")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("client-root", "package-root", "config", "model-dir", "corpus",
                 "transfer-corpus", "previous-study", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--coordinate", choices=("log-order", "angle", "window"), default="log-order")
    args = parser.parse_args()
    assert not args.output.exists()
    assert Path(st.__file__).resolve().parent == (args.package_root / "spiraltorch").resolve()
    sys.path.insert(0, str(args.client_root))
    if args.coordinate == "window":
        import hf_fractional_window_study as client
    elif args.coordinate == "angle":
        import hf_fractional_angle_study as client
    else:
        import hf_fractional_gain_study as client
    driver, pilot = client.study, client.study.pilot
    config = json.loads(args.config.read_bytes())
    client.validate_protocol(config)
    previous = json.loads((args.previous_study / "plan.json").read_bytes())
    journal = json.loads((args.previous_study / "journal.json").read_bytes())
    assert driver.completed_result(args.previous_study, journal, previous) is not None
    assert args.model_dir.name == config["model_snapshot"]
    torch.set_num_threads(config["threads"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    train, _, _, data = driver.prepare_data(tokenizer, args.corpus.read_bytes(),
                                           args.transfer_corpus.read_bytes(), config)
    assert data == previous["data"]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    model.config.use_cache = False
    base_hash = pilot.model_digest(model)
    assert base_hash == previous["base_parameter_sha256"]
    parent_name, child = config["block"].rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original = parent.get_submodule(child)
    seed = config["seeds"][0]
    batches = pilot.schedule(seed, len(train), config["steps"]+1, config["batch_size"])
    assert batches == previous["batch_schedules"][str(seed)]
    captured = []
    hook = original.register_forward_hook(lambda _module, _args, output: captured.append(output.detach()))
    try:
        with torch.no_grad():
            model(train[batches[0]])
    finally:
        hook.remove()
    assert len(captured) == 1
    hidden = captured[0]
    assert list(hidden.shape) == [config["batch_size"], config["block_size"], config["features"]]
    ordinary = client.adapter_for(client.ARMS[0], config, seed)
    angular = args.coordinate in ("angle", "window")
    target = (ordinary._history(hidden, ordinary._alpha_tensor()) if args.coordinate == "window"
              else ordinary.history(hidden)).detach()
    report = {}
    for arm in client.ARMS:
        adapter = client.adapter_for(arm, config, seed)
        assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == 2*config["features"]+2
        initial_hash = pilot.model_digest(adapter)
        with torch.no_grad():
            assert torch.equal(adapter(hidden), hidden)
            observed = (adapter.history(hidden) if arm == client.ARMS[0] and args.coordinate != "window"
                        else adapter._history(hidden, adapter._alpha_tensor() if angular
                                              else adapter.log_alpha.exp()))
            assert torch.allclose(observed, target, rtol=2e-6, atol=2e-7)
        optimizer = driver.make_optimizer(adapter, arm, config, None)
        records = []
        parent.add_module(child, torch.nn.Sequential(original, adapter))
        try:
            for indices in batches[:2]:
                records.append(pilot.update(model, adapter, optimizer, train[indices]))
            saved = copy.deepcopy(adapter.state_dict()), copy.deepcopy(optimizer.state_dict())
            reference_update = pilot.update(model, adapter, optimizer, train[batches[2]])
            restored = client.adapter_for(arm, config, seed)
            restored.load_state_dict(saved[0])
            resumed = driver.make_optimizer(restored, arm, config, None)
            resumed.load_state_dict(saved[1])
            parent.add_module(child, torch.nn.Sequential(original, restored))
            replay_update = pilot.update(model, restored, resumed, train[batches[2]])
            assert reference_update == replay_update
            assert pilot.equal_state(adapter.state_dict(), restored.state_dict())
            assert pilot.model_digest(adapter) == pilot.model_digest(restored)
            assert pilot.equal_state(optimizer.state_dict(), resumed.state_dict())
        finally:
            parent.add_module(child, original)
        assert records[0]["log_gain_gradient"] == 0 and records[1]["log_gain_gradient"] != 0
        shape = "history_angle" if angular or arm == client.ARMS[0] else "log_alpha"
        assert records[0][f"{shape}_gradient"] == 0 and records[1][f"{shape}_gradient"] != 0
        report[arm] = {"records": records, "initial_parameter_sha256": initial_hash,
                       "initial_filter_close": True, "next_update_and_adam_equal": True,
                       "parameter_count": sum(p.numel() for p in restored.parameters())}
        print(f"{arm}: two auxiliary training updates and exact continuation verified", flush=True)
    assert pilot.model_digest(model) == base_hash and all(p.grad is None for p in model.parameters())
    if args.coordinate != "window":
        assert report[client.ARMS[1]]["initial_parameter_sha256"] == report[client.ARMS[2]]["initial_parameter_sha256"]
    if angular:
        assert len({r["initial_parameter_sha256"] for r in report.values()}) == 1
    first = report[client.ARMS[0]]["records"][0]
    for arm in client.ARMS:
        row = report[arm]["records"][0]
        assert row["loss"] == first["loss"]
        for field in ("gate_gradient_l2", "local_gate_gradient_l2"):
            assert abs(row[field]-first[field]) <= 2e-7 + 2e-6*abs(first[field])
    payload = {"schema": ("spiraltorch.fractional_window_study_preflight.v1" if args.coordinate == "window" else
                         "spiraltorch.fractional_angle_study_preflight.v1" if args.coordinate == "angle"
                          else "spiraltorch.fractional_gain_preflight.v1"), "status": "passed",
               "config_sha256": sha(args.config), "preflight_source_sha256": sha(Path(__file__)),
               "native_sha256": sha(Path(native.__file__)), "base_parameter_sha256": base_hash,
               "train_tokens_sha256": data["train_tokens_sha256"],
               "source_sha256": {p.name: sha(p) for p in args.client_root.glob("*.py")},
               "seed": seed, "actual_shape": list(hidden.shape), "auxiliary_training_updates": 2*len(client.ARMS),
               "continuation_only_updates": 2*len(client.ARMS), "frozen_base_unchanged": True,
               "heldout_losses_computed": False, "runs": report,
               "scope": "Training-only connection check; not primary training, endpoint outcomes or speed evidence"}
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
