import builtins
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def client(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_fractional_history_factorial.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("history_factorial_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def summary_module():
    source = Path(__file__).resolve().parents[3] / "tools" / "summarize_wave_gate_long_horizon.py"
    spec = importlib.util.spec_from_file_location("history_factorial_summary_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    source = Path(client.__file__).with_name("hf_fractional_pride_history_factorial.json")
    config = json.loads(source.read_text())
    config.update(features=8, steps=4, block_size=6, batch_size=2, seeds=[41],
                  checkpoint_every=2, evaluate_every=2, learning_rate=.01)
    return config


def test_identity_capacity_rng_recipe_separation_and_common_initial_filter(client):
    config = recipe(client)
    outputs, derivatives, hashes, adapters = [], [], [], []
    torch.set_num_threads(2)
    for arm in client.ARMS:
        rng = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(p.device.type == "cpu" and p.requires_grad for p in adapter.parameters())
        assert sum(p.numel() for p in adapter.parameters()) == 17
        hashes.append(client.study.pilot.model_digest(adapter))
        x = torch.linspace(-.9, .8, 96).reshape(2, 6, 8).transpose(0, 1).contiguous().transpose(0, 1)
        x.requires_grad_()
        assert torch.equal(adapter(x), x)
        extra = adapter.get_extra_state()
        assert extra["kernel"]["kernel_len"] == (3 if arm.endswith("_short") else 32)
        assert ("gain" in extra) == arm.startswith("history_l2_")
        assert adapter.execution_backend == "rust_f32_cpu"
        restored = client.adapter_for(arm, config, 47)
        restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
        assert client.study.pilot.equal_state(adapter.state_dict(), restored.state_dict())
        with torch.no_grad():
            adapter.gate.copy_(torch.linspace(-.4, .5, 8))
            adapter.local_gate.copy_(torch.linspace(.3, -.2, 8))
        y = adapter(x)
        outputs.append(y)
        derivatives.append(torch.autograd.grad(y, (x, adapter.gate, adapter.local_gate), x.cos()))
        work = x.double()
        first = torch.cat((torch.zeros_like(work[:, :1]), work[:, :-1]), dim=1)
        second = torch.cat((torch.zeros_like(work[:, :2]), work[:, :-2]), dim=1)
        ordinary = (x + .1*adapter.local_gate.tanh()*x
                    + .1*adapter.gate.tanh()*(-2*first+second).float())
        assert torch.equal(y, ordinary)
        adapters.append(adapter)
    assert len(set(hashes)) == 1
    for output, gradient in zip(outputs[1:], derivatives[1:]):
        assert torch.equal(output, outputs[0])
        assert all(torch.equal(left, right) for left, right in zip(gradient, derivatives[0]))
    for left in adapters:
        for right in adapters:
            if left.arm != right.arm:
                with pytest.raises(ValueError, match="recipe"):
                    left.load_state_dict(right.state_dict())


def test_long_history_integer_tail_has_a_real_learning_direction(client):
    config = recipe(client)
    config["features"] = 1
    x = torch.tensor([1., 0., 0., 0.]).reshape(1, 4, 1)
    for arm in client.ARMS:
        adapter = client.adapter_for(arm, config, 41)
        with torch.no_grad():
            adapter.gate.fill_(.3)
        y = adapter(x)
        assert y[0, 3, 0] == 0
        order = torch.autograd.grad(y[0, 3, 0], adapter.log_alpha)[0]
        if arm.endswith("_short"):
            assert order == 0
        else:
            assert order == pytest.approx(-2*.1*math.tanh(.3)/3, rel=2e-6)


@pytest.mark.parametrize("corruption", ["schema", "arms", "reference", "alpha", "bool_alpha",
                                       "gain", "bool_gain", "short", "bool_short", "full", "step", "bool_step"])
def test_protocol_rejects_changed_comparison(client, corruption):
    config = recipe(client)
    changes = {"schema": ("schema", "old"), "arms": ("arms", client.ARMS[:-1]),
               "reference": ("reference_arm", "history_l2_full"),
               "alpha": ("initial_alpha", 1.), "bool_alpha": ("initial_alpha", True),
               "gain": ("history_l2_gain", 1.), "bool_gain": ("history_l2_gain", True),
               "short": ("short_kernel_len", 4), "bool_short": ("short_kernel_len", True)}
    if corruption in changes:
        key, value = changes[corruption]
        config[key] = value
    elif corruption == "full":
        config["kernel"]["kernel_len"] = 3
    else:
        config["kernel"]["step"] = True if corruption == "bool_step" else .5
    with pytest.raises(ValueError):
        client.adapter_for(client.ARMS[0], config, 41)


def test_source_binding_and_protocol_keep_previous_data_budget(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kwargs: captured.update(kwargs))
    client.main()
    assert captured["arms"] == client.ARMS
    assert set(captured["adapter_sources"]) == {"history_factorial", "fractional_bridge"}
    assert all(path.is_file() for path in captured["adapter_sources"].values())
    source = Path(client.__file__)
    config = json.loads(source.with_name("hf_fractional_pride_history_factorial.json").read_text())
    previous = json.loads(source.with_name("hf_fractional_pride_two_lag.json").read_text())
    for key in ("model_snapshot", "corpus_sha256", "transfer_sha256", "block", "features",
                "seeds", "kernel", "steps", "batch_size", "block_size", "development_blocks",
                "transfer_blocks", "evaluate_every", "checkpoint_every", "learning_rate", "strength", "threads"):
        assert config[key] == previous[key]
    client.validate_protocol(config)


def test_four_hf_arms_resume_and_raw_full_replays_previous_recipe(client, summary_module, tmp_path, monkeypatch):
    driver, config = client.study, recipe(client)
    tokens = torch.arange(48).reshape(8, 6) % 32

    def setup(name):
        directory = tmp_path / name
        directory.mkdir()
        torch.manual_seed(139)
        torch.set_num_threads(2)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
            resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
        )).eval().requires_grad_(False)
        parent = model.transformer.h[0]
        plan = {"study_id": "history-factorial-test", "config": config,
                "base_parameter_sha256": driver.pilot.model_digest(model),
                "batch_schedules": {"41": driver.pilot.schedule(41, 8, 5, 2)}}
        journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
        return [directory, model, parent, parent.mlp, plan, journal]

    def run(state, callback=None):
        directory, model, parent, original, plan, journal = state
        driver.run_training(model, parent, "mlp", original, tokens, tokens[:2], plan,
                            directory, journal, after_checkpoint=callback,
                            adapter_factory=client.adapter_for)

    full = setup("full")
    run(full)
    interrupted = setup("interrupted")

    def stop(key, cursor):
        if key == "41:history_l2_full" and cursor == 2:
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        run(interrupted, stop)
    with pytest.raises(ValueError, match="endpoint evaluation is locked"):
        driver.endpoint_gate(interrupted[-1], interrupted[-2], interrupted[0])
    resumed = setup("resumed")
    resumed[0] = interrupted[0]
    resumed[-1] = json.loads((interrupted[0] / "journal.json").read_text())
    run(resumed)
    saved = {}
    for key, entry in full[-1]["runs"].items():
        left = driver.load_checkpoint(full[0], entry["checkpoint"], full[-2]["study_id"], key)
        right = driver.load_checkpoint(resumed[0], resumed[-1]["runs"][key]["checkpoint"], resumed[-2]["study_id"], key)
        assert driver.pilot.equal_state(left, right)
        assert entry["frozen_base_unchanged"] and entry["resume_next_update_equal"]
        assert all(r["gate_gradient_l2"] > 0 and r["local_gate_gradient_l2"] > 0 for r in left["records"])
        assert left["records"][0]["log_alpha_gradient"] == 0
        assert all(r["log_alpha_trainable"] for r in left["records"])
        assert any(r["log_alpha_gradient"] != 0 for r in left["records"][1:])
        saved[key] = left

    import hf_fractional_two_lag_study as previous

    old_config = {**config, "schema": "spiraltorch.fractional_two_lag_protocol.v1",
                  "initial_orders": previous.INITIAL_ORDERS}
    old = previous.adapter_for("history_learned_two", old_config, 41)
    optimizer = driver.make_optimizer(old, "history_learned_two", old_config, None)
    directory, model, parent, original, plan, journal = full
    parent.mlp = torch.nn.Sequential(original, old)
    records = []
    try:
        for i, indices in enumerate(plan["batch_schedules"]["41"][:config["steps"]]):
            row = driver.pilot.update(model, old, optimizer, tokens[indices])
            row.update(step=i+1, batch_indices=indices)
            records.append(row)
    finally:
        parent.mlp = original
    current = saved["41:history_raw_full"]
    assert records == current["records"]
    for name, parameter in old.named_parameters():
        assert torch.equal(parameter, current["adapter"][name])
    assert driver.pilot.equal_state(optimizer.state_dict(), current["optimizer"])

    result = driver.run_endpoints(model, parent, "mlp", original, {"tail": tokens[:2]},
                                  plan, directory, journal, adapter_factory=client.adapter_for,
                                  result_schema="spiraltorch.fractional_history_factorial_study.v1")
    assert driver.completed_result(directory, journal, plan) == result
    plan["data"] = {"evaluation_block_hashes": {"tail": driver.block_hashes(tokens[:2])}}
    original_import = builtins.__import__

    def no_torch_import(name, *args, **kwargs):
        assert name != "torch", "receipt summary must not import Torch"
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_torch_import)
        report = summary_module.summarize(plan, result, journal, journal["results_sha256"])
    assert report["primary_updates"] == 16 and report["initial_filter_receipt_status"] == "passed"
    assert len(report["paired_factorial_contrasts"]["tail"]) == 5
    assert set(report["order_trajectories"]) == set(journal["runs"])
    assert all(row["nonzero_order_gradient_steps"] > 0 for row in report["order_trajectories"].values())


def receipts(client):
    config = recipe(client)
    config["steps"] = 2
    runs = {}
    for arm in client.ARMS:
        initial = float(torch.tensor(math.log(2.)))
        after = initial + .01
        records = [{"loss": 2., "gate_gradient_l2": .1, "local_gate_gradient_l2": .1,
                    "gate_before_update_l2": 0., "local_gate_before_update_l2": 0.,
                    "log_alpha_trainable": True, "log_alpha_gradient": 0.,
                    "log_alpha_before_update": initial, "alpha_before_update": 2.,
                    "log_alpha_after_update": initial, "alpha_after_update": 2.},
                   {"loss": 1.9, "gate_gradient_l2": .1, "local_gate_gradient_l2": .1,
                    "gate_before_update_l2": .1, "local_gate_before_update_l2": .1,
                    "log_alpha_trainable": True, "log_alpha_gradient": .1,
                    "log_alpha_before_update": initial, "alpha_before_update": 2.,
                    "log_alpha_after_update": after, "alpha_after_update": math.exp(after)}]
        runs[f"41:{arm}"] = {"initial_parameter_sha256": "a"*64, "parameter_count": 17,
                              "trainable_parameter_count": 17, "records": records,
                              "final_log_alpha": after, "final_alpha": math.exp(after)}
    measured = {f"41:{arm}": {"tail": v} for arm, v in zip(client.ARMS, (2., 1.9, 1.8, 1.6))}
    return config, runs, measured


def test_factorial_contrasts_and_failed_initial_control_are_not_suppressed(client, summary_module):
    config, runs, measured = receipts(client)
    report, trajectories, parity = summary_module.history_factorial_report(config, runs, measured, {"tail": []})
    for name, expected in zip(("raw_full_minus_raw_short", "l2_full_minus_l2_short",
                              "l2_short_minus_raw_short", "l2_full_minus_raw_full",
                              "length_by_normalization_interaction"), (-.1, -.2, -.2, -.3, -.1)):
        assert report["tail"][name]["mean_ce_difference"] == pytest.approx(expected)
    assert len(trajectories) == 4 and parity["41"]["status"] == "passed"
    runs["41:history_l2_full"]["records"][0]["gate_gradient_l2"] += .001
    _, _, parity = summary_module.history_factorial_report(config, runs, measured, {"tail": []})
    assert parity["41"] == {"status": "failed", "first_update_receipts_equal": False}


@pytest.mark.parametrize("corruption", ["pairing", "count", "mode", "empty", "gradient", "initial_gate",
                                       "endpoint", "continuity", "gain", "kernel", "initial", "receipt"])
def test_summary_rejects_broken_factorial_controls(client, summary_module, corruption):
    config, runs, measured = receipts(client)
    row = runs["41:history_l2_full"]
    if corruption == "pairing":
        row["initial_parameter_sha256"] = "b"*64
    elif corruption == "count":
        row["trainable_parameter_count"] -= 1
    elif corruption == "mode":
        row["records"][0]["log_alpha_trainable"] = False
    elif corruption == "empty":
        row["records"] = []
    elif corruption == "gradient":
        row["records"][1]["log_alpha_gradient"] = float("nan")
    elif corruption == "initial_gate":
        row["records"][0]["gate_before_update_l2"] = .1
    elif corruption == "endpoint":
        row["final_alpha"] *= 1.1
    elif corruption == "continuity":
        row["records"][1]["log_alpha_before_update"] += .1
    elif corruption == "gain":
        config["history_l2_gain"] = 1.
    elif corruption == "kernel":
        config["short_kernel_len"] = 4
    elif corruption == "initial":
        config["initial_alpha"] = True
    else:
        row["records"][1]["loss"] = float("inf")
    with pytest.raises(ValueError):
        summary_module.history_factorial_report(config, runs, measured, {"tail": []})
