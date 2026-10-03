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
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_fractional_lag_study.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("lag_study_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def summary_module():
    source = Path(__file__).resolve().parents[3] / "tools" / "summarize_wave_gate_long_horizon.py"
    spec = importlib.util.spec_from_file_location("lag_summary_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {"schema": "spiraltorch.fractional_lag_protocol.v1",
            "features": 8, "strength": .1, "initial_orders": dict(client.INITIAL_ORDERS),
            "kernel": {"kernel_len": 4, "step": 1.0}, "learning_rate": .01,
            "steps": 4, "batch_size": 2, "checkpoint_every": 2, "evaluate_every": 2,
            "arms": client.ARMS, "seeds": [41], "reference_arm": "lag1"}


def test_identity_counts_rng_and_strict_arm_recipes(client):
    config = recipe(client)
    for arm in client.ARMS:
        before = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(before, torch.get_rng_state())
        assert all(p.device.type == "cpu" for p in adapter.parameters())
        assert sum(p.numel() for p in adapter.parameters()) == (16 if arm == "lag1" else 17)
        count = 17 if arm.startswith("history_learned_") else 16
        assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == count
        x = torch.linspace(-1, 1, 96).reshape(2, 6, 8)
        assert torch.equal(adapter(x), x)
        restored = client.adapter_for(arm, config, 47)
        restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
        assert client.study.pilot.equal_state(adapter.state_dict(), restored.state_dict())
    adapters = {arm: client.adapter_for(arm, config, 41) for arm in client.ARMS[1:]}
    assert (client.study.pilot.model_digest(adapters["history_fixed_one"])
            == client.study.pilot.model_digest(adapters["history_learned_one"]))
    for left in adapters:
        for right in adapters:
            if left != right:
                with pytest.raises(ValueError, match="recipe"):
                    adapters[left].load_state_dict(adapters[right].state_dict())
    with pytest.raises(ValueError, match="unrecognized"):
        client.adapter_for("off", config, 41)


@pytest.mark.parametrize("time,features", [(1, 3), (6, 3), (128, 768)])
def test_shift_and_rust_alpha_one_outputs_input_and_gate_gradients(client, time, features):
    config = recipe(client)
    config.update(features=features, kernel={"kernel_len": 32, "step": 1.0})
    lag = client.adapter_for("lag1", config, 41)
    fixed = client.adapter_for("history_fixed_one", config, 41)
    with torch.no_grad():
        for module in (lag, fixed):
            module.gate.copy_(torch.linspace(-.6, .7, features))
            module.local_gate.copy_(torch.linspace(.4, -.3, features))
    generator = torch.Generator().manual_seed(223)
    # Noncontiguous, including the real training dimensions.
    x = torch.randn(2, features, time, generator=generator).transpose(1, 2).requires_grad_()
    upstream = torch.randn(x.shape, generator=generator)
    ordinary, rust = lag(x), fixed(x)
    assert torch.equal(ordinary, rust)
    a = torch.autograd.grad(ordinary, (x, lag.gate, lag.local_gate), upstream)
    b = torch.autograd.grad(rust, (x, fixed.gate, fixed.local_gate), upstream)
    torch.testing.assert_close(a[0], b[0], rtol=3e-6, atol=3e-6)
    assert all(torch.equal(l, r) for l, r in zip(a[1:], b[1:]))


def test_shift_does_not_wrap_or_mix_lanes_and_integer_order_still_has_tail_derivative(client):
    lag = client.CausalLagGate(3, strength=.1, kernel_len=4)
    x = torch.arange(36, dtype=torch.float32).reshape(2, 6, 3).requires_grad_()
    y = lag.history(x)
    assert torch.count_nonzero(y[:, 0]) == 0
    assert torch.equal(y[:, 1:], -x[:, :-1])
    modified = x.detach().clone()
    modified[0, 2:] = 90
    modified[1] = -80
    assert torch.equal(y[0, :3], lag.history(modified)[0, :3])
    gradient = torch.autograd.grad(y[0, 2, 1], x)[0]
    assert gradient[0, 1, 1] == -1 and torch.count_nonzero(gradient) == 1
    impulse = torch.tensor([1., 0., 0., 0.]).reshape(1, 4, 1)
    alpha = torch.tensor(1., requires_grad=True)
    history = client.st.fractional_gl_history_autograd(
        impulse, alpha, axis=1, kernel=client.st.FractionalGlKernel(kernel_len=4))
    assert history[0, 2, 0] == 0
    assert torch.autograd.grad(history[0, 2, 0], alpha)[0] == .5


@pytest.mark.parametrize("invalid", ["boolean_step", "step", "taps", "bool_order", "order", "arm_order"])
def test_recipe_rejects_changed_semantics(client, invalid):
    config = recipe(client)
    if invalid == "boolean_step":
        config["kernel"]["step"] = True
    elif invalid == "step":
        config["kernel"]["step"] = .5
    elif invalid == "taps":
        config["kernel"]["kernel_len"] = 1
    else:
        config["initial_orders"]["history_fixed_one" if invalid == "bool_order"
                                 else "history_learned_half"] = True if invalid == "bool_order" else .75
        if invalid == "arm_order":
            config["initial_orders"]["surprise"] = .5
    with pytest.raises(ValueError):
        client.adapter_for("lag1", config, 41)


def test_shift_validates_domain_and_budget(client):
    lag = client.CausalLagGate(3, strength=.1, kernel_len=4, max_values=36, max_products=144)
    for x in (torch.zeros(2, 3), torch.zeros(0, 6, 3), torch.zeros(2, 0, 3),
              torch.zeros(2, 6, 4), torch.full((2, 6, 3), float("nan")),
              torch.zeros(2, 6, 3, dtype=torch.float64), torch.zeros(3, 6, 3)):
        with pytest.raises((ValueError, TypeError)):
            lag(x)
    with torch.no_grad():
        lag.gate[0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        lag(torch.zeros(2, 6, 3))
    with pytest.raises(ValueError, match="budget"):
        client.CausalLagGate(3, strength=.1, kernel_len=4, max_products=100)(torch.zeros(2, 6, 3))


def test_protocol_binds_sources_and_comparable_preceding_inputs(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kwargs: captured.update(kwargs))
    client.main()
    assert captured["arms"] == client.ARMS
    assert set(captured["adapter_sources"]) == {"lag_study", "fractional_bridge"}
    assert all(p.is_file() for p in captured["adapter_sources"].values())
    source = Path(client.__file__)
    config = json.loads(source.with_name("hf_fractional_pride_lag.json").read_text())
    previous = json.loads(source.with_name("hf_fractional_pride_history.json").read_text())
    assert config["arms"] == client.ARMS and config["initial_orders"] == client.INITIAL_ORDERS
    assert config["reference_arm"] == "lag1"
    for key in ("model_snapshot", "corpus_sha256", "transfer_sha256", "block", "features",
                "seeds", "kernel", "steps", "batch_size", "block_size", "development_blocks",
                "transfer_blocks", "evaluate_every", "checkpoint_every", "learning_rate",
                "strength", "threads"):
        assert config[key] == previous[key]


def test_four_hf_arms_resume_and_same_math_updates_match_exactly(client, summary_module, tmp_path):
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
        plan = {"study_id": "lag-study-test", "config": config,
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
        if key == "41:history_learned_one" and cursor == 2:
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        run(interrupted, stop)
    with pytest.raises(ValueError, match="all planned runs must finish"):
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
        saved[key] = left
        rows = left["records"]
        assert all(r["gate_gradient_l2"] > 0 and r["local_gate_gradient_l2"] > 0 for r in rows)
        if key.endswith("history_fixed_one"):
            assert all(r["log_alpha_gradient"] is None and not r["log_alpha_trainable"] for r in rows)
            assert all(r["alpha_after_update"] == 1. for r in rows)
        elif "history_learned_" in key:
            assert rows[0]["log_alpha_gradient"] == 0
            assert all(r["log_alpha_gradient"] != 0 for r in rows[1:])
            assert rows[0]["log_alpha_before_update"] != rows[-1]["log_alpha_after_update"]
    ordinary, fixed = (saved[f"41:{arm}"] for arm in ("lag1", "history_fixed_one"))
    for field in ("gate", "local_gate"):
        assert torch.equal(ordinary["adapter"][field], fixed["adapter"][field])
    # Optimizer IDs follow registration order, including the frozen alpha.
    for lag_id, gl_id in ((0, 0), (1, 2)):
        assert driver.pilot.equal_state(ordinary["optimizer"]["state"][lag_id], fixed["optimizer"]["state"][gl_id])
    assert 1 not in fixed["optimizer"]["state"]
    directory, model, parent, original, plan, journal = full
    result = driver.run_endpoints(model, parent, "mlp", original, {"tail": tokens[:2]},
                                  plan, directory, journal, adapter_factory=client.adapter_for,
                                  result_schema="spiraltorch.fractional_lag_study.v1")
    assert driver.completed_result(directory, journal, plan) == result
    plan["data"] = {"evaluation_block_hashes": {"tail": driver.block_hashes(tokens[:2])}}
    report = summary_module.summarize(plan, result, journal, journal["results_sha256"])
    assert report["order_trajectories"]["41:history_fixed_one"]["final_alpha"] == 1.
    assert report["order_trajectories"]["41:history_learned_one"]["nonzero_order_gradient_steps"] == 3
    assert report["order_trajectories"]["41:history_learned_half"]["nonzero_order_gradient_steps"] == 3
    assert report["same_math_receipt_parity"]["41"] == {
        "update_receipts_equal": True, "development_equal": True,
        "endpoint_block_losses_equal": {"tail": True}}


def receipts(client):
    config, runs = recipe(client), {}
    for arm in client.ARMS:
        learned = arm.startswith("history_learned_")
        initial = math.log(config["initial_orders"].get(arm, 1.))
        final = initial + .1 if learned else initial
        row = {"initial_parameter_sha256": "a" * 64,
               "parameter_count": 16 + int(arm != "lag1"),
               "trainable_parameter_count": 16 + int(learned),
               "records": [{"loss": 2., "gate_gradient_l2": 1., "local_gate_gradient_l2": 2.,
                            "gate_before_update_l2": 0., "local_gate_before_update_l2": 0.,
                            "log_alpha_before_update": a, "log_alpha_after_update": b,
                            "alpha_before_update": math.exp(a), "alpha_after_update": math.exp(b),
                            "log_alpha_trainable": learned,
                            "log_alpha_gradient": (0. if i == 0 else .1) if learned else None}
                           for i, (a, b) in enumerate([(initial, initial), (initial, final)])],
               "final_log_alpha": final, "final_alpha": math.exp(final),
               "development": [], "scores": {"tail": {"block_losses": [2.]}}}
        runs[f"41:{arm}"] = row
    return config, runs


@pytest.mark.parametrize("corruption", ["initial", "pairing", "count", "mode", "fixed", "gradient",
                                         "endpoint", "continuity", "empty", "nan", "receipt", "kernel"])
def test_summary_rejects_broken_lag_controls(client, summary_module, corruption):
    config, runs = receipts(client)
    row = runs["41:history_fixed_one"]
    receipt = row["records"][0]
    if corruption == "initial":
        config["initial_orders"]["history_learned_half"] = 1.
    elif corruption == "pairing":
        row["initial_parameter_sha256"] = "b" * 64
    elif corruption == "count":
        row["trainable_parameter_count"] = 17
    elif corruption == "mode":
        receipt["log_alpha_trainable"] = True
    elif corruption == "fixed":
        row["records"][1].update(log_alpha_after_update=.1, alpha_after_update=math.exp(.1))
        row.update(final_log_alpha=.1, final_alpha=math.exp(.1))
    elif corruption == "gradient":
        runs["41:history_learned_one"]["records"][0]["log_alpha_gradient"] = .1
    elif corruption == "endpoint":
        row["final_alpha"] = .6
    elif corruption == "continuity":
        row["records"][1].update(log_alpha_before_update=.1, alpha_before_update=math.exp(.1))
    elif corruption == "empty":
        row["records"] = []
    elif corruption == "nan":
        receipt["alpha_before_update"] = float("nan")
    elif corruption == "receipt":
        receipt.pop("local_gate_gradient_l2")
    else:
        config["kernel"]["step"] = True
    with pytest.raises(ValueError):
        summary_module.fractional_lag_report(config, runs, {}, {})


def test_summary_keeps_parity_failures_visible_and_initialization_separate(client, summary_module):
    config, runs = receipts(client)
    measured = {f"41:{arm}": {"tail": value} for arm, value in zip(client.ARMS, (2., 2., 1.8, 1.9))}
    contrasts, orders, parity = summary_module.fractional_lag_report(config, runs, measured, {"tail": []})
    assert contrasts["tail"]["fixed_one_minus_lag1"]["mean_ce_difference"] == 0
    assert contrasts["tail"]["learned_one_minus_fixed_one"]["mean_ce_difference"] == pytest.approx(-.2)
    assert contrasts["tail"]["learned_half_minus_learned_one"]["mean_ce_difference"] == pytest.approx(.1)
    assert orders["41:history_learned_one"]["initial_alpha"] == 1.
    assert orders["41:history_learned_half"]["initial_alpha"] == .5
    assert parity["41"]["update_receipts_equal"]
    runs["41:history_fixed_one"]["records"][1]["loss"] += .1
    runs["41:history_fixed_one"]["scores"]["tail"]["block_losses"] = [2.1]
    _, _, parity = summary_module.fractional_lag_report(config, runs, measured, {"tail": []})
    assert not parity["41"]["update_receipts_equal"]
    assert not parity["41"]["endpoint_block_losses_equal"]["tail"]
