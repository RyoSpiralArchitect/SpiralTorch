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
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_fractional_history_study.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("history_study_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def summary_module():
    source = Path(__file__).resolve().parents[3] / "tools" / "summarize_wave_gate_long_horizon.py"
    spec = importlib.util.spec_from_file_location("history_summary_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {"schema": "spiraltorch.fractional_history_protocol.v1",
            "features": 8, "strength": .1, "initial_alpha": .5, "initial_decay": .5,
            "kernel": {"kernel_len": 4, "step": 1.0}, "learning_rate": .01,
            "steps": 4, "batch_size": 2, "checkpoint_every": 2, "evaluate_every": 2,
            "arms": client.ARMS, "seeds": [41], "reference_arm": "pointwise"}


def test_identity_real_parameter_counts_rng_and_recipes(client):
    config = recipe(client)
    for arm in client.ARMS:
        before = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(before, torch.get_rng_state())
        assert all(p.device.type == "cpu" for p in adapter.parameters())
        assert sum(p.numel() for p in adapter.parameters()) == (8 if arm == "pointwise" else 17)
        expected_trainable = 8 if arm == "pointwise" else (16 if arm == "history_fixed" else 17)
        assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == expected_trainable
        x = torch.linspace(-1, 1, 96).reshape(2, 6, 8)
        assert torch.equal(adapter(x), x)
        restored = client.adapter_for(arm, config, 47)
        restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
        assert client.study.pilot.equal_state(adapter.state_dict(), restored.state_dict())
    fixed = client.adapter_for("history_fixed", config, 41)
    learned = client.adapter_for("history_learned", config, 41)
    assert client.study.pilot.model_digest(fixed) == client.study.pilot.model_digest(learned)
    with pytest.raises(ValueError, match="recipe"):
        fixed.load_state_dict(learned.state_dict())
    with pytest.raises(ValueError, match="unrecognized"):
        client.adapter_for("off", config, 41)


@pytest.mark.parametrize("time", [1, 6])
@pytest.mark.parametrize("initial_decay", [.1, .5, .9])
def test_ordinary_ema_output_and_gradients_match_dense_f64_sum(client, time, initial_decay):
    adapter = client.CausalEmaGate(3, strength=.1, initial_decay=initial_decay, kernel_len=4)
    x = torch.linspace(-.8, .9, 6*time).reshape(2, 3, time).transpose(1, 2).requires_grad_()
    actual = adapter.history(x)
    d = adapter.logit_decay.double().sigmoid()
    reference = torch.stack([
        x[:, t].double() * 0 + d * 0 + sum((1-d)*d**(k-1)*x[:, t-k].double()
                                           for k in range(1, min(t+1, 4)))
        for t in range(time)
    ], dim=1).float()
    torch.testing.assert_close(actual, reference, rtol=3e-6, atol=3e-6)
    actual_gradients = torch.autograd.grad(actual, (x, adapter.logit_decay), x.detach().cos())
    reference_gradients = torch.autograd.grad(reference, (x, adapter.logit_decay), x.detach().cos())
    for a, r in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(a, r, rtol=3e-5, atol=3e-6)


def test_ordinary_ema_is_strictly_past_lane_isolated_and_bounded(client):
    adapter = client.CausalEmaGate(3, strength=.1, initial_decay=.5, kernel_len=4)
    x = torch.linspace(-1, 1, 36).reshape(2, 6, 3).requires_grad_()
    y = adapter.history(x)
    modified = x.detach().clone()
    modified[0, 1:] = 90
    modified[1] = -80
    assert torch.equal(y[0, :2], adapter.history(modified)[0, :2])
    gradient = torch.autograd.grad(y[0, 1, 0], x)[0]
    assert gradient[0, 0, 0] != 0 and torch.count_nonzero(gradient) == 1
    old = x.detach().clone()
    old[:, 0] += 70
    assert torch.equal(y[:, 4:], adapter.history(old)[:, 4:])
    for invalid in [0., 1., -1., float("nan"), True]:
        with pytest.raises(ValueError, match="decay"):
            client.CausalEmaGate(3, strength=.1, initial_decay=invalid, kernel_len=4)
    with pytest.raises(ValueError, match="budget"):
        client.CausalEmaGate(3, strength=.1, initial_decay=.5, kernel_len=4,
                            max_products=100)(x)
    with pytest.raises(ValueError, match="nonfinite"):
        adapter(torch.full_like(x, float("nan")))


@pytest.mark.parametrize("order", [.5, 1.0])
def test_gl_history_at_training_shape_matches_torch_polynomial(client, order):
    generator = torch.Generator().manual_seed(217)
    x = torch.randn(2, 128, 768, generator=generator, requires_grad=True)
    a = torch.tensor(order, requires_grad=True)
    upstream = torch.randn(x.shape, generator=generator)
    actual = client.st.fractional_gl_history_autograd(
        x, a, axis=1, kernel=client.st.FractionalGlKernel(kernel_len=32))
    alpha = a.double()
    coefficients = [torch.ones_like(alpha)]
    for k in range(1, 32):
        coefficients.append(coefficients[-1] * (k-1-alpha) / k)
    coefficients[0] = alpha * 0
    weight = torch.stack(coefficients).flip(0).reshape(1, 1, 32).repeat(768, 1, 1)
    reference = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x.double().transpose(1, 2), (31, 0)), weight,
        groups=768).transpose(1, 2).float()
    torch.testing.assert_close(actual, reference, rtol=3e-6, atol=3e-6)
    actual_gradients = torch.autograd.grad(actual, (x, a), upstream)
    reference_gradients = torch.autograd.grad(reference, (x, a), upstream)
    for observed, wanted in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(observed, wanted, rtol=3e-5, atol=3e-5)


def test_protocol_binds_every_custom_source_and_freezes_four_arms(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kwargs: captured.update(kwargs))
    client.main()
    assert captured["arms"] == client.ARMS
    assert set(captured["adapter_sources"]) == {"history_study", "memory_study", "fractional_bridge"}
    assert all(p.is_file() for p in captured["adapter_sources"].values())
    config = json.loads(Path(client.__file__).with_name("hf_fractional_pride_history.json").read_text())
    assert config["arms"] == client.ARMS and config["steps"] == 512
    assert config["seeds"] == [41, 43, 47] and config["reference_arm"] == "pointwise"
    assert config["initial_decay"] == config["initial_alpha"] == .5


def test_all_four_hf_arms_learn_and_resume_without_opening_endpoints(client, summary_module, tmp_path):
    driver, config = client.study, recipe(client)
    tokens = torch.arange(48).reshape(8, 6) % 32

    def setup(name):
        directory = tmp_path / name
        directory.mkdir()
        torch.manual_seed(137)
        torch.set_num_threads(2)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
            resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
        )).eval().requires_grad_(False)
        parent = model.transformer.h[0]
        plan = {"study_id": "history-study-test", "config": config,
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
        if key == "41:history_learned" and cursor == 2:
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        run(interrupted, stop)
    with pytest.raises(ValueError, match="until training completes"):
        driver.endpoint_gate(interrupted[-1], interrupted[-2], interrupted[0])
    resumed = setup("resumed")
    resumed[0] = interrupted[0]
    resumed[-1] = json.loads((interrupted[0] / "journal.json").read_text())
    run(resumed)
    for key, entry in full[-1]["runs"].items():
        left = driver.load_checkpoint(full[0], entry["checkpoint"], full[-2]["study_id"], key)
        right = driver.load_checkpoint(resumed[0], resumed[-1]["runs"][key]["checkpoint"], resumed[-2]["study_id"], key)
        assert driver.pilot.equal_state(left, right)
        assert entry["frozen_base_unchanged"] and entry["resume_next_update_equal"]
        if key.endswith("pointwise"):
            continue
        rows = left["records"]
        assert all(r["gate_gradient_l2"] > 0 and r["local_gate_gradient_l2"] > 0 for r in rows)
        name = "logit_decay" if key.endswith("ema_learned") else "log_alpha"
        if key.endswith("history_fixed"):
            assert all(r["log_alpha_gradient"] is None and not r["log_alpha_trainable"] for r in rows)
            assert all(r["alpha_after_update"] == .5 for r in rows)
        else:
            assert rows[0][f"{name}_gradient"] == 0
            assert any(r[f"{name}_gradient"] != 0 for r in rows[1:])
            assert rows[0][f"{name}_before_update"] != rows[-1][f"{name}_after_update"]
    directory, model, parent, original, plan, journal = full
    result = driver.run_endpoints(model, parent, "mlp", original, {"tail": tokens[:2]},
                                  plan, directory, journal, adapter_factory=client.adapter_for,
                                  result_schema="spiraltorch.fractional_history_study.v1")
    assert driver.completed_result(directory, journal, plan) == result
    plan["data"] = {"evaluation_block_hashes": {"tail": driver.block_hashes(tokens[:2])}}
    report = summary_module.summarize(plan, result, journal, journal["results_sha256"])
    assert report["order_trajectories"]["41:history_fixed"]["final_alpha"] == .5
    assert report["order_trajectories"]["41:history_learned"]["nonzero_order_gradient_steps"] == 3
    assert report["decay_trajectories"]["41:ema_learned"]["nonzero_decay_gradient_steps"] == 3
    assert "learned_minus_ema" in report["paired_fractional_contrasts"]["tail"]


@pytest.mark.parametrize("corruption", ["empty", "nonfinite", "continuity", "initial", "first_gradient", "decay", "endpoint"])
def test_summary_rejects_broken_ema_trajectory(summary_module, corruption):
    row = {"records": [
        {"logit_decay_before_update": 0., "logit_decay_after_update": 0.,
         "logit_decay_gradient": 0., "decay_after_update": .5},
        {"logit_decay_before_update": 0., "logit_decay_after_update": .1,
         "logit_decay_gradient": -.2, "decay_after_update": 1/(1+math.exp(-.1))}],
        "final_logit_decay": .1, "final_decay": 1/(1+math.exp(-.1))}
    config = {"seeds": [41], "initial_decay": .5}
    assert summary_module.ema_trajectories(config, {"41:ema_learned": row})
    if corruption == "empty":
        row["records"] = []
    elif corruption == "nonfinite":
        row["records"][1]["logit_decay_gradient"] = float("nan")
    elif corruption == "continuity":
        row["records"][1]["logit_decay_before_update"] = .2
    elif corruption == "initial":
        config["initial_decay"] = .4
    elif corruption == "first_gradient":
        row["records"][0]["logit_decay_gradient"] = .1
    elif corruption == "decay":
        row["records"][1]["decay_after_update"] = .8
        row["final_decay"] = .8
    else:
        row["final_logit_decay"] = .2
    with pytest.raises(ValueError):
        summary_module.ema_trajectories(config, {"41:ema_learned": row})
