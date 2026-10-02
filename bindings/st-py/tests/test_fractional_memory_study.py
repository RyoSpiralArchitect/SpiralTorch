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
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_fractional_memory_study.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("fractional_study_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {"features": 8, "strength": 0.1, "initial_alpha": 0.5,
            "kernel": {"kernel_len": 4, "step": 1.0}, "learning_rate": 0.01,
            "steps": 4, "batch_size": 2, "checkpoint_every": 2, "evaluate_every": 2,
            "arms": client.ARMS, "seeds": [41]}


@pytest.fixture
def summary_module():
    source = Path(__file__).resolve().parents[3] / "tools" / "summarize_wave_gate_long_horizon.py"
    spec = importlib.util.spec_from_file_location("fractional_summary_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_parameter_budgets_rng_and_fixed_order_recipe(client):
    config = recipe(client)
    for arm in client.ARMS:
        before = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(torch.get_rng_state(), before)
        assert all(p.device.type == "cpu" for p in adapter.parameters())
        assert sum(p.numel() for p in adapter.parameters()) == (8 if arm == "pointwise" else 9)
        assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == (9 if arm == "fractional_learned" else 8)
        x = torch.arange(96, dtype=torch.float32).reshape(2, 6, 8) / 100
        assert torch.equal(adapter(x), x)
        restored = client.adapter_for(arm, config, 47)
        restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
        assert client.study.pilot.equal_state(adapter.state_dict(), restored.state_dict())
    fixed = client.adapter_for("fractional_fixed", config, 41)
    learned = client.adapter_for("fractional_learned", config, 41)
    assert client.study.pilot.model_digest(fixed) == client.study.pilot.model_digest(learned)
    with pytest.raises(ValueError, match="recipe"):
        fixed.load_state_dict(learned.state_dict())
    with pytest.raises(ValueError, match="unrecognized"):
        client.adapter_for("off", config, 41)


def test_controls_isolate_history_and_order_gradients(client):
    config = recipe(client)
    x = torch.linspace(-1, 1, 96).reshape(2, 6, 8).requires_grad_()
    altered = x.detach().clone()
    altered[0, 0] += 10
    outputs = {}
    for arm in client.ARMS:
        adapter = client.adapter_for(arm, config, 41)
        with torch.no_grad():
            adapter.gate.fill_(0.4)
        output = adapter(x)
        outputs[arm] = output.detach()
        torch.testing.assert_close(output[1], adapter(altered)[1], rtol=0, atol=0)
        if arm == "pointwise":
            assert torch.equal(output[0, 1:], adapter(altered)[0, 1:])
        else:
            assert not torch.equal(output[0, 1:4], adapter(altered)[0, 1:4])
        output.square().sum().backward()
        if arm != "pointwise":
            assert (adapter.log_alpha.grad is not None) == (arm == "fractional_learned")
            if arm == "fractional_learned":
                assert adapter.log_alpha.grad.item() != 0
    assert torch.equal(outputs["fractional_fixed"], outputs["fractional_learned"])


def test_protocol_binds_real_bridge_and_separates_extra_scalar(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kwargs: captured.update(kwargs))
    client.main()
    assert captured["arms"] == client.ARMS
    assert set(captured["adapter_sources"]) == {"fractional_study", "fractional_bridge"}
    assert all(p.is_file() for p in captured["adapter_sources"].values())
    config = json.loads(Path(client.__file__).with_name("hf_fractional_pride_memory.json").read_text())
    assert config["steps"] == 512 and config["seeds"] == [41, 43, 47]
    assert config["reference_arm"] == "pointwise"
    assert config["kernel"]["kernel_len"] == 32 and config["kernel"]["step"] == 1


def test_real_hf_all_arms_preserve_frozen_order_and_resume_exactly(client, summary_module, tmp_path):
    driver = client.study
    config = recipe(client)
    tokens = torch.arange(48).reshape(8, 6) % 32

    def setup(name):
        directory = tmp_path / name
        directory.mkdir()
        torch.manual_seed(127)
        torch.set_num_threads(2)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
            resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
        )).eval().requires_grad_(False)
        parent = model.transformer.h[0]
        plan = {"study_id": "fractional-study-test", "config": config,
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
        if key == "41:fractional_learned" and cursor == 2:
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
        rows = left["records"]
        if key.endswith("fractional_fixed"):
            assert all(r["log_alpha_trainable"] is False and r["log_alpha_gradient"] is None for r in rows)
            assert all(r["log_alpha_before_update"] == r["log_alpha_after_update"] == rows[0]["log_alpha_before_update"] for r in rows)
        elif key.endswith("fractional_learned"):
            assert rows[0]["log_alpha_gradient"] == 0
            assert all(r["log_alpha_trainable"] is True for r in rows)
            assert any(r["log_alpha_gradient"] != 0 for r in rows[1:])
            assert rows[0]["log_alpha_before_update"] != rows[-1]["log_alpha_after_update"]
        if "fractional_" in key:
            assert [r["log_alpha_after_update"] for r in rows[:-1]] == [r["log_alpha_before_update"] for r in rows[1:]]
    directory, model, parent, original, plan, journal = full
    result = driver.run_endpoints(model, parent, "mlp", original, {"tail": tokens[:2]},
                                  plan, directory, journal, adapter_factory=client.adapter_for,
                                  result_schema="spiraltorch.fractional_memory_study.v1")
    for row in result["runs"]:
        assert row["trainable_parameter_count"] == (9 if row["run_key"].endswith("fractional_learned") else 8)
    assert driver.completed_result(directory, journal, plan) == result
    config.update(schema="spiraltorch.fractional_memory_protocol.v1", reference_arm="pointwise")
    plan["data"] = {"evaluation_block_hashes": {"tail": driver.block_hashes(tokens[:2])}}
    report = summary_module.summarize(plan, result, journal, journal["results_sha256"])
    assert report["order_trajectories"]["41:fractional_fixed"]["final_alpha"] == 0.5
    assert report["order_trajectories"]["41:fractional_learned"]["nonzero_order_gradient_steps"] == 3


def order_receipts():
    config = {"schema": "spiraltorch.fractional_memory_protocol.v1",
              "arms": ["pointwise", "fractional_fixed", "fractional_learned"],
              "reference_arm": "pointwise", "features": 8, "initial_alpha": 0.5, "seeds": [41]}
    runs = {}
    for arm in config["arms"]:
        learned = arm == "fractional_learned"
        initial, final = math.log(.5), math.log(.6 if learned else .5)
        runs[f"41:{arm}"] = {
            "initial_parameter_sha256": "a" * 64, "parameter_count": 8 + int(arm != "pointwise"),
            "trainable_parameter_count": 8 + int(learned),
            "records": [{"log_alpha_before_update": a, "log_alpha_after_update": b,
                         "alpha_before_update": math.exp(a), "alpha_after_update": math.exp(b),
                         "log_alpha_trainable": learned,
                         "log_alpha_gradient": (0. if i == 0 else .1) if learned else None}
                        for i, (a, b) in enumerate([(initial, initial), (initial, final)])],
            "final_log_alpha": final, "final_alpha": math.exp(final),
        }
    return config, runs


@pytest.mark.parametrize("corruption", ["pairing", "count", "mode", "alpha", "nan", "fixed", "gradient", "endpoint", "continuity"])
def test_summary_rejects_misreported_order_learning(summary_module, corruption):
    config, runs = order_receipts()
    row = runs["41:fractional_fixed"]
    receipt = row["records"][0]
    if corruption == "pairing":
        row["initial_parameter_sha256"] = "b" * 64
    elif corruption == "count":
        row["trainable_parameter_count"] = 9
    elif corruption == "mode":
        receipt["log_alpha_trainable"] = True
    elif corruption == "alpha":
        receipt["alpha_after_update"] = .4
    elif corruption == "nan":
        receipt["log_alpha_before_update"] = float("nan")
    elif corruption == "fixed":
        row["records"][1].update(log_alpha_after_update=math.log(.6), alpha_after_update=.6)
        row.update(final_log_alpha=math.log(.6), final_alpha=.6)
    elif corruption == "gradient":
        runs["41:fractional_learned"]["records"][1]["log_alpha_gradient"] = None
    elif corruption == "endpoint":
        row["final_alpha"] = .6
    else:
        row["records"][1].update(log_alpha_before_update=math.log(.6), alpha_before_update=.6)
    with pytest.raises(ValueError):
        summary_module.fractional_report(config, runs, {}, {})


def test_summary_keeps_order_effect_separate_from_history(summary_module):
    config, runs = order_receipts()
    measured = {"41:pointwise": {"tail": 2.}, "41:fractional_fixed": {"tail": 1.8},
                "41:fractional_learned": {"tail": 1.9}}
    contrasts, trajectories = summary_module.fractional_report(config, runs, measured, {"tail": []})
    assert contrasts["tail"]["fixed_minus_pointwise"]["mean_ce_difference"] == pytest.approx(-.2)
    assert contrasts["tail"]["learned_minus_fixed"]["mean_ce_difference"] == pytest.approx(.1)
    assert trajectories["41:fractional_fixed"]["nonzero_order_gradient_steps"] == 0
