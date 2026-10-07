import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import spiraltorch as st


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_fractional_gain_study.py"
    monkeypatch.syspath_prepend(str(source.parent))
    return load_module(source, "gain_study_test")


def recipe(client):
    config = json.loads(Path(client.__file__).with_name("hf_fractional_pride_gain.json").read_text())
    config.update(features=8, steps=4, block_size=6, batch_size=2, seeds=[41],
                  checkpoint_every=2, evaluate_every=2, learning_rate=.01)
    return config


def setup(client, directory):
    directory.mkdir()
    torch.manual_seed(197)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    config = recipe(client)
    driver = client.study
    plan = {"study_id": "gain-study-test", "config": config,
            "base_parameter_sha256": driver.pilot.model_digest(model),
            "batch_schedules": {"41": driver.pilot.schedule(41, 8, 5, 2)}}
    journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
    parent = model.transformer.h[0]
    return directory, model, parent, parent.mlp, plan, journal


def run(client, state, callback=None, optimizer_factory=None):
    directory, model, parent, original, plan, journal = state
    tokens = torch.arange(48).reshape(8, 6) % 32
    client.study.run_training(model, parent, "mlp", original, tokens, tokens[:2], plan,
                              directory, journal, after_checkpoint=callback,
                              adapter_factory=client.adapter_for, optimizer_factory=optimizer_factory)


def test_native_gain_observation_reuses_rust_not_torch_exp(monkeypatch):
    adapter = st.FractionalGainHistoryAdapter(2, initial_gain=1.5)
    expected = st.FractionalGlKernel.gain_from_log_gain(float(adapter.log_gain.detach()))
    def forbidden(*args, **kwargs):
        raise AssertionError("gain observation reconstructed Torch exp")
    monkeypatch.setattr(torch.Tensor, "exp", forbidden)
    assert adapter.gain == expected
    with torch.no_grad():
        adapter.log_gain.fill_(.3)
    assert adapter.gain == st.FractionalGlKernel.gain_from_log_gain(.3)


def test_frozen_gain_metadata_is_constant_and_legacy_records_stay_unchanged(client):
    assert client.study.pilot.gain_snapshot(st.FractionalHistoryAdapter(2)) == {}
    adapter = st.FractionalGainHistoryAdapter(2, kernel_len=3)
    adapter.log_gain.requires_grad_(False)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    class Loss:
        def __call__(self, value, labels):
            return SimpleNamespace(loss=adapter(value).square().mean())
    batch = torch.linspace(-.3, .8, 16).reshape(2, 4, 2)
    records = [client.study.pilot.update(Loss(), adapter, optimizer, batch) for _ in range(2)]
    for row in records:
        assert row["log_gain_trainable"] is False and row["log_gain_gradient"] is None
        assert row["log_gain_before_update"] == row["log_gain_after_update"] == 0
        assert row["gain_before_update"] == row["gain_after_update"] == 1
    assert records[0]["effective_history_gate_l2_before_update"] == 0
    assert records[1]["effective_history_gate_l2_after_update"] > 0


def test_identity_capacity_initial_maps_vjps_and_recipe_separation(client):
    config = recipe(client)
    outputs, gradients, adapters, hashes = [], [], [], []
    for arm in client.ARMS:
        rng = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(rng, torch.get_rng_state())
        assert sum(p.numel() for p in adapter.parameters()) == 18
        assert all(p.device.type == "cpu" and p.requires_grad for p in adapter.parameters())
        hashes.append(client.study.pilot.model_digest(adapter))
        x = torch.linspace(-.9, .8, 96).reshape(2, 6, 8).transpose(0, 1).contiguous().transpose(0, 1)
        x.requires_grad_()
        assert torch.equal(adapter(x), x)
        assert adapter.gain == pytest.approx(5**.5, rel=2e-7)
        with torch.no_grad():
            adapter.gate.copy_(torch.linspace(-.4, .5, 8))
            adapter.local_gate.copy_(torch.linspace(.3, -.2, 8))
        y = adapter(x)
        outputs.append(y)
        gradients.append(torch.autograd.grad(y, (x, adapter.gate, adapter.local_gate), x.cos()))
        work = x.double()
        first = torch.cat((torch.zeros_like(work[:, :1]), work[:, :-1]), 1)
        second = torch.cat((torch.zeros_like(work[:, :2]), work[:, :-2]), 1)
        reference = x + .1*adapter.local_gate.tanh()*x + .1*adapter.gate.tanh()*(-2*first+second).float()
        assert torch.allclose(y, reference, atol=2e-7, rtol=2e-6)
        restored = client.adapter_for(arm, config, 47)
        restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
        assert client.study.pilot.equal_state(adapter.state_dict(), restored.state_dict())
        adapters.append(adapter)
    assert hashes[1] == hashes[2] and hashes[0] != hashes[1]
    for y, ds in zip(outputs[1:], gradients[1:]):
        assert torch.allclose(y, outputs[0], atol=2e-7, rtol=2e-6)
        assert all(torch.allclose(a, b, atol=2e-7, rtol=2e-6) for a, b in zip(ds, gradients[0]))
    for i, adapter in enumerate(adapters):
        for j, other in enumerate(adapters):
            if i != j:
                with pytest.raises((ValueError, RuntimeError)):
                    adapter.load_state_dict(other.state_dict())


def test_ordinary_filter_causality_shape_and_gain_differentials(client):
    adapter = client.adapter_for(client.ARMS[0], recipe(client), 41)
    x = torch.linspace(-.4, .7, 96).reshape(2, 6, 8)
    with torch.no_grad():
        adapter.history_angle.fill_(.8)
        adapter.log_gain.fill_(.3)
    upstream = x.cos()
    y = adapter.history(x)
    gradients = torch.autograd.grad(y, (adapter.history_angle, adapter.log_gain), upstream)
    for parameter, gradient in zip((adapter.history_angle, adapter.log_gain), gradients):
        old = parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(old + .001)
            plus = float((adapter.history(x)*upstream).sum())
            parameter.copy_(old - .001)
            minus = float((adapter.history(x)*upstream).sum())
            parameter.copy_(old)
        assert float(gradient) == pytest.approx((plus-minus)/.002, rel=3e-3, abs=3e-3)
    changed = x.clone()
    changed[:, 3:] += 10
    assert torch.equal(adapter.history(changed)[:, :4], y[:, :4])
    assert torch.equal(adapter.history(x[:, :1]), torch.zeros_like(x[:, :1]))
    with pytest.raises(ValueError, match="BTF"):
        adapter.history(torch.zeros(2, 0, 8))


@pytest.mark.parametrize("key,value", [("schema", "old"), ("arms", []),
    ("reference_arm", "history_gain_full"), ("initial_alpha", 1.), ("initial_alpha", True),
    ("initial_gain", 1.), ("initial_gain", True), ("initial_history_angle", 0.),
    ("short_kernel_len", 4), ("short_kernel_len", True)])
def test_invalid_protocol_is_rejected(client, key, value):
    config = recipe(client)
    config[key] = value
    with pytest.raises(ValueError):
        client.adapter_for(client.ARMS[0], config, 41)


def test_source_binding_and_reused_budget_are_explicit(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kwargs: captured.update(kwargs))
    client.main()
    assert captured["arms"] == client.ARMS
    assert set(captured["adapter_sources"]) == {"gain_study", "lag_control", "fractional_bridge"}
    assert all(path.is_file() for path in captured["adapter_sources"].values())
    root = Path(client.__file__).parent
    config = json.loads((root / "hf_fractional_pride_gain.json").read_text())
    previous = json.loads((root / "hf_fractional_pride_history_factorial.json").read_text())
    for key in ("model_snapshot", "corpus_sha256", "transfer_sha256", "block", "features", "seeds",
                "kernel", "steps", "batch_size", "block_size", "development_blocks", "transfer_blocks",
                "evaluate_every", "checkpoint_every", "learning_rate", "strength", "threads"):
        assert config[key] == previous[key]
    client.validate_protocol(config)


def test_three_hf_arms_resume_gain_metadata_and_torch_free_summary(client, tmp_path, monkeypatch):
    driver = client.study
    full = setup(client, tmp_path / "full")
    run(client, full)
    interrupted = setup(client, tmp_path / "interrupted")
    def stop(key, cursor):
        if key == "41:history_gain_full" and cursor == 2:
            raise RuntimeError("controlled interruption")
    with pytest.raises(RuntimeError, match="controlled interruption"):
        run(client, interrupted, stop)
    with pytest.raises(ValueError, match="endpoint evaluation is locked"):
        driver.endpoint_gate(interrupted[-1], interrupted[-2], interrupted[0])
    resumed = list(setup(client, tmp_path / "resumed"))
    resumed[0], resumed[-1] = interrupted[0], json.loads((interrupted[0] / "journal.json").read_text())
    run(client, resumed)
    for key, entry in full[-1]["runs"].items():
        left = driver.load_checkpoint(full[0], entry["checkpoint"], full[-2]["study_id"], key)
        right = driver.load_checkpoint(resumed[0], resumed[-1]["runs"][key]["checkpoint"], resumed[-2]["study_id"], key)
        for field in ("adapter", "optimizer", "records", "development"):
            assert driver.pilot.equal_state(left[field], right[field])
        assert entry["resume_next_update_equal"] and entry["frozen_base_unchanged"]
        assert left["records"][0]["log_gain_gradient"] == 0
        assert any(row["log_gain_gradient"] != 0 for row in left["records"][1:])
    directory, model, parent, original, plan, journal = full
    evaluation = {"probe": torch.arange(24).reshape(4, 6) % 32}
    report = driver.run_endpoints(model, parent, "mlp", original, evaluation, plan,
                                       directory, journal, adapter_factory=client.adapter_for,
                                       result_schema="spiraltorch.fractional_gain_study.v1")
    assert driver.completed_result(directory, journal, plan) == report
    plan["data"] = {"evaluation_block_hashes": {name: driver.block_hashes(blocks)
                                               for name, blocks in evaluation.items()}}
    path = Path(__file__).resolve().parents[3] / "tools" / "summarize_wave_gate_long_horizon.py"
    summary = load_module(path, "gain_summary_test")
    import builtins
    real_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("summary imports Torch")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    digest = driver.pilot.digest((directory / "results.json").read_bytes())
    observed = summary.summarize(plan, report, journal, digest)
    assert observed["initial_filter_receipt_status"] == "passed"
    assert len(observed["gain_trajectories"]) == 3
    assert len(observed["order_trajectories"]) == 2 and len(observed["angle_trajectories"]) == 1
    assert set(observed["paired_gain_contrasts"]["probe"]) == {
        "gl_short_minus_ordinary_short", "gl_full_minus_ordinary_short", "gl_full_minus_gl_short"}
    rows = {row["run_key"]: row for row in report["runs"]}
    row = rows["41:history_gain_full"]
    for mutation in ("endpoint", "continuity", "gain", "gradient", "scale", "mode", "count"):
        changed = copy.deepcopy(rows)
        bad = changed[row["run_key"]]
        if mutation == "endpoint": bad["final_gain"] += .1
        elif mutation == "continuity": bad["records"][1]["log_gain_before_update"] += .01
        elif mutation == "gain": bad["records"][0]["gain_before_update"] = 0
        elif mutation == "gradient": bad["records"][1]["log_gain_gradient"] = None
        elif mutation == "scale": bad["records"][0]["effective_history_gate_l2_before_update"] = -1
        elif mutation == "mode": bad["records"][0]["log_gain_trainable"] = False
        else: bad["trainable_parameter_count"] -= 1
        with pytest.raises(ValueError):
            summary.gain_study_report(plan["config"], changed, {}, ["probe"])
    changed = copy.deepcopy(rows)
    changed[row["run_key"]]["records"][0]["gate_gradient_l2"] += 1
    measured = {key: {"probe": r["scores"]["probe"]["mean"]} for key, r in rows.items()}
    assert summary.gain_study_report(plan["config"], changed, measured, ["probe"])[-1]["41"]["status"] == "failed"


@pytest.mark.parametrize("bad", [100., -200.])
def test_invalid_gain_after_update_cannot_publish_checkpoint(client, tmp_path, bad):
    state = setup(client, tmp_path / "failed")
    def optimizer_factory(adapter, arm, config):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
        if arm == "history_gain_full":
            step = optimizer.step
            def corrupt():
                result = step()
                with torch.no_grad():
                    adapter.log_gain.fill_(bad)
                return result
            optimizer.step = corrupt
        return optimizer
    with pytest.raises(ValueError, match="fractional history log-gain"):
        run(client, state, optimizer_factory=optimizer_factory)
    assert set(state[-1]["runs"]) == {"41:ordinary_gain_short", "41:history_gain_short"}
    assert state[2].mlp is state[3]
