import importlib.util
import json
from pathlib import Path

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "EllipticCausalLearningBatch"),
    reason="native causal batch required",
)


@pytest.fixture
def client(monkeypatch):
    source = (
        Path(__file__).resolve().parents[1] / "examples" / "hf_elliptic_causal_study.py"
    )
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("causal_controls_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {
        "features": 8,
        "strength": 0.1,
        "warp": {"curvature_radius": 1.0, "sheet_count": 2, "spin_harmonics": 1},
        "learning_rate": 0.001,
        "steps": 4,
        "batch_size": 2,
        "checkpoint_every": 2,
        "evaluate_every": 2,
        "arms": client.ARMS,
        "seeds": [41, 43],
    }


def test_all_four_arms_pair_initial_weights_and_preserve_rng(client, monkeypatch):
    calls = []
    for module, method in ((torch.cuda, "manual_seed_all"), (torch.mps, "manual_seed")):
        monkeypatch.setattr(module, "_is_in_bad_fork", lambda: False)
        monkeypatch.setattr(module, method, lambda seed: calls.append(seed))
    x = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 48
    hashes = []
    for seed in recipe(client)["seeds"]:
        initial = []
        for arm in client.ARMS:
            rng = torch.get_rng_state().clone()
            with torch.device("meta"):
                adapter = client.adapter_for(arm, recipe(client), seed)
                assert torch.empty(1).device.type == "meta"
            assert torch.equal(torch.get_rng_state(), rng)
            assert all(p.device.type == "cpu" for p in adapter.parameters())
            assert all(b.device.type == "cpu" for b in adapter.buffers())
            assert sum(p.numel() for p in adapter.parameters()) == 11 * 8 + 2
            assert torch.equal(adapter(x), x)
            initial.append(client.study.pilot.model_digest(adapter))
        assert len(set(initial)) == 1
        hashes.append(initial[0])
    assert hashes[0] != hashes[1]
    assert calls == []


def test_causal_control_uses_same_tangent_and_tied_mixing(client):
    config = recipe(client)
    control = client.adapter_for("causal_tangent", config, 41)
    pointwise = client.adapter_for("tangent", config, 41)
    assert torch.equal(control.anchor, pointwise.anchor)
    assert torch.equal(control.tangent, pointwise.tangent)
    torch.manual_seed(101)
    coordinates = torch.randn(2, 4, 2, requires_grad=True)
    other = coordinates.detach().clone().requires_grad_()
    orientation = torch.cat((torch.ones_like(coordinates[..., :1]), coordinates), -1)
    actual = control._map_features(orientation)
    expected = client.causal_mix(pointwise.chart_features(other))
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(coordinates.grad, other.grad)


def test_causal_oracle_matches_rust_geometric_forward_and_gradient(client):
    torch.manual_seed(103)
    x = torch.randn(2, 5, 3)
    x[..., 0] = 1
    x.requires_grad_()
    other = x.detach().clone().requires_grad_()
    warp = st.EllipticWarp(1.0, 2, 1)
    actual = st.elliptic_causal_autograd(warp, x)
    expected = client.causal_mix(st.elliptic_warp_autograd(warp, other))
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("arm", ["causal_tangent", "causal_elliptic"])
def test_causal_paths_share_guards_and_do_not_leak_future_or_other_batch(client, arm):
    adapter = client.adapter_for(arm, recipe(client), 41)
    with torch.no_grad():
        adapter.readout.weight.fill_(0.1)
    torch.manual_seed(107)
    x = torch.randn(2, 4, 8, requires_grad=True)
    full = adapter(x)
    changed = x.detach().clone()
    changed[0, 2:] += 3
    changed[1] -= 2
    assert torch.equal(full[0, :2], adapter(changed)[0, :2])
    torch.testing.assert_close(full[0, :2], adapter(x[:1, :2])[0])
    full[0, 1].sum().backward()
    assert x.grad[0, 0].abs().sum() > 0
    assert torch.count_nonzero(x.grad[0, 2:]) == 0
    assert torch.count_nonzero(x.grad[1]) == 0
    for invalid in (torch.ones(2, 8), torch.ones(0, 2, 8)):
        with pytest.raises(ValueError, match="nonempty"):
            adapter(invalid)
    with pytest.raises(ValueError, match="budget"):
        adapter(torch.ones(1, 1025, 8))


def test_checkpoint_cannot_change_factorial_arm(client):
    adapters = {arm: client.adapter_for(arm, recipe(client), 41) for arm in client.ARMS}
    for arm, source in adapters.items():
        for other in client.ARMS:
            target = client.adapter_for(other, recipe(client), 41)
            if arm == other:
                target.load_state_dict(source.state_dict())
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    target.load_state_dict(source.state_dict())


def test_protocol_and_entrypoint_bind_all_adapter_sources(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kw: captured.update(kw))
    client.main()
    assert captured["arms"] == client.ARMS
    assert captured["adapter_factory"] is client.adapter_for
    assert captured["result_schema"] == "spiraltorch.elliptic_causal_study.v1"
    assert captured["adapter_sources"] == {
        "study_adapter": Path(client.__file__),
        "pointwise_adapter": Path(client.pointwise.__file__),
        "elliptic_bridge": Path(client.pointwise.elliptic_bridge.__file__),
    }
    config = json.loads(
        Path(client.__file__).with_name("hf_elliptic_pride_causal.json").read_text()
    )
    assert config["arms"] == client.ARMS
    assert config["steps"] == 512 and config["seeds"] == [41, 43, 47]


def test_actual_hf_all_arms_train_and_interrupted_run_resumes_exactly(client, tmp_path):
    driver = client.study
    config = recipe(client)
    config["seeds"] = [41]
    tokens = torch.arange(48).reshape(8, 6) % 32

    def setup(name):
        directory = tmp_path / name
        directory.mkdir()
        torch.manual_seed(109)
        torch.set_num_threads(2)
        model = (
            transformers.GPT2LMHeadModel(
                transformers.GPT2Config(
                    vocab_size=32,
                    n_positions=8,
                    n_embd=8,
                    n_head=2,
                    n_layer=1,
                    resid_pdrop=0,
                    attn_pdrop=0,
                    embd_pdrop=0,
                    use_cache=False,
                )
            )
            .eval()
            .requires_grad_(False)
        )
        parent = model.transformer.h[0]
        plan = {
            "study_id": "causal-factorial-test",
            "config": config,
            "base_parameter_sha256": driver.pilot.model_digest(model),
            "batch_schedules": {"41": driver.pilot.schedule(41, 8, 5, 2)},
        }
        journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
        return directory, model, parent, parent.mlp, plan, journal

    def run(state, callback=None):
        directory, model, parent, original, plan, journal = state
        driver.run_training(
            model,
            parent,
            "mlp",
            original,
            tokens,
            tokens[:2],
            plan,
            directory,
            journal,
            after_checkpoint=callback,
            adapter_factory=client.adapter_for,
        )

    full = setup("full")
    run(full)
    interrupted = setup("interrupted")

    def stop(key, cursor):
        if key == "41:causal_tangent" and cursor == 2:
            raise RuntimeError("controlled interrupt")

    with pytest.raises(RuntimeError, match="controlled interrupt"):
        run(interrupted, stop)
    restored = list(setup("restored"))
    restored[0] = interrupted[0]
    restored[-1] = json.loads((interrupted[0] / "journal.json").read_text())
    run(restored)
    initial = set()
    for key, entry in full[-1]["runs"].items():
        left = driver.load_checkpoint(
            full[0], entry["checkpoint"], full[-2]["study_id"], key
        )
        right = driver.load_checkpoint(
            restored[0],
            restored[-1]["runs"][key]["checkpoint"],
            restored[-2]["study_id"],
            key,
        )
        assert driver.pilot.equal_state(left, right)
        assert entry["resume_next_update_equal"] and entry["frozen_base_unchanged"]
        assert left["records"][0]["orientation.weight_gradient_l2"] == 0
        assert all(row["readout.weight_gradient_l2"] > 0 for row in left["records"])
        assert all(
            row["orientation.weight_gradient_l2"] > 0 for row in left["records"][1:]
        )
        initial.add(left["initial_parameter_sha256"])
    assert len(initial) == 1
