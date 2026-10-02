import importlib.util
import json
from pathlib import Path

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def client(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_elliptic_anchored_study.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("anchored_controls_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {
        "features": 8, "strength": 0.1,
        "warp": {"curvature_radius": 1.0, "sheet_count": 2, "spin_harmonics": 1},
        "learning_rate": 0.001, "steps": 4, "batch_size": 2,
        "checkpoint_every": 2, "evaluate_every": 2,
        "arms": client.ARMS, "seeds": [41],
    }


def test_all_parameters_are_paired_without_accelerator_rng_or_dummy_parameters(client, monkeypatch):
    calls = []
    for module, method in ((torch.cuda, "manual_seed_all"), (torch.mps, "manual_seed")):
        monkeypatch.setattr(module, "_is_in_bad_fork", lambda: False)
        monkeypatch.setattr(module, method, lambda seed: calls.append(seed))
    hashes, projections = [], []
    for arm in client.ARMS:
        rng = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, recipe(client), 41)
            assert torch.empty(1).device.type == "meta"
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(p.device.type == "cpu" for p in adapter.parameters())
        assert all(b.device.type == "cpu" for b in adapter.buffers())
        assert sum(p.numel() for p in adapter.parameters()) == 91
        assert adapter.raw_mix.item() == 0
        hashes.append(client.study.pilot.model_digest(adapter))
        projections.append(client.study.pilot.model_digest(adapter, exclude={"raw_mix"}))
    assert len(set(hashes)) == len(set(projections)) == 1
    assert calls == []
    with pytest.raises(ValueError, match="unrecognized"):
        client.adapter_for("invented", recipe(client), 41)


@pytest.mark.parametrize("raw", [-0.7, 0.0, 0.8])
def test_anchored_oracle_matches_native_at_training_shape(client, raw):
    generator = torch.Generator().manual_seed(113)
    x = torch.randn(2, 128, 3, generator=generator)
    x[..., 0] = 1
    x.requires_grad_()
    other = x.detach().clone().requires_grad_()
    gate = torch.tensor(raw, requires_grad=True)
    reference_gate = gate.detach().clone().requires_grad_()
    warp = st.EllipticWarp(1.0, 2, 1)
    anchor = torch.tensor(warp.map_orientations_batch([1.0, 0.0, 0.0]).features)
    actual = st.elliptic_anchored_autograd(warp, x, gate)
    expected = client.anchored_mix(st.elliptic_warp_autograd(warp, other), anchor, reference_gate)
    seed = torch.randn(actual.shape, generator=generator)
    (actual * seed).sum().backward()
    (expected * seed).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(x.grad, other.grad, atol=3e-5, rtol=1e-4)
    torch.testing.assert_close(gate.grad, reference_gate.grad, atol=3e-5, rtol=1e-4)


def test_control_is_affine_rowwise_and_schema_isolated(client):
    config = recipe(client)
    ordinary = client.adapter_for("anchored_tangent", config, 41)
    pointwise = client.gated.causal.pointwise.adapter_for("tangent", config, 41)
    coordinates = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16
    orientation = torch.cat((torch.ones_like(coordinates[..., :1]), coordinates), -1)
    assert torch.equal(ordinary._map_features(orientation), pointwise.chart_features(coordinates))
    with torch.no_grad():
        ordinary.raw_mix.fill_(-0.7)
    features = ordinary._map_features(orientation)
    expected = ordinary.anchor + (1 - ordinary.raw_mix.tanh()) * (coordinates @ ordinary.tangent.T)
    torch.testing.assert_close(features, expected)
    assert torch.equal(features[:, -1], ordinary._map_features(orientation[:, -1]))
    for arm in client.ARMS:
        source = client.adapter_for(arm, config, 41)
        for other in client.ARMS:
            target = client.adapter_for(other, config, 41)
            if arm == other:
                target.load_state_dict(source.state_dict())
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    target.load_state_dict(source.state_dict())


def test_protocol_and_transitive_sources_are_frozen(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kw: captured.update(kw))
    client.main()
    assert captured["arms"] == client.ARMS
    assert captured["result_schema"] == "spiraltorch.elliptic_anchored_study.v1"
    assert set(captured["adapter_sources"]) == {
        "study_adapter", "gated_control", "causal_control", "pointwise_adapter", "elliptic_bridge"
    }
    assert all(p.is_file() for p in captured["adapter_sources"].values())
    config = json.loads(Path(client.__file__).with_name("hf_elliptic_pride_anchored.json").read_text())
    old = json.loads(Path(client.gated.__file__).with_name("hf_elliptic_pride_gated.json").read_text())
    assert config["arms"] == client.ARMS
    assert config["reference_arm"] == "anchored_tangent"
    assert config["steps"] == 512 and config["seeds"] == [41, 43, 47]
    for key in old.keys() - {"schema", "scope", "arms", "summary_schema", "comparison_notes"}:
        assert config[key] == old[key]


def test_actual_hf_all_arms_train_and_interrupt_resume_exactly(client, tmp_path):
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
        plan = {
            "study_id": "anchored-study-test", "config": config,
            "base_parameter_sha256": driver.pilot.model_digest(model),
            "batch_schedules": {"41": driver.pilot.schedule(41, 8, 5, 2)},
        }
        journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
        return [directory, model, parent, parent.mlp, plan, journal]

    def run(state, callback=None):
        directory, model, parent, original, plan, journal = state
        driver.run_training(
            model, parent, "mlp", original, tokens, tokens[:2], plan, directory,
            journal, after_checkpoint=callback, adapter_factory=client.adapter_for,
        )

    full = setup("full")
    run(full)
    interrupted = setup("interrupted")

    def stop(key, cursor):
        if key == "41:anchored_elliptic" and cursor == 2:
            raise RuntimeError("controlled interrupt")

    with pytest.raises(RuntimeError, match="controlled interrupt"):
        run(interrupted, stop)
    restored = setup("restored")
    restored[0] = interrupted[0]
    restored[-1] = json.loads((interrupted[0] / "journal.json").read_text())
    run(restored)
    hashes = set()
    for key, entry in full[-1]["runs"].items():
        left = driver.load_checkpoint(full[0], entry["checkpoint"], full[-2]["study_id"], key)
        right = driver.load_checkpoint(
            restored[0], restored[-1]["runs"][key]["checkpoint"], restored[-2]["study_id"], key
        )
        assert driver.pilot.equal_state(left, right)
        assert entry["resume_next_update_equal"] and entry["frozen_base_unchanged"]
        hashes.add(left["initial_parameter_sha256"])
        records = left["records"]
        assert records[0]["raw_mix_before_update"] == records[0]["raw_mix_gradient"] == 0
        assert all(r["raw_mix_gradient_l2"] > 0 for r in records[1:])
        assert [r["raw_mix_after_update"] for r in records[:-1]] == [
            r["raw_mix_before_update"] for r in records[1:]
        ]
    assert len(hashes) == 1
    directory, model, parent, original, plan, journal = full
    result = driver.run_endpoints(
        model, parent, "mlp", original, {"tail": tokens[:2]}, plan, directory, journal,
        adapter_factory=client.adapter_for, result_schema="spiraltorch.elliptic_anchored_study.v1",
    )
    for row in result["runs"]:
        assert row["final_raw_mix"] == row["records"][-1]["raw_mix_after_update"]
        assert row["parameter_count"] == 91
    assert driver.completed_result(directory, journal, plan) == result
