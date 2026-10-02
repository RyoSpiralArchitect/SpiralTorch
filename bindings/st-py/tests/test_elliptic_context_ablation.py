import copy
import importlib.util
import json
from pathlib import Path

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def client(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_elliptic_context_ablation.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("context_ablation_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {
        "features": 8, "strength": 0.1,
        "warp": {"curvature_radius": 1.0, "sheet_count": 2, "spin_harmonics": 1},
        "learning_rate": 0.001, "steps": 3, "batch_size": 2,
        "checkpoint_every": 1, "evaluate_every": 3,
        "arms": client.gated.ARMS, "seeds": [41],
    }


@pytest.mark.parametrize("arm", ["gated_tangent", "gated_elliptic"])
@pytest.mark.parametrize("raw", [-0.6, 0.0, 0.3])
def test_control_formulas_causality_and_frozen_state(client, arm, raw):
    config = recipe(client)
    source = client.gated.adapter_for(arm, config, 41).eval()
    with torch.no_grad():
        source.raw_mix.fill_(raw)
        source.readout.weight.fill_(0.02)
    generator = torch.Generator().manual_seed(113)
    orientation = torch.randn(2, 4, 3, generator=generator)
    orientation[..., 0] = 1
    if arm == "gated_tangent":
        local = source.anchor + orientation[..., 1:] @ source.tangent.T
    else:
        local = st.elliptic_warp_autograd(source._warp, orientation)
    anchor = torch.tensor(source._warp.map_orientations_batch([1., 0., 0.]).features).double()
    gate = source.raw_mix.detach().double().tanh()
    prefix = (local.double().cumsum(1) / torch.arange(1, 5).reshape(1, 4, 1)).float().double()
    context = {"gain_only": torch.zeros_like(local), "anchor": anchor, "prefix_mean": prefix}
    for mode in client.MODES:
        rng = torch.get_rng_state().clone()
        adapter = client.adapter_for(arm, config, 41, mode)
        assert torch.equal(rng, torch.get_rng_state())
        adapter.load_state_dict(source.state_dict())
        before = copy.deepcopy(adapter.state_dict())
        with torch.no_grad():
            actual = adapter._map_features(orientation)
            if mode == "native":
                expected = source._map_features(orientation)
            elif mode == "local":
                expected = local
            else:
                expected = ((1 - gate) * local.double() + gate * context[mode]).float()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            changed = orientation.clone()
            changed[:, -1, 1:] += 1
            changed[1, :, 1:] += 0.7
            torch.testing.assert_close(
                adapter._map_features(changed)[0, :3], actual[0, :3], atol=0, rtol=0,
            )
            inputs = torch.randn(2, 4, 8, generator=generator)
            adapter(inputs)
        assert client.pilot.equal_state(before, adapter.state_dict())


def test_controls_are_eval_only_and_keep_full_context_guards(client):
    adapter = client.adapter_for("gated_elliptic", recipe(client), 41, "prefix_mean")
    with pytest.raises(ValueError, match="no retraining"):
        adapter(torch.zeros(1, 2, 8))
    with torch.no_grad():
        adapter.train()
        with pytest.raises(ValueError, match="no retraining"):
            adapter(torch.zeros(1, 2, 8))
        adapter.eval()
        with pytest.raises(ValueError, match="nonempty"):
            adapter(torch.zeros(2, 8))
        with pytest.raises(ValueError, match="budget"):
            adapter(torch.zeros(1, 1025, 8))
    with pytest.raises(ValueError, match="unknown"):
        client.adapter_for("gated_tangent", recipe(client), 41, "future_mean")


@pytest.fixture
def learned(client, tmp_path):
    config = recipe(client)
    directory = tmp_path / "parent"
    directory.mkdir()
    torch.manual_seed(127)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    parent = model.transformer.h[0]
    original = parent.mlp
    tokens = torch.arange(48).reshape(8, 6) % 32
    plan = {
        "study_id": "context-test", "config": config,
        "base_parameter_sha256": client.pilot.model_digest(model),
        "batch_schedules": {"41": client.pilot.schedule(41, 8, 4, 2)},
    }
    journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
    client.study.run_training(
        model, parent, "mlp", original, tokens, tokens[:2], plan, directory,
        journal, adapter_factory=client.gated.adapter_for,
    )
    evaluation = {"tail": tokens[:3], "transfer": tokens[3:5]}
    result = client.study.run_endpoints(
        model, parent, "mlp", original, evaluation, plan, directory, journal,
        adapter_factory=client.gated.adapter_for,
        result_schema="spiraltorch.elliptic_gated_study.v1",
    )
    return dict(
        model=model, parent=parent, child="mlp", original=original,
        evaluation=evaluation, parent_plan=plan, parent_journal=journal,
        parent_result=result, parent_dir=directory,
        binding={"config": config, "ablation_id": "test-ablation"},
    )


def test_actual_hf_replay_ablation_interrupt_resume_and_completed_noop(client, learned, tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    before = {p.name: p.read_bytes() for p in learned["parent_dir"].iterdir()}
    result = client.run_ablation(**learned, directory=full)
    assert result["status"] == "completed" and result["base_unchanged"]
    assert len(result["conditions"]) == 10
    assert learned["parent"].mlp is learned["original"]
    assert all(p.read_bytes() == before[p.name] for p in learned["parent_dir"].iterdir())
    assert all(row["mode"] == "native" for row in result["conditions"][:2])
    resumed = tmp_path / "resumed"
    resumed.mkdir()

    def stop(key, mode):
        if mode == "local" and key == "41:gated_tangent":
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        client.run_ablation(**learned, directory=resumed, after_condition=stop)
    completed = client.run_ablation(**learned, directory=resumed)
    assert completed == result
    frozen = (resumed / "results.json").read_bytes()
    assert client.run_ablation(**learned, directory=resumed) == completed
    assert (resumed / "results.json").read_bytes() == frozen
    summary = client.summarize(completed)
    for arm in client.ARMS:
        for score in summary["contrasts"][arm]["native"].values():
            assert score["mean_delta_from_native"] == 0
    for mutate, match in [
        (lambda x: x["conditions"].pop(), "incomplete"),
        (lambda x: x["conditions"].append(x["conditions"][0]), "duplicate"),
        (lambda x: x["conditions"][0]["scores"]["tail"]["block_losses"].__setitem__(0, float("nan")), "block scores"),
        (lambda x: x["conditions"][0].__setitem__("raw_mix", 99), "gate differs"),
    ]:
        bad = copy.deepcopy(completed)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            client.validate_report(bad, learned["binding"], learned["parent_result"])
    bad = json.loads(frozen)
    bad["conditions"][-1]["scores"]["tail"]["mean"] += 1
    (resumed / "results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="hash mismatch"):
        client.run_ablation(**learned, directory=resumed)


def test_native_mismatch_blocks_interventions_and_restores_model(client, learned, tmp_path):
    directory = tmp_path / "bad-replay"
    directory.mkdir()
    for row in learned["parent_result"]["runs"]:
        if row["run_key"] == "41:gated_tangent":
            row["scores"]["tail"]["mean"] += 1
    with pytest.raises(ValueError, match="native replay differs"):
        client.run_ablation(**learned, directory=directory)
    assert learned["parent"].mlp is learned["original"]
    assert not (directory / "results.json").exists()
