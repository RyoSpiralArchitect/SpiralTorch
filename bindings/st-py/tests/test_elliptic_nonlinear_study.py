import importlib.util
import json
from pathlib import Path

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "EllipticLearningBatch"), reason="native elliptic batch required"
)


@pytest.fixture
def client(monkeypatch):
    source = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "hf_elliptic_nonlinear_study.py"
    )
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("elliptic_controls_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(client):
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


def tiny_model():
    torch.manual_seed(23)
    torch.set_num_threads(2)
    return (
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


def test_controls_pair_parameters_preserve_rng_and_identity(client):
    recipe = config(client)
    hashes = []
    x = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 48
    for seed in recipe["seeds"]:
        initial = []
        for arm in client.ARMS:
            rng = torch.get_rng_state().clone()
            adapter = client.adapter_for(arm, recipe, seed)
            assert torch.equal(torch.get_rng_state(), rng)
            assert sum(p.numel() for p in adapter.parameters()) == 11 * 8 + 2
            assert torch.equal(adapter(x), x)
            initial.append(client.study.pilot.model_digest(adapter))
        assert len(set(initial)) == 1
        hashes.append(initial[0])
    assert hashes[0] != hashes[1]


def test_factory_does_not_seed_accelerator_generators(client, monkeypatch):
    calls = []
    for module, method in ((torch.cuda, "manual_seed_all"), (torch.mps, "manual_seed")):
        monkeypatch.setattr(module, "_is_in_bad_fork", lambda: False)
        monkeypatch.setattr(module, method, lambda seed: calls.append(seed))
    state = torch.get_rng_state().clone()
    for arm in client.ARMS:
        client.adapter_for(arm, config(client), 41)
    assert calls == []
    assert torch.equal(state, torch.get_rng_state())


def test_factory_constructs_on_cpu_without_changing_callers_default_device(client):
    with torch.device("meta"):
        for arm in client.ARMS:
            adapter = client.adapter_for(arm, config(client), 41)
            assert all(p.device.type == "cpu" for p in adapter.parameters())
            assert all(b.device.type == "cpu" for b in adapter.buffers())
        assert torch.empty(1).device.type == "meta"


def test_entrypoint_binds_factory_and_geometry_bridge_sources(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        client.study, "main", lambda **options: captured.update(options)
    )
    client.main()
    assert captured["arms"] == client.ARMS
    assert captured["adapter_factory"] is client.adapter_for
    assert captured["result_schema"] == "spiraltorch.elliptic_nonlinear_study.v1"
    assert captured["adapter_sources"]["study_adapter"] == Path(client.__file__)
    assert captured["adapter_sources"]["elliptic_bridge"] == Path(
        client.elliptic_bridge.__file__
    )


@pytest.mark.parametrize("mode", ["tangent", "tanh_control"])
def test_ordinary_controls_match_anchor_and_local_differential(client, mode):
    adapter = client.adapter_for(mode, config(client), 41)
    zero = torch.zeros(2, requires_grad=True)
    assert torch.equal(adapter.chart_features(zero), adapter.anchor)
    jacobian = torch.autograd.functional.jacobian(adapter.chart_features, zero)
    assert torch.equal(jacobian, adapter.tangent)
    value = torch.tensor([0.3, -0.4], requires_grad=True)
    weight = torch.arange(9, dtype=torch.float32) / 9
    derivative = torch.autograd.grad(
        (adapter.chart_features(value) * weight).sum(), value
    )[0]
    displacement = adapter.tangent @ value.detach()
    slope = (
        torch.ones_like(displacement)
        if mode == "tangent"
        else 1 - displacement.tanh().square()
    )
    assert torch.allclose(derivative, (weight * slope) @ adapter.tangent, atol=1e-6)
    opposite = client.adapter_for(
        "tanh_control" if mode == "tangent" else "tangent", config(client), 41
    )
    with pytest.raises(ValueError, match="control differs"):
        opposite.load_state_dict(adapter.state_dict())


def test_actual_hf_updates_resume_and_initial_hashes_are_paired(client, tmp_path):
    driver = client.study
    recipe = config(client)
    tokens = torch.arange(48).reshape(8, 6) % 32

    def setup(name):
        directory = tmp_path / name
        directory.mkdir()
        model = tiny_model()
        parent = model.transformer.h[0]
        plan = {
            "study_id": "elliptic-test",
            "config": recipe,
            "base_parameter_sha256": driver.pilot.model_digest(model),
            "batch_schedules": {
                str(seed): driver.pilot.schedule(seed, 8, 5, 2)
                for seed in recipe["seeds"]
            },
        }
        journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
        return directory, model, parent, parent.mlp, plan, journal

    full, model, parent, original, plan, journal = setup("full")
    driver.run_training(
        model,
        parent,
        "mlp",
        original,
        tokens,
        tokens[:2],
        plan,
        full,
        journal,
        adapter_factory=client.adapter_for,
    )
    interrupted, other, other_parent, other_original, other_plan, other_journal = setup(
        "interrupted"
    )

    def stop(key, cursor):
        if key == "41:tanh_control" and cursor == 2:
            raise RuntimeError("controlled interrupt")

    with pytest.raises(RuntimeError, match="controlled interrupt"):
        driver.run_training(
            other,
            other_parent,
            "mlp",
            other_original,
            tokens,
            tokens[:2],
            other_plan,
            interrupted,
            other_journal,
            after_checkpoint=stop,
            adapter_factory=client.adapter_for,
        )
    restored = tiny_model()
    restored_parent = restored.transformer.h[0]
    restored_original = restored_parent.mlp
    other_journal = json.loads((interrupted / "journal.json").read_text())
    driver.run_training(
        restored,
        restored_parent,
        "mlp",
        restored_original,
        tokens,
        tokens[:2],
        other_plan,
        interrupted,
        other_journal,
        adapter_factory=client.adapter_for,
    )
    initial = {}
    for key in journal["runs"]:
        left = driver.load_checkpoint(
            full, journal["runs"][key]["checkpoint"], plan["study_id"], key
        )
        right = driver.load_checkpoint(
            interrupted, other_journal["runs"][key]["checkpoint"], plan["study_id"], key
        )
        assert driver.pilot.equal_state(left, right)
        records = left["records"]
        assert records[0]["orientation.weight_gradient_l2"] == 0
        assert any(x["orientation.weight_gradient_l2"] > 0 for x in records[1:])
        assert records[0]["readout.weight_gradient_l2"] > 0
        seed, _ = key.split(":")
        initial.setdefault(seed, set()).add(left["initial_parameter_sha256"])
    assert all(len(values) == 1 for values in initial.values())
    assert initial["41"] != initial["43"]
    result = driver.run_endpoints(
        restored,
        restored_parent,
        "mlp",
        restored_original,
        {"sealed": tokens[:2]},
        other_plan,
        interrupted,
        other_journal,
        adapter_factory=client.adapter_for,
        result_schema="spiraltorch.elliptic_nonlinear_study.v1",
    )
    assert result["schema"] == "spiraltorch.elliptic_nonlinear_study.v1"
    assert all(
        x["parameter_count"] == 90 and x["final_log_radius"] is None
        for x in result["runs"]
    )
    assert driver.completed_result(interrupted, other_journal, other_plan) == result
