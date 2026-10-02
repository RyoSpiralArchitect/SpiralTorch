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
    path = Path(__file__).parents[1] / "examples" / "hf_elliptic_chart_step_study.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("chart_step_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {
        "features": 8,
        "strength": 0.1,
        "warp": {"curvature_radius": 1.0, "sheet_count": 2, "spin_harmonics": 1},
        "learning_rate": 0.001,
        "relative_damping": 0.1,
        "steps": 4,
        "batch_size": 2,
        "checkpoint_every": 2,
        "evaluate_every": 2,
        "seeds": [41],
        "arms": client.ARMS,
    }


def test_protocol_keeps_original_data_and_budgets_and_binds_optimizer(
    client, monkeypatch
):
    path = Path(client.__file__)
    config = json.loads(path.with_name("hf_elliptic_pride_chart_step.json").read_text())
    original = json.loads(path.with_name("hf_elliptic_pride_anchored.json").read_text())
    for key in original.keys() - {
        "schema",
        "scope",
        "arms",
        "reference_arm",
        "summary_schema",
        "comparison_notes",
    }:
        assert config[key] == original[key]
    assert config["relative_damping"] == 0.1 and config["arms"] == client.ARMS
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kw: captured.update(kw))
    client.main()
    assert captured["optimizer_factory"] == client.optimizer_for
    assert captured["adapter_sources"]["chart_study"] == path


def test_native_chart_step_matches_independent_dense_solve():
    batch = st.EllipticWarp(1.3, 3, 2).map_orientations_batch(
        [1.0, 16.0, 5.0, 1.0, -12.0, 3.0]
    )
    proposal = torch.tensor([[0.01, -0.03, 0.02], [0.04, -0.02, 0.01]])
    result = batch.chart_step(proposal.flatten().tolist(), 0.1)
    assert isinstance(result, st.EllipticChartStep)
    basis = [
        batch.jvp([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        batch.jvp([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    ]
    j = torch.tensor(basis).T.double()
    gram = j.T @ j / 2
    torch.testing.assert_close(
        torch.tensor(result.metric, dtype=torch.float64).reshape(2, 2),
        gram,
        rtol=1e-12,
        atol=1e-12,
    )
    matrix = gram + float(torch.tensor(0.1)) * gram.trace() * 0.5 * torch.eye(
        2, dtype=torch.float64
    )
    expected = torch.linalg.solve(matrix, proposal.double())
    expected *= proposal.double().norm() / expected.norm()
    torch.testing.assert_close(
        torch.tensor(result.values).reshape_as(proposal),
        expected.float(),
        rtol=2e-6,
        atol=1e-9,
    )
    assert result.step_l2 == pytest.approx(result.proposal_l2, rel=1e-7)
    assert result.damped_condition == pytest.approx(
        float(torch.linalg.cond(matrix)), rel=1e-10
    )


@pytest.mark.parametrize("arm", ["adam_tangent", "adam_elliptic"])
def test_disabled_chart_is_exactly_adam(client, arm):
    config = recipe(client)
    left = client.adapter_for(arm, config, 41)
    right = client.adapter_for(arm, config, 41)
    opt = client.optimizer_for(left, arm, config)
    reference = torch.optim.Adam(right.parameters(), lr=config["learning_rate"])
    x = torch.linspace(-1, 1, 48).reshape(2, 3, 8)
    for _ in range(4):
        for adapter, optimizer in ((left, opt), (right, reference)):
            optimizer.zero_grad()
            (adapter(x) - x.sin()).square().mean().backward()
            optimizer.step()
        assert client.study.pilot.equal_state(left.state_dict(), right.state_dict())
        actual = opt.state_dict()
        actual.pop("chart_recipe")
        assert client.study.pilot.equal_state(actual, reference.state_dict())


def test_chart_rollback_freshness_and_recipe_checks(client):
    config = recipe(client)
    adapter = client.adapter_for("chart_elliptic", config, 41)
    opt = client.optimizer_for(adapter, "chart_elliptic", config)
    with pytest.raises(ValueError, match="fresh forward"):
        opt.step()
    before_params, before_opt = copy.deepcopy(adapter.state_dict()), copy.deepcopy(
        opt.state_dict()
    )
    x = torch.ones(2, 3, 8)
    opt.zero_grad()
    adapter(x).sum().backward()
    original = adapter._warp

    class FailedSnapshot:
        def chart_step(self, *args):
            raise ValueError("injected native rejection")

    class FailedWarp:
        def map_orientations_batch(self, *args):
            return FailedSnapshot()

    adapter._warp = FailedWarp()
    with pytest.raises(ValueError, match="injected native rejection"):
        opt.step()
    adapter._warp = original
    assert client.study.pilot.equal_state(adapter.state_dict(), before_params)
    assert client.study.pilot.equal_state(opt.state_dict(), before_opt)
    assert adapter._chart_orientation is None
    corrupted = copy.deepcopy(before_opt)
    corrupted["chart_recipe"]["relative_damping"] = 0.2
    with pytest.raises(ValueError, match="recipe mismatch"):
        opt.load_state_dict(corrupted)
    opt.zero_grad()
    (adapter(x).sum() + adapter(x).sum()).backward()
    with pytest.raises(ValueError, match="accumulation"):
        opt.step()


def test_all_hf_arms_interrupt_resume_with_unchanged_base_and_step_budget(
    client, tmp_path
):
    driver, config = client.study, recipe(client)
    tokens = torch.arange(48).reshape(8, 6) % 32

    def setup(name):
        directory = tmp_path / name
        directory.mkdir()
        torch.manual_seed(127)
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
            "study_id": "chart-study-test",
            "config": config,
            "base_parameter_sha256": driver.pilot.model_digest(model),
            "batch_schedules": {"41": driver.pilot.schedule(41, 8, 5, 2)},
        }
        journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
        return [directory, model, parent, parent.mlp, plan, journal]

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
            optimizer_factory=client.optimizer_for,
        )

    full, interrupted = setup("full"), setup("interrupted")
    run(full)

    def stop(key, cursor):
        if key == "41:chart_elliptic" and cursor == 2:
            raise RuntimeError("controlled interrupt")

    with pytest.raises(RuntimeError, match="controlled interrupt"):
        run(interrupted, stop)
    restored = setup("restored")
    restored[0], restored[-1] = interrupted[0], json.loads(
        (interrupted[0] / "journal.json").read_text()
    )
    run(restored)
    hashes = set()
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
        hashes.add(left["initial_parameter_sha256"])
        for row in left["records"]:
            receipt = row["optimizer_step"]
            if receipt["enabled"]:
                assert receipt["step_l2"] == pytest.approx(
                    receipt["proposal_l2"], rel=1e-7
                )
                assert receipt["applied_step_l2"] == pytest.approx(
                    receipt["proposal_l2"], rel=2e-5, abs=1e-8
                )
    assert len(hashes) == 1
    directory, model, parent, original, plan, journal = full
    result = driver.run_endpoints(
        model,
        parent,
        "mlp",
        original,
        {"tail": tokens[:2]},
        plan,
        directory,
        journal,
        adapter_factory=client.adapter_for,
        result_schema="spiraltorch.elliptic_chart_step_study.v1",
    )
    assert driver.completed_result(directory, journal, plan) == result
