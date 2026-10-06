"""Read-only intervention semantics and fail-closed replay, not quality evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    examples = ROOT / "bindings/st-py/examples"
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import hf_fractional_angle_study as client
    spec = importlib.util.spec_from_file_location("window_diagnostic", ROOT / "tools/diagnose_fractional_history_windows.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    config = json.loads((examples / "hf_fractional_pride_angle.json").read_text())
    config.update(features=8, seeds=[41, 43], block_size=6, steps=2, batch_size=2)
    config["kernel"]["kernel_len"] = 8
    torch.manual_seed(93)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    evaluation = {"probe": torch.arange(24).reshape(4, 6) % 32}
    driver = client.study
    plan = {"study_id": "fixture", "config": config,
            "base_parameter_sha256": driver.pilot.model_digest(model),
            "data": {"evaluation_block_hashes": {"probe": driver.block_hashes(evaluation["probe"])}}}
    rows, journal = {}, {"runs": {}}
    for seed in config["seeds"]:
        key = f"{seed}:{tool.ARM}"
        adapter = client.adapter_for(tool.ARM, config, seed)
        initial = driver.pilot.model_digest(adapter)
        with torch.no_grad():
            adapter.gate.copy_(torch.linspace(-.3, .4, 8))
            adapter.local_gate.copy_(torch.linspace(-.2, .1, 8))
            adapter.history_angle.fill_(-.2)
            adapter.log_gain.fill_(.5)
        saved = {"study_id": "fixture", "run_key": key, "cursor": 2, "records": [],
                 "development": [], "initial_parameter_sha256": initial,
                 "adapter": copy.deepcopy(adapter.state_dict())}
        checkpoint = driver.save_checkpoint(tmp_path, saved)
        journal["runs"][key] = {"checkpoint": checkpoint}
        rows[key] = {"checkpoint": checkpoint, "records": [], "development": [],
                     "scores": tool.score(model, config, evaluation, adapter, driver)}
    return tool, client, model, evaluation, (plan, journal, {}, rows), tmp_path


def test_all_modes_preserve_parameters_recipes_and_files_without_training(fixture, monkeypatch):
    tool, client, model, evaluation, current, directory = fixture
    before = {p.name: tool.digest(p) for p in directory.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must not train or use an optimizer")
    monkeypatch.setattr(client.study.pilot, "update", forbidden)
    monkeypatch.setattr(client.study, "make_optimizer", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    snapshots = []
    report = tool.evaluate(model, evaluation, current, directory, client, {},
                           lambda r: snapshots.append(copy.deepcopy(r)))
    assert report["status"] == "completed" and report["frozen_base_unchanged"]
    assert all(len(row["modes"]) == 1 for row in snapshots[1]["runs"].values())
    for row in report["runs"].values():
        assert row["full_replay"]["exact"] is True
        modes = row["modes"]
        assert set(modes) == {"full", "retained_short", "retained_tail", "local_only"}
        assert len({v["receipt"]["parameter_sha256"] for v in modes.values()}) == 1
        assert len({v["receipt"]["intervention_recipe_sha256"] for v in modes.values()}) == 4
        for mode in modes:
            assert modes[mode]["receipt"]["normalization_kernel_len"] == 8
        assert any(x != 0 for x in modes["local_only"]["delta_from_full"]["probe"]["block_deltas"])
    assert before == {p.name: tool.digest(p) for p in directory.iterdir()}


@pytest.mark.parametrize("seed_index", [0, 1])
def test_any_full_score_mismatch_withholds_every_intervention(fixture, seed_index):
    tool, client, model, evaluation, current, directory = fixture
    key = f"{current[0]['config']['seeds'][seed_index]}:{tool.ARM}"
    current[3][key]["scores"]["probe"]["block_losses"][0] += 1e-9
    report = tool.evaluate(model, evaluation, current, directory, client, {}, lambda r: None)
    assert report["status"] == "blocked_full_replay"
    assert all(set(row["modes"]) == {"full"} for row in report["runs"].values())


def test_empty_window_is_local_gate_not_an_identity_baseline(fixture):
    tool, client, _, _, current, directory = fixture
    plan, journal, _, _ = current
    key = f"41:{tool.ARM}"
    saved = client.study.load_checkpoint(directory, journal["runs"][key]["checkpoint"], "fixture", key)
    untouched = copy.deepcopy(saved)
    value = torch.linspace(-1., 1., 96).reshape(2, 6, 8)
    adapters = {mode: tool.intervention(client, plan["config"], saved, mode)[0]
                for mode in tool.windows(8)}
    local = adapters["local_only"](value)
    assert not torch.equal(local, value)
    expected = value + plan["config"]["strength"] * adapters["local_only"].local_gate.tanh() * value
    assert torch.equal(local, expected)
    assert torch.allclose(adapters["full"](value), adapters["retained_short"](value)
                          + adapters["retained_tail"](value) - local, atol=3e-7, rtol=3e-6)
    assert tool.equal(saved, untouched)
    with pytest.raises(ValueError, match="recipe"):
        adapters["retained_short"].load_state_dict(saved["adapter"])


def test_different_evaluation_order_is_rejected_before_scoring(fixture, monkeypatch):
    tool, client, model, evaluation, current, directory = fixture
    monkeypatch.setattr(tool, "score", lambda *a: pytest.fail("must not score changed blocks"))
    with pytest.raises(ValueError, match="block identity"):
        tool.evaluate(model, {"probe": evaluation["probe"].flip(0)}, current, directory, client, {}, lambda r: None)


def test_scoring_exception_restores_original_module(fixture, monkeypatch):
    tool, client, model, evaluation, current, directory = fixture
    plan, journal, _, _ = current
    key = f"41:{tool.ARM}"
    saved = client.study.load_checkpoint(directory, journal["runs"][key]["checkpoint"], "fixture", key)
    adapter, _ = tool.intervention(client, plan["config"], saved, "retained_short")
    original = model.transformer.h[0].mlp
    def fail(*args, **kwargs):
        raise RuntimeError("injected scoring failure")
    monkeypatch.setattr(client.study, "per_block_loss", fail)
    with pytest.raises(RuntimeError, match="injected"):
        tool.score(model, plan["config"], evaluation, adapter, client.study)
    assert model.transformer.h[0].mlp is original
    assert client.study.pilot.model_digest(model) == plan["base_parameter_sha256"]


@pytest.mark.parametrize("field", ["recipe", "dtype", "nan"])
def test_invalid_saved_state_is_not_coerced_into_a_diagnostic(fixture, field):
    tool, client, _, _, current, directory = fixture
    plan, journal, _, _ = current
    key = f"41:{tool.ARM}"
    saved = client.study.load_checkpoint(directory, journal["runs"][key]["checkpoint"], "fixture", key)
    if field == "recipe":
        saved["adapter"]["_extra_state"]["lag_window"] = [1, 3]
    elif field == "dtype":
        saved["adapter"]["gate"] = saved["adapter"]["gate"].double()
    else:
        saved["adapter"]["gate"][0] = float("nan")
    with pytest.raises(ValueError):
        tool.intervention(client, plan["config"], saved, "retained_short")


def test_output_protection_includes_symlink_resolution(fixture, tmp_path):
    tool = fixture[0]
    protected = tmp_path / "frozen"
    protected.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(protected, target_is_directory=True)
    for path in (protected, protected / "result", link / "result"):
        with pytest.raises(ValueError, match="outside"):
            tool.outside_inputs(path, [protected])
    tool.outside_inputs(tmp_path / "new-diagnostic", [protected])
