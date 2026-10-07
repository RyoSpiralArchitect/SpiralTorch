import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/hf_fractional_angle_study.py"
    monkeypatch.syspath_prepend(str(path.parent))
    return load(path, "angle_study_test")


def recipe(client):
    config = json.loads(Path(client.__file__).with_name("hf_fractional_pride_angle.json").read_text())
    config.update(features=8, steps=4, block_size=6, batch_size=2, seeds=[41],
                  checkpoint_every=2, evaluate_every=2, learning_rate=.01)
    return config


def setup(client, path):
    path.mkdir()
    torch.manual_seed(197)
    torch.set_num_threads(2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    config = recipe(client)
    plan = {"study_id": "angle-study-test", "config": config,
            "base_parameter_sha256": client.study.pilot.model_digest(model),
            "batch_schedules": {"41": client.study.pilot.schedule(41, 8, 5, 2)}}
    journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
    return path, model, model.transformer.h[0], model.transformer.h[0].mlp, plan, journal


def run(client, state, **kwargs):
    path, model, parent, original, plan, journal = state
    tokens = torch.arange(48).reshape(8, 6) % 32
    client.study.run_training(model, parent, "mlp", original, tokens, tokens[:2], plan, path,
                              journal, adapter_factory=client.adapter_for, **kwargs)


def test_all_arms_share_initial_parameters_but_reject_cross_recipe_state(client):
    config, adapters, hashes = recipe(client), [], []
    for arm in client.ARMS:
        rng = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(rng, torch.get_rng_state())
        assert list(dict(adapter.named_parameters())) == ["gate", "local_gate", "history_angle", "log_gain"]
        assert sum(p.numel() for p in adapter.parameters()) == 18
        assert all(p.device.type == "cpu" and p.requires_grad for p in adapter.parameters())
        assert adapter.alpha == 2.
        hashes.append(client.study.pilot.model_digest(adapter))
        x = torch.linspace(-.9, .8, 96).reshape(2, 6, 8)
        assert torch.equal(adapter(x), x)
        restored = client.adapter_for(arm, config, 47)
        restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
        assert client.study.pilot.equal_state(restored.state_dict(), adapter.state_dict())
        adapters.append(adapter)
    assert len(set(hashes)) == 1
    for i, a in enumerate(adapters):
        for j, b in enumerate(adapters):
            if i != j:
                with pytest.raises((ValueError, RuntimeError)):
                    a.load_state_dict(b.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        adapters[0].load_state_dict(client.gain.OrdinaryGainShort(
            8, strength=.1, kernel_len=3).state_dict())


@pytest.mark.parametrize("bad", [-.5, 1.6, float("nan"), float("inf")])
@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_domain_failure_is_terminal_and_never_publishes_invalid_checkpoint(client, tmp_path, bad, arm_index):
    state = setup(client, tmp_path / "study")
    arm = client.ARMS[arm_index]
    def optimizer_factory(adapter, name, config):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
        if name == arm:
            step, counter = optimizer.step, [0]
            def corrupt():
                result = step()
                counter[0] += 1
                if counter[0] == 3:
                    with torch.no_grad(): adapter.history_angle.fill_(bad)
                return result
            optimizer.step = corrupt
        return optimizer
    with pytest.raises(client.study.TerminalStudyError, match="domain exit"):
        run(client, state, optimizer_factory=optimizer_factory)
    journal = json.loads((state[0] / "journal.json").read_text())
    assert journal == state[-1] and journal["status"] == "terminal_failure"
    failure = journal["terminal_failure"]
    assert failure["run_key"] == f"41:{arm}" and failure["phase"] == "primary_update"
    assert failure["attempted_step"] == 3 and failure["completed_primary_updates_in_run"] == 2
    assert failure["failed_update_resumable"] is False
    assert failure["details"]["condition"] == "angle_domain_exit"
    assert failure["details"]["policy"] == client.DOMAIN_POLICY
    assert state[2].mlp is state[3] and not (state[0] / "results.json").exists()
    entry = journal["runs"][f"41:{arm}"]
    saved = client.study.load_checkpoint(state[0], entry["checkpoint"], state[-2]["study_id"], f"41:{arm}")
    assert saved["cursor"] == entry["cursor"] == 2
    hashes = {p.name: client.study.pilot.digest(p.read_bytes()) for p in state[0].iterdir()}
    with pytest.raises(ValueError, match="cannot resume"): run(client, state)
    with pytest.raises(ValueError, match="terminal protocol failure"):
        client.study.endpoint_gate(journal, state[-2], state[0])
    assert hashes == {p.name: client.study.pilot.digest(p.read_bytes()) for p in state[0].iterdir()}


@pytest.mark.parametrize("replay", [False, True])
def test_continuation_domain_failure_locks_endpoints_even_with_final_checkpoints(client, tmp_path, replay):
    state = setup(client, tmp_path / "study")
    calls = [0]
    def factory(adapter, arm, config):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
        if arm == client.ARMS[-1]:
            calls[0] += 1
            instance, step, counter = calls[0], optimizer.step, [0]
            def corrupt():
                result = step()
                counter[0] += 1
                if (not replay and instance == 1 and counter[0] == 5) or (replay and instance == 2):
                    with torch.no_grad(): adapter.history_angle.fill_(-.5)
                return result
            optimizer.step = corrupt
        return optimizer
    with pytest.raises(client.study.TerminalStudyError): run(client, state, optimizer_factory=factory)
    failure = state[-1]["terminal_failure"]
    assert failure["phase"] == ("continuation_replay" if replay else "continuation_reference")
    assert failure["attempted_step"] == 5 and failure["completed_primary_updates_in_run"] == 4
    assert all(entry["cursor"] == 4 for entry in state[-1]["runs"].values())
    with pytest.raises(ValueError, match="terminal protocol failure"):
        client.study.endpoint_gate(state[-1], state[-2], state[0])


def test_three_arms_resume_and_torch_free_chart_summary(client, tmp_path, monkeypatch):
    full = setup(client, tmp_path / "full")
    run(client, full)
    interrupted = setup(client, tmp_path / "interrupted")
    def stop(key, cursor):
        if key == f"41:{client.ARMS[-1]}" and cursor == 2:
            raise RuntimeError("controlled infrastructure interruption")
    with pytest.raises(RuntimeError, match="infrastructure"):
        run(client, interrupted, after_checkpoint=stop)
    assert "terminal_failure" not in interrupted[-1]
    resumed = list(setup(client, tmp_path / "fresh-process"))
    resumed[0], resumed[-1] = interrupted[0], json.loads((interrupted[0] / "journal.json").read_text())
    run(client, resumed)
    for key, entry in full[-1]["runs"].items():
        a = client.study.load_checkpoint(full[0], entry["checkpoint"], full[-2]["study_id"], key)
        b = client.study.load_checkpoint(resumed[0], resumed[-1]["runs"][key]["checkpoint"], full[-2]["study_id"], key)
        for field in ("adapter", "optimizer", "records", "development"):
            assert client.study.pilot.equal_state(a[field], b[field])
        assert a["records"][0]["history_angle_gradient"] == 0
        assert any(row["history_angle_gradient"] != 0 for row in a["records"][1:])
    directory, model, parent, original, plan, journal = full
    evaluation = {"probe": torch.arange(24).reshape(4, 6) % 32}
    report = client.study.run_endpoints(model, parent, "mlp", original, evaluation, plan, directory,
                                        journal, adapter_factory=client.adapter_for,
                                        result_schema="spiraltorch.fractional_angle_study.v1")
    assert client.study.completed_result(directory, journal, plan) == report
    plan["data"] = {"evaluation_block_hashes": {key: client.study.block_hashes(value)
                                               for key, value in evaluation.items()}}
    summary = load(Path(__file__).resolve().parents[3] / "tools/summarize_wave_gate_long_horizon.py",
                   "angular_summary_test")
    import builtins
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("summary must remain Torch-free")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    observed = summary.summarize(plan, report, journal, client.study.pilot.digest((directory / "results.json").read_bytes()))
    assert observed["initial_filter_receipt_status"] == "passed"
    assert observed["order_trajectories"] == {} and len(observed["angle_trajectories"]) == 3
    assert len(observed["angular_order_trajectories"]) == 3
    assert set(observed["paired_gain_contrasts"]["probe"]) == {
        "gl_short_minus_ordinary_short", "gl_full_minus_ordinary_short", "gl_full_minus_gl_short"}
    for mutation in ("initial", "order", "domain", "continuity", "endpoint"):
        row = copy.deepcopy(report["runs"][0])
        if mutation == "initial": row["records"][0]["alpha_before_update"] = 1.
        elif mutation == "order": row["records"][1]["alpha_after_update"] += .1
        elif mutation == "domain": row["records"][1]["history_angle_after_update"] = -.5
        elif mutation == "continuity":
            row["records"][1]["history_angle_before_update"] = 0.
            row["records"][1]["alpha_before_update"] = 1.
        else: row["final_alpha"] += .1
        with pytest.raises(ValueError): summary.angular_order_trajectory(row)


@pytest.mark.parametrize("key,value", [("schema", "old"), ("arms", []),
    ("reference_arm", "history_angle_full"), ("angle_domain_policy", "clip"),
    ("initial_alpha", 1.), ("initial_gain", 1.), ("initial_history_angle", 0.),
    ("short_kernel_len", 4), ("short_kernel_len", True)])
def test_invalid_recipe_is_rejected(client, key, value):
    config = recipe(client)
    config[key] = value
    with pytest.raises(ValueError): client.adapter_for(client.ARMS[0], config, 41)


def test_sources_and_same_budget_are_bound(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kw: captured.update(kw))
    client.main()
    assert captured["arms"] == client.ARMS
    assert set(captured["adapter_sources"]) == {"angle_study", "gain_control", "lag_control", "fractional_bridge"}
    assert all(path.is_file() for path in captured["adapter_sources"].values())
    root = Path(client.__file__).parent
    config = json.loads((root / "hf_fractional_pride_angle.json").read_text())
    previous = json.loads((root / "hf_fractional_pride_gain.json").read_text())
    for key in ("model_snapshot", "corpus_sha256", "transfer_sha256", "block", "features", "seeds",
                "kernel", "steps", "batch_size", "block_size", "development_blocks", "transfer_blocks",
                "evaluate_every", "checkpoint_every", "learning_rate", "strength", "threads"):
        assert config[key] == previous[key]
