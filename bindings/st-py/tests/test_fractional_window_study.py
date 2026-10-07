import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "bindings/st-py/examples"))
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return load(ROOT / "bindings/st-py/examples/hf_fractional_window_study.py", "window_study_test")


def recipe():
    config = json.loads((ROOT / "bindings/st-py/examples/hf_fractional_pride_window.json").read_text())
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
    config = recipe()
    tokens = torch.arange(48).reshape(8, 6) % 32
    binding = {"config": config, "base_parameter_sha256": client.study.pilot.model_digest(model),
               "data": {"evaluation_block_hashes": {"probe": client.study.block_hashes(tokens[:2])}}}
    plan = {**binding, "study_id": client.study.identity(binding), "source_revision": "a"*40,
            "batch_schedules": {"41": client.study.pilot.schedule(41, 8, 5, 2)}}
    journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
    client.study.atomic_json(path / "plan.json", plan)
    return path, model, model.transformer.h[0], model.transformer.h[0].mlp, plan, journal


def run(client, state, **kwargs):
    path, model, parent, original, plan, journal = state
    tokens = torch.arange(48).reshape(8, 6) % 32
    client.study.run_training(model, parent, "mlp", original, tokens, tokens[:2], plan, path,
                              journal, adapter_factory=client.adapter_for, **kwargs)


def test_same_initial_parameters_and_full_normalization_but_different_order_derivatives(client):
    config = recipe()
    adapters = []
    for arm in client.ARMS:
        rng = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(torch.get_rng_state(), rng)
        assert list(dict(adapter.named_parameters())) == ["gate", "local_gate", "history_angle", "log_gain"]
        assert all(p.device.type == "cpu" and p.requires_grad for p in adapter.parameters())
        assert adapter.alpha == 2. and adapter.get_extra_state()["kernel"]["kernel_len"] == 32
        assert sum(p.numel() for p in adapter.parameters()) == 18
        adapters.append(adapter)
    assert len({client.study.pilot.model_digest(a) for a in adapters}) == 1
    x = torch.linspace(-.9, .8, 96).reshape(2, 6, 8)
    histories, derivatives = [], []
    for adapter in adapters:
        assert torch.equal(adapter(x), x)
        history = adapter._history(x, adapter._alpha_tensor())
        histories.append(history)
        derivatives.append(torch.autograd.grad(history.square().sum(), adapter.history_angle)[0])
    assert torch.equal(*histories)
    assert all(d != 0 for d in derivatives) and derivatives[0] != derivatives[1]
    for source, target in ((adapters[0], adapters[1]), (adapters[1], adapters[0])):
        with pytest.raises(ValueError): target.load_state_dict(source.state_dict())
    previous = client.angle.adapter_for(client.angle.ARMS[-1],
        {**config, "schema": "spiraltorch.fractional_angle_protocol.v1",
         "arms": client.angle.ARMS, "reference_arm": client.angle.ARMS[0]}, 41)
    with pytest.raises(ValueError): adapters[-1].load_state_dict(previous.state_dict())


@pytest.mark.parametrize("arm_index", [0, 1])
@pytest.mark.parametrize("bad", [-.5, 1.6, float("nan"), float("inf")])
def test_domain_exit_preserves_last_valid_state_and_locks_endpoints(client, tmp_path, arm_index, bad):
    state = setup(client, tmp_path / "study")
    def factory(adapter, arm, config):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
        if arm == client.ARMS[arm_index]:
            step, count = optimizer.step, [0]
            def corrupt():
                result = step()
                count[0] += 1
                if count[0] == 3:
                    with torch.no_grad(): adapter.history_angle.fill_(bad)
                return result
            optimizer.step = corrupt
        return optimizer
    with pytest.raises(client.study.TerminalStudyError): run(client, state, optimizer_factory=factory)
    entry = state[-1]["runs"][f"41:{client.ARMS[arm_index]}"]
    assert entry["cursor"] == 2 and state[-1]["terminal_failure"]["attempted_step"] == 3
    assert state[2].mlp is state[3]
    before = {p.name: p.read_bytes() for p in state[0].iterdir()}
    with pytest.raises(ValueError, match="cannot resume"): run(client, state)
    with pytest.raises(ValueError, match="terminal protocol failure"):
        client.study.endpoint_gate(state[-1], state[-2], state[0])
    assert before == {p.name: p.read_bytes() for p in state[0].iterdir()}


def test_resume_summary_and_saved_state_verifier_cover_both_windows(client, tmp_path):
    full = setup(client, tmp_path / "full")
    run(client, full)
    interrupted = setup(client, tmp_path / "interrupted")
    def stop(key, cursor):
        if key == f"41:{client.ARMS[-1]}" and cursor == 2:
            raise RuntimeError("infrastructure interruption")
    with pytest.raises(RuntimeError, match="interruption"):
        run(client, interrupted, after_checkpoint=stop)
    assert "terminal_failure" not in interrupted[-1]
    resumed = list(setup(client, tmp_path / "fresh-process"))
    resumed[0], resumed[-1] = interrupted[0], json.loads((interrupted[0] / "journal.json").read_bytes())
    run(client, resumed)
    for key, entry in full[-1]["runs"].items():
        left = client.study.load_checkpoint(full[0], entry["checkpoint"], full[-2]["study_id"], key)
        right = client.study.load_checkpoint(resumed[0], resumed[-1]["runs"][key]["checkpoint"], full[-2]["study_id"], key)
        for field in ("adapter", "optimizer", "records", "development"):
            assert client.study.pilot.equal_state(left[field], right[field])
        assert left["records"][0]["history_angle_gradient"] == left["records"][0]["log_gain_gradient"] == 0
        assert any(r["history_angle_gradient"] != 0 for r in left["records"][1:])
    directory, model, parent, original, plan, journal = full
    report = client.study.run_endpoints(model, parent, "mlp", original,
        {"probe": torch.arange(12).reshape(2, 6) % 32}, plan, directory, journal,
        adapter_factory=client.adapter_for, result_schema="spiraltorch.fractional_window_study.v1")
    summary = load(ROOT / "tools/summarize_wave_gate_long_horizon.py", "window_summary")
    verifier = load(ROOT / "tools/verify_fractional_gain_study.py", "window_verify")
    observed = summary.summarize(plan, report, journal, journal["results_sha256"])
    assert observed["initial_filter_receipt_status"] == "passed"
    assert set(observed["paired_window_contrasts"]["probe"]) == {"full_minus_retained_short"}
    assert len(observed["angular_order_trajectories"]) == len(observed["gain_trajectories"]) == 2
    observed["input_sha256"] = {key: verifier.digest(directory / f"{key}.json") for key in ("plan", "results", "journal")}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(observed, indent=2, allow_nan=False) + "\n")
    saved_report = verifier.verify(directory, path, client, summary)
    assert saved_report["planned_runs_verified"] == 2 and saved_report["primary_updates"] == 8
    assert saved_report["continuation_only_updates"] == 4 and saved_report["status"] == "passed"
    assert "paired_short_states" not in saved_report
    current = verifier.common.completed(directory, client)
    for key, row in current[-1].items():
        entry = journal["runs"][key]
        for corruption in ("window", "normalization", "moment"):
            saved = client.study.load_checkpoint(directory, entry["checkpoint"], plan["study_id"], key)
            if corruption == "window": saved["adapter"]["_extra_state"]["lag_window"] = [3, 32]
            elif corruption == "normalization": saved["adapter"]["_extra_state"]["kernel"]["kernel_len"] = 3
            else: saved["optimizer"]["state"][0]["exp_avg_sq"][0] = -1.
            with pytest.raises(ValueError): verifier.inspect_run(client, plan, row, entry, saved)


@pytest.mark.parametrize("key,value", [("schema", "old"), ("arms", []),
    ("reference_arm", "history_window_full"), ("angle_domain_policy", "clip"),
    ("initial_alpha", 1.), ("initial_gain", 1.), ("initial_history_angle", 0.),
    ("normalization_policy", "window"), ("lag_windows", {"history_window_short": [True, 3], "history_window_full": None}),
    ("lag_windows", {"history_window_short": [1, 3.], "history_window_full": None}),
    ("lag_windows", {"history_window_short": [1, 4], "history_window_full": None})])
def test_invalid_protocol_cannot_start(client, key, value):
    config = recipe()
    config[key] = value
    with pytest.raises(ValueError): client.adapter_for(client.ARMS[0], config, 41)


def test_same_budget_and_complete_executable_source_bindings(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kw: captured.update(kw))
    client.main()
    assert set(captured["adapter_sources"]) == {"window_study", "angle_control", "gain_control", "lag_control", "fractional_bridge"}
    assert all(path.is_file() for path in captured["adapter_sources"].values())
    config = json.loads((ROOT / "bindings/st-py/examples/hf_fractional_pride_window.json").read_text())
    old = json.loads((ROOT / "bindings/st-py/examples/hf_fractional_pride_angle.json").read_text())
    for key in ("model_snapshot", "corpus_sha256", "transfer_sha256", "features", "seeds", "kernel",
                "steps", "batch_size", "block_size", "development_blocks", "transfer_blocks",
                "evaluate_every", "checkpoint_every", "learning_rate", "strength", "threads", "block"):
        assert config[key] == old[key]
