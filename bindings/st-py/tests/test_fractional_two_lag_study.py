import builtins
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def client(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "examples" / "hf_fractional_two_lag_study.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("two_lag_study_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def summary_module():
    source = Path(__file__).resolve().parents[3] / "tools" / "summarize_wave_gate_long_horizon.py"
    spec = importlib.util.spec_from_file_location("two_lag_summary_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(client):
    return {"schema": "spiraltorch.fractional_two_lag_protocol.v1",
            "features": 8, "strength": .1, "initial_orders": dict(client.INITIAL_ORDERS),
            "kernel": {"kernel_len": 4, "step": 1.0}, "learning_rate": .01,
            "steps": 4, "batch_size": 2, "checkpoint_every": 2, "evaluate_every": 2,
            "arms": client.ARMS, "seeds": [41], "reference_arm": "lag2"}


def test_identity_capacity_rng_and_recipe_separation(client):
    config = recipe(client)
    adapters = {}
    for arm in client.ARMS:
        before = torch.get_rng_state().clone()
        with torch.device("meta"):
            adapter = client.adapter_for(arm, config, 41)
        assert torch.equal(before, torch.get_rng_state())
        assert all(p.device.type == "cpu" for p in adapter.parameters())
        assert sum(p.numel() for p in adapter.parameters()) == 16 + int(arm != "lag2")
        learned = arm.startswith("history_learned_")
        assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == 16 + int(learned)
        x = torch.linspace(-1, 1, 96).reshape(2, 6, 8)
        assert torch.equal(adapter(x), x)
        restored = client.adapter_for(arm, config, 47)
        restored.load_state_dict(copy.deepcopy(adapter.state_dict()))
        assert client.study.pilot.equal_state(adapter.state_dict(), restored.state_dict())
        adapters[arm] = adapter
    digest = client.study.pilot.model_digest
    assert digest(adapters["history_fixed_two"]) == digest(adapters["history_learned_two"])
    for left in client.ARMS:
        for right in client.ARMS:
            if left != right:
                with pytest.raises((ValueError, RuntimeError)):
                    adapters[left].load_state_dict(adapters[right].state_dict())
    old = client.one_lag.CausalLagGate(8, strength=.1, kernel_len=4)
    with pytest.raises(ValueError, match="recipe"):
        adapters["lag2"].load_state_dict(old.state_dict())


@pytest.mark.parametrize("time,features", [(1, 3), (2, 3), (3, 3), (6, 3), (128, 768), (256, 768)])
def test_two_tap_and_rust_outputs_gate_and_input_gradients(client, time, features):
    config = recipe(client)
    config.update(features=features, kernel={"kernel_len": 32, "step": 1.0})
    ordinary = client.adapter_for("lag2", config, 41)
    fixed = client.adapter_for("history_fixed_two", config, 41)
    with torch.no_grad():
        for adapter in (ordinary, fixed):
            adapter.gate.copy_(torch.linspace(-.6, .7, features))
            adapter.local_gate.copy_(torch.linspace(.4, -.3, features))
    generator = torch.Generator().manual_seed(239)
    torch.set_num_threads(2)
    x = torch.randn(2, features, time, generator=generator).transpose(1, 2).requires_grad_()
    upstream = torch.randn(x.shape, generator=generator)
    left, right = ordinary(x), fixed(x)
    assert torch.equal(left, right)
    a = torch.autograd.grad(left, (x, ordinary.gate, ordinary.local_gate), upstream)
    b = torch.autograd.grad(right, (x, fixed.gate, fixed.local_gate), upstream)
    torch.testing.assert_close(a[0], b[0], rtol=3e-6, atol=3e-6)
    assert all(torch.equal(l, r) for l, r in zip(a[1:], b[1:]))


def test_causal_boundaries_and_integer_order_tail_derivative(client):
    ordinary = client.CausalTwoLagGate(3, strength=.1, kernel_len=4)
    x = torch.arange(36, dtype=torch.float32).reshape(2, 6, 3).requires_grad_()
    y = ordinary.history(x)
    assert torch.count_nonzero(y[:, 0]) == 0
    assert torch.equal(y[:, 1], -2*x[:, 0])
    assert torch.equal(y[:, 2:], -2*x[:, 1:-1] + x[:, :-2])
    modified = x.detach().clone()
    modified[0, 2:] = 90
    modified[1] = -80
    assert torch.equal(y[0, :3], ordinary.history(modified)[0, :3])
    gradient = torch.autograd.grad(y[0, 2, 1], x)[0]
    assert gradient[0, 1, 1] == -2 and gradient[0, 0, 1] == 1
    assert torch.count_nonzero(gradient) == 2
    impulse = torch.tensor([1., 0., 0., 0.]).reshape(1, 4, 1)
    alpha = torch.tensor(2., requires_grad=True)
    history = client.st.fractional_gl_history_autograd(
        impulse, alpha, axis=1, kernel=client.st.FractionalGlKernel(kernel_len=4))
    assert history[0, 3, 0] == 0
    assert torch.autograd.grad(history[0, 3, 0], alpha)[0] == torch.tensor(-1/3)


def test_double_reference_preserves_f32_limit_cancellation_and_rejects_overflow(client):
    ordinary = client.CausalTwoLagGate(1, strength=.1, kernel_len=4)
    largest = torch.finfo(torch.float32).max
    value = torch.tensor([largest/2, largest*.75, largest*.75, 0.]).reshape(1, 4, 1)
    native = client.st.fractional_gl_history_autograd(
        value, torch.tensor(2.), axis=1, kernel=client.st.FractionalGlKernel(kernel_len=4))
    assert torch.isfinite(native).all()
    assert torch.equal(ordinary.history(value), native)
    assert not torch.isfinite(-2*value).all()
    bad = torch.tensor([largest, 0., 0.]).reshape(1, 3, 1)
    with pytest.raises(ValueError, match="nonfinite"):
        ordinary.history(bad)
    with pytest.raises(ValueError):
        client.st.fractional_gl_history_autograd(
            bad, torch.tensor(2.), axis=1, kernel=client.st.FractionalGlKernel(kernel_len=4))


@pytest.mark.parametrize("invalid", ["protocol", "boolean_step", "step", "taps", "bool_order", "order", "extra_order"])
def test_two_lag_recipe_rejects_changed_semantics(client, invalid):
    config = recipe(client)
    if invalid == "protocol":
        config["schema"] = "spiraltorch.fractional_lag_protocol.v1"
    elif invalid in ("boolean_step", "step"):
        config["kernel"]["step"] = True if invalid == "boolean_step" else .5
    elif invalid == "taps":
        config["kernel"]["kernel_len"] = 2
    elif invalid == "bool_order":
        config["initial_orders"]["history_learned_one"] = True
    elif invalid == "order":
        config["initial_orders"]["history_learned_two"] = 1.
    else:
        config["initial_orders"]["surprise"] = .5
    with pytest.raises(ValueError):
        client.adapter_for("lag2", config, 41)


def test_two_lag_domain_budget_and_transitive_source_binding(client, monkeypatch):
    ordinary = client.CausalTwoLagGate(3, strength=.1, kernel_len=4, max_values=36, max_products=144)
    for x in (torch.zeros(2, 3), torch.zeros(0, 6, 3), torch.zeros(2, 0, 3),
              torch.zeros(2, 6, 4), torch.full((2, 6, 3), float("nan")),
              torch.zeros(2, 6, 3, dtype=torch.float64), torch.zeros(3, 6, 3)):
        with pytest.raises((ValueError, TypeError)):
            ordinary(x)
    with torch.no_grad():
        ordinary.gate[0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        ordinary(torch.zeros(2, 6, 3))
    with pytest.raises(ValueError, match="three taps"):
        client.CausalTwoLagGate(3, strength=.1, kernel_len=2)
    captured = {}
    monkeypatch.setattr(client.study, "main", lambda **kwargs: captured.update(kwargs))
    client.main()
    assert captured["arms"] == client.ARMS
    assert set(captured["adapter_sources"]) == {"two_lag_study", "one_lag_control", "fractional_bridge"}
    assert all(p.is_file() for p in captured["adapter_sources"].values())
    source = Path(client.__file__)
    config = json.loads(source.with_name("hf_fractional_pride_two_lag.json").read_text())
    previous = json.loads(source.with_name("hf_fractional_pride_lag.json").read_text())
    assert config["arms"] == client.ARMS and config["initial_orders"] == client.INITIAL_ORDERS
    assert config["reference_arm"] == "lag2"
    for key in ("model_snapshot", "corpus_sha256", "transfer_sha256", "block", "features",
                "seeds", "kernel", "steps", "batch_size", "block_size", "development_blocks",
                "transfer_blocks", "evaluate_every", "checkpoint_every", "learning_rate", "strength", "threads"):
        assert config[key] == previous[key]


def test_four_hf_arms_resume_and_same_math_training_matches(client, summary_module, tmp_path, monkeypatch):
    driver, config = client.study, recipe(client)
    tokens = torch.arange(48).reshape(8, 6) % 32

    def setup(name):
        directory = tmp_path / name
        directory.mkdir()
        torch.manual_seed(139)
        torch.set_num_threads(2)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
            resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
        )).eval().requires_grad_(False)
        parent = model.transformer.h[0]
        plan = {"study_id": "two-lag-study-test", "config": config,
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
        if key == "41:history_learned_two" and cursor == 2:
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        run(interrupted, stop)
    with pytest.raises(ValueError, match="all planned runs must finish"):
        driver.endpoint_gate(interrupted[-1], interrupted[-2], interrupted[0])
    resumed = setup("resumed")
    resumed[0] = interrupted[0]
    resumed[-1] = json.loads((interrupted[0] / "journal.json").read_text())
    run(resumed)
    saved = {}
    for key, entry in full[-1]["runs"].items():
        left = driver.load_checkpoint(full[0], entry["checkpoint"], full[-2]["study_id"], key)
        right = driver.load_checkpoint(resumed[0], resumed[-1]["runs"][key]["checkpoint"], resumed[-2]["study_id"], key)
        assert driver.pilot.equal_state(left, right)
        assert entry["frozen_base_unchanged"] and entry["resume_next_update_equal"]
        saved[key] = left
        rows = left["records"]
        assert all(r["gate_gradient_l2"] > 0 and r["local_gate_gradient_l2"] > 0 for r in rows)
        if key.endswith("history_fixed_two"):
            assert all(r["log_alpha_gradient"] is None and not r["log_alpha_trainable"] for r in rows)
            assert all(r["alpha_after_update"] == 2. for r in rows)
        elif "history_learned_" in key:
            assert rows[0]["log_alpha_gradient"] == 0
            assert any(r["log_alpha_gradient"] != 0 for r in rows[1:])
            assert rows[0]["log_alpha_before_update"] != rows[-1]["log_alpha_after_update"]
    ordinary, fixed = (saved[f"41:{arm}"] for arm in ("lag2", "history_fixed_two"))
    for field in ("gate", "local_gate"):
        assert torch.equal(ordinary["adapter"][field], fixed["adapter"][field])
    for lag_id, gl_id in ((0, 0), (1, 2)):
        assert driver.pilot.equal_state(ordinary["optimizer"]["state"][lag_id], fixed["optimizer"]["state"][gl_id])
    assert 1 not in fixed["optimizer"]["state"]
    directory, model, parent, original, plan, journal = full
    result = driver.run_endpoints(model, parent, "mlp", original, {"tail": tokens[:2]},
                                  plan, directory, journal, adapter_factory=client.adapter_for,
                                  result_schema="spiraltorch.fractional_two_lag_study.v1")
    assert driver.completed_result(directory, journal, plan) == result
    plan["data"] = {"evaluation_block_hashes": {"tail": driver.block_hashes(tokens[:2])}}
    original_import = builtins.__import__

    def no_torch_import(name, *args, **kwargs):
        assert name != "torch", "receipt-only summary must not import Torch"
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_torch_import)
        report = summary_module.summarize(plan, result, journal, journal["results_sha256"])
    assert report["same_math_parity_status"] == "unverified"
    assert report["same_math_state_parity"]["41"] == {
        "status": "unverified", "saved_gates_equal": None, "saved_named_adam_equal": None}
    assert report["order_trajectories"]["41:history_fixed_two"]["final_alpha"] == 2.
    assert report["order_trajectories"]["41:history_learned_two"]["nonzero_order_gradient_steps"] > 0
    assert report["same_math_receipt_parity"]["41"] == {
        "update_receipts_equal": True, "development_equal": True,
        "endpoint_block_losses_equal": {"tail": True}}
    verified = summary_module.summarize(plan, result, journal, journal["results_sha256"],
                                        checkpoint_dir=directory)
    assert verified["same_math_parity_status"] == "passed"
    states = verified["same_math_state_parity"]["41"]
    assert states["saved_gates_equal"] and states["saved_named_adam_equal"]
    assert set(states["states"]["history_fixed_two"]["named_adam"]) == {"gate", "local_gate"}
    driver.atomic_json(directory / "plan.json", plan)
    output = directory / "summary.json"
    argv = [summary_module.__file__]
    for name in ("plan", "results", "journal"):
        argv.extend([f"--{name}", str(directory / f"{name}.json")])
    argv.extend(["--output", str(output), "--checkpoint-dir", str(directory)])
    monkeypatch.setattr(sys, "argv", argv)
    summary_module.main()
    cli_report = json.loads(output.read_text())
    assert cli_report["same_math_state_parity"] == verified["same_math_state_parity"]
    assert cli_report["same_math_parity_status"] == "passed"
    assert cli_report["input_sha256"]["results"] == journal["results_sha256"]
    # Equal scalar receipts and vector norms must not hide changed vector content.
    for path in (("adapter", "gate"), ("adapter", "local_gate"),
                 ("optimizer", "state", 0, "exp_avg"),
                 ("optimizer", "state", 2, "exp_avg_sq")):
        altered = copy.deepcopy(fixed)
        tensor = altered
        for part in path:
            tensor = tensor[part]
        before = tensor.clone()
        tensor[0].neg_()
        assert not torch.equal(before, tensor) and torch.equal(before.norm(), tensor.norm())
        receipt = driver.save_checkpoint(directory, altered)
        changed_result, changed_journal = copy.deepcopy(result), copy.deepcopy(journal)
        key = "41:history_fixed_two"
        next(row for row in changed_result["runs"] if row["run_key"] == key)["checkpoint"] = receipt
        changed_journal["runs"][key]["checkpoint"] = receipt
        changed_hash = driver.pilot.digest(json.dumps(changed_result, sort_keys=True).encode())
        changed_journal["results_sha256"] = changed_hash
        failed = summary_module.summarize(plan, changed_result, changed_journal, changed_hash,
                                          checkpoint_dir=directory)
        assert failed["same_math_receipt_parity"] == report["same_math_receipt_parity"]
        assert failed["same_math_parity_status"] == "failed"
        parity = failed["same_math_state_parity"]["41"]
        assert parity["saved_gates_equal"] is (path[0] != "adapter")
        assert parity["saved_named_adam_equal"] is (path[0] != "optimizer")


def receipts(client):
    config, runs = recipe(client), {}
    for arm in client.ARMS:
        learned = arm.startswith("history_learned_")
        initial = math.log(config["initial_orders"].get(arm, 2.))
        final = initial + .1 if learned else initial
        runs[f"41:{arm}"] = {
            "initial_parameter_sha256": "a" * 64,
            "parameter_count": 16 + int(arm != "lag2"), "trainable_parameter_count": 16 + int(learned),
            "records": [{"loss": 2., "gate_gradient_l2": 1., "local_gate_gradient_l2": 2.,
                         "gate_before_update_l2": 0., "local_gate_before_update_l2": 0.,
                         "log_alpha_before_update": a, "log_alpha_after_update": b,
                         "alpha_before_update": math.exp(a), "alpha_after_update": math.exp(b),
                         "log_alpha_trainable": learned,
                         "log_alpha_gradient": (0. if i == 0 else .1) if learned else None}
                        for i, (a, b) in enumerate([(initial, initial), (initial, final)])],
            "final_log_alpha": final, "final_alpha": math.exp(final),
            "development": [], "scores": {"tail": {"block_losses": [2.]}}}
    return config, runs


@pytest.mark.parametrize("corruption", ["initial", "pairing", "count", "mode", "fixed", "gradient",
                                         "endpoint", "continuity", "empty", "nan", "receipt", "kernel"])
def test_summary_rejects_broken_two_lag_controls(client, summary_module, corruption):
    config, runs = receipts(client)
    row = runs["41:history_fixed_two"]
    receipt = row["records"][0]
    if corruption == "initial":
        config["initial_orders"]["history_learned_one"] = 2.
    elif corruption == "pairing":
        row["initial_parameter_sha256"] = "b" * 64
    elif corruption == "count":
        row["trainable_parameter_count"] = 17
    elif corruption == "mode":
        receipt["log_alpha_trainable"] = True
    elif corruption == "fixed":
        row["records"][1].update(log_alpha_after_update=.1, alpha_after_update=math.exp(.1))
        row.update(final_log_alpha=.1, final_alpha=math.exp(.1))
    elif corruption == "gradient":
        runs["41:history_learned_two"]["records"][0]["log_alpha_gradient"] = .1
    elif corruption == "endpoint":
        row["final_alpha"] = .6
    elif corruption == "continuity":
        row["records"][1].update(log_alpha_before_update=.1, alpha_before_update=math.exp(.1))
    elif corruption == "empty":
        row["records"] = []
    elif corruption == "nan":
        receipt["alpha_before_update"] = float("nan")
    elif corruption == "receipt":
        receipt.pop("local_gate_gradient_l2")
    else:
        config["kernel"]["kernel_len"] = 2
    with pytest.raises(ValueError):
        summary_module.fractional_lag_report(config, runs, {}, {})


def test_summary_keeps_two_lag_parity_failures_and_initialization_separate(client, summary_module):
    config, runs = receipts(client)
    measured = {f"41:{arm}": {"tail": v} for arm, v in zip(client.ARMS, (2., 2., 1.8, 1.9))}
    contrasts, orders, parity = summary_module.fractional_lag_report(config, runs, measured, {"tail": []})
    assert contrasts["tail"]["fixed_two_minus_lag2"]["mean_ce_difference"] == 0
    assert contrasts["tail"]["learned_two_minus_fixed_two"]["mean_ce_difference"] == pytest.approx(-.2)
    assert contrasts["tail"]["learned_one_minus_learned_two"]["mean_ce_difference"] == pytest.approx(.1)
    assert orders["41:history_learned_two"]["initial_alpha"] == 2.
    assert orders["41:history_learned_one"]["initial_alpha"] == 1.
    assert parity["41"]["update_receipts_equal"]
    runs["41:history_fixed_two"]["records"][1]["loss"] += .1
    runs["41:history_fixed_two"]["scores"]["tail"]["block_losses"] = [2.1]
    _, _, parity = summary_module.fractional_lag_report(config, runs, measured, {"tail": []})
    assert not parity["41"]["update_receipts_equal"]
    assert not parity["41"]["endpoint_block_losses_equal"]["tail"]


@pytest.fixture
def saved_controls(client, tmp_path):
    config, runs = receipts(client)
    config["steps"] = 2
    plan = {"study_id": "state-parity-fixture", "config": config}
    saved = {}
    for arm in ("lag2", "history_fixed_two"):
        adapter = client.adapter_for(arm, config, 41)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
        for _ in range(config["steps"]):
            for parameter in adapter.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        key = f"41:{arm}"
        if arm == "history_fixed_two":
            log_alpha = float(adapter.log_alpha.detach())
            runs[key]["final_log_alpha"] = log_alpha
        saved[key] = {"study_id": plan["study_id"], "run_key": key, "cursor": config["steps"],
                      "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(),
                      "frozen_base_verified": True, "initial_parameter_sha256": "a" * 64,
                      "records": runs[key]["records"], "development": runs[key]["development"]}
        runs[key]["checkpoint"] = client.study.save_checkpoint(tmp_path, saved[key])
    return plan, runs, saved, tmp_path


@pytest.mark.parametrize("corruption", ["hash", "escape", "symlink", "identity", "cursor", "records",
                                         "initial", "base", "recipe", "names", "dtype", "shape", "nan",
                                         "frozen_adam", "missing_adam", "adam_step", "adam_ids", "adam_lr"])
def test_state_parity_rejects_unbound_or_incomplete_checkpoints(client, summary_module, saved_controls, corruption):
    plan, runs, saved, directory = saved_controls
    key = "41:history_fixed_two"
    value, row = saved[key], runs[key]
    if corruption == "hash":
        row["checkpoint"]["sha256"] = "0" * 64
    elif corruption == "escape":
        row["checkpoint"]["filename"] = "../" + row["checkpoint"]["filename"]
    elif corruption == "symlink":
        link = directory / "checkpoint-link.pt"
        link.symlink_to(directory / row["checkpoint"]["filename"])
        row["checkpoint"]["filename"] = link.name
    else:
        value = copy.deepcopy(value)
        if corruption == "identity":
            value["run_key"] = "43:history_fixed_two"
        elif corruption == "cursor":
            value["cursor"] -= 1
        elif corruption == "records":
            value["records"][0]["loss"] += .1
        elif corruption == "initial":
            value["initial_parameter_sha256"] = "b" * 64
        elif corruption == "base":
            value["frozen_base_verified"] = False
        elif corruption == "recipe":
            value["adapter"]["_extra_state"]["arm"] = "history_learned_two"
        elif corruption == "names":
            value["adapter"]["renamed_gate"] = value["adapter"].pop("gate")
        elif corruption == "dtype":
            value["adapter"]["gate"] = value["adapter"]["gate"].double()
        elif corruption == "shape":
            value["adapter"]["gate"] = value["adapter"]["gate"][None]
        elif corruption == "nan":
            value["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
        elif corruption == "frozen_adam":
            value["optimizer"]["state"][1] = copy.deepcopy(value["optimizer"]["state"][0])
        elif corruption == "missing_adam":
            value["optimizer"]["state"].pop(2)
        elif corruption == "adam_step":
            value["optimizer"]["state"][2]["step"] -= 1
        elif corruption == "adam_ids":
            value["optimizer"]["param_groups"][0]["params"] = [0, 0, 2]
        else:
            value["optimizer"]["param_groups"][0]["lr"] *= 2
        row["checkpoint"] = client.study.save_checkpoint(directory, value)
    with pytest.raises(ValueError):
        summary_module.two_lag_state_parity(plan, runs, directory)


def test_state_parity_uses_named_adam_not_raw_ids_and_checks_group_settings(client, summary_module, saved_controls):
    plan, runs, saved, directory = saved_controls
    key = "41:history_fixed_two"
    optimizer = saved[key]["optimizer"]
    optimizer["param_groups"][0]["params"] = [30, 31, 32]
    optimizer["state"] = {30+i: value for i, value in optimizer["state"].items()}
    runs[key]["checkpoint"] = client.study.save_checkpoint(directory, saved[key])
    assert summary_module.two_lag_state_parity(plan, runs, directory)["41"]["status"] == "passed"
    optimizer["param_groups"][0]["eps"] *= 2
    runs[key]["checkpoint"] = client.study.save_checkpoint(directory, saved[key])
    result = summary_module.two_lag_state_parity(plan, runs, directory)["41"]
    assert result["status"] == "failed" and not result["saved_named_adam_equal"]
    assert result["saved_gates_equal"]


@pytest.mark.parametrize("arm", ["lag2", "history_fixed_two"])
@pytest.mark.parametrize("corruption", ["missing_schema", "wrong_schema", "extra_field",
                                         "missing_kernel_budget", "extra_kernel_field"])
def test_state_parity_requires_the_complete_adapter_recipe(client, summary_module, saved_controls, arm, corruption):
    plan, runs, saved, directory = saved_controls
    key = f"41:{arm}"
    extra = saved[key]["adapter"]["_extra_state"]
    if corruption == "missing_schema":
        extra.pop("schema")
    elif corruption == "wrong_schema":
        extra["schema"] = "spiraltorch.fractional_memory_adapter.v1"
    elif corruption == "extra_field":
        extra["unsupported"] = True
    elif corruption == "missing_kernel_budget":
        extra["kernel"].pop("max_values")
    else:
        extra["kernel"]["unsupported"] = True
    with pytest.raises(ValueError, match="recipe"):
        client.adapter_for(arm, plan["config"], 41).load_state_dict(saved[key]["adapter"])
    runs[key]["checkpoint"] = client.study.save_checkpoint(directory, saved[key])
    with pytest.raises(ValueError, match="recipe"):
        summary_module.two_lag_state_parity(plan, runs, directory)


@pytest.mark.parametrize("field", ["lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
                                   "foreach", "capturable", "differentiable", "fused"])
def test_state_parity_rejects_matching_incomplete_adam_groups(client, summary_module, saved_controls, field):
    plan, runs, saved, directory = saved_controls
    for key, value in saved.items():
        value["optimizer"]["param_groups"][0].pop(field)
        runs[key]["checkpoint"] = client.study.save_checkpoint(directory, value)
    with pytest.raises(ValueError, match="Adam configuration"):
        summary_module.two_lag_state_parity(plan, runs, directory)


@pytest.mark.parametrize("field,value", [
    ("lr", True), ("lr", float("nan")), ("betas", []), ("betas", [.9]),
    ("betas", [.9, 1.]), ("betas", [-.1, .999]), ("betas", [True, .999]),
    ("betas", [.9, float("nan")]), ("betas", [.9, "0.999"]),
    ("eps", -1.), ("eps", float("nan")), ("eps", True),
    ("weight_decay", -1.), ("weight_decay", float("inf")),
    ("amsgrad", 0), ("maximize", "false"), ("capturable", 0),
    ("differentiable", None), ("foreach", "auto"), ("fused", 0),
    ("decoupled_weight_decay", "false"),
])
def test_state_parity_rejects_matching_invalid_adam_groups(client, summary_module, saved_controls, field, value):
    plan, runs, saved, directory = saved_controls
    for key, state in saved.items():
        state["optimizer"]["param_groups"][0][field] = value
        runs[key]["checkpoint"] = client.study.save_checkpoint(directory, state)
    with pytest.raises(ValueError, match="Adam configuration"):
        summary_module.two_lag_state_parity(plan, runs, directory)


def test_checkpoint_option_does_not_silently_ignore_other_protocols(summary_module):
    plan = {"study_id": "s", "config": {"schema": "spiraltorch.fractional_lag_protocol.v1"}}
    result = {"status": "completed", "study_id": "s"}
    journal = {**result, "results_sha256": "a" * 64}
    with pytest.raises(ValueError, match="only for the two-lag protocol"):
        summary_module.summarize(plan, result, journal, "a" * 64, checkpoint_dir=".")
