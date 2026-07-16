from __future__ import annotations

import pytest

import spiraltorch as st


class _UnitRateControl:
    def hyper_rate_scale(self) -> float:
        return 1.0

    def real_rate_scale(self) -> float:
        return 1.0


def _require_native() -> None:
    try:
        st.hypergrad((1, 1))
        st.realgrad((1, 1))
    except Exception as exc:  # pragma: no cover - depends on native build
        pytest.skip(f"native SpiralTorch extension required: {exc}", allow_module_level=True)


def test_amegagrad_available_in_optim_module() -> None:
    _require_native()
    assert hasattr(st, "optim")
    assert hasattr(st.optim, "Amegagrad")
    assert hasattr(st.optim, "amegagrad")
    assert "compare_amegagrad_topos_training_traces" in st.__all__
    assert "trace_amegagrad_topos_training_sweep" in st.__all__


def test_amegagrad_step_updates_weights() -> None:
    _require_native()

    opt = st.optim.Amegagrad((1, 3), curvature=-0.9, hyper_learning_rate=0.03, real_learning_rate=0.02)
    assert opt.shape() == (1, 3)

    weights = st.Tensor((1, 3), data=[0.2, -0.1, 0.05])
    signal = st.Tensor((1, 3), data=[0.4, -0.6, 0.2])
    opt.accumulate_wave(signal)

    hyper_grad = opt.hyper.gradient()
    real_grad = opt.real.gradient()
    assert len(hyper_grad) == 3
    assert len(real_grad) == 3
    assert any(abs(value) > 0.0 for value in hyper_grad)
    assert any(abs(value) > 0.0 for value in real_grad)

    before = weights.tolist()
    updated = opt.step(weights, tune=False)
    after = weights.tolist()
    assert updated is weights
    assert after != before


def test_amegagrad_step_rolls_back_late_real_tape_failure() -> None:
    _require_native()

    opt = st.optim.Amegagrad(
        (1, 1),
        hyper_learning_rate=0.05,
        real_learning_rate=3.4028235e38,
    )
    weights = st.Tensor((1, 1), data=[0.0])
    opt.accumulate_wave(st.Tensor((1, 1), data=[2.0]))
    weights_before = weights.tolist()
    hyper_gradient_before = opt.hyper.gradient()
    real_gradient_before = opt.real.gradient()
    hyper_momentum_before = opt.hyper.optimizer_momentum()
    real_momentum_before = opt.real.optimizer_momentum()

    with pytest.raises(ValueError, match="realgrad_delta"):
        opt.step(weights, tune=False)

    assert weights.tolist() == weights_before
    assert opt.hyper.gradient() == hyper_gradient_before
    assert opt.real.gradient() == real_gradient_before
    assert opt.hyper.optimizer_momentum() == hyper_momentum_before
    assert opt.real.optimizer_momentum() == real_momentum_before


def test_amegagrad_absorb_text_handles_variable_length() -> None:
    _require_native()
    if not hasattr(st, "LanguageWaveEncoder"):
        pytest.skip("LanguageWaveEncoder unavailable in this build")

    encoder = st.LanguageWaveEncoder(-1.0, 0.5)
    rows, cols = encoder.encode_z_space("seed").shape()

    opt = st.optim.Amegagrad((rows, cols), curvature=float(encoder.curvature()))
    weights = st.Tensor(rows, cols, [0.0] * (rows * cols))

    opt.absorb_text(encoder, "SpiralTorch wheels smoke test")
    before = weights.tolist()
    opt.step(weights, tune=False)
    after = weights.tolist()
    assert after != before


def test_amegagrad_applies_topos_training_hints_to_tune() -> None:
    _require_native()

    guard = st.hypergrad_topos(
        curvature=-0.9,
        tolerance=1e-4,
        saturation=2.0,
        max_depth=8,
        max_volume=16,
    )
    opt = st.optim.Amegagrad(
        (1, 4),
        curvature=-0.9,
        hyper_learning_rate=0.04,
        real_learning_rate=0.02,
        topos=guard,
        topos_control_gain=1.0,
        topos_observed_depth=4,
        topos_visited_volume=8,
    )

    hints = opt.topos_training_hints()
    expected_scale = hints["learning_rate_scale"]

    control = _UnitRateControl()
    returned = opt.tune(control=control, use_topos=True)

    assert returned is control
    assert opt.hyper.learning_rate() == pytest.approx(0.04 * expected_scale)
    assert opt.real.learning_rate() == pytest.approx(0.02 * expected_scale)
    diagnostics = opt.topos_diagnostics()
    assert diagnostics["snapshot"]["kind"] == "spiraltorch.topos_optimizer_snapshot"
    assert diagnostics["snapshot"]["sequence"] == 1
    assert diagnostics["training_hints"]["clip_scale"] == pytest.approx(
        hints["clip_scale"]
    )
    assert diagnostics["training_plan"]["rate_scale"] == pytest.approx(expected_scale)
    assert diagnostics["runtime_profile"]["training_rate_scale"] == pytest.approx(
        expected_scale
    )
    assert diagnostics["runtime_profile"]["control_energy"] > 0.0
    assert diagnostics["effect"]["rate_scale"] == pytest.approx(expected_scale)
    assert diagnostics["training_plan"]["raw_rate_scale"] == pytest.approx(
        hints["learning_rate_scale"]
    )
    assert diagnostics["effect"]["effective_gradient_clip_scale"] == pytest.approx(
        diagnostics["training_plan"]["effective_gradient_clip_scale"]
    )
    assert diagnostics["effect"]["gradient_clip_normalization"] == "biased_gradient_rms"
    assert diagnostics["training_plan"]["effective_gradient_bias_scale"] > 0.0
    assert diagnostics["effect"]["effective_gradient_bias_scale"] == pytest.approx(
        diagnostics["training_plan"]["effective_gradient_bias_scale"]
    )
    assert diagnostics["effect"]["effective_momentum_damping"] == pytest.approx(
        diagnostics["training_plan"]["effective_momentum_damping"]
    )
    assert diagnostics["effect"]["gradient_bias_normalization"] == "raw_gradient_rms"
    assert opt.hyper.optimizer_state_control() == opt.real.optimizer_state_control()
    assert opt.hyper.optimizer_state_control()[
        "effective_gradient_bias_scale"
    ] == pytest.approx(diagnostics["effect"]["effective_gradient_bias_scale"])

    telemetry_contract = opt.topos_telemetry_contract()
    assert telemetry_contract["kind"] == "spiraltorch.zspace_telemetry_fusion"
    assert telemetry_contract["contract_version"] == (
        "spiraltorch.zspace_telemetry_fusion.v1"
    )
    assert telemetry_contract["semantic_owner"] == "st-core::telemetry::zspace_fusion"
    assert telemetry_contract["semantic_backend"] == "rust"
    assert telemetry_contract["input_count"] == 1
    telemetry = opt.topos_telemetry_payload()
    assert telemetry == telemetry_contract["payload"]
    assert telemetry["topos.closure_pressure"] == pytest.approx(0.5)
    assert telemetry["topos.training_hints.clip_scale"] == pytest.approx(
        hints["clip_scale"]
    )
    assert telemetry["topos.optimizer_effect.rate_scale"] == pytest.approx(
        expected_scale
    )
    assert telemetry["topos.training_plan.raw_rate_scale"] == pytest.approx(
        diagnostics["training_plan"]["raw_rate_scale"]
    )
    assert telemetry[
        "topos.training_plan.effective_gradient_bias_scale"
    ] == pytest.approx(diagnostics["training_plan"]["effective_gradient_bias_scale"])
    assert telemetry[
        "topos.optimizer_effect.effective_gradient_bias_scale"
    ] == pytest.approx(diagnostics["effect"]["effective_gradient_bias_scale"])
    assert telemetry[
        "topos.optimizer_effect.effective_momentum_damping"
    ] == pytest.approx(diagnostics["effect"]["effective_momentum_damping"])
    assert telemetry["topos.optimizer_snapshot.sequence"] == 1.0
    assert telemetry["topos.runtime_profile.training_rate_scale"] == pytest.approx(
        expected_scale
    )
    assert telemetry["topos.runtime_profile.control_energy"] == pytest.approx(
        diagnostics["runtime_profile"]["control_energy"]
    )


def test_amegagrad_topos_telemetry_projects_one_authoritative_snapshot() -> None:
    _require_native()

    guard = st.hypergrad_topos(max_depth=8, max_volume=16)
    opt = st.optim.Amegagrad((1, 1), topos=guard, topos_control_gain=1.0)
    opt.tune(
        control=_UnitRateControl(),
        use_topos=True,
        observed_depth=4,
        visited_volume=1,
    )
    contract = opt.topos_telemetry_contract()

    assert contract["input_count"] == 1
    assert contract["payload"]["topos.optimizer_snapshot.sequence"] == 1.0
    assert contract["payload"]["topos.optimizer_effect.rate_scale"] == pytest.approx(
        contract["payload"]["topos.training_plan.rate_scale"]
    )
    assert contract["payload"]["topos.runtime_profile.control_energy"] == pytest.approx(
        opt.topos_diagnostics()["runtime_profile"]["control_energy"]
    )
    assert contract["payload"][
        "topos.optimizer_effect.effective_gradient_bias_scale"
    ] == pytest.approx(
        contract["payload"]["topos.training_plan.effective_gradient_bias_scale"]
    )


def test_amegagrad_rejects_incomplete_rust_optimizer_snapshot(monkeypatch) -> None:
    _require_native()

    opt = st.optim.Amegagrad((1, 1), topos_control_gain=1.0)
    monkeypatch.setattr(
        st,
        "topos_optimizer_snapshot",
        lambda *_args, **_kwargs: {
            "sequence": 1,
            "control": {"training_plan": {"rate_scale": 0.5}},
            "optimizer_application": {"rate_scale": 0.5},
        },
    )

    with pytest.raises(RuntimeError, match="hyper_learning_rate"):
        opt.tune(
            control=_UnitRateControl(),
            use_topos=True,
            topos_hints={"learning_rate_scale": 0.5, "clip_scale": 1.0},
        )


def test_amegagrad_custom_hints_cannot_reuse_a_stale_topos_signal() -> None:
    _require_native()

    guard = st.hypergrad_topos(max_depth=8, max_volume=16)
    opt = st.optim.Amegagrad(
        (1, 2),
        topos=guard,
        topos_control_gain=1.0,
    )
    old_signal = opt.topos_control_signal(observed_depth=1, visited_volume=1)
    assert old_signal["observed_depth"] == 1
    old_signal["observed_depth"] = 0
    assert opt.last_topos_signal["observed_depth"] == 1

    opt.tune(
        control=_UnitRateControl(),
        use_topos=True,
        topos_hints={"learning_rate_scale": 0.5, "clip_scale": 0.8},
        observed_depth=7,
        visited_volume=5,
    )
    first = opt.topos_diagnostics()["snapshot"]
    assert first["sequence"] == 1
    assert first["control"]["observed_depth"] == 7
    assert first["control"]["visited_volume"] == 5
    assert first["control"]["training_hints"]["learning_rate_scale"] == pytest.approx(
        0.5
    )
    first["control"]["observed_depth"] = 0
    assert opt.topos_diagnostics()["snapshot"]["control"]["observed_depth"] == 7

    opt.tune(control=_UnitRateControl(), use_topos=True, observed_depth=6)
    assert opt.topos_diagnostics()["snapshot"]["sequence"] == 2


def test_amegagrad_does_not_publish_a_failed_native_configuration(
    monkeypatch,
) -> None:
    _require_native()

    guard = st.hypergrad_topos(max_depth=8, max_volume=16)
    opt = st.optim.Amegagrad(
        (1, 2),
        hyper_learning_rate=0.04,
        real_learning_rate=0.02,
        topos=guard,
        topos_control_gain=1.0,
    )
    optimizer_globals = st.optim.Amegagrad.tune.__globals__
    original_configure = optimizer_globals["_configure_amegagrad_optimizer"]
    hyper_state_before = opt.hyper.optimizer_state_control()
    real_state_before = opt.real.optimizer_state_control()
    hyper_momentum_before = opt.hyper.optimizer_momentum()
    real_momentum_before = opt.real.optimizer_momentum()

    def _fail_configuration(*_args, **_kwargs) -> None:
        raise RuntimeError("injected native configuration failure")

    monkeypatch.setitem(
        optimizer_globals,
        "_configure_amegagrad_optimizer",
        _fail_configuration,
    )
    with pytest.raises(RuntimeError, match="injected native configuration failure"):
        opt.tune(control=_UnitRateControl(), use_topos=True, observed_depth=4)

    assert opt.hyper.learning_rate() == pytest.approx(0.04)
    assert opt.real.learning_rate() == pytest.approx(0.02)
    assert opt.hyper.optimizer_state_control() == hyper_state_before
    assert opt.real.optimizer_state_control() == real_state_before
    assert opt.hyper.optimizer_momentum() == hyper_momentum_before
    assert opt.real.optimizer_momentum() == real_momentum_before
    assert opt.topos_diagnostics()["snapshot"] is None

    monkeypatch.setitem(
        optimizer_globals,
        "_configure_amegagrad_optimizer",
        original_configure,
    )
    opt.tune(control=_UnitRateControl(), use_topos=True, observed_depth=4)
    assert opt.topos_diagnostics()["snapshot"]["sequence"] == 1


def test_gradient_tapes_reject_rewritten_topos_optimizer_rules() -> None:
    _require_native()

    snapshot = st.topos_optimizer_snapshot(
        sequence=1,
        hyper_learning_rate=0.04,
        real_learning_rate=0.02,
        gain=1.0,
        observed_depth=1,
        visited_volume=1,
    )
    application = dict(snapshot["optimizer_application"])
    tape = st.hypergrad((1, 1))

    application["gradient_clip_rule"] = "python_reconstruction"
    with pytest.raises(ValueError, match="gradient_clip_rule"):
        tape.configure_optimizer_state(application)

    application = dict(snapshot["optimizer_application"])
    application["momentum_rule"] = "python_reconstruction"
    with pytest.raises(ValueError, match="momentum_rule"):
        tape.configure_optimizer_state(application)


def test_amegagrad_keeps_default_tune_non_topos_by_default() -> None:
    _require_native()

    guard = st.hypergrad_topos(max_depth=8, max_volume=16)
    opt = st.optim.Amegagrad(
        (1, 2),
        curvature=-0.9,
        hyper_learning_rate=0.04,
        real_learning_rate=0.02,
        topos=guard,
    )

    opt.tune(control=_UnitRateControl())

    assert opt.hyper.learning_rate() == pytest.approx(0.04)
    assert opt.real.learning_rate() == pytest.approx(0.02)
    assert opt.topos_diagnostics()["effect"] is None
    assert opt.topos_training_hints()["clip_scale"] <= 1.0


def test_amegagrad_session_writes_topos_training_trace(tmp_path) -> None:
    _require_native()

    guard = st.hypergrad_topos(
        curvature=-0.9,
        tolerance=1e-4,
        saturation=2.0,
        max_depth=8,
        max_volume=16,
    )
    session = st.amegagrad_session(
        (1, 4),
        curvature=-0.9,
        hyper_learning_rate=0.04,
        real_learning_rate=0.02,
        topos=guard,
        topos_control_gain=1.0,
        topos_observed_depth=4,
        topos_visited_volume=8,
        telemetry=False,
        z_lam_frac=0.0,
    )
    wave = st.Tensor((1, 4), data=[0.4, -0.6, 0.2, 0.1])

    session.step_wave(wave)
    metrics = session.last_step_metrics
    trace_path = tmp_path / "amegagrad-topos-trace.jsonl"
    session.write_trainer_trace_event(trace_path, mode="w")

    assert metrics["realgrad.l2"] > 0.0
    assert metrics["realgrad.mean_abs"] > 0.0
    assert metrics["topos.closure_pressure"] == pytest.approx(0.5)
    assert metrics["topos.optimizer_effect.rate_scale"] < 1.0
    assert metrics["topos.runtime_profile.training_rate_scale"] == pytest.approx(
        metrics["topos.optimizer_effect.rate_scale"]
    )
    assert metrics["topos.runtime_profile.control_energy"] > 0.0
    assert metrics["hypergrad.learning_rate"] == pytest.approx(
        metrics["topos.optimizer_effect.hyper_learning_rate"]
    )
    assert metrics["realgrad.learning_rate"] == pytest.approx(
        metrics["topos.optimizer_effect.real_learning_rate"]
    )

    events = st.load_trainer_trace_events(trace_path)
    summary = st.summarize_trainer_trace_events(trace_path)

    assert events[0]["step"] == 1
    assert events[0]["metrics"]["extra"]["topos.optimizer_effect.rate_scale"] == pytest.approx(
        metrics["topos.optimizer_effect.rate_scale"]
    )
    assert summary["count"] == 1
    assert summary["metrics"]["realgrad.l2"]["samples"] == 1
    assert summary["metrics"]["realgrad.l2"]["nonzero"] == 1
    assert summary["topos_context"]["observed_count"] == 1
    assert summary["topos_context"]["optimizer_rate_scale"]["last"] == pytest.approx(
        metrics["topos.optimizer_effect.rate_scale"]
    )
    assert summary["topos_context"]["runtime_profile_training_rate_scale"][
        "last"
    ] == pytest.approx(metrics["topos.optimizer_effect.rate_scale"])


def test_amegagrad_topos_training_sweep_compares_runs(tmp_path) -> None:
    _require_native()

    result = st.trace_amegagrad_topos_training_sweep(tmp_path / "sweep", steps=2)
    comparison = result["comparison"]

    assert result["kind"] == "spiraltorch.amegagrad_topos_training_trace_sweep"
    assert set(result["trace_paths"]) == {"guard_only", "topos_tuned"}
    assert comparison["kind"] == "spiraltorch.amegagrad_topos_training_trace_comparison"
    assert comparison["count"] == 2
    assert comparison["winners"]["lowest_optimizer_rate_scale"] == "topos_tuned"

    tuned_summary = result["summaries"]["topos_tuned"]
    tuned_context = tuned_summary["topos_context"]
    assert tuned_summary["count"] == 2
    assert tuned_summary["metrics"]["realgrad.l2"]["nonzero"] == 2
    assert tuned_context["observed_count"] == 2
    assert tuned_context["optimizer_rate_scale"]["last"] < 1.0
    assert tuned_context["training_plan_raw_rate_scale"]["last"] < 1.0
    assert tuned_context["training_plan_effective_gradient_bias_scale"]["mean"] > 0.0
    assert tuned_context["runtime_profile_training_rate_scale"]["last"] < 1.0
    assert tuned_context["runtime_profile_control_energy"]["mean"] > 0.0

    guard_row = next(row for row in comparison["runs"] if row["label"] == "guard_only")
    tuned_row = next(row for row in comparison["runs"] if row["label"] == "topos_tuned")
    assert guard_row["realgrad_l2_mean"] > 0.0
    assert tuned_row["realgrad_l2_mean"] > 0.0
    assert guard_row["topos_optimizer_rate_scale_mean"] is None
    assert guard_row["topos_runtime_training_rate_scale_mean"] == pytest.approx(1.0)
    assert tuned_row["topos_optimizer_rate_scale_mean"] == pytest.approx(
        tuned_context["optimizer_rate_scale"]["mean"]
    )
    assert tuned_row["topos_runtime_training_rate_scale_mean"] == pytest.approx(
        tuned_context["runtime_profile_training_rate_scale"]["mean"]
    )
    assert tuned_row["topos_training_plan_raw_rate_scale_mean"] == pytest.approx(
        tuned_context["training_plan_raw_rate_scale"]["mean"]
    )
    assert tuned_row["topos_training_plan_rate_scale_mean"] == pytest.approx(
        tuned_context["training_plan_rate_scale"]["mean"]
    )
    assert tuned_row[
        "topos_training_plan_effective_gradient_bias_scale_mean"
    ] == pytest.approx(
        tuned_context["training_plan_effective_gradient_bias_scale"]["mean"]
    )
    assert tuned_row["topos_optimizer_raw_rate_scale_mean"] is None
    assert tuned_row[
        "topos_optimizer_effective_gradient_bias_scale_mean"
    ] == pytest.approx(
        tuned_context["optimizer_effective_gradient_bias_scale"]["mean"]
    )
    assert tuned_row[
        "topos_optimizer_effective_momentum_damping_mean"
    ] == pytest.approx(tuned_context["optimizer_effective_momentum_damping"]["mean"])

    reloaded = st.compare_amegagrad_topos_training_traces(result["trace_paths"])
    assert reloaded["winners"]["lowest_optimizer_rate_scale"] == "topos_tuned"
    assert reloaded["winners"]["lowest_training_plan_rate_scale"] == "topos_tuned"
    assert reloaded["winners"]["highest_planned_gradient_bias"] == "topos_tuned"
    assert reloaded["winners"]["lowest_optimizer_raw_rate_scale"] is None
    assert reloaded["winners"]["highest_effective_gradient_bias"] == "topos_tuned"
    assert reloaded["winners"]["lowest_runtime_training_rate_scale"] == "topos_tuned"
