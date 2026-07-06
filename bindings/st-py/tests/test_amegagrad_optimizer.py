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
    expected_scale = hints["learning_rate_scale"] * hints["clip_scale"]

    control = _UnitRateControl()
    returned = opt.tune(control=control, use_topos=True)

    assert returned is control
    assert opt.hyper.learning_rate() == pytest.approx(0.04 * expected_scale)
    assert opt.real.learning_rate() == pytest.approx(0.02 * expected_scale)
    diagnostics = opt.topos_diagnostics()
    assert diagnostics["training_hints"]["clip_scale"] == pytest.approx(hints["clip_scale"])
    assert diagnostics["effect"]["rate_scale"] == pytest.approx(expected_scale)

    telemetry = opt.topos_telemetry_payload()
    assert telemetry["topos.closure_pressure"] == pytest.approx(0.5)
    assert telemetry["topos.training_hints.clip_scale"] == pytest.approx(hints["clip_scale"])
    assert telemetry["topos.optimizer_effect.rate_scale"] == pytest.approx(expected_scale)


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

    assert metrics["topos.closure_pressure"] == pytest.approx(0.5)
    assert metrics["topos.optimizer_effect.rate_scale"] < 1.0
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
    assert summary["topos_context"]["observed_count"] == 1
    assert summary["topos_context"]["optimizer_rate_scale"]["last"] == pytest.approx(
        metrics["topos.optimizer_effect.rate_scale"]
    )


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
    assert tuned_context["observed_count"] == 2
    assert tuned_context["optimizer_rate_scale"]["last"] < 1.0

    guard_row = next(row for row in comparison["runs"] if row["label"] == "guard_only")
    tuned_row = next(row for row in comparison["runs"] if row["label"] == "topos_tuned")
    assert guard_row["topos_optimizer_rate_scale_mean"] is None
    assert tuned_row["topos_optimizer_rate_scale_mean"] == pytest.approx(
        tuned_context["optimizer_rate_scale"]["mean"]
    )

    reloaded = st.compare_amegagrad_topos_training_traces(result["trace_paths"])
    assert reloaded["winners"]["lowest_optimizer_rate_scale"] == "topos_tuned"
