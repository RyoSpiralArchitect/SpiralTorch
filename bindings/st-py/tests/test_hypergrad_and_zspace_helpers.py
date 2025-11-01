from __future__ import annotations

import spiraltorch as st

from spiral.hypergrad import hypergrad_summary_dict, suggest_hypergrad_operator


def test_hypergrad_helper_accepts_tuple_shape() -> None:
    tape = st.hypergrad((1, 4))
    assert tape.shape() == (1, 4)
    assert tape.curvature() == -1.0


def test_hypergrad_helper_accepts_tensor_shape() -> None:
    tensor = st.Tensor((2, 3))
    tape = st.hypergrad(tensor, learning_rate=0.01)
    assert tape.shape() == tensor.shape()
    assert tape.learning_rate() == 0.01


def test_hypergrad_scale_gradient_tracks_summary() -> None:
    tape = st.hypergrad(1, 3, curvature=-0.9, learning_rate=0.05)
    tensor = st.Tensor((1, 3), data=[0.4, -0.6, 0.2])
    tape.accumulate_wave(tensor)
    before = tape.gradient()
    tape.scale_gradient(-0.5)
    after = tape.gradient()
    for prev, current in zip(before, after):
        assert abs(current - (-0.5 * prev)) < 1e-6
    summary = tape.summary()
    expected_l2 = sum(value * value for value in after) ** 0.5
    assert abs(summary.l2() - expected_l2) < 1e-6
    assert summary.count() == len(after)


def test_hypergrad_rescale_rms_targets_value() -> None:
    tape = st.hypergrad(1, 4, curvature=-0.88, learning_rate=0.04)
    tensor = st.Tensor((1, 4), data=[0.35, -0.45, 0.25, -0.15])
    tape.accumulate_wave(tensor)
    base = tape.summary()
    target = base.rms() * 0.3
    factor = tape.rescale_rms(target)
    assert factor > 0.0
    summary = tape.summary()
    assert abs(summary.rms() - target) < 5e-3


def test_hypergrad_helper_accepts_mapping_topos() -> None:
    tape = st.hypergrad(
        1,
        3,
        curvature=-0.9,
        learning_rate=0.02,
        topos={
            "curvature": -0.9,
            "tolerance": 1e-3,
            "saturation": 0.8,
            "depth": 4,
            "volume": 16,
        },
    )
    guard = tape.topos()
    assert guard.curvature() == -0.9
    assert guard.max_depth() == 4
    assert guard.max_volume() == 16


def test_hypergrad_telemetry_reports_metrics() -> None:
    tape = st.hypergrad(1, 3, curvature=-0.95, learning_rate=0.04)
    tensor = st.Tensor((1, 3), data=[0.5, -0.25, 0.75])
    tape.accumulate_wave(tensor)
    telemetry = tape.telemetry()
    assert telemetry.shape() == (1, 3)
    assert telemetry.volume() == 3
    assert tape.non_finite_count() == 0
    assert not tape.has_non_finite()
    assert abs(tape.non_finite_ratio()) < 1e-6
    assert telemetry.curvature() == -0.95
    assert telemetry.learning_rate() == 0.04
    summary = telemetry.summary()
    assert summary.count() == 3
    assert telemetry.finite_count() == summary.count()
    assert telemetry.non_finite_count() == 0
    assert telemetry.non_finite_ratio() == 0.0
    assert telemetry.tolerance() > 0.0
    assert telemetry.saturation() > 0.0
    assert telemetry.max_volume() >= telemetry.volume()
    assert summary.std() > 0.0
    assert summary.variance() > 0.0
    assert summary.kurtosis() >= 0.0
    assert summary.activation() > 0.0
    assert summary.support_width() > 0.0


def test_hypergrad_desire_feedback_interfaces() -> None:
    tape = st.hypergrad(1, 2, curvature=-0.9, learning_rate=0.05)
    tensor = st.Tensor((1, 2), data=[0.7, -0.3])
    tape.accumulate_wave(tensor)
    real = st.GradientSummary.from_values([0.35, -0.15])
    interpretation = tape.desire_interpretation(real)
    assert interpretation.hyper_pressure() > interpretation.real_pressure()
    assert interpretation.hyper_std() > 0.0
    assert interpretation.real_std() >= 0.0
    assert interpretation.sharpness() >= 0.0
    assert interpretation.activation() > 0.0
    assert interpretation.sign_alignment() >= 0.0
    assert interpretation.sign_entropy() >= 0.0
    control = tape.desire_control(real)
    damped = tape.desire_control(real, gain=0.5)
    assert control.penalty_gain() >= damped.penalty_gain()
    assert control.hyper_rate_scale() >= damped.hyper_rate_scale()
    assert "lr" in " ".join(control.events())


def test_hypergrad_summary_dict_reports_moments() -> None:
    tape = st.hypergrad(1, 2, curvature=-0.85, learning_rate=0.05)
    tensor = st.Tensor((1, 2), data=[0.2, -0.6])
    tape.accumulate_wave(tensor)
    payload = hypergrad_summary_dict(tape)
    summary = payload["summary"]
    assert summary["std"] > 0.0
    assert "skewness" in summary
    assert "kurtosis" in summary
    assert summary["sum_cubes"] != 0.0
    assert "activation" in summary
    assert "sign_entropy" in summary
    operator = suggest_hypergrad_operator(payload)
    assert "std" in operator
    assert "skewness" in operator
    assert "kurtosis" in operator
    assert "activation" in operator


def test_hypergrad_topos_factory_returns_guard() -> None:
    guard = st.hypergrad_topos(
        curvature=-0.8,
        tolerance=5e-4,
        saturation=0.7,
        max_depth=8,
        max_volume=32,
    )
    assert guard.curvature() == -0.8
    assert guard.tolerance() == 5e-4
    assert guard.max_depth() == 8
    assert guard.max_volume() == 32


def test_hypergrad_notation_square_brackets() -> None:
    tape = st.hg[2, 3](learning_rate=0.03)
    assert tape.shape() == (2, 3)
    assert tape.learning_rate() == 0.03


def test_hypergrad_notation_slice_bindings() -> None:
    tape = st.hg[1:4](curvature=-0.75)
    assert tape.shape() == (1, 4)
    assert tape.curvature() == -0.75


def test_hypergrad_notation_topos_alias() -> None:
    guard = st.hg.topos(curvature=-0.82, tolerance=2e-3, saturation=0.65, max_depth=6, max_volume=24)
    assert guard.curvature() == -0.82
    assert guard.max_depth() == 6


def test_hypergrad_partial_with_inline_topos() -> None:
    weights = st.Tensor((1, 4))
    tape = st.hg[weights].with_topos(curvature=-0.88, tolerance=1.5e-3, saturation=0.7, max_depth=5, max_volume=20)
    assert tape.shape() == weights.shape()
    guard = tape.topos()
    assert guard.curvature() == -0.88
    assert guard.max_volume() == 20


def test_hypergrad_partial_accepts_existing_topos() -> None:
    guard = st.hg.topos(curvature=-0.8, max_depth=4, max_volume=12)
    tape = st.hg[2, 2].with_topos(topos=guard)
    assert tape.topos().max_depth() == 4


def test_z_metrics_aliases_normalise_inputs() -> None:
    metrics = st.z_metrics(
        velocity=0.5,
        mem=0.25,
        stab=0.9,
        drift=0.1,
        grad=[1, -2, 3],
    )
    assert metrics.speed == 0.5
    assert metrics.memory == 0.25
    assert metrics.stability == 0.9
    assert metrics.drs == 0.1
    assert metrics.gradient == [1.0, -2.0, 3.0]


def test_encode_zspace_returns_tensor() -> None:
    tensor = st.encode_zspace("hypergrad keeps z-space grounded", temperature=0.35)
    assert isinstance(tensor, st.Tensor)
    rows, cols = tensor.shape()
    assert rows == 1
    assert cols > 0


def test_z_notation_bracket_temperature() -> None:
    tensor = st.z["hg keeps us centred", 0.42]
    assert isinstance(tensor, st.Tensor)
    assert tensor.shape()[0] == 1


def test_z_notation_metrics_helper() -> None:
    metrics = st.z.metrics(velocity=0.4, drift=0.12)
    assert metrics.speed == 0.4
    assert metrics.drs == 0.12


def test_z_partial_accepts_keyword_metrics() -> None:
    partial = st.z.partial(speed=0.7, mem=0.4, frac=0.3, origin="telemetry", weight=2.5)
    assert isinstance(partial, st.ZSpacePartialBundle)
    resolved = partial.resolved()
    assert resolved["speed"] == 0.7
    assert resolved["memory"] == 0.4
    assert resolved["frac"] == 0.3
    assert partial.weight == 2.5
    assert partial.origin == "telemetry"


def test_z_partial_merges_metrics_and_telemetry() -> None:
    base = st.z.partial(speed=0.2, telemetry={"psi": {"mean": 0.5}})
    combined = st.z.partial(base, stability=0.9, telemetry={"z": {"bias": 0.1}})
    resolved = combined.resolved()
    assert resolved["speed"] == 0.2
    assert resolved["stability"] == 0.9
    payload = combined.telemetry_payload()
    assert payload["psi.mean"] == 0.5
    assert payload["z.bias"] == 0.1


def test_z_partial_accepts_gradient_sequences() -> None:
    partial = st.z.partial(gradient=[1, -2, 3], speed=0.6)
    resolved = partial.resolved()
    assert resolved["gradient"] == [1.0, -2.0, 3.0]
    assert resolved["speed"] == 0.6


def test_z_bundle_weighted_mean() -> None:
    first = st.z.partial(speed=0.6, memory=0.2)
    second = st.z.partial(speed=0.2, memory=0.4, weight=0.5)
    bundle = st.z.bundle(first, second)
    assert abs(bundle["speed"] - ((0.6 + 0.1) / 1.5)) < 1e-6
    assert abs(bundle["memory"] - ((0.2 + 0.2) / 1.5)) < 1e-6
