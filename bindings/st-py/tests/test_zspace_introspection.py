"""Smoke tests for the newly exposed Z-space introspection helpers."""

import spiraltorch as st


def test_zspace_describe_returns_none_when_feedback_unset() -> None:
    assert st.z.describe() is None
    assert st.z.feedback() is None
    assert st.z.snapshot() is None
    assert st.z.signal() is None


def test_telemetry_current_is_none_without_feedback() -> None:
    assert st.telemetry.current() is None


def test_nn_softlogic_signal_forwarding() -> None:
    assert st.nn.softlogic_signal() is None


def test_curvature_scheduler_configuration_roundtrip() -> None:
    scheduler = st.nn.CurvatureScheduler(min_curvature=-2.0, max_curvature=-0.2, target_pressure=0.05)
    assert scheduler.min_curvature <= scheduler.current <= scheduler.max_curvature
    scheduler.set_step(0.1)
    scheduler.set_tolerance(0.02)
    scheduler.set_smoothing(0.3)
    scheduler.set_target_pressure(0.1)
    scheduler.set_bounds(-1.5, -0.3)
    scheduler.sync(-0.6)
    decision = scheduler.observe_pressure(0.2)
    assert isinstance(decision.raw_pressure, float)
    assert isinstance(decision.curvature, float)
