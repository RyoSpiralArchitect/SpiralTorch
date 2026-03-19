from __future__ import annotations

import types

import pytest

from ._native_loader import require_native


def test_mps_probe_exposes_placeholder_surface() -> None:
    st = require_native()

    assert hasattr(st, "mps_probe")
    assert hasattr(st, "planner")
    assert hasattr(st.planner, "mps_probe")

    report = st.mps_probe()
    planner_report = st.planner.mps_probe()
    assert report["backend"] == "mps"
    assert planner_report["backend"] == "mps"
    assert planner_report["planner_surrogate_backend"] == report["planner_surrogate_backend"]
    assert "feature_enabled" in report
    assert "platform_supported" in report
    assert report["status"] in {
        "build-feature-disabled",
        "unsupported-host",
        "placeholder",
    }
    assert isinstance(report["host_class"], str)
    assert report["backend_wired"] is False
    assert report["placeholder"] is True
    assert report["available"] is False
    assert report["initialized"] is False
    assert report["planner_surrogate_backend"] in {"wgpu", "cpu"}
    assert report["planner_route"] in {"metal-via-wgpu", "cpu-fallback"}
    assert report["recommended_backend"] == report["planner_surrogate_backend"]
    planner_caps = report["planner_caps"]
    assert planner_caps["backend"] == report["planner_surrogate_backend"]
    assert "lane_width" in planner_caps
    assert "max_workgroup" in planner_caps
    assert "subgroup" in planner_caps
    assert "shared_mem_per_workgroup" in planner_caps
    assert isinstance(report["recommendation"], str)
    assert report["devices"] == []
    assert isinstance(report["host_os"], str)
    assert isinstance(report["host_arch"], str)
    assert isinstance(report["error"], str)


def test_describe_device_accepts_mps_backend() -> None:
    st = require_native()

    report = st.describe_device("mps", workgroup=300, cols=4096)
    assert report["backend"] == "mps"
    assert report["status"] in {
        "build-feature-disabled",
        "unsupported-host",
        "placeholder",
    }
    assert report["planner_surrogate_backend"] in {"wgpu", "cpu"}
    assert report["planner_route"] in {"metal-via-wgpu", "cpu-fallback"}
    assert report["recommended_backend"] == report["planner_surrogate_backend"]
    assert report["backend_wired"] is False
    assert report["placeholder"] is True
    assert report["available"] is False
    assert report["initialized"] is False
    assert "lane_width" in report
    assert "max_workgroup" in report
    assert "subgroup" in report
    assert "shared_mem_per_workgroup" in report
    assert "aligned_workgroup" in report
    assert "occupancy_score" in report
    assert "preferred_tile" in report
    assert "preferred_compaction_tile" in report
    assert isinstance(report["recommendation"], str)
    assert isinstance(report["error"], str)


def test_plan_accepts_mps_backend_and_exposes_surrogate_route() -> None:
    st = require_native()

    plan = st.plan("topk", 16, 128, 8, backend="mps")
    assert plan.kind == "topk"
    assert plan.requested_backend == "mps"
    assert plan.effective_backend in {"wgpu", "cpu"}
    assert int(plan.rows) == 16
    assert int(plan.cols) == 128
    assert int(plan.k) == 8
    assert int(plan.workgroup) >= 1
    assert int(plan.lanes) >= 1


def test_init_backend_and_session_expose_mps_preflight() -> None:
    st = require_native()

    assert st.init_backend("mps") is False

    session = st.SpiralSession(backend="mps")
    assert session.backend == "mps"
    assert session.requested_backend == "mps"
    assert session.effective_backend in {"wgpu", "cpu"}
    assert session.device in {"metal-via-wgpu", "cpu-fallback"}

    report = session.device_preflight
    assert report["backend"] == "mps"
    assert report["planner_surrogate_backend"] == session.effective_backend
    assert report["planner_route"] == session.device
    assert report["backend_wired"] is False
    assert report["placeholder"] is True
    assert report["available"] is False
    assert "lane_width" in report


def test_public_mps_probe_requires_matching_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    st = require_native()

    monkeypatch.setattr(st, "_native_mps_probe", None, raising=False)
    with pytest.raises(RuntimeError, match="matching SpiralTorch native extension"):
        st.mps_probe()


def test_public_mps_describe_device_requires_matching_native_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()

    def _raise_unknown_backend(*_args: object, **_kwargs: object) -> types.SimpleNamespace:
        raise ValueError("unknown backend label 'mps'")

    monkeypatch.setattr(st, "_native_describe_device", _raise_unknown_backend, raising=False)
    with pytest.raises(RuntimeError, match="matching SpiralTorch native extension"):
        st.describe_device("mps")


def test_plan_retains_legacy_mps_routing_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    st = require_native()
    native_plan = st._native_plan

    def _patched_plan(kind: str, rows: int, cols: int, k: int, *, backend: str | None = None, **kwargs):
        raw = "" if backend is None else str(backend).strip().lower()
        if raw == "mps":
            raise ValueError("unknown backend label 'mps'")
        return native_plan(kind, rows, cols, k, backend=backend, **kwargs)

    monkeypatch.setattr(st, "_native_plan", _patched_plan, raising=False)
    monkeypatch.setattr(st, "_native_mps_probe", None, raising=False)

    plan = st.plan("topk", 16, 128, 8, backend="mps")
    assert plan.requested_backend == "mps"
    assert plan.effective_backend in {"wgpu", "cpu"}


def test_session_retains_legacy_mps_runtime_entry_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    native_describe_device = st._native_describe_device

    def _patched_describe_device(backend: str = "wgpu", **kwargs):
        raw = "" if backend is None else str(backend).strip().lower()
        if raw == "mps":
            raise ValueError("unknown backend label 'mps'")
        return native_describe_device(backend, **kwargs)

    monkeypatch.setattr(st, "_native_describe_device", _patched_describe_device, raising=False)
    monkeypatch.setattr(st, "_native_mps_probe", None, raising=False)

    session = st.SpiralSession(backend="mps")
    assert session.requested_backend == "mps"
    assert session.effective_backend in {"wgpu", "cpu"}
    assert session.device in {"metal-via-wgpu", "cpu-fallback"}
