from __future__ import annotations

from types import SimpleNamespace

import pytest

from ._native_loader import require_native, require_wgpu_runtime


def test_probe_gpu_path_exposes_runtime_route_visibility() -> None:
    st = require_native()

    assert hasattr(st, "probe_gpu_path")
    assert hasattr(st, "planner")
    assert hasattr(st.planner, "probe_gpu_path")

    report = st.probe_gpu_path("bottomk", backend="cuda", rows=2, cols=5, k=2)
    assert report["backend"] == "cuda"
    assert report["kind"] == "bottomk"
    assert int(report["rows"]) == 2
    assert int(report["cols"]) == 5
    assert int(report["k"]) == 2
    assert "strict_success" in report
    assert "non_strict_success" in report
    assert "gpu_path_available" in report
    assert "used_fallback" in report


def test_describe_device_explicit_wgpu_backend() -> None:
    st = require_native()

    report = st.describe_device("wgpu", workgroup=300, cols=4096)
    assert report["backend"] == "wgpu"
    assert "lane_width" in report
    assert "max_workgroup" in report
    assert "subgroup" in report
    assert "shared_mem_per_workgroup" in report
    assert "aligned_workgroup" in report
    assert "occupancy_score" in report
    assert "preferred_tile" in report
    assert "preferred_compaction_tile" in report


def test_describe_device_auto_backend_uses_effective_wgpu_label() -> None:
    st = require_native()

    report = st.describe_device("auto", workgroup=300, cols=4096)
    assert report["backend"] == "wgpu"
    assert "lane_width" in report
    assert "max_workgroup" in report
    assert "subgroup" in report
    assert "shared_mem_per_workgroup" in report


def test_plan_explicit_wgpu_backend() -> None:
    st = require_native()

    plan = st.plan("topk", 16, 128, 8, backend="wgpu")
    assert plan.kind == "topk"
    assert plan.requested_backend == "wgpu"
    assert plan.effective_backend == "wgpu"
    assert int(plan.rows) == 16
    assert int(plan.cols) == 128
    assert int(plan.k) == 8
    assert int(plan.workgroup) >= 1
    assert int(plan.lanes) >= 1


def test_init_backend_and_session_explicit_wgpu_backend_when_runtime_is_enabled() -> None:
    st = require_native()
    require_wgpu_runtime(st)

    assert st.init_backend("wgpu") is True

    session = st.SpiralSession(backend="wgpu")
    assert session.backend == "wgpu"
    assert session.requested_backend == "wgpu"
    assert session.effective_backend == "wgpu"
    assert session.device == "wgpu"
    assert session.device_preflight["backend"] == "wgpu"
    assert "lane_width" in session.device_preflight


def test_session_auto_prefers_wgpu_backend_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[str] = []

    def _patched_init_backend(backend: str) -> bool:
        calls.append(str(backend))
        return True

    def _patched_describe_device(backend: str = "wgpu", **_kwargs: object):
        return {"backend": str(backend), "lane_width": 32}

    monkeypatch.setattr(st, "init_backend", _patched_init_backend, raising=False)
    monkeypatch.setattr(st, "describe_device", _patched_describe_device, raising=False)

    session = st.SpiralSession()

    assert session.backend == "auto"
    assert session.requested_backend == "auto"
    assert session.effective_backend == "wgpu"
    assert session.device == "wgpu"
    assert session.device_preflight["backend"] == "wgpu"
    assert calls == ["wgpu"]


def test_session_auto_falls_back_to_cpu_when_wgpu_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[str] = []

    def _patched_init_backend(backend: str) -> bool:
        raw = str(backend)
        calls.append(raw)
        if raw == "wgpu":
            raise RuntimeError("wgpu unavailable")
        return True

    def _patched_describe_device(backend: str = "wgpu", **_kwargs: object):
        return {"backend": str(backend), "lane_width": 1}

    monkeypatch.setattr(st, "init_backend", _patched_init_backend, raising=False)
    monkeypatch.setattr(st, "describe_device", _patched_describe_device, raising=False)

    session = st.SpiralSession()

    assert session.backend == "auto"
    assert session.requested_backend == "auto"
    assert session.effective_backend == "cpu"
    assert session.device == "cpu"
    assert session.device_preflight["backend"] == "cpu"
    assert calls == ["wgpu", "cpu"]


def test_trace_wgpu_first_runtime_captures_session_plan_and_tensor_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[str] = []

    def _patched_init_backend(backend: str) -> bool:
        calls.append(str(backend))
        return True

    def _patched_describe_device(backend: str = "wgpu", **_kwargs: object):
        return {"backend": str(backend), "lane_width": 32}

    def _patched_plan_topk(*, rows: int, cols: int, k: int, backend: str):
        return SimpleNamespace(
            kind="topk",
            requested_backend=backend,
            effective_backend=backend,
            rows=rows,
            cols=cols,
            k=k,
            workgroup=128,
            lanes=32,
        )

    class _FakeTensor:
        def __init__(self, rows: int, cols: int, data: object) -> None:
            self._shape = (rows, cols)

        def row_softmax(self, *, backend: str):
            assert backend == "wgpu"
            return self

        def shape(self) -> tuple[int, int]:
            return self._shape

        def tolist(self) -> list[list[float]]:
            rows, cols = self._shape
            value = 1.0 / float(cols)
            return [[value] * cols for _ in range(rows)]

    events: list[dict[str, object]] = []

    def _subscribe(event_type: str, callback):
        assert event_type == "TensorOpMeta"
        event = {"type": "TensorOpMeta", "payload": {"op_name": "row_softmax"}}
        events.append(event)
        callback(event)
        return 7

    unsubscribed: list[tuple[str, int]] = []

    def _unsubscribe(event_type: str, subscription_id: int) -> bool:
        unsubscribed.append((event_type, subscription_id))
        return True

    monkeypatch.setattr(st, "init_backend", _patched_init_backend, raising=False)
    monkeypatch.setattr(st, "describe_device", _patched_describe_device, raising=False)
    monkeypatch.setattr(st, "plan_topk", _patched_plan_topk, raising=False)
    monkeypatch.setattr(st, "Tensor", _FakeTensor, raising=False)
    monkeypatch.setattr(
        st,
        "plugin",
        SimpleNamespace(subscribe=_subscribe, unsubscribe=_unsubscribe),
        raising=False,
    )

    report = st.trace_wgpu_first_runtime(rows=2, cols=4, k=8)

    assert report["requested_backend"] == "auto"
    assert report["effective_backend"] == "wgpu"
    assert report["device_preflight"]["backend"] == "wgpu"
    assert report["planner"]["k"] == 4
    assert report["planner"]["effective_backend"] == "wgpu"
    assert report["tensor_operation"]["requested_backend"] == "wgpu"
    assert report["tensor_operation"]["ok"] is True
    assert report["tensor_operation"]["row_sums"] == pytest.approx([1.0, 1.0])
    assert report["tensor_meta_events"] == events
    assert unsubscribed == [("TensorOpMeta", 7)]
    assert calls == ["wgpu"]
