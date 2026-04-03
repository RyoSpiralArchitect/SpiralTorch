from __future__ import annotations

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
