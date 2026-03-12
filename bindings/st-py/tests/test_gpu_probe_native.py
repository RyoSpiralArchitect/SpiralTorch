from __future__ import annotations

import importlib
import sys
import types

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def _load_native() -> types.ModuleType | None:
    _ensure_torch_stub()
    try:
        module = importlib.import_module("spiraltorch")
    except Exception:
        return None

    for native_name in (
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    ):
        try:
            importlib.import_module(native_name)
        except Exception:
            continue
        return module
    return None


def test_probe_gpu_path_exposes_runtime_route_visibility() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

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


def test_mps_probe_exposes_placeholder_surface() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

    assert hasattr(st, "mps_probe")
    assert hasattr(st, "planner")
    assert hasattr(st.planner, "mps_probe")

    report = st.mps_probe()
    assert report["backend"] == "mps"
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
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

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


def test_plan_accepts_mps_backend() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

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
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

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
