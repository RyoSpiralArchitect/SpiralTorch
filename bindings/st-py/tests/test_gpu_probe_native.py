from __future__ import annotations

import json
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
    assert report["kind"] == "spiraltorch.runtime_device_probe"
    assert report["contract_version"] == "spiraltorch.runtime_device_probe.v1"
    assert report["semantic_owner"] == "st-core::backend::runtime_probe"
    assert report["semantic_backend"] == "rust"
    assert report["execution_client"] == "python"
    assert report["committed"] is True
    assert len(report["request_sha256"]) == 64
    assert len(report["output_sha256"]) == 64
    assert report["contract"]["output_sha256"] == report["output_sha256"]
    assert "validate_runtime_device_probe_contract" in st.__all__
    assert st.validate_runtime_device_probe_contract(report) == report["contract"]
    assert (
        st.validate_runtime_device_probe_contract(
            report,
            request=report["request"],
        )
        == report["contract"]
    )
    assert report["backend"] == "wgpu"
    assert report["requested_backend"] == "wgpu"
    assert report["effective_backend"] == "wgpu"
    assert report["runtime_ready"] == report["effective_backend_runtime_ready"]
    assert report["runtime_status"] == report["effective_backend_runtime_status"]
    assert report["runtime_status"] in {
        "ready",
        "initialization_failed",
        "feature_disabled",
    }
    assert report["requested_backend_runtime_status"] == report["runtime_status"]
    assert report["requested_backend_runtime_ready"] == report["runtime_ready"]
    assert report["route_evidence"]["runtime_ready"] == report["runtime_ready"]
    assert (
        report["route_evidence"]["effective_backend_runtime_ready"]
        == report["effective_backend_runtime_ready"]
    )
    assert report["request"]["caps"]["backend"] == "wgpu"
    assert isinstance(report["effective_backend_integration_compiled"], bool)
    assert isinstance(report["effective_backend_runtime_initialized"], bool)
    if report["runtime_status"] == "initialization_failed":
        assert report["runtime_ready"] is False
        assert isinstance(report["effective_backend_runtime_error"], str)

    tampered = json.loads(json.dumps(report["contract"]))
    tampered["aligned_workgroup"] = 1
    with pytest.raises(ValueError, match="runtime-device probe validation failed"):
        st.validate_runtime_device_probe_contract(tampered)
    assert isinstance(report["runtime_recommendation"], str)
    assert isinstance(report["effective_backend_runtime_recommendation"], str)
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
    assert report["requested_backend"] == "wgpu"
    assert report["effective_backend"] == "wgpu"
    assert report["runtime_ready"] == report["effective_backend_runtime_ready"]
    assert report["effective_backend_runtime_status"] in {
        "ready",
        "initialization_failed",
        "feature_disabled",
    }
    assert "lane_width" in report
    assert "max_workgroup" in report
    assert "subgroup" in report
    assert "shared_mem_per_workgroup" in report


def test_describe_runtime_devices_collects_backend_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[tuple[str, dict[str, object]]] = []

    def _patched_describe_device(backend: str = "wgpu", **kwargs: object):
        calls.append((backend, dict(kwargs)))
        if backend == "mps":
            raise RuntimeError("mps placeholder")
        return {
            "backend": backend,
            "requested_backend": backend,
            "effective_backend": backend,
            "runtime_ready": backend == "wgpu",
            "runtime_status": "kernel_wired" if backend == "wgpu" else "cpu",
        }

    monkeypatch.setattr(st, "describe_device", _patched_describe_device, raising=False)

    summary = st.describe_runtime_devices(
        ["wgpu", "cpu", "mps"],
        required_ready_backends=["wgpu", "mps"],
        workgroup=128,
    )

    assert "describe_runtime_devices" in st.__all__
    assert st.planner.describe_runtime_devices is st.describe_runtime_devices
    assert summary["backends"] == ["wgpu", "cpu", "mps"]
    assert summary["kind"] == "spiraltorch.runtime_device_route"
    assert summary["contract_version"] == "spiraltorch.runtime_device_route.v4"
    assert summary["semantic_owner"] == "st-core::backend::runtime_route"
    assert summary["semantic_backend"] == "rust"
    assert summary["execution_client"] == "python"
    assert summary["committed"] is True
    assert len(summary["request_sha256"]) == 64
    assert len(summary["output_sha256"]) == 64
    assert summary["requested_backends"] == ["wgpu", "cpu", "mps"]
    assert [row["requested_backend"] for row in summary["evidence"]] == [
        "wgpu",
        "cpu",
        "mps",
    ]
    assert summary["ready_backends"] == ["wgpu"]
    assert summary["not_ready_backends"] == ["cpu", "mps"]
    assert summary["error_backends"] == ["mps"]
    assert summary["status_by_backend"] == {
        "wgpu": "kernel_wired",
        "cpu": "cpu",
        "mps": "error",
    }
    assert summary["all_ready"] is False
    assert summary["has_errors"] is True
    assert summary["runtime_readiness"] == "not_ready"
    assert summary["runtime_ready"] is False
    assert summary["runtime_ready_basis"] == "required_ready_backends"
    assert summary["runtime_missing_ready_backends"] == ["mps"]
    assert summary["reports"][2]["error"] == "mps placeholder"
    assert st.validate_runtime_device_route_contract(summary) == summary
    assert calls == [
        ("wgpu", {"workgroup": 128}),
        ("cpu", {"workgroup": 128}),
        ("mps", {"workgroup": 128}),
    ]

    with pytest.raises(RuntimeError, match="mps placeholder"):
        st.describe_runtime_devices(["mps"], continue_on_error=False)


def test_runtime_device_route_distinguishes_native_and_surrogate_readiness() -> None:
    st = require_native()

    contract = st.evaluate_runtime_device_route(
        [
            {
                "requested_backend": "mps",
                "effective_backend": "wgpu",
                "runtime_ready": True,
                "requested_backend_runtime_ready": False,
                "effective_backend_runtime_ready": True,
                "runtime_status": "kernel_wired",
                "requested_backend_runtime_status": "placeholder",
                "effective_backend_runtime_status": "kernel_wired",
                "error": "native MPS kernels are not wired",
            }
        ],
        requested_backends=["mps"],
        required_available_backends=["mps"],
        required_ready_backends=["mps"],
    )

    assert "evaluate_runtime_device_route" in st.__all__
    assert contract["ready_backends"] == ["mps"]
    assert contract["native_ready_backends"] == []
    assert contract["native_not_ready_backends"] == ["mps"]
    assert contract["fallback_backends"] == ["mps"]
    assert contract["error_backends"] == []
    assert contract["routes"][0]["route"] == "surrogate"
    assert contract["routes"][0]["native_readiness"] == "not_ready"
    assert contract["routes"][0]["route_readiness"] == "ready"
    assert contract["routes"][0]["diagnostic"] == "native MPS kernels are not wired"
    assert contract["execution_client"] == "python"
    assert contract["committed"] is True
    assert contract["passed"] is True


def test_runtime_device_route_preserves_unknown_readiness() -> None:
    st = require_native()

    contract = st.evaluate_runtime_device_route(
        [{"requested_backend": "cpu", "runtime_status": "cpu"}],
        requested_backends=["cpu"],
        required_ready_backends=["cpu"],
    )

    route = contract["routes"][0]
    assert route["native_readiness"] == "unknown"
    assert route["native_ready"] is None
    assert route["route_readiness"] == "unknown"
    assert route["route_ready"] is False
    assert route["route_status"] == "unknown"
    assert contract["native_readiness_unknown_backends"] == ["cpu"]
    assert contract["route_readiness_unknown_backends"] == ["cpu"]
    assert contract["required_ready_backends_unknown"] == ["cpu"]
    assert contract["required_ready_backends_passed"] is False
    assert contract["failures"] == ["runtime_device_readiness_unknown:cpu"]


def test_runtime_device_route_rejects_conflicting_readiness() -> None:
    st = require_native()

    with pytest.raises(ValueError, match="disagrees on route readiness"):
        st.evaluate_runtime_device_route(
            [
                {
                    "requested_backend": "wgpu",
                    "runtime_ready": True,
                    "effective_backend_runtime_ready": False,
                }
            ]
        )


def test_runtime_device_route_rejects_cross_report_effective_backend_drift() -> None:
    st = require_native()

    with pytest.raises(ValueError, match="effective backend 'wgpu' readiness"):
        st.evaluate_runtime_device_route(
            [
                {
                    "requested_backend": "mps",
                    "effective_backend": "wgpu",
                    "runtime_ready": True,
                    "requested_backend_runtime_ready": False,
                    "effective_backend_runtime_ready": True,
                    "runtime_status": "kernel_wired",
                },
                {
                    "requested_backend": "wgpu",
                    "runtime_ready": False,
                    "runtime_status": "feature_disabled",
                },
            ]
        )


def test_runtime_device_route_contract_validation_is_rust_owned() -> None:
    st = require_native()
    request = {
        "reports": [
            {
                "requested_backend": "cpu",
                "runtime_ready": True,
                "runtime_status": "cpu",
            }
        ],
        "requested_backends": ["cpu"],
        "required_available_backends": [],
        "required_ready_backends": ["cpu"],
    }
    contract = st.evaluate_runtime_device_route(
        request["reports"],
        requested_backends=request["requested_backends"],
        required_ready_backends=request["required_ready_backends"],
    )

    assert "validate_runtime_device_route_contract" in st.__all__
    assert st.validate_runtime_device_route_contract(contract) == contract
    assert (
        st.validate_runtime_device_route_contract(contract, request=request)
        == contract
    )

    tampered = json.loads(json.dumps(contract))
    tampered["routes"][0]["route_ready"] = False
    with pytest.raises(ValueError, match="derived fields do not match replay"):
        st.validate_runtime_device_route_contract(tampered)


def test_runtime_execution_plan_is_rust_owned_and_replayable() -> None:
    st = require_native()
    probe = st.describe_device("cpu")

    plan = st.evaluate_runtime_execution_plan(
        probe,
        accelerator_fallback="allow",
        tensor_util_values=2048,
        required_native_components=["softmax", "dense_matmul", "dense_matmul"],
    )

    assert "evaluate_runtime_execution_plan" in st.__all__
    assert "validate_runtime_execution_plan_contract" in st.__all__
    assert plan["kind"] == "spiraltorch.runtime_execution_plan"
    assert plan["contract_version"] == "spiraltorch.runtime_execution_plan.v1"
    assert plan["semantic_owner"] == "st-core::backend::execution_plan"
    assert plan["semantic_backend"] == "rust"
    assert plan["execution_client"] == "python"
    assert plan["committed"] is True
    assert len(plan["request_sha256"]) == 64
    assert len(plan["output_sha256"]) == 64
    assert plan["requested_backend"] == "cpu"
    assert plan["effective_backend"] == "cpu"
    assert plan["runtime_ready"] is True
    assert plan["surrogate"] is False
    assert plan["execution_allowed"] is True
    assert plan["status"] == "ready"
    assert plan["all_components_native"] is True
    assert plan["automatic_components"] == []
    assert plan["policy"]["dense_matmul"] == "faer"
    assert plan["policy"]["softmax"] == "cpu"
    assert plan["request"]["runtime_probe"].get("execution_client") is None
    assert plan["runtime_route"].get("execution_client") is None
    assert plan["request"]["required_native_components"] == [
        "dense_matmul",
        "softmax",
    ]
    assert (
        plan["runtime_probe_output_sha256"]
        == plan["request"]["runtime_probe"]["output_sha256"]
    )
    assert (
        plan["runtime_route_output_sha256"]
        == plan["runtime_route"]["output_sha256"]
    )
    assert st.validate_runtime_execution_plan_contract(plan) == plan
    assert (
        st.validate_runtime_execution_plan_contract(plan, request=plan["request"])
        == plan
    )

    tampered = json.loads(json.dumps(plan))
    tampered["component_routes"][0]["selected_backend"] = "auto"
    with pytest.raises(ValueError, match="runtime execution-plan validation failed"):
        st.validate_runtime_execution_plan_contract(tampered)


def test_runtime_execution_plan_exposes_threshold_and_strict_surrogate_gates() -> None:
    st = require_native()
    wgpu_probe = st.describe_device("wgpu")
    thresholded = st.evaluate_runtime_execution_plan(
        wgpu_probe,
        tensor_util_wgpu_min_values=1024,
        tensor_util_values=8,
        required_native_components=["tensor_util"],
    )
    tensor_util = next(
        row
        for row in thresholded["component_routes"]
        if row["component"] == "tensor_util"
    )
    assert tensor_util == {
        "component": "tensor_util",
        "requested_backend": "wgpu",
        "selected_backend": "cpu",
        "route": "cpu_threshold_fallback",
        "native": False,
        "fallback": True,
        "values": 8,
        "threshold": 1024,
    }
    assert thresholded["required_native_components_missing"] == ["tensor_util"]
    assert thresholded["execution_allowed"] is False
    assert "native_component_unavailable:tensor_util" in thresholded["blockers"]

    mps_probe = st.describe_device("mps")
    strict = st.evaluate_runtime_execution_plan(
        mps_probe,
        accelerator_fallback="forbid",
    )
    assert strict["surrogate"] is True
    assert strict["execution_allowed"] is False
    assert strict["status"] == "blocked"
    assert any(
        blocker.startswith("surrogate_forbidden:mps->")
        for blocker in strict["blockers"]
    )


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


def test_rank_plan_exposes_the_rust_owned_audit_contract() -> None:
    st = require_native()

    plan = st.plan("midk", 4, 128, 8, backend="wgpu")
    contract = plan.contract()

    assert contract["kind"] == "spiraltorch.rank_plan"
    assert contract["contract_version"] == "spiraltorch.rank_plan.v1"
    assert contract["semantic_owner"] == "st-core::ops::rank_entry"
    assert contract["semantic_backend"] == "rust"
    assert contract["execution_client"] == "python"
    assert contract["requested_backend"] == "wgpu"
    assert contract["effective_backend"] == "wgpu"
    assert contract["rank_kind"] == "midk"
    assert contract["input_elements"] == 512
    assert contract["output_elements"] == 32
    assert contract["device_caps"]["backend"] == "wgpu"
    assert contract["choice"]["workgroup"] == plan.workgroup
    assert contract["choice"]["compaction_tile"] == plan.compaction_tile


def test_rank_plan_rejects_invalid_shape_and_caps_in_rust() -> None:
    st = require_native()

    with pytest.raises(ValueError, match="dimension 'rows' must be positive"):
        st.plan("topk", 0, 8, 2, backend="wgpu")
    with pytest.raises(ValueError, match="k=9 exceeds cols=8"):
        st.plan("topk", 2, 8, 9, backend="wgpu")
    with pytest.raises(ValueError, match="lane_width.*must be positive"):
        st.plan("topk", 2, 8, 2, backend="wgpu", lane_width=0)
    with pytest.raises(ValueError, match="lane_width=64 exceeds max_workgroup=32"):
        st.describe_device("wgpu", lane_width=64, max_workgroup=32)

    narrow = st.plan(
        "topk",
        2,
        128,
        8,
        backend="wgpu",
        lane_width=32,
        max_workgroup=32,
    )
    assert narrow.workgroup == 32
    assert narrow.contract()["device_caps"]["max_workgroup"] == 32


def test_spiralk_rewrite_uses_validated_rust_rank_semantics() -> None:
    st = require_native()

    top = st.plan("topk", 64, 4096, 32, backend="wgpu")
    rewritten = top.rewrite_with_spiralk(
        "algo: 2; tile_cols: 2048; radix: 2; segments: 2;"
    )
    assert rewritten.merge_strategy == "bitonic"
    assert rewritten.merge_detail == "bitonic"
    assert rewritten.fft_tile == 2048
    assert rewritten.fft_radix == 2
    assert rewritten.fft_segments == 2
    assert rewritten.spiralk_context().tile_cols == 2048
    assert rewritten.contract()["semantic_owner"] == "st-core::ops::rank_entry"

    bottom = st.plan("bottomk", 256, 65536, 1024, backend="wgpu")
    assert bottom.rewrite_with_spiralk("bottomk: 2;").use_two_stage is True

    with pytest.raises(ValueError, match="choice 'workgroup' must be positive"):
        top.rewrite_with_spiralk("wg: 0;")
    with pytest.raises(ValueError, match="choice 'fft_radix'=3 is invalid"):
        top.rewrite_with_spiralk("radix: 3;")
    with pytest.raises(ValueError, match="disagree on two-stage execution"):
        bottom.rewrite_with_spiralk("u2: false; bottomk: 2;")


def test_rank_plan_exposes_the_captured_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    monkeypatch.setenv("SPIRALTORCH_STRICT_GPU", "1")
    monkeypatch.setenv("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "37")

    strict_plan = st.plan("topk", 2, 8, 2, backend="cpu")

    monkeypatch.setenv("SPIRALTORCH_STRICT_GPU", "0")
    monkeypatch.setenv("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "91")
    fallback_plan = st.plan("topk", 2, 8, 2, backend="cpu")

    assert strict_plan.accelerator_fallback == "forbid"
    assert int(strict_plan.tensor_util_wgpu_min_values) == 37
    assert fallback_plan.accelerator_fallback == "allow"
    assert int(fallback_plan.tensor_util_wgpu_min_values) == 91


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

    def _patched_resolve_rs_attr(candidate: str):
        if candidate == "build_info":
            return lambda: {"features": {"logic": True, "wgpu": True}}
        return None

    monkeypatch.delattr(st, "build_info", raising=False)
    monkeypatch.setattr(st, "init_backend", _patched_init_backend, raising=False)
    monkeypatch.setattr(st, "describe_device", _patched_describe_device, raising=False)
    monkeypatch.setattr(st, "plan_topk", _patched_plan_topk, raising=False)
    monkeypatch.setattr(st, "Tensor", _FakeTensor, raising=False)
    monkeypatch.setattr(st, "_resolve_rs_attr", _patched_resolve_rs_attr, raising=False)
    monkeypatch.setattr(
        st,
        "plugin",
        SimpleNamespace(subscribe=_subscribe, unsubscribe=_unsubscribe),
        raising=False,
    )

    report = st.trace_wgpu_first_runtime(rows=2, cols=4, k=8)

    assert report["requested_backend"] == "auto"
    assert report["effective_backend"] == "wgpu"
    assert report["build_features"] == {"logic": True, "wgpu": True}
    assert report["device_preflight"]["backend"] == "wgpu"
    assert report["planner"]["k"] == 4
    assert report["planner"]["effective_backend"] == "wgpu"
    assert report["tensor_operation"]["requested_backend"] == "wgpu"
    assert report["tensor_operation"]["ok"] is True
    assert report["tensor_operation"]["row_sums"] == pytest.approx([1.0, 1.0])
    assert report["tensor_meta_events"] == events
    assert unsubscribed == [("TensorOpMeta", 7)]
    assert calls == ["wgpu"]


def test_trace_wgpu_first_runtime_matrix_collects_backend_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    st = require_native()
    calls: list[tuple[str, int, int, int]] = []

    def _patched_trace(backend: str = "auto", *, rows: int, cols: int, k: int):
        calls.append((backend, rows, cols, k))
        if backend == "mps":
            raise RuntimeError("mps unavailable")
        effective = "wgpu" if backend in {"auto", "wgpu"} else "cpu"
        return {
            "requested_backend": backend,
            "effective_backend": effective,
            "device": effective,
            "device_preflight": {"backend": effective},
            "planner": {"effective_backend": effective},
            "tensor_operation": {
                "op": "row_softmax",
                "requested_backend": effective,
                "ok": True,
            },
            "tensor_meta_events": [],
        }

    monkeypatch.setattr(st, "trace_wgpu_first_runtime", _patched_trace, raising=False)

    matrix = st.trace_wgpu_first_runtime_matrix(
        ["auto", "wgpu", "mps", "cpu"],
        rows=3,
        cols=4,
        k=8,
    )

    assert matrix["kind"] == "wgpu_first_runtime_matrix"
    assert matrix["requested_backends"] == ["auto", "wgpu", "mps", "cpu"]
    assert matrix["k"] == 4
    assert matrix["summary"]["runs"] == 4
    assert matrix["summary"]["ok"] == 3
    assert matrix["summary"]["errors"] == 1
    assert matrix["summary"]["effective_backends"] == {"wgpu": 2, "cpu": 1}
    assert matrix["errors"] == [
        {"requested_backend": "mps", "error": "mps unavailable"}
    ]
    assert [run["matrix_status"] for run in matrix["runs"]] == [
        "ok",
        "ok",
        "error",
        "ok",
    ]
    assert calls == [
        ("auto", 3, 4, 8),
        ("wgpu", 3, 4, 8),
        ("mps", 3, 4, 8),
        ("cpu", 3, 4, 8),
    ]

    output_path = tmp_path / "wgpu-runtime-matrix.json"
    written = st.write_wgpu_first_runtime_matrix(
        output_path,
        ["auto"],
        rows=1,
        cols=2,
        k=1,
    )
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert written["artifact_path"] == str(output_path)
    assert loaded["artifact_path"] == str(output_path)
    assert loaded["summary"]["runs"] == 1
