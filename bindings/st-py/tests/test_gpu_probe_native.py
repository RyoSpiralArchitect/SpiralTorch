from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ._native_loader import require_native, require_wgpu_runtime


def _install_fake_session_execution_plan_runtime(monkeypatch, st) -> None:
    def _evaluate(report, **kwargs):
        backend = str(report["backend"])
        marker = "a" if backend == "wgpu" else "b"
        return {
            "requested_backend": backend,
            "effective_backend": backend,
            "output_sha256": marker * 64,
            "request": {
                "execution_config": {
                    "accelerator_fallback": kwargs.get(
                        "accelerator_fallback", "allow"
                    ),
                    "tensor_util_wgpu_min_values": kwargs.get(
                        "tensor_util_wgpu_min_values", 1024
                    ),
                },
                "component_resolution": kwargs.get(
                    "component_resolution", "concrete"
                ),
            },
        }

    monkeypatch.setattr(
        st,
        "evaluate_runtime_execution_plan",
        _evaluate,
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "require_executable_runtime_execution_plan",
        lambda payload: dict(payload),
        raising=False,
    )


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


def test_public_runtime_device_observer_delegates_resolution_to_rust() -> None:
    st = require_native()

    report = st.observe_runtime_device_probe(
        "cpu",
        max_workgroup=64,
        requested_workgroup=63,
        cols=1024,
    )

    assert "observe_runtime_device_probe" in st.__all__
    assert report["semantic_owner"] == "st-core::backend::runtime_probe"
    assert report["semantic_backend"] == "rust"
    assert report["execution_client"] == "python"
    assert report["requested_backend"] == "cpu"
    assert report["effective_backend"] == "cpu"
    assert report["request"]["caps"]["backend"] == "cpu"
    assert report["request"]["caps"]["max_workgroup"] == 64
    assert report["requested_runtime"] == report["effective_runtime"]
    assert st.validate_runtime_device_probe_contract(report) == report["contract"]

    with pytest.raises(ValueError, match="max_workgroup.*must be positive"):
        st.observe_runtime_device_probe("cpu", max_workgroup=0)

    with pytest.raises(ValueError, match="unknown field"):
        st._rs._runtime_device_probe_observe(
            {
                "requested_backend": "cpu",
                "effective_backend": "wgpu",
            }
        )


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
    assert summary["contract_version"] == "spiraltorch.runtime_device_route.v5"
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


def test_committed_probe_routes_do_not_rebuild_evidence_in_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    import spiraltorch.runtime_imports as runtime_imports

    def _unexpected_compatibility_projection(*_args: object, **_kwargs: object):
        raise AssertionError("committed probes must bypass Python evidence projection")

    monkeypatch.setattr(
        runtime_imports,
        "_runtime_device_route_evidence",
        _unexpected_compatibility_projection,
    )

    probe = st.observe_runtime_device_probe("cpu")
    canonical_probe = st.validate_runtime_device_probe_contract(probe)
    route = st.evaluate_runtime_device_route_from_probes(
        [probe],
        required_ready_backends=["cpu"],
    )
    public_route = st.evaluate_runtime_device_route(
        [probe],
        required_ready_backends=["cpu"],
    )
    canonical_route = st.evaluate_runtime_device_route(
        [canonical_probe],
        required_ready_backends=["cpu"],
    )
    summary = st.describe_runtime_devices(["cpu"])

    assert "evaluate_runtime_device_route_from_probes" in st.__all__
    assert route["evidence"] == [probe["route_evidence"]]
    assert route["requested_backends"] == ["cpu"]
    assert route["selection"]["requested_backend"] == "cpu"
    assert route["execution_client"] == "python"
    assert public_route == route
    assert canonical_route == route
    assert summary["evidence"] == [summary["reports"][0]["route_evidence"]]

    tampered = json.loads(json.dumps(probe))
    tampered["contract"]["route_evidence"]["runtime_ready"] = False
    with pytest.raises(ValueError, match="probe 0 failed committed validation"):
        st.evaluate_runtime_device_route_from_probes([tampered])


def test_mixed_device_reports_never_downgrade_committed_probe_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    probe = st.observe_runtime_device_probe("cpu")
    tampered_transport = json.loads(json.dumps(probe))
    tampered_transport["route_evidence"]["runtime_ready"] = False

    def _mixed_describe_device(backend: str = "cpu", **_kwargs: object):
        if backend == "cpu":
            return tampered_transport
        raise RuntimeError("diagnostic-only probe failure")

    monkeypatch.setattr(st, "describe_device", _mixed_describe_device, raising=False)

    summary = st.describe_runtime_devices(
        ["cpu", "mps"],
        required_ready_backends=["cpu", "mps"],
    )

    assert summary["evidence"][0] == probe["contract"]["route_evidence"]
    assert summary["evidence"][1]["requested_backend"] == "mps"
    assert summary["evidence"][1]["runtime_status"] == "error"
    assert summary["requested_backends"] == ["cpu", "mps"]
    assert summary["ready_backends"] == ["cpu"]
    assert summary["error_backends"] == ["mps"]
    assert summary["status_by_backend"]["mps"] == "error"
    assert summary["has_errors"] is True
    assert summary["routes"][0]["requested_backend"] == "cpu"
    assert summary["routes"][0]["route_ready"] is True
    assert summary["selection"] is None
    assert summary["required_ready_backends_passed"] is False
    assert summary["runtime_missing_ready_backends"] == ["mps"]
    assert summary["reports"][1]["error"] == "diagnostic-only probe failure"
    assert summary["reports"][0]["route_evidence"]["runtime_ready"] is False
    assert st.validate_runtime_device_route_contract(summary) == summary


def test_malformed_probe_envelopes_never_fall_back_to_transport_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    malformed = json.loads(json.dumps(st.observe_runtime_device_probe("cpu")))
    malformed["contract"]["kind"] = "forged.runtime_device_probe"
    malformed["route_evidence"]["runtime_ready"] = False

    monkeypatch.setattr(
        st,
        "describe_device",
        lambda _backend="cpu", **_kwargs: malformed,
        raising=False,
    )

    with pytest.raises(ValueError, match="kind.*must be"):
        st.describe_runtime_devices(["cpu"])


@pytest.mark.parametrize("kind_mutation", ["corrupt", "remove"])
def test_canonical_probe_contracts_never_downgrade_when_kind_is_damaged(
    kind_mutation: str,
) -> None:
    st = require_native()
    canonical = st.validate_runtime_device_probe_contract(
        st.observe_runtime_device_probe("cpu")
    )
    if kind_mutation == "corrupt":
        canonical["kind"] = "forged.runtime_device_probe"
    else:
        canonical.pop("kind")
    canonical["route_evidence"]["runtime_ready"] = False

    with pytest.raises(ValueError, match="kind|probe"):
        st.evaluate_runtime_device_route([canonical])


def test_naked_route_evidence_never_enters_compatibility_projection() -> None:
    st = require_native()
    canonical = st.validate_runtime_device_probe_contract(
        st.observe_runtime_device_probe("cpu")
    )
    route_evidence = dict(canonical["route_evidence"])
    route_evidence["runtime_ready"] = False

    with pytest.raises(ValueError, match="kind|probe"):
        st.evaluate_runtime_device_route([{"route_evidence": route_evidence}])


def test_runtime_preflight_applies_required_gates_to_committed_probes() -> None:
    st = require_native()

    fields = st.runtime_device_report_fields(
        {
            "runtime_device_backends": ["cpu"],
            "required_runtime_device_backends": ["cpu"],
            "required_runtime_device_ready_backends": ["cpu"],
        }
    )
    contract = json.loads(fields["runtime_device_route_contract_json"])
    reports = json.loads(fields["runtime_device_reports_json"])

    assert "reports" not in contract
    assert reports
    assert st.validate_runtime_device_route_contract(contract) == contract
    assert contract["required_available_backends"] == ["cpu"]
    assert contract["required_ready_backends"] == ["cpu"]
    assert contract["required_available_backends_passed"] is True
    assert contract["required_ready_backends_passed"] is True
    assert fields["required_runtime_device_backends_passed"] is True
    assert fields["required_runtime_device_ready_backends_passed"] is True


def test_runtime_preflight_custom_collectors_preserve_committed_probe_ingress() -> None:
    st = require_native()
    probe = st.observe_runtime_device_probe("cpu")
    tampered_transport = json.loads(json.dumps(probe))
    tampered_transport["route_evidence"]["runtime_ready"] = False

    def _custom_collector(_backends: object, **_kwargs: object):
        return {
            "reports": [
                tampered_transport,
                {
                    "backend": "mps",
                    "requested_backend": "mps",
                    "runtime_ready": False,
                    "runtime_status": "error",
                    "error": "diagnostic-only custom collector failure",
                },
            ]
        }

    fields = st.runtime_device_report_fields(
        {
            "runtime_device_backends": ["cpu", "mps"],
            "required_runtime_device_ready_backends": ["cpu", "mps"],
        },
        describe_runtime_devices=_custom_collector,
    )
    contract = json.loads(fields["runtime_device_route_contract_json"])
    reports = json.loads(fields["runtime_device_reports_json"])

    assert contract["evidence"][0] == probe["contract"]["route_evidence"]
    assert contract["evidence"][1]["requested_backend"] == "mps"
    assert contract["evidence"][1]["runtime_status"] == "error"
    assert contract["ready_backends"] == ["cpu"]
    assert contract["error_backends"] == ["mps"]
    assert contract["status_by_backend"]["mps"] == "error"
    assert contract["has_errors"] is True
    assert fields["runtime_device_report_error_backends"] == "mps"
    assert contract["required_ready_backends_passed"] is False
    assert contract["runtime_missing_ready_backends"] == ["mps"]
    assert reports[0]["route_evidence"]["runtime_ready"] is False
    assert reports[1]["error"] == "diagnostic-only custom collector failure"


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
    assert contract["successful_probe_backends"] == ["mps"]
    assert contract["available_backends"] == []
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
    assert contract["required_available_backends_passed"] is False
    assert contract["runtime_readiness"] == "not_ready"
    assert contract["runtime_ready"] is False
    assert contract["runtime_ready_basis"] == "required_available_and_ready_backends"
    assert contract["failures"] == ["runtime_device_unavailable:mps"]
    assert contract["selection"] is None
    assert contract["passed"] is False

    route_only = st.evaluate_runtime_device_route(
        contract["evidence"],
        requested_backends=["mps"],
        required_ready_backends=["mps"],
    )
    assert route_only["passed"] is True
    assert route_only["selection"]["requested_backend"] == "mps"
    assert route_only["selection"]["effective_backend"] == "wgpu"


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


def test_runtime_device_route_does_not_hide_duplicate_python_labels() -> None:
    st = require_native()

    with pytest.raises(ValueError, match="appears more than once"):
        st.evaluate_runtime_device_route(
            [{"requested_backend": "wgpu", "runtime_ready": True}],
            requested_backends=[" WGPU ", "wgpu"],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        st.evaluate_runtime_device_route([], requested_backends="")


def test_describe_runtime_devices_validates_labels_before_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[str] = []

    def _unexpected_probe(backend: str, **_kwargs: object):
        calls.append(backend)
        raise AssertionError("invalid labels must fail before probing")

    monkeypatch.setattr(st, "describe_device", _unexpected_probe, raising=False)

    with pytest.raises(ValueError, match="appears more than once"):
        st.describe_runtime_devices([" WGPU ", "wgpu"])
    with pytest.raises(ValueError, match="must not be empty"):
        st.describe_runtime_devices("")
    assert calls == []


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
        component_workloads=[
            {
                "component": "dense_matmul",
                "rows": 2,
                "inner": 3,
                "cols": 4,
            },
            {
                "component": "prepacked_matmul",
                "rows": 2,
                "inner": 3,
                "cols": 4,
                "bias": True,
            },
            {"component": "layer_norm", "rows": 2, "cols": 4},
            {
                "component": "attention",
                "contexts": 1,
                "sequence": 2,
                "head_dim": 4,
                "z_bias": True,
                "attn_bias": True,
            },
            {"component": "softmax", "rows": 2, "cols": 8},
            {
                "component": "tensor_util",
                "operation": "scale",
                "rows": 32,
                "cols": 64,
            },
        ],
        required_native_components=["softmax", "dense_matmul", "dense_matmul"],
    )

    assert "evaluate_runtime_execution_plan" in st.__all__
    assert "observe_runtime_execution_plan_capabilities" in st.__all__
    assert "validate_runtime_execution_plan_contract" in st.__all__
    assert plan["kind"] == "spiraltorch.runtime_execution_plan"
    assert plan["contract_version"] == "spiraltorch.runtime_execution_plan.v6"
    assert plan["runtime_route"]["contract_version"] == "spiraltorch.runtime_device_route.v5"
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
    assert plan["request"]["component_resolution"] == "concrete"
    assert plan["runtime_route"].get("execution_client") is None
    assert plan["request"]["required_native_components"] == [
        "dense_matmul",
        "softmax",
    ]
    observation = plan["request"]["component_capability_observation"]
    assert observation["kind"] == "spiraltorch.runtime_component_capability_observation"
    assert (
        observation["contract_version"]
        == "spiraltorch.runtime_component_capability_observation.v2"
    )
    assert observation["semantic_owner"] == "st-core::backend::execution_capability"
    assert observation["committed"] is True
    assert observation["request"]["policy"] == plan["policy"]
    assert len(observation["request_sha256"]) == 64
    assert len(observation["output_sha256"]) == 64
    assert [
        evidence["status"]
        for evidence in observation["capabilities"]
    ] == ["ready"] * 6
    assert [
        evidence["ready_proof"]
        for evidence in observation["capabilities"]
    ] == ["static_host_contract"] * 6
    assert [
        route["capability_state"]
        for route in plan["component_routes"]
        if route["component"] in {"dense_matmul", "softmax"}
    ] == ["ready", "ready"]
    assert (
        plan["runtime_probe_output_sha256"]
        == plan["request"]["runtime_probe"]["output_sha256"]
    )
    assert (
        plan["component_capability_observation_output_sha256"]
        == observation["output_sha256"]
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

    naked_claim = json.loads(json.dumps(plan["request"]))
    naked_claim["component_capabilities"] = [
        {
            "workload": {"component": "softmax", "rows": 2, "cols": 8},
            "backend": "cpu",
            "status": "ready",
        }
    ]
    with pytest.raises(ValueError, match="runtime execution-plan replay request"):
        st.validate_runtime_execution_plan_contract(plan, request=naked_claim)

    tampered_observation = json.loads(json.dumps(plan["request"]))
    tampered_observation["component_capability_observation"]["capabilities"][0][
        "status"
    ] = "not_built"
    with pytest.raises(ValueError, match="runtime execution-plan replay failed"):
        st.validate_runtime_execution_plan_contract(
            plan,
            request=tampered_observation,
        )

    tampered_proof = json.loads(json.dumps(plan["request"]))
    tampered_proof["component_capability_observation"]["capabilities"][0][
        "ready_proof"
    ] = "runtime_dispatch_sentinel"
    with pytest.raises(ValueError, match="runtime execution-plan replay failed"):
        st.validate_runtime_execution_plan_contract(plan, request=tampered_proof)

    tampered = json.loads(json.dumps(plan))
    tampered["component_routes"][0]["selected_backend"] = "auto"
    with pytest.raises(ValueError, match="runtime execution-plan validation failed"):
        st.validate_runtime_execution_plan_contract(tampered)

    legacy = json.loads(json.dumps(plan))
    legacy["contract_version"] = "spiraltorch.runtime_execution_plan.v5"
    with pytest.raises(ValueError, match="contract_version"):
        st.validate_runtime_execution_plan_contract(legacy)


def test_tensor_execution_receipt_is_validated_only_by_rust() -> None:
    st = require_native()
    receipt = {
        "kind": "spiraltorch.tensor_execution_receipt",
        "contract_version": "spiraltorch.tensor_execution_receipt.v1",
        "semantic_owner": "st-tensor::execution",
        "component": "softmax",
        "operation": "row_softmax",
        "workload": {"component": "softmax", "rows": 2, "cols": 3},
        "requested_backend": "cpu",
        "selected_backend": "cpu",
        "executed_backend": "cpu",
        "route_status": "direct",
    }

    assert "validate_tensor_execution_receipt" in st.__all__
    assert st.validate_tensor_execution_receipt(receipt) == receipt

    tampered = dict(receipt)
    tampered["executed_backend"] = "wgpu"
    with pytest.raises(ValueError, match="receipt validation failed"):
        st.validate_tensor_execution_receipt(tampered)

    unsupported = dict(receipt)
    unsupported["requested_backend"] = "faer"
    unsupported["selected_backend"] = "faer"
    unsupported["executed_backend"] = "faer"
    with pytest.raises(ValueError, match="receipt validation failed"):
        st.validate_tensor_execution_receipt(unsupported)

    reconstructed = dict(receipt)
    reconstructed["python_route"] = "exploratory"
    with pytest.raises(ValueError, match="invalid tensor execution receipt"):
        st.validate_tensor_execution_receipt(reconstructed)


def test_python_transports_wgpu_dispatch_proofs_from_rust() -> None:
    st = require_native()
    require_wgpu_runtime(st)

    plan = st.evaluate_runtime_execution_plan(
        st.describe_device("wgpu"),
        accelerator_fallback="forbid",
        component_workloads=[
            {
                "component": "dense_matmul",
                "rows": 2,
                "inner": 3,
                "cols": 4,
            },
            {
                "component": "prepacked_matmul",
                "rows": 2,
                "inner": 3,
                "cols": 4,
                "bias": True,
            },
            {"component": "layer_norm", "rows": 2, "cols": 4},
            {
                "component": "attention",
                "contexts": 1,
                "sequence": 2,
                "head_dim": 4,
                "z_bias": True,
                "attn_bias": True,
            },
            {"component": "softmax", "rows": 2, "cols": 8},
            {
                "component": "tensor_util",
                "operation": "scale",
                "rows": 32,
                "cols": 64,
            },
        ],
    )

    observation = plan["request"]["component_capability_observation"]
    assert plan["execution_allowed"] is True
    assert [
        evidence["backend"] for evidence in observation["capabilities"]
    ] == ["wgpu"] * 6
    assert [
        evidence["status"] for evidence in observation["capabilities"]
    ] == ["ready"] * 6
    assert [
        evidence["ready_proof"] for evidence in observation["capabilities"]
    ] == ["runtime_dispatch_sentinel"] * 6
    assert st.validate_runtime_execution_plan_contract(plan) == plan


def test_runtime_execution_config_defaults_are_captured_by_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    monkeypatch.setenv("SPIRALTORCH_STRICT_GPU", "1")
    monkeypatch.setenv("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "37")

    captured = st.resolve_runtime_execution_config()
    overridden = st.resolve_runtime_execution_config(
        accelerator_fallback="allow",
        tensor_util_wgpu_min_values=91,
    )

    assert captured == {
        "accelerator_fallback": "forbid",
        "tensor_util_wgpu_min_values": 37,
    }
    assert overridden == {
        "accelerator_fallback": "allow",
        "tensor_util_wgpu_min_values": 91,
    }
    with pytest.raises(ValueError, match="invalid accelerator fallback override"):
        st.resolve_runtime_execution_config(accelerator_fallback="sometimes")


def test_session_captures_the_rust_environment_execution_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    monkeypatch.setenv("SPIRALTORCH_STRICT_GPU", "1")
    monkeypatch.setenv("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "37")

    session = st.SpiralSession(backend="cpu")

    assert session.runtime_execution_plan["request"]["execution_config"] == {
        "accelerator_fallback": "forbid",
        "tensor_util_wgpu_min_values": 37,
    }


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
        "capability_state": "not_applicable",
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
    assert contract["contract_version"] == "spiraltorch.rank_plan.v2"
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


def test_strict_wgpu_session_commits_a_deferred_rust_plan() -> None:
    st = require_native()
    require_wgpu_runtime(st)

    session = st.SpiralSession(
        backend="wgpu",
        accelerator_fallback="forbid",
        tensor_util_wgpu_min_values=37,
    )
    plan = session.runtime_execution_plan

    assert plan["request"]["execution_config"] == {
        "accelerator_fallback": "forbid",
        "tensor_util_wgpu_min_values": 37,
    }
    assert plan["request"]["component_resolution"] == "deferred"
    assert plan["execution_allowed"] is True
    assert plan["status"] == "ready"
    assert plan["blockers"] == []
    assert plan["all_components_native"] is False
    assert plan["native_components"] == []
    assert plan["conditional_components"] == []


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
    _install_fake_session_execution_plan_runtime(monkeypatch, st)

    session = st.SpiralSession()

    assert session.backend == "auto"
    assert session.requested_backend == "wgpu"
    assert session.effective_backend == "wgpu"
    assert session.device == "wgpu"
    assert session.device_preflight["backend"] == "wgpu"
    assert calls == ["wgpu"]


def test_session_auto_falls_back_to_cpu_when_rust_reports_wgpu_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[str] = []

    def _patched_init_backend(backend: str) -> bool:
        raw = str(backend)
        calls.append(raw)
        if raw == "wgpu":
            return False
        return True

    def _patched_describe_device(backend: str = "wgpu", **_kwargs: object):
        return {"backend": str(backend), "lane_width": 1}

    monkeypatch.setattr(st, "init_backend", _patched_init_backend, raising=False)
    monkeypatch.setattr(st, "describe_device", _patched_describe_device, raising=False)
    _install_fake_session_execution_plan_runtime(monkeypatch, st)

    session = st.SpiralSession()

    assert session.backend == "auto"
    assert session.requested_backend == "cpu"
    assert session.effective_backend == "cpu"
    assert session.device == "cpu"
    assert session.device_preflight["backend"] == "cpu"
    assert calls == ["wgpu", "cpu"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_session_auto_does_not_hide_execution_plan_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
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
    _install_fake_session_execution_plan_runtime(monkeypatch, st)

    def _reject_plan(_payload: object) -> dict[str, object]:
        raise error_type("runtime execution-plan validation failed")

    monkeypatch.setattr(
        st,
        "require_executable_runtime_execution_plan",
        _reject_plan,
        raising=False,
    )

    with pytest.raises(error_type, match="runtime execution-plan validation failed"):
        st.SpiralSession()

    assert calls == ["wgpu"]


def test_session_auto_does_not_fall_back_to_cpu_under_strict_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[str] = []
    monkeypatch.setenv("SPIRALTORCH_STRICT_GPU", "1")

    def _patched_init_backend(backend: str) -> bool:
        calls.append(str(backend))
        raise RuntimeError("wgpu unavailable")

    monkeypatch.setattr(st, "init_backend", _patched_init_backend, raising=False)

    with pytest.raises(RuntimeError, match="wgpu unavailable"):
        st.SpiralSession()

    assert calls == ["wgpu"]


def test_session_auto_does_not_hide_python_plumbing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = require_native()
    calls: list[str] = []

    def _patched_init_backend(backend: str) -> bool:
        calls.append(str(backend))
        raise AttributeError("broken Python adapter")

    monkeypatch.setattr(st, "init_backend", _patched_init_backend, raising=False)

    with pytest.raises(AttributeError, match="broken Python adapter"):
        st.SpiralSession()

    assert calls == ["wgpu"]


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

    def _patched_plan_topk(
        rows: int,
        cols: int,
        k: int,
        *,
        runtime_execution_plan,
    ):
        backend = str(runtime_execution_plan["effective_backend"])
        return SimpleNamespace(
            kind="topk",
            requested_backend=backend,
            effective_backend=backend,
            rows=rows,
            cols=cols,
            k=k,
            workgroup=128,
            lanes=32,
            runtime_execution_plan_output_sha256=runtime_execution_plan[
                "output_sha256"
            ],
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
    _install_fake_session_execution_plan_runtime(monkeypatch, st)
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
    assert report["planner"]["runtime_execution_plan_output_sha256"] == "a" * 64
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
