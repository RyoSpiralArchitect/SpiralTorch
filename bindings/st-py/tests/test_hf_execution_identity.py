from __future__ import annotations

import json

import pytest

from spiraltorch.hf_execution_identity import (
    HF_FINETUNE_EXECUTION_IDENTITY_SCHEMA,
    hf_finetune_execution_identity_lines,
    hf_finetune_execution_identity_report,
)


_BUILD_ID = "sha256:" + "1" * 64
_PYTHON_RUNTIME = {
    "implementation": "CPython",
    "python_version": "3.12.6",
    "cache_tag": "cpython-312",
    "byteorder": "little",
    "system": "Darwin",
    "release": "test",
    "machine": "arm64",
    "processor": "arm",
    "libc": None,
    "libc_version": None,
}
_TORCH_CAPABILITIES = {
    "default_dtype": "torch.float32",
    "deterministic_algorithms": False,
    "float32_matmul_precision": "highest",
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_devices": [],
    "cuda_runtime_version": None,
    "hip_runtime_version": None,
    "torch_git_version": "torch-test",
    "torch_debug_build": False,
    "mps_available": True,
    "mps_built": True,
}
_DISTRIBUTIONS = {
    "torch": {
        "distribution": "torch",
        "version": "2.12.1",
        "record_sha256": "a" * 64,
        "record_file_count": 20,
    },
    "transformers": {
        "distribution": "transformers",
        "version": "5.13.0",
        "record_sha256": "b" * 64,
        "record_file_count": 30,
    },
}
_SPIRALTORCH_DISTRIBUTION = {
    "distribution": "spiraltorch",
    "version": "0.4.13",
    "record_sha256": "d" * 64,
    "record_file_count": 40,
}


def _preflight(*, root: str = "/first/site-packages") -> dict[str, object]:
    return {
        "runtime_import_preflight_passed": True,
        "runtime_imports_json": json.dumps(
            [
                {
                    "module": "transformers",
                    "imported": True,
                    "version": "5.13.0",
                    "module_file": f"{root}/transformers/__init__.py",
                },
                {
                    "module": "torch",
                    "imported": True,
                    "version": "2.12.1",
                    "module_file": f"{root}/torch/__init__.py",
                },
            ]
        ),
        "runtime_device_report_requested": True,
        "runtime_device_reports_json": json.dumps(
            [
                {
                    "requested_backend": "cpu",
                    "effective_backend": "cpu",
                    "runtime_ready": True,
                    "runtime_status": "cpu",
                    "lane_width": 1,
                    "max_workgroup": 128,
                    "runtime_recommendation": "volatile prose is ignored",
                }
            ]
        ),
    }


def _report(
    preflight: dict[str, object] | None = None,
    *,
    expected_identity_id: str | None = None,
    distributions: dict[str, object] | None = None,
    environment: dict[str, str] | None = None,
    torch_capabilities: dict[str, object] | None = None,
    spiraltorch_build_fingerprint: str = _BUILD_ID,
    spiraltorch_distribution: dict[str, object] | None = None,
) -> dict[str, object]:
    return hf_finetune_execution_identity_report(
        preflight or _preflight(),
        spiraltorch_version="0.4.13",
        spiraltorch_build_fingerprint=spiraltorch_build_fingerprint,
        spiraltorch_distribution_fingerprint=(
            spiraltorch_distribution or _SPIRALTORCH_DISTRIBUTION
        ),
        python_runtime=_PYTHON_RUNTIME,
        torch_capabilities=torch_capabilities or _TORCH_CAPABILITIES,
        distribution_fingerprints=distributions or _DISTRIBUTIONS,
        environment=environment or {},
        expected_identity_id=expected_identity_id,
    )


def test_execution_identity_is_path_independent_and_verifiable() -> None:
    first = _report(_preflight(root="/first/site-packages"))
    relocated = _report(
        _preflight(root="/relocated/site-packages"),
        expected_identity_id=first["observed_identity_id"],
    )

    assert first["schema"] == HF_FINETUNE_EXECUTION_IDENTITY_SCHEMA
    assert first["status"] == "ready"
    assert first["distribution_count"] == 2
    assert relocated["status"] == "ready"
    assert relocated["expected_identity_verified"] is True
    assert relocated["observed_identity_id"] == first["observed_identity_id"]
    assert "status=ready" in hf_finetune_execution_identity_lines(first)[0]


def test_execution_identity_ignores_volatile_runtime_build_manifest() -> None:
    first = _report(spiraltorch_build_fingerprint="sha256:" + "1" * 64)
    restarted = _report(
        spiraltorch_build_fingerprint="sha256:" + "2" * 64,
        expected_identity_id=first["observed_identity_id"],
    )

    assert restarted["status"] == "ready"
    assert restarted["observed_identity_id"] == first["observed_identity_id"]
    assert (
        first["spiraltorch"]["runtime_build_fingerprint"]
        != restarted["spiraltorch"]["runtime_build_fingerprint"]
    )


@pytest.mark.parametrize("drift", ["distribution", "device", "environment"])
def test_execution_identity_rejects_hidden_runtime_drift(drift: str) -> None:
    ready = _report()
    preflight = _preflight()
    distributions = {key: dict(value) for key, value in _DISTRIBUTIONS.items()}
    environment: dict[str, str] = {}
    torch_capabilities = dict(_TORCH_CAPABILITIES)
    if drift == "distribution":
        distributions["torch"]["record_sha256"] = "c" * 64
    elif drift == "device":
        torch_capabilities["mps_available"] = False
    else:
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    blocked = _report(
        preflight,
        expected_identity_id=ready["observed_identity_id"],
        distributions=distributions,
        environment=environment,
        torch_capabilities=torch_capabilities,
    )

    assert blocked["status"] == "blocked"
    assert blocked["expected_identity_verified"] is False


def test_execution_identity_requires_complete_distribution_evidence() -> None:
    incomplete = _report(
        distributions={
            **_DISTRIBUTIONS,
            "torch": {
                "distribution": "torch",
                "version": "2.12.1",
                "record_sha256": None,
                "record_file_count": 0,
            },
        }
    )

    assert incomplete["status"] == "evidence_incomplete"
    assert incomplete["observed_identity_id"] is None
    assert any("RECORD hash" in error for error in incomplete["errors"])


def test_execution_identity_validates_expected_id() -> None:
    with pytest.raises(ValueError, match="sha256"):
        _report(expected_identity_id="not-an-identity")
