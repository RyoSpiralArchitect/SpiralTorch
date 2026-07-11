"""Content identity for the software and device basis of HF fine-tuning."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "HF_FINETUNE_EXECUTION_IDENTITY_SCHEMA",
    "hf_finetune_execution_identity_lines",
    "hf_finetune_execution_identity_report",
]


HF_FINETUNE_EXECUTION_IDENTITY_SCHEMA = "spiraltorch.hf_finetune_execution_identity.v1"
_HF_FINETUNE_EXECUTION_BUNDLE_SCHEMA = "spiraltorch.hf_finetune_execution_bundle.v1"
_MODULE_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "evaluate": "evaluate",
    "peft": "peft",
    "pyarrow": "pyarrow",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "tqdm": "tqdm",
    "trackio": "trackio",
    "transformers": "transformers",
    "trl": "trl",
}
_EXECUTION_ENVIRONMENT_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "HIP_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "RAYON_NUM_THREADS",
    "ROCR_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_OFFLINE",
)
_DEVICE_IDENTITY_FIELDS = (
    "requested_backend",
    "backend",
    "effective_backend",
    "requested_backend_feature_enabled",
    "effective_backend_feature_enabled",
    "requested_backend_placeholder",
    "effective_backend_placeholder",
    "requested_backend_real_kernels_compiled",
    "effective_backend_real_kernels_compiled",
    "requested_backend_runtime_ready",
    "effective_backend_runtime_ready",
    "runtime_ready",
    "requested_backend_runtime_status",
    "effective_backend_runtime_status",
    "runtime_status",
    "lane_width",
    "max_workgroup",
    "shared_mem_per_workgroup",
    "subgroup",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_identity_id(value: object | None) -> str | None:
    if value is None:
        return None
    identity_id = str(value).strip()
    digest = identity_id[7:] if identity_id.startswith("sha256:") else ""
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            "expected_identity_id must be a lowercase sha256:<64 hex> identity id"
        )
    return identity_id


def _json_rows(value: object) -> list[dict[str, object]]:
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            return []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _runtime_import_rows(
    preflight: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    rows = _json_rows(preflight.get("runtime_imports_json"))
    errors: list[str] = []
    packages: list[dict[str, object]] = []
    for row in rows:
        module = str(row.get("module") or "").strip()
        if not module:
            errors.append("runtime import row is missing its module name")
            continue
        if row.get("imported") is not True:
            errors.append(f"runtime package {module} was not imported")
            continue
        version = row.get("version")
        if version is None or not str(version).strip():
            errors.append(f"runtime package {module} did not expose a version")
            continue
        packages.append(
            {
                "module": module,
                "version": str(version).strip(),
            }
        )
    packages.sort(key=lambda row: str(row["module"]))
    if not packages:
        errors.append("runtime preflight did not expose imported package evidence")
    if preflight.get("runtime_import_preflight_passed") is not True:
        errors.append("runtime import preflight is not ready")
    return packages, errors


def _record_rows(value: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path, digest, size, *_extra in csv.reader(io.StringIO(value)):
        normalized = str(path).replace("\\", "/")
        if normalized.endswith(".dist-info/RECORD"):
            continue
        if normalized.endswith(".dist-info/direct_url.json"):
            continue
        if normalized.endswith(".dist-info/INSTALLER"):
            continue
        if normalized.endswith(".dist-info/REQUESTED"):
            continue
        rows.append(
            {
                "path": normalized,
                "digest": digest or None,
                "size": size or None,
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def _distribution_fingerprint(module: str, version: str) -> dict[str, object]:
    distribution_name = _MODULE_DISTRIBUTIONS.get(module, module)
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "status": "missing",
            "module": module,
            "distribution": distribution_name,
            "version": version,
            "record_sha256": None,
            "record_file_count": 0,
        }
    observed_version = str(distribution.version or "").strip() or None
    record = distribution.read_text("RECORD")
    rows = [] if record is None else _record_rows(record)
    return {
        "status": (
            "ready"
            if observed_version == version and rows
            else "version_mismatch"
            if observed_version != version
            else "record_missing"
        ),
        "module": module,
        "distribution": distribution_name,
        "version": observed_version,
        "record_sha256": (
            None if not rows else _sha256_bytes(_canonical_json_bytes(rows))
        ),
        "record_file_count": len(rows),
    }


def _distribution_rows(
    packages: list[dict[str, object]],
    overrides: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for package in packages:
        module = str(package["module"])
        version = str(package["version"])
        override = None if overrides is None else overrides.get(module)
        if isinstance(override, Mapping):
            row = dict(override)
            row.setdefault("module", module)
            row.setdefault("version", version)
            row.setdefault("status", "ready")
        else:
            row = _distribution_fingerprint(module, version)
        if row.get("status") != "ready":
            errors.append(
                f"runtime distribution {module} is not content-addressable: "
                f"{row.get('status')}"
            )
        if row.get("version") != version:
            errors.append(f"runtime distribution {module} version does not match")
        if not row.get("record_sha256"):
            errors.append(f"runtime distribution {module} RECORD hash is missing")
        rows.append(
            {
                "module": module,
                "distribution": row.get("distribution") or module,
                "version": row.get("version"),
                "record_sha256": row.get("record_sha256"),
                "record_file_count": row.get("record_file_count"),
            }
        )
    rows.sort(key=lambda row: str(row["module"]))
    return rows, errors


def _device_rows(
    preflight: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    requested = preflight.get("runtime_device_report_requested") is True
    raw_rows = _json_rows(preflight.get("runtime_device_reports_json"))
    rows = [
        {key: row.get(key) for key in _DEVICE_IDENTITY_FIELDS if key in row}
        for row in raw_rows
    ]
    rows.sort(
        key=lambda row: str(
            row.get("requested_backend")
            or row.get("backend")
            or row.get("effective_backend")
            or ""
        )
    )
    errors = []
    if requested and not rows:
        errors.append("runtime device preflight did not expose backend evidence")
    return rows, errors


def _safe_call(value: object, name: str, default: object = None) -> object:
    function = getattr(value, name, None)
    if not callable(function):
        return default
    try:
        return function()
    except Exception:
        return default


def _torch_capability_payload(torch_module: Any) -> dict[str, object] | None:
    if torch_module is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    version = getattr(torch_module, "version", None)
    cuda_available = bool(_safe_call(cuda, "is_available", False))
    cuda_count = int(_safe_call(cuda, "device_count", 0) or 0)
    cuda_devices: list[dict[str, object]] = []
    for index in range(cuda_count if cuda_available else 0):
        get_name = getattr(cuda, "get_device_name", None)
        get_capability = getattr(cuda, "get_device_capability", None)
        try:
            name = get_name(index) if callable(get_name) else None
        except Exception:
            name = None
        try:
            capability = get_capability(index) if callable(get_capability) else None
        except Exception:
            capability = None
        cuda_devices.append(
            {
                "index": index,
                "name": None if name is None else str(name),
                "capability": (None if capability is None else list(capability)),
            }
        )
    return {
        "default_dtype": str(_safe_call(torch_module, "get_default_dtype")),
        "deterministic_algorithms": bool(
            _safe_call(torch_module, "are_deterministic_algorithms_enabled", False)
        ),
        "float32_matmul_precision": _safe_call(
            torch_module,
            "get_float32_matmul_precision",
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
        "cuda_devices": cuda_devices,
        "cuda_runtime_version": getattr(version, "cuda", None),
        "hip_runtime_version": getattr(version, "hip", None),
        "torch_git_version": getattr(version, "git_version", None),
        "torch_debug_build": getattr(version, "debug", None),
        "mps_available": bool(_safe_call(mps, "is_available", False)),
        "mps_built": bool(_safe_call(mps, "is_built", False)),
    }


def _python_runtime_payload() -> dict[str, object]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "cache_tag": getattr(sys.implementation, "cache_tag", None),
        "byteorder": sys.byteorder,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libc": libc_name or None,
        "libc_version": libc_version or None,
    }


def _spiraltorch_runtime(
    version: object | None,
    build_fingerprint: object | None,
) -> dict[str, object]:
    if version is None or build_fingerprint is None:
        try:
            spiraltorch = importlib.import_module("spiraltorch")
        except Exception:
            spiraltorch = None
        if version is None and spiraltorch is not None:
            version = getattr(spiraltorch, "__version__", None)
        if build_fingerprint is None and spiraltorch is not None:
            build_fingerprint = getattr(spiraltorch, "BUILD_FINGERPRINT", None)
    return {
        "version": None if version is None else str(version),
        "build_fingerprint": (
            None if build_fingerprint is None else str(build_fingerprint)
        ),
    }


def hf_finetune_execution_identity_report(
    runtime_preflight: Mapping[str, object],
    *,
    spiraltorch_version: object | None = None,
    spiraltorch_build_fingerprint: object | None = None,
    spiraltorch_distribution_fingerprint: Mapping[str, object] | None = None,
    torch_module: Any = None,
    environment: Mapping[str, str] | None = None,
    python_runtime: Mapping[str, object] | None = None,
    torch_capabilities: Mapping[str, object] | None = None,
    distribution_fingerprints: Mapping[str, object] | None = None,
    expected_identity_id: str | None = None,
    phase: str = "pre_model_load",
) -> dict[str, object]:
    """Fingerprint the software, platform, and accelerator basis of one FT run."""

    expected_id = _validated_identity_id(expected_identity_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")
    errors: list[str] = []
    packages, package_errors = _runtime_import_rows(runtime_preflight)
    errors.extend(package_errors)
    distributions, distribution_errors = _distribution_rows(
        packages,
        distribution_fingerprints,
    )
    errors.extend(distribution_errors)
    devices, device_errors = _device_rows(runtime_preflight)
    errors.extend(device_errors)
    spiral = _spiraltorch_runtime(
        spiraltorch_version,
        spiraltorch_build_fingerprint,
    )
    if not spiral.get("version"):
        errors.append("SpiralTorch version is missing")
    spiral_version = str(spiral.get("version") or "")
    if isinstance(spiraltorch_distribution_fingerprint, Mapping):
        spiral_distribution = dict(spiraltorch_distribution_fingerprint)
        spiral_distribution.setdefault("module", "spiraltorch")
        spiral_distribution.setdefault("distribution", "spiraltorch")
        spiral_distribution.setdefault("version", spiral_version)
        spiral_distribution.setdefault("status", "ready")
    else:
        spiral_distribution = _distribution_fingerprint(
            "spiraltorch",
            spiral_version,
        )
    if spiral_distribution.get("status") != "ready":
        errors.append(
            "SpiralTorch distribution is not content-addressable: "
            f"{spiral_distribution.get('status')}"
        )
    if spiral_distribution.get("version") != spiral_version:
        errors.append("SpiralTorch distribution version does not match")
    if not spiral_distribution.get("record_sha256"):
        errors.append("SpiralTorch distribution RECORD hash is missing")
    spiral_identity = {
        "version": spiral.get("version"),
        "distribution": spiral_distribution.get("distribution") or "spiraltorch",
        "record_sha256": spiral_distribution.get("record_sha256"),
        "record_file_count": spiral_distribution.get("record_file_count"),
    }
    resolved_torch_capabilities = (
        dict(torch_capabilities)
        if isinstance(torch_capabilities, Mapping)
        else _torch_capability_payload(torch_module)
    )
    if any(row.get("module") == "torch" for row in packages) and not isinstance(
        resolved_torch_capabilities,
        Mapping,
    ):
        errors.append("Torch capability evidence is missing")
    source_environment = os.environ if environment is None else environment
    selected_environment = {
        key: (
            None
            if source_environment.get(key) is None
            else str(source_environment.get(key))
        )
        for key in _EXECUTION_ENVIRONMENT_KEYS
    }
    identity_payload = {
        "schema": _HF_FINETUNE_EXECUTION_BUNDLE_SCHEMA,
        "spiraltorch": spiral_identity,
        "python_runtime": (
            dict(python_runtime)
            if isinstance(python_runtime, Mapping)
            else _python_runtime_payload()
        ),
        "distributions": distributions,
        "runtime_devices": devices,
        "torch_capabilities": resolved_torch_capabilities,
        "environment": selected_environment,
    }
    observed_id = (
        None
        if errors
        else f"sha256:{_sha256_bytes(_canonical_json_bytes(identity_payload))}"
    )
    if expected_id is not None and observed_id != expected_id:
        errors.append("HF fine-tune execution identity does not match expected id")
    if expected_id is not None and errors:
        status = "blocked"
    elif errors:
        status = "evidence_incomplete"
    else:
        status = "ready"
    return {
        "row_type": "hf_finetune_execution_identity",
        "schema": HF_FINETUNE_EXECUTION_IDENTITY_SCHEMA,
        "status": status,
        "phase": resolved_phase,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "expected_identity_id": expected_id,
        "observed_identity_id": observed_id,
        "expected_identity_verified": (
            None if expected_id is None else observed_id == expected_id and not errors
        ),
        "identity_verified": status == "ready",
        "path_independent": True,
        "spiraltorch": {
            **spiral_identity,
            "runtime_build_fingerprint": spiral.get("build_fingerprint"),
        },
        "python_runtime": identity_payload["python_runtime"],
        "distributions": distributions,
        "distribution_count": len(distributions),
        "runtime_devices": devices,
        "runtime_device_count": len(devices),
        "torch_capabilities": resolved_torch_capabilities,
        "environment": selected_environment,
        "identity_payload": identity_payload if observed_id is not None else None,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_finetune_execution_identity_lines(
    report: Mapping[str, object],
) -> list[str]:
    spiral = report.get("spiraltorch")
    spiral_payload = dict(spiral) if isinstance(spiral, Mapping) else {}
    python = report.get("python_runtime")
    python_payload = dict(python) if isinstance(python, Mapping) else {}
    return [
        "hf_finetune_execution_identity "
        f"status={report.get('status')} "
        f"phase={report.get('phase')} "
        f"verified={report.get('identity_verified')} "
        f"observed={report.get('observed_identity_id')} "
        f"expected={report.get('expected_identity_id')} "
        f"spiraltorch={spiral_payload.get('version')} "
        f"wheel_record={spiral_payload.get('record_sha256')} "
        f"python={python_payload.get('python_version')} "
        f"system={python_payload.get('system')} "
        f"machine={python_payload.get('machine')} "
        f"distributions={report.get('distribution_count')} "
        f"devices={report.get('runtime_device_count')} "
        f"errors={report.get('error_count')}"
    ]
