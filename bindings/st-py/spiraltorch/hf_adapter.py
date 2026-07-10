"""Lineage and promotion contracts for Hugging Face PEFT adapters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hf_ft import summarize_hf_finetune_run_card
from .hf_peft import hf_causal_lm_artifact_report

__all__ = [
    "HF_ADAPTER_LINEAGE_FILENAME",
    "HF_ADAPTER_LINEAGE_SCHEMA",
    "HF_ADAPTER_PROMOTION_FILENAME",
    "HF_ADAPTER_PROMOTION_SCHEMA",
    "hf_adapter_fingerprint",
    "hf_adapter_lineage_lines",
    "hf_adapter_lineage_report",
    "hf_adapter_promotion_lines",
    "hf_adapter_promotion_report",
    "load_hf_adapter_lineage",
    "load_hf_adapter_promotion",
    "write_hf_adapter_lineage",
    "write_hf_adapter_promotion",
]


HF_ADAPTER_LINEAGE_SCHEMA = "spiraltorch.hf_adapter_lineage.v1"
HF_ADAPTER_LINEAGE_FILENAME = "spiraltorch-hf-adapter-lineage.json"
HF_ADAPTER_PROMOTION_SCHEMA = "spiraltorch.hf_adapter_promotion.v1"
HF_ADAPTER_PROMOTION_FILENAME = "spiraltorch-hf-adapter-promotion.json"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(
    value: Mapping[str, object] | str | Path,
) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return dict(payload), str(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _artifact_directory(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ValueError(f"adapter directory does not exist: {path}")
    return path.resolve()


def _adapter_config(report: Mapping[str, object]) -> dict[str, object]:
    config = report.get("adapter_config")
    return dict(config) if isinstance(config, Mapping) else {}


def _adapter_weight_paths(
    directory: Path,
    artifact: Mapping[str, object],
) -> list[tuple[str, Path]]:
    names = {str(name) for name in artifact.get("adapter_weight_files", [])}
    for index_name in list(names):
        if not index_name.endswith(".index.json"):
            continue
        index_payload, _ = _json_mapping(directory / index_name)
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"adapter weight index has no weight_map: {index_name}")
        for raw_name in weight_map.values():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(
                    f"adapter weight index has an invalid shard: {index_name}"
                )
            names.add(raw_name)
    paths: list[tuple[str, Path]] = []
    for name in sorted(names):
        path = (directory / name).resolve()
        try:
            relative = path.relative_to(directory)
        except ValueError as exc:
            raise ValueError(
                f"adapter weight resolves outside its directory: {name}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"adapter weight file does not exist: {name}")
        paths.append((relative.as_posix(), path))
    return paths


def hf_adapter_fingerprint(adapter: str | Path) -> dict[str, object]:
    """Return a path-independent SHA-256 identity for one local PEFT adapter."""

    directory = _artifact_directory(adapter)
    artifact = hf_causal_lm_artifact_report(
        directory,
        artifact_kind="peft_adapter",
    )
    if artifact.get("status") != "ready":
        errors = "; ".join(str(item) for item in artifact.get("errors", []))
        raise ValueError(f"invalid PEFT adapter: {errors}")
    config_path = directory / "adapter_config.json"
    config_sha256 = _sha256_file(config_path)
    weight_rows: list[dict[str, object]] = []
    for name, path in _adapter_weight_paths(directory, artifact):
        weight_rows.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    identity_payload = {
        "schema": "spiraltorch.hf_adapter_identity.v1",
        "adapter_config_sha256": config_sha256,
        "weights": [
            {"name": row["name"], "sha256": row["sha256"]} for row in weight_rows
        ],
    }
    config = _adapter_config(artifact)
    digest = _sha256_bytes(_canonical_json_bytes(identity_payload))
    return {
        "row_type": "hf_adapter_fingerprint",
        "status": "ready",
        "adapter_path": str(directory),
        "adapter_id": f"sha256:{digest}",
        "adapter_sha256": digest,
        "adapter_config_sha256": config_sha256,
        "adapter_weight_files": weight_rows,
        "adapter_weight_file_count": len(weight_rows),
        "adapter_weight_bytes": sum(int(row["size_bytes"]) for row in weight_rows),
        "base_model_name_or_path": artifact.get("base_model_name_or_path"),
        "base_model_revision": artifact.get("base_model_revision"),
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "rank": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules"),
        "identity_payload": identity_payload,
    }


def _manifest_path(value: str | Path, filename: str) -> Path:
    path = Path(value)
    return path / filename if path.is_dir() else path


def load_hf_adapter_lineage(value: str | Path) -> dict[str, object]:
    path = _manifest_path(value, HF_ADAPTER_LINEAGE_FILENAME)
    payload, _ = _json_mapping(path)
    if payload.get("schema") != HF_ADAPTER_LINEAGE_SCHEMA:
        raise ValueError(
            f"unsupported HF adapter lineage schema: {payload.get('schema')}"
        )
    payload["manifest_path"] = str(path.resolve())
    return payload


def load_hf_adapter_promotion(value: str | Path) -> dict[str, object]:
    path = _manifest_path(value, HF_ADAPTER_PROMOTION_FILENAME)
    payload, _ = _json_mapping(path)
    if payload.get("schema") != HF_ADAPTER_PROMOTION_SCHEMA:
        raise ValueError(
            f"unsupported HF adapter promotion schema: {payload.get('schema')}"
        )
    payload["report_path"] = str(path.resolve())
    return payload


def _run_card_payload(
    value: Mapping[str, object] | str | Path | None,
) -> tuple[dict[str, Any], str | None]:
    if value is None:
        return {}, None
    payload, source = _json_mapping(value)
    payload.pop("adapter_lineage", None)
    payload.pop("adapter_promotion", None)
    return payload, source


def _canonical_run_card_row_type(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith("hf_gpt2_finetune_"):
        return "hf_finetune_" + value.removeprefix("hf_gpt2_finetune_")
    if value.startswith("hf_gpt2_ft_"):
        return "hf_ft_" + value.removeprefix("hf_gpt2_ft_")
    return value


def _canonical_run_card_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (
                _canonical_run_card_row_type(item)
                if key == "row_type"
                else _canonical_run_card_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_run_card_payload(item) for item in value]
    return value


def _run_card_sha256(value: Mapping[str, object] | str | Path | None) -> str | None:
    payload, _ = _run_card_payload(value)
    canonical = _canonical_run_card_payload(payload)
    return _sha256_bytes(_canonical_json_bytes(canonical)) if payload else None


def hf_adapter_lineage_report(
    adapter: str | Path,
    *,
    parent_adapter: str | Path | None = None,
    run_card: Mapping[str, object] | str | Path | None = None,
    run_card_path: str | Path | None = None,
) -> dict[str, object]:
    """Build one adapter lineage node from local artifacts and run provenance."""

    current = hf_adapter_fingerprint(adapter)
    parent = None
    parent_lineage = None
    errors: list[str] = []
    if parent_adapter is not None:
        parent = hf_adapter_fingerprint(parent_adapter)
        if current.get("adapter_path") == parent.get("adapter_path"):
            errors.append("adapter and parent directories must differ")
        parent_manifest = (
            _artifact_directory(parent_adapter) / HF_ADAPTER_LINEAGE_FILENAME
        )
        if parent_manifest.is_file():
            parent_lineage = load_hf_adapter_lineage(parent_manifest)
            if parent_lineage.get("adapter_id") != parent.get("adapter_id"):
                errors.append(
                    "parent lineage fingerprint does not match parent adapter"
                )
        if current.get("base_model_name_or_path") != parent.get(
            "base_model_name_or_path"
        ):
            errors.append("candidate and parent adapters resolve different base models")

    if parent_lineage is not None:
        parent_depth = int(parent_lineage.get("lineage_depth") or 0)
        lineage_depth = parent_depth + 1
        root_adapter_id = parent_lineage.get("root_adapter_id") or parent.get(
            "adapter_id"
        )
        ancestor_ids = list(parent_lineage.get("ancestor_adapter_ids") or [])
        ancestor_ids.append(str(parent.get("adapter_id")))
    elif parent is not None:
        lineage_depth = 1
        root_adapter_id = parent.get("adapter_id")
        ancestor_ids = [str(parent.get("adapter_id"))]
    else:
        lineage_depth = 0
        root_adapter_id = current.get("adapter_id")
        ancestor_ids = []

    card_payload, detected_card_path = _run_card_payload(run_card)
    start = card_payload.get("finetune_start_report")
    start_report = dict(start) if isinstance(start, Mapping) else {}
    parent_reference = (
        None if parent is None else parent.get("adapter_path")
    ) or start_report.get("adapter_weights_source")
    if parent is None and parent_reference is not None:
        lineage_depth = None
        root_adapter_id = None
    resolved_card_path = (
        str(Path(run_card_path)) if run_card_path is not None else detected_card_path
    )
    return {
        "row_type": "hf_adapter_lineage",
        "schema": HF_ADAPTER_LINEAGE_SCHEMA,
        "status": "invalid" if errors else "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter_path": current.get("adapter_path"),
        "adapter_id": current.get("adapter_id"),
        "adapter_sha256": current.get("adapter_sha256"),
        "adapter_config_sha256": current.get("adapter_config_sha256"),
        "adapter_weight_files": current.get("adapter_weight_files"),
        "adapter_weight_bytes": current.get("adapter_weight_bytes"),
        "base_model_name_or_path": current.get("base_model_name_or_path"),
        "base_model_revision": current.get("base_model_revision"),
        "peft_type": current.get("peft_type"),
        "task_type": current.get("task_type"),
        "rank": current.get("rank"),
        "lora_alpha": current.get("lora_alpha"),
        "target_modules": current.get("target_modules"),
        "lineage_depth": lineage_depth,
        "root_adapter_id": root_adapter_id,
        "parent_adapter_path": None if parent is None else parent.get("adapter_path"),
        "parent_adapter_id": None if parent is None else parent.get("adapter_id"),
        "parent_adapter_reference": parent_reference,
        "parent_fingerprint_verified": parent is not None,
        "weights_changed_from_parent": (
            None
            if parent is None
            else current.get("adapter_id") != parent.get("adapter_id")
        ),
        "parent_lineage_present": parent_lineage is not None,
        "parent_lineage_depth": (
            None if parent_lineage is None else parent_lineage.get("lineage_depth")
        ),
        "ancestor_adapter_ids": ancestor_ids,
        "run_card_path": resolved_card_path,
        "run_card_sha256": _run_card_sha256(card_payload),
        "finetune_start_mode": start_report.get("mode"),
        "trainer_checkpoint_resume": start_report.get("trainer_checkpoint_resume"),
        "weights_only_warm_start": start_report.get("weights_only_warm_start"),
        "errors": errors,
    }


def write_hf_adapter_lineage(
    adapter_or_report: str | Path | Mapping[str, object],
    *,
    parent_adapter: str | Path | None = None,
    run_card: Mapping[str, object] | str | Path | None = None,
    run_card_path: str | Path | None = None,
    out: str | Path | None = None,
) -> dict[str, object]:
    report = (
        dict(adapter_or_report)
        if isinstance(adapter_or_report, Mapping)
        else hf_adapter_lineage_report(
            adapter_or_report,
            parent_adapter=parent_adapter,
            run_card=run_card,
            run_card_path=run_card_path,
        )
    )
    if report.get("status") != "ready":
        errors = "; ".join(str(item) for item in report.get("errors", []))
        raise ValueError(f"cannot write invalid adapter lineage: {errors}")
    adapter_path = Path(str(report["adapter_path"]))
    path = Path(out) if out is not None else adapter_path / HF_ADAPTER_LINEAGE_FILENAME
    report["manifest_path"] = str(path.resolve())
    _atomic_write_json(path, report)
    return report


def hf_adapter_lineage_lines(
    report_or_adapter: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_adapter)
        if isinstance(report_or_adapter, Mapping)
        else load_hf_adapter_lineage(report_or_adapter)
    )
    return [
        (
            "hf_adapter_lineage "
            f"status={report.get('status')} "
            f"depth={report.get('lineage_depth')} "
            f"adapter={report.get('adapter_id')} "
            f"parent={report.get('parent_adapter_id')} "
            f"root={report.get('root_adapter_id')} "
            f"base={report.get('base_model_name_or_path')} "
            f"start={report.get('finetune_start_mode')}"
        )
    ]


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check(
    name: str,
    *,
    passed: bool | None,
    required: bool = True,
    observed: object = None,
    threshold: object = None,
    message: str | None = None,
) -> dict[str, object]:
    if passed is True:
        status = "passed"
    elif passed is False:
        status = "failed"
    else:
        status = "missing" if required else "skipped"
    return {
        "name": name,
        "status": status,
        "required": required,
        "observed": observed,
        "threshold": threshold,
        "message": message,
    }


def hf_adapter_promotion_report(
    candidate_adapter: str | Path,
    run_card: Mapping[str, object] | str | Path,
    *,
    parent_adapter: str | Path | None = None,
    max_eval_loss_regression: float = 0.0,
    require_eval: bool = True,
    require_generation_changed: bool = False,
    require_weight_change: bool = True,
) -> dict[str, object]:
    """Gate adapter promotion on lineage integrity and before/after FT evidence."""

    regression_limit = float(max_eval_loss_regression)
    if not math.isfinite(regression_limit):
        raise ValueError("max_eval_loss_regression must be finite")
    candidate_path = _artifact_directory(candidate_adapter)
    fingerprint = hf_adapter_fingerprint(candidate_path)
    lineage_path = candidate_path / HF_ADAPTER_LINEAGE_FILENAME
    lineage = load_hf_adapter_lineage(lineage_path) if lineage_path.is_file() else None
    card_payload, card_path = _run_card_payload(run_card)
    summary = summarize_hf_finetune_run_card(card_payload)
    resolved_parent = parent_adapter
    if resolved_parent is None and lineage is not None:
        resolved_parent = lineage.get("parent_adapter_path")
    parent = (
        hf_adapter_fingerprint(str(resolved_parent))
        if resolved_parent is not None and Path(str(resolved_parent)).is_dir()
        else None
    )

    before_loss = _finite_number(summary.get("eval_before_loss"))
    after_loss = _finite_number(summary.get("effective_eval_after_loss"))
    eval_regression = (
        None if before_loss is None or after_loss is None else after_loss - before_loss
    )
    trainer_loss = _finite_number(summary.get("trainer_train_loss"))
    generation_changed = summary.get("generation_continuation_changed")
    checks = [
        _check(
            "lineage_manifest",
            passed=(None if lineage is None else lineage.get("status") == "ready"),
            observed={
                "path": str(lineage_path),
                "status": None if lineage is None else lineage.get("status"),
            },
            message="candidate must carry a ready lineage manifest",
        ),
        _check(
            "candidate_fingerprint",
            passed=(
                None
                if lineage is None
                else lineage.get("adapter_id") == fingerprint.get("adapter_id")
            ),
            observed=fingerprint.get("adapter_id"),
            threshold=None if lineage is None else lineage.get("adapter_id"),
        ),
        _check(
            "parent_fingerprint",
            passed=(
                None
                if lineage is None or lineage.get("parent_adapter_id") is None
                else parent is not None
                and parent.get("adapter_id") == lineage.get("parent_adapter_id")
            ),
            required=(
                lineage is not None
                and (
                    lineage.get("parent_adapter_id") is not None
                    or lineage.get("parent_adapter_reference") is not None
                )
            ),
            observed=None if parent is None else parent.get("adapter_id"),
            threshold=None if lineage is None else lineage.get("parent_adapter_id"),
        ),
        _check(
            "weight_change",
            passed=(
                None
                if parent is None
                else fingerprint.get("adapter_id") != parent.get("adapter_id")
            ),
            required=require_weight_change and parent is not None,
            observed=fingerprint.get("adapter_id"),
            threshold=None if parent is None else parent.get("adapter_id"),
        ),
        _check(
            "run_card_digest",
            passed=(
                None
                if lineage is None or lineage.get("run_card_sha256") is None
                else lineage.get("run_card_sha256") == _run_card_sha256(card_payload)
            ),
            observed=_run_card_sha256(card_payload),
            threshold=None if lineage is None else lineage.get("run_card_sha256"),
        ),
        _check(
            "training_completed",
            passed=(
                not summary.get("failure_stage")
                and summary.get("model_saved") is True
                and summary.get("adapter_saved") is True
            ),
            observed={
                "failure_stage": summary.get("failure_stage"),
                "model_saved": summary.get("model_saved"),
                "adapter_saved": summary.get("adapter_saved"),
            },
        ),
        _check(
            "trainer_loss_finite",
            passed=trainer_loss is not None,
            observed=trainer_loss,
        ),
        _check(
            "eval_evidence",
            passed=(None if before_loss is None or after_loss is None else True),
            required=require_eval,
            observed={"before": before_loss, "after": after_loss},
        ),
        _check(
            "eval_loss_regression",
            passed=(
                None if eval_regression is None else eval_regression <= regression_limit
            ),
            required=require_eval,
            observed=eval_regression,
            threshold=regression_limit,
        ),
        _check(
            "generation_changed",
            passed=(
                generation_changed is True if generation_changed is not None else None
            ),
            required=require_generation_changed,
            observed=generation_changed,
        ),
    ]
    required_checks = [row for row in checks if row["required"]]
    failed = [row for row in required_checks if row["status"] == "failed"]
    missing = [row for row in required_checks if row["status"] == "missing"]
    promotion_ready = not failed and not missing
    if promotion_ready:
        status = "ready"
        recommendation = "promote_candidate"
    elif failed:
        status = "blocked"
        recommendation = "keep_parent"
    else:
        status = "needs_evidence"
        recommendation = "run_before_after_evaluation"
    return {
        "row_type": "hf_adapter_promotion",
        "schema": HF_ADAPTER_PROMOTION_SCHEMA,
        "status": status,
        "promotion_ready": promotion_ready,
        "recommendation": recommendation,
        "candidate_adapter_path": str(candidate_path),
        "candidate_adapter_id": fingerprint.get("adapter_id"),
        "parent_adapter_path": None if parent is None else parent.get("adapter_path"),
        "parent_adapter_id": None if parent is None else parent.get("adapter_id"),
        "lineage_manifest_path": str(lineage_path),
        "lineage_depth": None if lineage is None else lineage.get("lineage_depth"),
        "run_card_path": card_path,
        "run_card_sha256": _run_card_sha256(card_payload),
        "eval_before_loss": before_loss,
        "eval_after_loss": after_loss,
        "eval_loss_regression": eval_regression,
        "max_eval_loss_regression": regression_limit,
        "generation_changed": generation_changed,
        "trainer_train_loss": trainer_loss,
        "require_eval": bool(require_eval),
        "require_generation_changed": bool(require_generation_changed),
        "require_weight_change": bool(require_weight_change),
        "check_count": len(checks),
        "required_check_count": len(required_checks),
        "passed_check_count": sum(row["status"] == "passed" for row in checks),
        "failed_checks": [row["name"] for row in failed],
        "missing_checks": [row["name"] for row in missing],
        "checks": checks,
        "run_summary": summary,
    }


def write_hf_adapter_promotion(
    report_or_candidate: Mapping[str, object] | str | Path,
    run_card: Mapping[str, object] | str | Path | None = None,
    *,
    parent_adapter: str | Path | None = None,
    max_eval_loss_regression: float = 0.0,
    require_eval: bool = True,
    require_generation_changed: bool = False,
    require_weight_change: bool = True,
    out: str | Path | None = None,
) -> dict[str, object]:
    if isinstance(report_or_candidate, Mapping):
        report = dict(report_or_candidate)
    else:
        if run_card is None:
            raise ValueError("run_card is required when building a promotion report")
        report = hf_adapter_promotion_report(
            report_or_candidate,
            run_card,
            parent_adapter=parent_adapter,
            max_eval_loss_regression=max_eval_loss_regression,
            require_eval=require_eval,
            require_generation_changed=require_generation_changed,
            require_weight_change=require_weight_change,
        )
    candidate = Path(str(report["candidate_adapter_path"]))
    path = Path(out) if out is not None else candidate / HF_ADAPTER_PROMOTION_FILENAME
    report["report_path"] = str(path.resolve())
    _atomic_write_json(path, report)
    return report


def hf_adapter_promotion_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_promotion(report_or_path)
    )
    lines = [
        (
            "hf_adapter_promotion "
            f"status={report.get('status')} "
            f"ready={report.get('promotion_ready')} "
            f"candidate={report.get('candidate_adapter_id')} "
            f"parent={report.get('parent_adapter_id')} "
            f"eval_before={report.get('eval_before_loss')} "
            f"eval_after={report.get('eval_after_loss')} "
            f"eval_regression={report.get('eval_loss_regression')} "
            f"recommendation={report.get('recommendation')}"
        )
    ]
    for raw_check in report.get("checks", []):
        if not isinstance(raw_check, Mapping):
            continue
        lines.append(
            "hf_adapter_promotion_check "
            f"name={raw_check.get('name')} "
            f"status={raw_check.get('status')} "
            f"required={raw_check.get('required')} "
            f"observed={raw_check.get('observed')} "
            f"threshold={raw_check.get('threshold')}"
        )
    return lines
