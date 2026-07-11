"""Content-addressed local input contracts for Hugging Face fine-tuning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "HF_FINETUNE_INPUT_IDENTITY_SCHEMA",
    "hf_finetune_input_identity_lines",
    "hf_finetune_input_identity_report",
]


HF_FINETUNE_INPUT_IDENTITY_SCHEMA = "spiraltorch.hf_finetune_input_identity.v1"
_HF_FINETUNE_INPUT_BUNDLE_SCHEMA = "spiraltorch.hf_finetune_input_bundle.v1"
_HF_FINETUNE_DIRECTORY_IDENTITY_SCHEMA = (
    "spiraltorch.hf_finetune_directory_identity.v1"
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


def _validated_input_id(value: object | None) -> str | None:
    if value is None:
        return None
    input_id = str(value).strip()
    digest = input_id.removeprefix("sha256:")
    if (
        not input_id.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "expected_input_id must be a lowercase sha256:<64 hex> input id"
        )
    return input_id


def _file_stat_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))


def _stable_file_identity(
    path: Path,
) -> tuple[dict[str, object], tuple[int, int, int]]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise RuntimeError(f"input file changed while hashing: {path}")
    return {
        "content_sha256": digest.hexdigest(),
        "size_bytes": int(after.st_size),
    }, (int(after.st_size), int(after.st_mtime_ns), int(after.st_ino))


def _directory_files(path: Path) -> list[Path]:
    return sorted(
        (entry for entry in path.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )


def _stable_directory_identity(path: Path) -> dict[str, object]:
    before = _directory_files(path)
    rows: list[dict[str, object]] = []
    signatures: list[tuple[int, int, int]] = []
    for entry in before:
        file_identity, signature = _stable_file_identity(entry)
        signatures.append(signature)
        rows.append(
            {
                "relative_path": entry.relative_to(path).as_posix(),
                "content_sha256": file_identity["content_sha256"],
                "size_bytes": file_identity["size_bytes"],
                "is_symlink": entry.is_symlink(),
            }
        )
    after = _directory_files(path)
    before_names = [entry.relative_to(path).as_posix() for entry in before]
    after_names = [entry.relative_to(path).as_posix() for entry in after]
    if before_names != after_names:
        raise RuntimeError(f"input directory changed while hashing: {path}")
    if any(
        _file_stat_signature(entry) != signature
        for entry, signature in zip(after, signatures)
    ):
        raise RuntimeError(f"input directory changed while hashing: {path}")
    identity_rows = [
        {
            "relative_path": row["relative_path"],
            "content_sha256": row["content_sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in rows
    ]
    payload = {
        "schema": _HF_FINETUNE_DIRECTORY_IDENTITY_SCHEMA,
        "files": identity_rows,
    }
    return {
        "content_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "file_count": len(rows),
        "files": rows,
    }


def _input_specs(
    *,
    model_configs: str | Path | None,
    train_files: Sequence[str | Path],
    validation_files: Sequence[str | Path],
    inference_distortion_sweep_report: str | Path | None,
    inference_distortion_probe: str | Path | None,
    resume_from_checkpoint: str | Path | None,
) -> list[tuple[str, int, Path, str]]:
    specs: list[tuple[str, int, Path, str]] = []
    if model_configs is not None:
        specs.append(("model_configs", 0, Path(model_configs), "file"))
    specs.extend(
        ("train_file", index, Path(value), "file")
        for index, value in enumerate(train_files)
    )
    specs.extend(
        ("validation_file", index, Path(value), "file")
        for index, value in enumerate(validation_files)
    )
    if inference_distortion_sweep_report is not None:
        specs.append(
            (
                "inference_distortion_sweep_report",
                0,
                Path(inference_distortion_sweep_report),
                "file",
            )
        )
    if inference_distortion_probe is not None:
        specs.append(
            (
                "inference_distortion_probe",
                0,
                Path(inference_distortion_probe),
                "file",
            )
        )
    if resume_from_checkpoint is not None:
        specs.append(
            (
                "resume_from_checkpoint",
                0,
                Path(resume_from_checkpoint),
                "directory",
            )
        )
    return specs


def hf_finetune_input_identity_report(
    *,
    model_configs: str | Path | None = None,
    train_files: Sequence[str | Path] = (),
    validation_files: Sequence[str | Path] = (),
    inference_distortion_sweep_report: str | Path | None = None,
    inference_distortion_probe: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    expected_input_id: str | None = None,
    phase: str = "preflight",
) -> dict[str, object]:
    """Fingerprint local FT inputs without binding their absolute locations."""

    expected_id = _validated_input_id(expected_input_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")
    specs = _input_specs(
        model_configs=model_configs,
        train_files=train_files,
        validation_files=validation_files,
        inference_distortion_sweep_report=inference_distortion_sweep_report,
        inference_distortion_probe=inference_distortion_probe,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for role, ordinal, source, expected_kind in specs:
        resolved = source.expanduser().resolve()
        row: dict[str, object] = {
            "role": role,
            "ordinal": ordinal,
            "source_path": str(source),
            "resolved_path": str(resolved),
            "expected_kind": expected_kind,
            "status": "blocked",
        }
        try:
            if expected_kind == "file":
                if not resolved.is_file():
                    raise FileNotFoundError(f"local input file does not exist: {source}")
                identity, _ = _stable_file_identity(resolved)
                row.update(identity)
                row["file_count"] = 1
            else:
                if not resolved.is_dir():
                    raise FileNotFoundError(
                        f"local input directory does not exist: {source}"
                    )
                row.update(_stable_directory_identity(resolved))
            row["status"] = "ready"
        except (OSError, RuntimeError) as exc:
            message = f"{role}[{ordinal}]: {exc}"
            row["error"] = message
            errors.append(message)
        rows.append(row)

    identity_inputs = [
        {
            "role": row["role"],
            "ordinal": row["ordinal"],
            "kind": row["expected_kind"],
            "content_sha256": row.get("content_sha256"),
            "size_bytes": row.get("size_bytes"),
            "file_count": row.get("file_count"),
        }
        for row in rows
        if row.get("status") == "ready"
    ]
    identity_payload = {
        "schema": _HF_FINETUNE_INPUT_BUNDLE_SCHEMA,
        "inputs": identity_inputs,
    }
    observed_id = (
        None
        if errors or not rows
        else f"sha256:{_sha256_bytes(_canonical_json_bytes(identity_payload))}"
    )
    if expected_id is not None and not rows:
        errors.append("expected input identity has no local inputs to verify")
    elif expected_id is not None and observed_id != expected_id:
        errors.append("fine-tune input fingerprint does not match expected input id")
    if errors:
        status = "blocked"
    elif not rows:
        status = "not_applicable"
    else:
        status = "ready"
    return {
        "row_type": "hf_finetune_input_identity",
        "schema": HF_FINETUNE_INPUT_IDENTITY_SCHEMA,
        "status": status,
        "phase": resolved_phase,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "observed_input_id": observed_id,
        "expected_input_id": expected_id,
        "expected_identity_verified": (
            None if expected_id is None else observed_id == expected_id and not errors
        ),
        "identity_verified": False if errors else (None if not rows else True),
        "identity_applicable": bool(rows),
        "path_independent": True,
        "input_count": len(rows),
        "ready_input_count": sum(row.get("status") == "ready" for row in rows),
        "file_count": sum(int(row.get("file_count") or 0) for row in rows),
        "total_bytes": sum(int(row.get("size_bytes") or 0) for row in rows),
        "inputs": rows,
        "identity_payload": identity_payload if rows else None,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_finetune_input_identity_lines(
    report_or_inputs: Mapping[str, object] | None = None,
    **kwargs: object,
) -> list[str]:
    report = (
        dict(report_or_inputs)
        if isinstance(report_or_inputs, Mapping)
        else hf_finetune_input_identity_report(**kwargs)
    )
    return [
        (
            "hf_finetune_input_identity "
            f"status={report.get('status')} "
            f"phase={report.get('phase')} "
            f"verified={report.get('identity_verified')} "
            f"observed={report.get('observed_input_id')} "
            f"expected={report.get('expected_input_id')} "
            f"inputs={report.get('ready_input_count')}/{report.get('input_count')} "
            f"files={report.get('file_count')} "
            f"bytes={report.get('total_bytes')} "
            f"errors={report.get('error_count')}"
        )
    ]
