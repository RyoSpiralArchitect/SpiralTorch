"""Lineage and promotion contracts for Hugging Face PEFT adapters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import tempfile
from collections.abc import Mapping, Sequence
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
    "HF_ADAPTER_PROMOTION_CHAIN_FILENAME",
    "HF_ADAPTER_PROMOTION_CHAIN_SCHEMA",
    "HF_ADAPTER_CONTINUATION_POLICY_FILENAME",
    "HF_ADAPTER_CONTINUATION_POLICY_SCHEMA",
    "hf_adapter_continuation_policy_lines",
    "hf_adapter_continuation_policy_report",
    "hf_adapter_fingerprint",
    "hf_adapter_lineage_lines",
    "hf_adapter_lineage_report",
    "hf_adapter_promotion_chain_lines",
    "hf_adapter_promotion_chain_report",
    "hf_adapter_promotion_lines",
    "hf_adapter_promotion_report",
    "load_hf_adapter_lineage",
    "load_hf_adapter_continuation_policy",
    "load_hf_adapter_promotion_chain",
    "load_hf_adapter_promotion",
    "write_hf_adapter_lineage",
    "write_hf_adapter_continuation_policy",
    "write_hf_adapter_promotion_chain",
    "write_hf_adapter_promotion",
]


HF_ADAPTER_LINEAGE_SCHEMA = "spiraltorch.hf_adapter_lineage.v1"
HF_ADAPTER_LINEAGE_FILENAME = "spiraltorch-hf-adapter-lineage.json"
HF_ADAPTER_PROMOTION_SCHEMA = "spiraltorch.hf_adapter_promotion.v1"
HF_ADAPTER_PROMOTION_FILENAME = "spiraltorch-hf-adapter-promotion.json"
HF_ADAPTER_PROMOTION_CHAIN_SCHEMA = "spiraltorch.hf_adapter_promotion_chain.v1"
HF_ADAPTER_PROMOTION_CHAIN_FILENAME = "spiraltorch-hf-adapter-promotion-chain.json"
HF_ADAPTER_CONTINUATION_POLICY_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_policy.v1"
)
HF_ADAPTER_CONTINUATION_POLICY_FILENAME = (
    "spiraltorch-hf-adapter-continuation-policy.json"
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


def _integer_number(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


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


def _artifact_probe_evidence(
    card_payload: Mapping[str, object],
    candidate_path: Path,
) -> tuple[dict[str, object], dict[str, object], bool | None, float | None]:
    raw_probe = card_payload.get("adapter_artifact_probe")
    if not isinstance(raw_probe, Mapping):
        return {}, {}, None, None
    probe = dict(raw_probe)
    raw_artifact = probe.get("artifact")
    artifact = dict(raw_artifact) if isinstance(raw_artifact, Mapping) else {}
    source = artifact.get("artifact_source")
    source_matches = False
    if source is not None:
        try:
            source_matches = (
                Path(str(source)).expanduser().resolve() == candidate_path.resolve()
            )
        except (OSError, RuntimeError, ValueError):
            source_matches = False
    new_token_count = _finite_number(probe.get("new_token_count"))
    return probe, artifact, source_matches, new_token_count


def hf_adapter_promotion_report(
    candidate_adapter: str | Path,
    run_card: Mapping[str, object] | str | Path,
    *,
    parent_adapter: str | Path | None = None,
    max_eval_loss_regression: float = 0.0,
    require_eval: bool = True,
    require_generation_changed: bool = False,
    require_weight_change: bool = True,
    require_artifact_probe: bool = False,
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
    artifact_probe, probed_artifact, probe_source_matches, probe_new_tokens = (
        _artifact_probe_evidence(card_payload, candidate_path)
    )
    raw_probe_generation = artifact_probe.get("generation")
    probe_generation = (
        dict(raw_probe_generation)
        if isinstance(raw_probe_generation, Mapping)
        else {}
    )
    raw_probe_process = artifact_probe.get("process_isolation")
    probe_process = (
        dict(raw_probe_process) if isinstance(raw_probe_process, Mapping) else {}
    )
    artifact_probe_present = bool(artifact_probe)
    artifact_reload_passed = (
        None
        if not artifact_probe_present
        else artifact_probe.get("status") == "ready"
        and probed_artifact.get("artifact_kind") == "peft_adapter"
        and probed_artifact.get("adapter_loaded") is True
        and probe_source_matches is True
        and artifact_probe.get("local_files_only") is True
    )
    artifact_generation_passed = (
        None
        if not artifact_probe_present
        else artifact_probe.get("status") == "ready"
        and probe_new_tokens is not None
        and probe_new_tokens > 0.0
        and probe_generation.get("do_sample") is False
    )
    probe_pid = _integer_number(probe_process.get("pid"))
    probe_parent_pid = _integer_number(probe_process.get("parent_pid"))
    worker_pid = _integer_number(artifact_probe.get("worker_pid"))
    artifact_process_isolation_passed = (
        None
        if not artifact_probe_present or not probe_process
        else probe_process.get("status") == "ready"
        and probe_process.get("fresh_process") is True
        and probe_process.get("runner_kind") == "python_module"
        and probe_process.get("worker_module")
        == "spiraltorch.hf_artifact_probe_worker"
        and probe_pid is not None
        and probe_pid > 0.0
        and probe_parent_pid is not None
        and probe_parent_pid > 0.0
        and probe_pid != probe_parent_pid
        and worker_pid == probe_pid
        and probe_process.get("worker_pid_matches") is True
        and _integer_number(probe_process.get("exit_code")) == 0
        and probe_process.get("timed_out") is False
    )
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
        _check(
            "artifact_reload",
            passed=artifact_reload_passed,
            required=require_artifact_probe,
            observed={
                "status": artifact_probe.get("status"),
                "artifact_kind": probed_artifact.get("artifact_kind"),
                "artifact_source": probed_artifact.get("artifact_source"),
                "adapter_loaded": probed_artifact.get("adapter_loaded"),
                "tokenizer_source_kind": probed_artifact.get(
                    "tokenizer_source_kind"
                ),
                "candidate_matches": probe_source_matches,
                "local_files_only": artifact_probe.get("local_files_only"),
            },
            threshold={
                "status": "ready",
                "artifact_kind": "peft_adapter",
                "artifact_source": str(candidate_path),
                "adapter_loaded": True,
                "local_files_only": True,
            },
            message=(
                "candidate must survive a fresh local-only PEFT artifact reload"
            ),
        ),
        _check(
            "artifact_generation",
            passed=artifact_generation_passed,
            required=require_artifact_probe,
            observed={
                "new_token_count": probe_new_tokens,
                "generated_text_changed": artifact_probe.get(
                    "generated_text_changed"
                ),
                "do_sample": probe_generation.get("do_sample"),
                "device": artifact_probe.get("device"),
            },
            threshold={"new_token_count_min": 1, "do_sample": False},
            message=(
                "freshly reloaded candidate must complete deterministic bounded "
                "generation"
            ),
        ),
        _check(
            "artifact_process_isolation",
            passed=artifact_process_isolation_passed,
            required=require_artifact_probe,
            observed={
                "status": probe_process.get("status"),
                "fresh_process": probe_process.get("fresh_process"),
                "runner_kind": probe_process.get("runner_kind"),
                "worker_module": probe_process.get("worker_module"),
                "parent_pid": probe_parent_pid,
                "pid": probe_pid,
                "worker_pid": worker_pid,
                "worker_pid_matches": probe_process.get("worker_pid_matches"),
                "exit_code": probe_process.get("exit_code"),
                "timed_out": probe_process.get("timed_out"),
            },
            threshold={
                "status": "ready",
                "fresh_process": True,
                "runner_kind": "python_module",
                "worker_module": "spiraltorch.hf_artifact_probe_worker",
                "pid_differs_from_parent": True,
                "worker_pid_matches": True,
                "exit_code": 0,
                "timed_out": False,
            },
            message="artifact qualification must complete in a fresh worker process",
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
        recommendation = (
            "run_artifact_reload_probe"
            if any(
                row["name"]
                in {
                    "artifact_reload",
                    "artifact_generation",
                    "artifact_process_isolation",
                }
                for row in missing
            )
            else "run_before_after_evaluation"
        )
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
        "artifact_probe_status": artifact_probe.get("status"),
        "artifact_probe_report_path": artifact_probe.get("report_path"),
        "artifact_probe_device": artifact_probe.get("device"),
        "artifact_probe_tokenizer_source_kind": probed_artifact.get(
            "tokenizer_source_kind"
        ),
        "artifact_probe_new_token_count": probe_new_tokens,
        "artifact_probe_candidate_matches": probe_source_matches,
        "artifact_probe_local_files_only": artifact_probe.get("local_files_only"),
        "artifact_probe_do_sample": probe_generation.get("do_sample"),
        "artifact_probe_process_status": probe_process.get("status"),
        "artifact_probe_process_fresh": probe_process.get("fresh_process"),
        "artifact_probe_process_parent_pid": probe_parent_pid,
        "artifact_probe_process_pid": probe_pid,
        "artifact_probe_process_exit_code": _integer_number(
            probe_process.get("exit_code")
        ),
        "artifact_probe_process_timed_out": probe_process.get("timed_out"),
        "artifact_probe_process_duration_seconds": _finite_number(
            probe_process.get("duration_seconds")
        ),
        "trainer_train_loss": trainer_loss,
        "require_eval": bool(require_eval),
        "require_generation_changed": bool(require_generation_changed),
        "require_weight_change": bool(require_weight_change),
        "require_artifact_probe": bool(require_artifact_probe),
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
    require_artifact_probe: bool = False,
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
            require_artifact_probe=require_artifact_probe,
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
            f"artifact_probe={report.get('artifact_probe_status')} "
            f"probe_process={report.get('artifact_probe_process_status')} "
            f"probe_pid={report.get('artifact_probe_process_pid')} "
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


def _chain_issue(
    code: str,
    message: str,
    *,
    severity: str = "error",
    path: str | Path | None = None,
    adapter_id: object = None,
) -> dict[str, object]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "path": None if path is None else str(path),
        "adapter_id": adapter_id,
    }


def _chain_source_paths(
    sources: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(sources, (str, Path)):
        values: Sequence[str | Path] = [sources]
    elif isinstance(sources, Sequence):
        values = sources
    else:
        raise TypeError("adapter chain sources must be paths or a sequence of paths")
    paths = [Path(value).expanduser() for value in values]
    if not paths:
        raise ValueError("at least one adapter chain source is required")
    return paths


def _discover_lineage_manifests(
    sources: str | Path | Sequence[str | Path],
    *,
    recursive: bool,
) -> tuple[list[Path], list[dict[str, object]]]:
    manifests: dict[Path, None] = {}
    issues: list[dict[str, object]] = []
    for source in _chain_source_paths(sources):
        if source.is_file():
            if source.name != HF_ADAPTER_LINEAGE_FILENAME:
                issues.append(
                    _chain_issue(
                        "unsupported_source_file",
                        "chain source files must be adapter lineage manifests",
                        path=source,
                    )
                )
                continue
            manifests[source.resolve()] = None
            continue
        if not source.is_dir():
            issues.append(
                _chain_issue(
                    "missing_source",
                    "adapter chain source does not exist",
                    path=source,
                )
            )
            continue
        before = len(manifests)
        direct = source / HF_ADAPTER_LINEAGE_FILENAME
        if direct.is_file():
            manifests[direct.resolve()] = None
        if recursive:
            for path in source.rglob(HF_ADAPTER_LINEAGE_FILENAME):
                manifests[path.resolve()] = None
        if len(manifests) == before:
            issues.append(
                _chain_issue(
                    "lineage_not_found",
                    "no adapter lineage manifests were found under this source",
                    path=source,
                )
            )

    # Pull local ancestors into the report even when the caller starts at a tip.
    pending = list(manifests)
    cursor = 0
    while cursor < len(pending):
        manifest = pending[cursor]
        cursor += 1
        try:
            payload, _ = _json_mapping(manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        parent_path = payload.get("parent_adapter_path")
        if parent_path is None:
            continue
        parent_manifest = (
            Path(str(parent_path)).expanduser() / HF_ADAPTER_LINEAGE_FILENAME
        )
        if not parent_manifest.is_file():
            continue
        resolved = parent_manifest.resolve()
        if resolved not in manifests:
            manifests[resolved] = None
            pending.append(resolved)
    return sorted(manifests, key=str), issues


def _chain_reference_path(value: object, *, anchor: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (anchor / path).resolve() if not path.is_absolute() else path.resolve()


def _chain_run_card(
    lineage: Mapping[str, object],
    adapter_path: Path,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    candidates: list[Path] = []
    referenced = _chain_reference_path(
        lineage.get("run_card_path"),
        anchor=adapter_path,
    )
    if referenced is not None:
        candidates.append(referenced)
    for filename in (
        "spiraltorch-hf-finetune-run-card.json",
        "spiraltorch-hf-gpt2-ft-run-card.json",
    ):
        candidate = adapter_path / filename
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload, _ = _json_mapping(candidate)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return None, candidate, f"{exc.__class__.__name__}: {exc}"
        return payload, candidate, None
    return None, referenced, None


def _chain_command(value: object) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    command = [str(item) for item in value]
    return command or None


def _chain_command_flag(command: Sequence[object], flag: str) -> str | None:
    values = [str(item) for item in command]
    for index, value in enumerate(values):
        if value == flag and index + 1 < len(values):
            return values[index + 1]
        prefix = f"{flag}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _chain_command_artifacts(
    values: Sequence[Mapping[str, object] | str | Path] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    for value in values or []:
        source_path: Path | None = None
        if isinstance(value, Mapping):
            payload = dict(value)
        else:
            source_path = Path(value).expanduser()
            try:
                payload, _ = _json_mapping(source_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append(
                    _chain_issue(
                        "invalid_command_artifact",
                        f"failed to load command artifact: {exc}",
                        path=source_path,
                    )
                )
                continue
        command = _chain_command(payload.get("command"))
        output_dir = (
            None
            if command is None
            else _chain_command_flag(
                command,
                "--output-dir",
            )
        )
        if command is None or output_dir is None:
            issues.append(
                _chain_issue(
                    "command_artifact_missing_output",
                    "command artifact must contain a command with --output-dir",
                    path=source_path,
                )
            )
            continue
        rows.append(
            {
                "command": command,
                "output_dir": str(Path(output_dir).expanduser().resolve()),
                "source_path": None
                if source_path is None
                else str(source_path.resolve()),
                "run_returncode": payload.get("run_returncode"),
                "preflight_status": payload.get("preflight_status"),
            }
        )
    return rows, issues


def _adapter_promotion_chain_node(manifest_path: Path) -> dict[str, object]:
    adapter_path = manifest_path.parent.resolve()
    issues: list[dict[str, object]] = []
    try:
        lineage = load_hf_adapter_lineage(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "row_type": "hf_adapter_promotion_chain_node",
            "status": "invalid",
            "adapter_path": str(adapter_path),
            "adapter_id": None,
            "lineage_manifest_path": str(manifest_path),
            "issues": [
                _chain_issue(
                    "invalid_lineage_manifest",
                    f"failed to load lineage manifest: {exc}",
                    path=manifest_path,
                )
            ],
        }

    adapter_id = lineage.get("adapter_id")
    if lineage.get("status") != "ready":
        issues.append(
            _chain_issue(
                "lineage_not_ready",
                "lineage manifest status is not ready",
                path=manifest_path,
                adapter_id=adapter_id,
            )
        )
    try:
        fingerprint = hf_adapter_fingerprint(adapter_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        fingerprint = {}
        issues.append(
            _chain_issue(
                "adapter_fingerprint_failed",
                f"failed to fingerprint adapter: {exc}",
                path=adapter_path,
                adapter_id=adapter_id,
            )
        )
    if fingerprint and fingerprint.get("adapter_id") != adapter_id:
        issues.append(
            _chain_issue(
                "adapter_fingerprint_mismatch",
                "adapter weights no longer match the lineage fingerprint",
                path=adapter_path,
                adapter_id=adapter_id,
            )
        )

    stored_adapter_path = _chain_reference_path(
        lineage.get("adapter_path"),
        anchor=adapter_path,
    )
    if stored_adapter_path is not None and stored_adapter_path != adapter_path:
        issues.append(
            _chain_issue(
                "adapter_path_relocated",
                "adapter directory moved after the lineage manifest was written",
                severity="warning",
                path=stored_adapter_path,
                adapter_id=adapter_id,
            )
        )

    run_card, run_card_path, run_card_error = _chain_run_card(lineage, adapter_path)
    if run_card_error is not None:
        issues.append(
            _chain_issue(
                "invalid_run_card",
                run_card_error,
                path=run_card_path,
                adapter_id=adapter_id,
            )
        )
    lineage_card_sha256 = lineage.get("run_card_sha256")
    observed_card_sha256 = None if run_card is None else _run_card_sha256(run_card)
    depth = lineage.get("lineage_depth")
    if depth not in (0, None) and run_card is None:
        issues.append(
            _chain_issue(
                "run_card_missing",
                "non-root lineage node is missing its fine-tune run card",
                path=run_card_path or adapter_path,
                adapter_id=adapter_id,
            )
        )
    if lineage_card_sha256 is not None and observed_card_sha256 != lineage_card_sha256:
        issues.append(
            _chain_issue(
                "run_card_digest_mismatch",
                "run card no longer matches the lineage digest",
                path=run_card_path,
                adapter_id=adapter_id,
            )
        )

    promotion_path = adapter_path / HF_ADAPTER_PROMOTION_FILENAME
    promotion: dict[str, object] | None = None
    if promotion_path.is_file():
        try:
            promotion = load_hf_adapter_promotion(promotion_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                _chain_issue(
                    "invalid_promotion_report",
                    f"failed to load promotion report: {exc}",
                    path=promotion_path,
                    adapter_id=adapter_id,
                )
            )
    elif depth not in (0, None):
        issues.append(
            _chain_issue(
                "promotion_report_missing",
                "non-root lineage node is missing its promotion report",
                path=promotion_path,
                adapter_id=adapter_id,
            )
        )
    if promotion is not None:
        for matches, code, message in (
            (
                promotion.get("candidate_adapter_id") == adapter_id,
                "promotion_candidate_mismatch",
                "promotion candidate does not match the lineage adapter",
            ),
            (
                promotion.get("parent_adapter_id") == lineage.get("parent_adapter_id"),
                "promotion_parent_mismatch",
                "promotion parent does not match the lineage parent",
            ),
            (
                promotion.get("lineage_depth") == depth,
                "promotion_depth_mismatch",
                "promotion depth does not match the lineage depth",
            ),
            (
                promotion.get("run_card_sha256") == lineage_card_sha256,
                "promotion_run_card_mismatch",
                "promotion run-card digest does not match lineage",
            ),
        ):
            if not matches:
                issues.append(
                    _chain_issue(
                        code,
                        message,
                        path=promotion_path,
                        adapter_id=adapter_id,
                    )
                )
        if depth is not None and (
            promotion.get("status") != "ready"
            or promotion.get("promotion_ready") is not True
        ):
            issues.append(
                _chain_issue(
                    "promotion_not_ready",
                    "promotion gate did not approve this lineage node",
                    path=promotion_path,
                    adapter_id=adapter_id,
                )
            )
    promotion_revalidation: dict[str, object] | None = None
    if promotion is not None and run_card is not None and depth is not None:
        parent_path = _chain_reference_path(
            lineage.get("parent_adapter_path"),
            anchor=adapter_path,
        )
        try:
            promotion_revalidation = hf_adapter_promotion_report(
                adapter_path,
                run_card,
                parent_adapter=(
                    parent_path
                    if parent_path is not None and parent_path.is_dir()
                    else None
                ),
                max_eval_loss_regression=float(
                    promotion.get("max_eval_loss_regression") or 0.0
                ),
                require_eval=promotion.get("require_eval") is not False,
                require_generation_changed=(
                    promotion.get("require_generation_changed") is True
                ),
                require_weight_change=(
                    promotion.get("require_weight_change") is not False
                ),
                require_artifact_probe=(
                    promotion.get("require_artifact_probe") is True
                ),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                _chain_issue(
                    "promotion_revalidation_failed",
                    f"failed to re-evaluate promotion gate: {exc}",
                    path=promotion_path,
                    adapter_id=adapter_id,
                )
            )
        else:
            revalidation_fields = (
                "status",
                "promotion_ready",
                "candidate_adapter_id",
                "parent_adapter_id",
                "lineage_depth",
                "run_card_sha256",
                "eval_loss_regression",
                "artifact_probe_status",
                "artifact_probe_new_token_count",
                "artifact_probe_candidate_matches",
                "artifact_probe_tokenizer_source_kind",
                "artifact_probe_local_files_only",
                "artifact_probe_do_sample",
                "artifact_probe_process_status",
                "artifact_probe_process_fresh",
                "artifact_probe_process_parent_pid",
                "artifact_probe_process_pid",
                "artifact_probe_process_exit_code",
                "artifact_probe_process_timed_out",
                "failed_checks",
                "missing_checks",
            )
            mismatched_fields = [
                field
                for field in revalidation_fields
                if promotion.get(field) != promotion_revalidation.get(field)
            ]
            if mismatched_fields:
                issues.append(
                    _chain_issue(
                        "promotion_revalidation_mismatch",
                        "stored promotion differs from live revalidation: "
                        + ",".join(mismatched_fields),
                        path=promotion_path,
                        adapter_id=adapter_id,
                    )
                )
            if promotion_revalidation.get("promotion_ready") is not True:
                issues.append(
                    _chain_issue(
                        "promotion_revalidation_not_ready",
                        "live promotion revalidation did not approve this node",
                        path=promotion_path,
                        adapter_id=adapter_id,
                    )
                )

    launch_command = (
        None if run_card is None else _chain_command(run_card.get("launch_command"))
    )
    return {
        "row_type": "hf_adapter_promotion_chain_node",
        "status": "pending",
        "adapter_path": str(adapter_path),
        "adapter_id": adapter_id,
        "parent_adapter_id": lineage.get("parent_adapter_id"),
        "parent_adapter_path": lineage.get("parent_adapter_path"),
        "root_adapter_id": lineage.get("root_adapter_id"),
        "ancestor_adapter_ids": list(lineage.get("ancestor_adapter_ids") or []),
        "lineage_depth": depth,
        "base_model_name_or_path": lineage.get("base_model_name_or_path"),
        "created_at": lineage.get("created_at"),
        "lineage_status": lineage.get("status"),
        "lineage_manifest_path": str(manifest_path),
        "parent_fingerprint_verified": lineage.get("parent_fingerprint_verified"),
        "weights_changed_from_parent": lineage.get("weights_changed_from_parent"),
        "run_card_path": None if run_card_path is None else str(run_card_path),
        "run_card_sha256": observed_card_sha256,
        "trainer_trace_jsonl": None
        if run_card is None
        else run_card.get("trainer_trace_jsonl"),
        "promotion_status": None if promotion is None else promotion.get("status"),
        "promotion_ready": None
        if promotion is None
        else promotion.get("promotion_ready"),
        "promotion_revalidated_ready": None
        if promotion_revalidation is None
        else promotion_revalidation.get("promotion_ready"),
        "promotion_report_path": str(promotion_path)
        if promotion_path.is_file()
        else None,
        "eval_before_loss": None
        if promotion is None
        else promotion.get("eval_before_loss"),
        "eval_after_loss": None
        if promotion is None
        else promotion.get("eval_after_loss"),
        "eval_loss_regression": None
        if promotion is None
        else promotion.get("eval_loss_regression"),
        "artifact_probe_status": None
        if promotion is None
        else promotion.get("artifact_probe_status"),
        "artifact_probe_report_path": None
        if promotion is None
        else promotion.get("artifact_probe_report_path"),
        "artifact_probe_device": None
        if promotion is None
        else promotion.get("artifact_probe_device"),
        "artifact_probe_tokenizer_source_kind": None
        if promotion is None
        else promotion.get("artifact_probe_tokenizer_source_kind"),
        "artifact_probe_new_token_count": None
        if promotion is None
        else promotion.get("artifact_probe_new_token_count"),
        "artifact_probe_local_files_only": None
        if promotion is None
        else promotion.get("artifact_probe_local_files_only"),
        "artifact_probe_do_sample": None
        if promotion is None
        else promotion.get("artifact_probe_do_sample"),
        "artifact_probe_process_status": None
        if promotion is None
        else promotion.get("artifact_probe_process_status"),
        "artifact_probe_process_fresh": None
        if promotion is None
        else promotion.get("artifact_probe_process_fresh"),
        "artifact_probe_process_parent_pid": None
        if promotion is None
        else promotion.get("artifact_probe_process_parent_pid"),
        "artifact_probe_process_pid": None
        if promotion is None
        else promotion.get("artifact_probe_process_pid"),
        "artifact_probe_process_exit_code": None
        if promotion is None
        else promotion.get("artifact_probe_process_exit_code"),
        "artifact_probe_process_timed_out": None
        if promotion is None
        else promotion.get("artifact_probe_process_timed_out"),
        "launch_command": launch_command,
        "launch_command_display": None
        if launch_command is None
        else shlex.join(launch_command),
        "launch_command_source": "run_card" if launch_command is not None else None,
        "issues": issues,
    }


def _chain_add_node_issue(
    node: dict[str, object],
    code: str,
    message: str,
    *,
    severity: str = "error",
    path: str | Path | None = None,
) -> None:
    issues = node.setdefault("issues", [])
    if isinstance(issues, list):
        issues.append(
            _chain_issue(
                code,
                message,
                severity=severity,
                path=path or node.get("adapter_path"),
                adapter_id=node.get("adapter_id"),
            )
        )


def _chain_inferred_root_node(
    child: Mapping[str, object],
) -> dict[str, object] | None:
    parent_id = child.get("parent_adapter_id")
    parent_path = _chain_reference_path(
        child.get("parent_adapter_path"),
        anchor=Path(str(child.get("adapter_path"))),
    )
    if (
        not isinstance(parent_id, str)
        or parent_path is None
        or not parent_path.is_dir()
    ):
        return None
    try:
        fingerprint = hf_adapter_fingerprint(parent_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if fingerprint.get("adapter_id") != parent_id:
        return None
    return {
        "row_type": "hf_adapter_promotion_chain_node",
        "status": "pending",
        "adapter_path": str(parent_path),
        "adapter_id": parent_id,
        "parent_adapter_id": None,
        "parent_adapter_path": None,
        "root_adapter_id": parent_id,
        "ancestor_adapter_ids": [],
        "lineage_depth": 0,
        "base_model_name_or_path": fingerprint.get("base_model_name_or_path"),
        "created_at": None,
        "lineage_status": "inferred_seed",
        "lineage_manifest_path": None,
        "parent_fingerprint_verified": None,
        "weights_changed_from_parent": None,
        "run_card_path": None,
        "run_card_sha256": None,
        "trainer_trace_jsonl": None,
        "promotion_status": None,
        "promotion_ready": None,
        "promotion_report_path": None,
        "eval_before_loss": None,
        "eval_after_loss": None,
        "eval_loss_regression": None,
        "launch_command": None,
        "launch_command_display": None,
        "launch_command_source": None,
        "issues": [
            _chain_issue(
                "root_lineage_inferred",
                (
                    "seed lineage manifest is absent; root identity was inferred "
                    "from matching local adapter weights"
                ),
                severity="warning",
                path=parent_path,
                adapter_id=parent_id,
            )
        ],
    }


def _chain_depth(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return None
    return depth if depth >= 0 and depth == value else None


def _adapter_promotion_chain_transition(
    parent: Mapping[str, object],
    child: Mapping[str, object],
    *,
    selected_path: bool,
) -> dict[str, object]:
    parent_depth = _chain_depth(parent.get("lineage_depth"))
    child_depth = _chain_depth(child.get("lineage_depth"))
    depth_step = (
        None
        if parent_depth is None or child_depth is None
        else child_depth - parent_depth
    )
    parent_eval_after = _finite_number(parent.get("eval_after_loss"))
    child_eval_before = _finite_number(child.get("eval_before_loss"))
    child_eval_after = _finite_number(child.get("eval_after_loss"))
    eval_handoff_delta = (
        None
        if parent_eval_after is None or child_eval_before is None
        else child_eval_before - parent_eval_after
    )
    child_eval_improvement = (
        None
        if child_eval_before is None or child_eval_after is None
        else child_eval_before - child_eval_after
    )
    root_matches = parent.get("root_adapter_id") == child.get("root_adapter_id")
    base_model_matches = parent.get("base_model_name_or_path") == child.get(
        "base_model_name_or_path"
    )
    lineage_ready = bool(
        depth_step == 1
        and root_matches
        and base_model_matches
        and child.get("parent_fingerprint_verified") is True
        and child.get("weights_changed_from_parent") is True
    )
    transition_ready = bool(
        lineage_ready
        and parent.get("chain_eligible") is True
        and child.get("chain_eligible") is True
    )
    return {
        "row_type": "hf_adapter_promotion_chain_transition",
        "status": "ready" if transition_ready else "rejected",
        "transition_ready": transition_ready,
        "selected_path": selected_path,
        "parent_adapter_id": parent.get("adapter_id"),
        "parent_adapter_path": parent.get("adapter_path"),
        "parent_lineage_depth": parent_depth,
        "parent_chain_eligible": parent.get("chain_eligible") is True,
        "child_adapter_id": child.get("adapter_id"),
        "child_adapter_path": child.get("adapter_path"),
        "child_lineage_depth": child_depth,
        "child_chain_eligible": child.get("chain_eligible") is True,
        "depth_step": depth_step,
        "root_adapter_id": child.get("root_adapter_id"),
        "root_matches": root_matches,
        "base_model_name_or_path": child.get("base_model_name_or_path"),
        "base_model_matches": base_model_matches,
        "lineage_ready": lineage_ready,
        "parent_fingerprint_verified": (
            child.get("parent_fingerprint_verified") is True
        ),
        "weights_changed_from_parent": (
            child.get("weights_changed_from_parent") is True
        ),
        "parent_eval_after_loss": parent_eval_after,
        "child_eval_before_loss": child_eval_before,
        "child_eval_after_loss": child_eval_after,
        "eval_handoff_observed": (
            parent_eval_after is not None and child_eval_before is not None
        ),
        "eval_handoff_delta": eval_handoff_delta,
        "child_eval_improvement": child_eval_improvement,
        "promotion_ready": child.get("promotion_ready") is True,
        "promotion_revalidated_ready": (
            child.get("promotion_revalidated_ready") is True
        ),
        "artifact_probe_status": child.get("artifact_probe_status"),
        "artifact_probe_process_status": child.get("artifact_probe_process_status"),
        "artifact_probe_process_pid": child.get("artifact_probe_process_pid"),
        "artifact_probe_process_exit_code": child.get(
            "artifact_probe_process_exit_code"
        ),
    }


def _continuation_policy_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if parsed < minimum or parsed != value:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return parsed


def _continuation_policy_float(
    name: str,
    value: object | None,
    *,
    minimum: float,
) -> float | None:
    if value is None:
        return None
    parsed = _finite_number(value)
    if parsed is None or parsed < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return parsed


def hf_adapter_continuation_policy_report(
    chain_or_path: Mapping[str, object] | str | Path,
    *,
    max_lineage_depth: int | None = None,
    target_eval_loss: float | None = None,
    min_eval_improvement: float | None = None,
    plateau_patience: int = 1,
) -> dict[str, object]:
    """Decide whether an audited adapter chain should train another generation."""

    chain, source_path = _json_mapping(chain_or_path)
    if chain.get("row_type") != "hf_adapter_promotion_chain":
        raise ValueError(
            "continuation policy requires an HF adapter promotion chain report"
        )
    if chain.get("schema") != HF_ADAPTER_PROMOTION_CHAIN_SCHEMA:
        raise ValueError(
            "unsupported HF adapter promotion chain schema: "
            f"{chain.get('schema')}"
        )
    resolved_max_depth = (
        None
        if max_lineage_depth is None
        else _continuation_policy_int(
            "max_lineage_depth",
            max_lineage_depth,
            minimum=0,
        )
    )
    resolved_target = _continuation_policy_float(
        "target_eval_loss",
        target_eval_loss,
        minimum=0.0,
    )
    resolved_min_improvement = _continuation_policy_float(
        "min_eval_improvement",
        min_eval_improvement,
        minimum=0.0,
    )
    resolved_patience = _continuation_policy_int(
        "plateau_patience",
        plateau_patience,
        minimum=1,
    )

    raw_nodes = chain.get("nodes")
    nodes = (
        [dict(node) for node in raw_nodes if isinstance(node, Mapping)]
        if isinstance(raw_nodes, Sequence) and not isinstance(raw_nodes, (str, bytes))
        else []
    )
    nodes_by_id = {
        str(node.get("adapter_id")): node
        for node in nodes
        if isinstance(node.get("adapter_id"), str)
    }
    raw_selected_path_ids = chain.get("selected_path_adapter_ids")
    selected_path_ids = (
        [
            str(adapter_id)
            for adapter_id in raw_selected_path_ids
            if isinstance(adapter_id, str)
        ]
        if isinstance(raw_selected_path_ids, Sequence)
        and not isinstance(raw_selected_path_ids, (str, bytes))
        else []
    )
    missing_path_adapter_ids = [
        adapter_id for adapter_id in selected_path_ids if adapter_id not in nodes_by_id
    ]
    selected_path = [
        nodes_by_id[adapter_id]
        for adapter_id in selected_path_ids
        if adapter_id in nodes_by_id
    ]
    observations: list[dict[str, object]] = []
    for node in selected_path:
        depth = _chain_depth(node.get("lineage_depth"))
        if depth in (0, None):
            continue
        before = _finite_number(node.get("eval_before_loss"))
        after = _finite_number(node.get("eval_after_loss"))
        improvement = None if before is None or after is None else before - after
        meets_minimum = (
            None
            if resolved_min_improvement is None or improvement is None
            else improvement >= resolved_min_improvement
        )
        observations.append(
            {
                "adapter_id": node.get("adapter_id"),
                "adapter_path": node.get("adapter_path"),
                "lineage_depth": depth,
                "eval_before_loss": before,
                "eval_after_loss": after,
                "eval_improvement": improvement,
                "min_eval_improvement": resolved_min_improvement,
                "meets_min_eval_improvement": meets_minimum,
            }
        )

    selected_depth = _chain_depth(chain.get("selected_lineage_depth"))
    selected_after = (
        _finite_number(observations[-1].get("eval_after_loss"))
        if observations
        else None
    )
    path_start_loss = (
        _finite_number(observations[0].get("eval_before_loss"))
        if observations
        else None
    )
    path_improvement = (
        None
        if path_start_loss is None or selected_after is None
        else path_start_loss - selected_after
    )

    consecutive_below_minimum = 0
    if resolved_min_improvement is not None:
        for observation in reversed(observations):
            meets = observation.get("meets_min_eval_improvement")
            if meets is not False:
                break
            consecutive_below_minimum += 1

    missing_evidence: list[dict[str, object]] = []
    if resolved_target is not None and selected_depth not in (0, None):
        if selected_after is None:
            missing_evidence.append(
                {
                    "field": "selected_eval_after_loss",
                    "message": "target_eval_loss requires final evaluation evidence",
                }
            )
    if resolved_min_improvement is not None and observations:
        required_tail = observations[-min(resolved_patience, len(observations)) :]
        for observation in required_tail:
            if observation.get("eval_improvement") is None:
                missing_evidence.append(
                    {
                        "field": "eval_improvement",
                        "adapter_id": observation.get("adapter_id"),
                        "lineage_depth": observation.get("lineage_depth"),
                        "message": (
                            "plateau detection requires before/after evaluation evidence"
                        ),
                    }
                )
    if resolved_min_improvement is not None and selected_depth not in (0, None):
        required_generation_count = min(resolved_patience, selected_depth)
        if len(observations) < required_generation_count:
            missing_evidence.append(
                {
                    "field": "selected_path_observations",
                    "observed": len(observations),
                    "threshold": required_generation_count,
                    "message": (
                        "plateau detection is missing selected-path generation "
                        "observations"
                    ),
                }
            )
    for adapter_id in missing_path_adapter_ids:
        missing_evidence.append(
            {
                "field": "selected_path_node",
                "adapter_id": adapter_id,
                "message": "selected adapter path references a missing chain node",
            }
        )

    stop_reasons: list[dict[str, object]] = []
    if (
        resolved_max_depth is not None
        and selected_depth is not None
        and selected_depth >= resolved_max_depth
    ):
        stop_reasons.append(
            {
                "code": "max_lineage_depth_reached",
                "observed": selected_depth,
                "threshold": resolved_max_depth,
                "message": "selected lineage depth reached the configured maximum",
            }
        )
    if (
        resolved_target is not None
        and selected_after is not None
        and selected_after <= resolved_target
    ):
        stop_reasons.append(
            {
                "code": "target_eval_loss_reached",
                "observed": selected_after,
                "threshold": resolved_target,
                "message": "selected adapter reached the target evaluation loss",
            }
        )
    if (
        resolved_min_improvement is not None
        and consecutive_below_minimum >= resolved_patience
    ):
        stop_reasons.append(
            {
                "code": "eval_improvement_plateau",
                "observed": consecutive_below_minimum,
                "threshold": resolved_patience,
                "message": (
                    "consecutive generations stayed below the minimum evaluation "
                    "improvement"
                ),
            }
        )

    policy_active = any(
        value is not None
        for value in (
            resolved_max_depth,
            resolved_target,
            resolved_min_improvement,
        )
    )
    if chain.get("chain_ready") is not True:
        status = "blocked"
        recommendation = "resolve_chain"
    elif stop_reasons:
        status = "stop"
        recommendation = "stop_training"
    elif missing_evidence:
        status = "needs_evidence"
        recommendation = "collect_eval_evidence"
    else:
        status = "continue"
        recommendation = "continue_training"
    return {
        "row_type": "hf_adapter_continuation_policy",
        "schema": HF_ADAPTER_CONTINUATION_POLICY_SCHEMA,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_chain_path": source_path or chain.get("report_path"),
        "chain_schema": chain.get("schema"),
        "chain_status": chain.get("status"),
        "chain_ready": chain.get("chain_ready") is True,
        "policy_active": policy_active,
        "continuation_allowed": status == "continue",
        "recommendation": recommendation,
        "selected_adapter_id": chain.get("selected_adapter_id"),
        "selected_adapter_path": chain.get("selected_adapter_path"),
        "selected_lineage_depth": selected_depth,
        "next_lineage_depth": None if selected_depth is None else selected_depth + 1,
        "selected_path_adapter_ids": selected_path_ids,
        "selected_path_generation_count": len(observations),
        "evaluable_generation_count": sum(
            observation.get("eval_improvement") is not None
            for observation in observations
        ),
        "selected_path_eval_start_loss": path_start_loss,
        "selected_path_eval_end_loss": selected_after,
        "selected_path_eval_improvement": path_improvement,
        "max_lineage_depth": resolved_max_depth,
        "target_eval_loss": resolved_target,
        "min_eval_improvement": resolved_min_improvement,
        "plateau_patience": resolved_patience,
        "consecutive_below_min_eval_improvement": consecutive_below_minimum,
        "stop_reason_count": len(stop_reasons),
        "stop_reason_codes": [str(reason.get("code")) for reason in stop_reasons],
        "stop_reasons": stop_reasons,
        "missing_evidence_count": len(missing_evidence),
        "missing_evidence": missing_evidence,
        "observations": observations,
    }


def hf_adapter_promotion_chain_report(
    sources: str | Path | Sequence[str | Path],
    *,
    recursive: bool = True,
    allow_inferred_roots: bool = True,
    select_adapter_id: str | None = None,
    command_artifacts: Sequence[Mapping[str, object] | str | Path] | None = None,
    max_lineage_depth: int | None = None,
    target_eval_loss: float | None = None,
    min_eval_improvement: float | None = None,
    plateau_patience: int = 1,
) -> dict[str, object]:
    """Audit a local adapter DAG and select one promotion-ready continuation tip."""

    manifest_paths, report_issues = _discover_lineage_manifests(
        sources,
        recursive=recursive,
    )
    nodes = [_adapter_promotion_chain_node(path) for path in manifest_paths]
    if allow_inferred_roots:
        known_ids = {
            node.get("adapter_id")
            for node in nodes
            if isinstance(node.get("adapter_id"), str)
        }
        inferred_roots: dict[str, dict[str, object]] = {}
        for child in nodes:
            if _chain_depth(child.get("lineage_depth")) != 1:
                continue
            parent_id = child.get("parent_adapter_id")
            if not isinstance(parent_id, str) or parent_id in known_ids:
                continue
            inferred = _chain_inferred_root_node(child)
            if inferred is not None:
                inferred_roots[parent_id] = inferred
        nodes.extend(inferred_roots.values())
    command_rows, command_issues = _chain_command_artifacts(command_artifacts)
    report_issues.extend(command_issues)
    for node in nodes:
        if node.get("launch_command") is not None:
            continue
        matches = [
            row
            for row in command_rows
            if row.get("output_dir") == node.get("adapter_path")
        ]
        if len(matches) > 1:
            _chain_add_node_issue(
                node,
                "multiple_command_artifacts",
                "multiple launch commands target this adapter directory",
                severity="warning",
            )
        if matches:
            selected_command = matches[-1]
            node["launch_command"] = selected_command.get("command")
            node["launch_command_display"] = shlex.join(
                [str(item) for item in selected_command.get("command") or []]
            )
            node["launch_command_source"] = "command_artifact"
            node["launch_command_artifact_path"] = selected_command.get("source_path")

    id_rows: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        adapter_id = node.get("adapter_id")
        if isinstance(adapter_id, str):
            id_rows.setdefault(adapter_id, []).append(node)
    for adapter_id, duplicates in id_rows.items():
        if len(duplicates) <= 1:
            continue
        for node in duplicates:
            _chain_add_node_issue(
                node,
                "duplicate_adapter_id",
                "the same adapter fingerprint appears in multiple directories",
            )

    unique_nodes = {
        adapter_id: rows[0] for adapter_id, rows in id_rows.items() if len(rows) == 1
    }
    children: dict[str, list[str]] = {adapter_id: [] for adapter_id in unique_nodes}
    for adapter_id, node in unique_nodes.items():
        depth = _chain_depth(node.get("lineage_depth"))
        node["lineage_depth"] = depth
        parent_id = node.get("parent_adapter_id")
        if depth is None:
            _chain_add_node_issue(
                node,
                "invalid_lineage_depth",
                "lineage depth must be a non-negative integer",
            )
            continue
        if depth == 0:
            if parent_id is not None:
                _chain_add_node_issue(
                    node,
                    "root_has_parent",
                    "depth-zero lineage nodes cannot have a parent",
                )
            if node.get("root_adapter_id") != adapter_id:
                _chain_add_node_issue(
                    node,
                    "root_identity_mismatch",
                    "depth-zero lineage root must identify itself",
                )
            if node.get("ancestor_adapter_ids"):
                _chain_add_node_issue(
                    node,
                    "root_has_ancestors",
                    "depth-zero lineage nodes cannot have ancestors",
                )
            continue
        if not isinstance(parent_id, str) or parent_id not in unique_nodes:
            _chain_add_node_issue(
                node,
                "parent_node_missing",
                "the declared parent lineage node was not found",
            )
            continue
        parent = unique_nodes[parent_id]
        children[parent_id].append(adapter_id)
        parent_depth = _chain_depth(parent.get("lineage_depth"))
        if parent_depth is None or depth != parent_depth + 1:
            _chain_add_node_issue(
                node,
                "lineage_depth_discontinuity",
                "child lineage depth must equal parent depth plus one",
            )
        if node.get("root_adapter_id") != parent.get("root_adapter_id"):
            _chain_add_node_issue(
                node,
                "root_lineage_mismatch",
                "child and parent lineage roots differ",
            )
        if node.get("base_model_name_or_path") != parent.get("base_model_name_or_path"):
            _chain_add_node_issue(
                node,
                "base_model_mismatch",
                "child and parent adapters resolve different base models",
            )
        expected_ancestors = [
            *list(parent.get("ancestor_adapter_ids") or []),
            parent_id,
        ]
        if node.get("ancestor_adapter_ids") != expected_ancestors:
            _chain_add_node_issue(
                node,
                "ancestor_sequence_mismatch",
                "ancestor IDs do not form the parent lineage prefix",
            )
        parent_path = _chain_reference_path(
            node.get("parent_adapter_path"),
            anchor=Path(str(node.get("adapter_path"))),
        )
        if parent_path is not None and str(parent_path) != parent.get("adapter_path"):
            _chain_add_node_issue(
                node,
                "parent_path_mismatch",
                "declared parent path resolves to a different adapter node",
                path=parent_path,
            )
        if node.get("parent_fingerprint_verified") is not True:
            _chain_add_node_issue(
                node,
                "parent_fingerprint_unverified",
                "non-root lineage parent fingerprint was not verified",
            )
        if node.get("weights_changed_from_parent") is not True:
            _chain_add_node_issue(
                node,
                "weights_unchanged",
                "non-root adapter weights did not change from the parent",
            )

    forks: list[dict[str, object]] = []
    for parent_id, child_ids in children.items():
        if len(child_ids) <= 1:
            continue
        forks.append(
            {
                "parent_adapter_id": parent_id,
                "child_adapter_ids": sorted(child_ids),
                "child_count": len(child_ids),
            }
        )
        report_issues.append(
            _chain_issue(
                "lineage_fork",
                "multiple child adapters branch from the same parent",
                severity="warning",
                adapter_id=parent_id,
            )
        )

    ordered_nodes = sorted(
        nodes,
        key=lambda node: (
            _chain_depth(node.get("lineage_depth"))
            if _chain_depth(node.get("lineage_depth")) is not None
            else 10**9,
            str(node.get("adapter_id") or ""),
            str(node.get("adapter_path") or ""),
        ),
    )
    eligible_ids: set[str] = set()
    for node in ordered_nodes:
        node_issues = node.get("issues") or []
        validation_ready = not any(
            isinstance(issue, Mapping) and issue.get("severity") == "error"
            for issue in node_issues
        )
        adapter_id = node.get("adapter_id")
        depth = _chain_depth(node.get("lineage_depth"))
        eligible = bool(validation_ready and isinstance(adapter_id, str))
        if eligible and depth not in (0, None):
            eligible = bool(
                node.get("promotion_ready") is True
                and node.get("parent_adapter_id") in eligible_ids
            )
        node["validation_ready"] = validation_ready
        node["chain_eligible"] = eligible
        node["status"] = "ready" if eligible else "rejected"
        if eligible and isinstance(adapter_id, str):
            eligible_ids.add(adapter_id)

    eligible_tips = [
        node
        for node in ordered_nodes
        if node.get("chain_eligible") is True
        and not any(
            child_id in eligible_ids
            for child_id in children.get(
                str(node.get("adapter_id")),
                [],
            )
        )
    ]
    selected: dict[str, object] | None = None
    if select_adapter_id is not None:
        candidate = unique_nodes.get(select_adapter_id)
        if candidate is not None and candidate.get("chain_eligible") is True:
            selected = candidate
            selection_status = "explicit"
        else:
            selection_status = "invalid_selection"
            report_issues.append(
                _chain_issue(
                    "selected_adapter_not_eligible",
                    "the requested adapter is absent or not chain-eligible",
                    adapter_id=select_adapter_id,
                )
            )
    elif eligible_tips:
        max_depth = max(int(node.get("lineage_depth") or 0) for node in eligible_tips)
        deepest = [
            node
            for node in eligible_tips
            if int(node.get("lineage_depth") or 0) == max_depth
        ]
        if len(deepest) == 1:
            selected = deepest[0]
            selection_status = "deepest_unique_tip"
        else:
            selection_status = "ambiguous_deepest_tips"
            report_issues.append(
                _chain_issue(
                    "ambiguous_continuation_tip",
                    "multiple promotion-ready tips share the deepest lineage depth",
                )
            )
    else:
        selection_status = "no_eligible_tip"
        report_issues.append(
            _chain_issue(
                "no_eligible_tip",
                "no lineage node is eligible for continuation",
            )
        )

    selected_path_ids: list[str] = []
    cursor_node = selected
    seen_ids: set[str] = set()
    while cursor_node is not None:
        cursor_id = cursor_node.get("adapter_id")
        if not isinstance(cursor_id, str) or cursor_id in seen_ids:
            break
        selected_path_ids.append(cursor_id)
        seen_ids.add(cursor_id)
        parent_id = cursor_node.get("parent_adapter_id")
        cursor_node = (
            unique_nodes.get(str(parent_id)) if parent_id is not None else None
        )
    selected_path_ids.reverse()
    selected_path_pairs = set(zip(selected_path_ids, selected_path_ids[1:]))
    transitions = []
    for child in ordered_nodes:
        child_id = child.get("adapter_id")
        parent_id = child.get("parent_adapter_id")
        if (
            not isinstance(child_id, str)
            or not isinstance(parent_id, str)
            or parent_id not in unique_nodes
        ):
            continue
        transitions.append(
            _adapter_promotion_chain_transition(
                unique_nodes[parent_id],
                child,
                selected_path=(parent_id, child_id) in selected_path_pairs,
            )
        )
    selected_path_transitions = [
        transition
        for transition in transitions
        if transition.get("selected_path") is True
    ]
    selected_path_transition_count = len(selected_path_transitions)
    expected_selected_path_transition_count = max(len(selected_path_ids) - 1, 0)
    selected_path_transitions_ready = bool(
        selected is not None
        and selected_path_transition_count == expected_selected_path_transition_count
        and all(
            transition.get("transition_ready") is True
            for transition in selected_path_transitions
        )
    )

    node_issues = [
        issue
        for node in ordered_nodes
        for issue in node.get("issues", [])
        if isinstance(issue, Mapping)
    ]
    all_issues = [*report_issues, *node_issues]
    error_count = sum(issue.get("severity") == "error" for issue in all_issues)
    warning_count = sum(issue.get("severity") == "warning" for issue in all_issues)
    rejected_count = sum(
        node.get("chain_eligible") is not True for node in ordered_nodes
    )
    chain_ready = selected is not None
    continuation_artifacts_ready = bool(
        selected is not None and selected.get("launch_command") is not None
    )
    if not chain_ready:
        status = "ambiguous" if selection_status.startswith("ambiguous") else "blocked"
    elif not continuation_artifacts_ready:
        status = "needs_command"
    elif rejected_count:
        status = "ready_with_rejections"
    else:
        status = "ready"

    roots = [
        node
        for node in ordered_nodes
        if node.get("lineage_depth") == 0 and node.get("adapter_id") is not None
    ]
    candidate = None if selected is None else dict(selected)
    report: dict[str, object] = {
        "row_type": "hf_adapter_promotion_chain",
        "schema": HF_ADAPTER_PROMOTION_CHAIN_SCHEMA,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chain_ready": chain_ready,
        "continuation_artifacts_ready": continuation_artifacts_ready,
        "continuation_ready": continuation_artifacts_ready,
        "selection_status": selection_status,
        "selected_adapter_id": None if selected is None else selected.get("adapter_id"),
        "selected_adapter_path": None
        if selected is None
        else selected.get("adapter_path"),
        "selected_lineage_depth": None
        if selected is None
        else selected.get("lineage_depth"),
        "selected_path_adapter_ids": selected_path_ids,
        "selected_path_transition_count": selected_path_transition_count,
        "selected_path_transitions_ready": selected_path_transitions_ready,
        "continuation_candidate": candidate,
        "continuation_candidate_command": None
        if selected is None
        else selected.get("launch_command"),
        "continuation_candidate_command_display": None
        if selected is None
        else selected.get("launch_command_display"),
        "node_count": len(ordered_nodes),
        "root_count": len(roots),
        "root_adapter_ids": [node.get("adapter_id") for node in roots],
        "eligible_node_count": len(eligible_ids),
        "eligible_tip_count": len(eligible_tips),
        "eligible_tip_adapter_ids": [node.get("adapter_id") for node in eligible_tips],
        "promotion_ready_node_count": sum(
            node.get("promotion_ready") is True for node in ordered_nodes
        ),
        "transition_count": len(transitions),
        "ready_transition_count": sum(
            transition.get("transition_ready") is True for transition in transitions
        ),
        "rejected_transition_count": sum(
            transition.get("transition_ready") is not True for transition in transitions
        ),
        "rejected_node_count": rejected_count,
        "launch_command_available_count": sum(
            node.get("launch_command") is not None for node in ordered_nodes
        ),
        "fork_count": len(forks),
        "forks": forks,
        "integrity_ready": error_count == 0,
        "issue_count": len(all_issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": report_issues,
        "nodes": ordered_nodes,
        "transitions": transitions,
    }
    policy = hf_adapter_continuation_policy_report(
        report,
        max_lineage_depth=max_lineage_depth,
        target_eval_loss=target_eval_loss,
        min_eval_improvement=min_eval_improvement,
        plateau_patience=plateau_patience,
    )
    continuation_allowed = policy.get("continuation_allowed") is True
    report["continuation_policy"] = policy
    report["continuation_policy_status"] = policy.get("status")
    report["continuation_allowed"] = continuation_allowed
    report["continuation_ready"] = bool(
        continuation_artifacts_ready and continuation_allowed
    )
    report["continuation_stop_reason_codes"] = list(
        policy.get("stop_reason_codes") or []
    )
    if chain_ready and policy.get("status") == "stop":
        report["status"] = "stopped_by_policy"
    elif chain_ready and policy.get("status") == "needs_evidence":
        report["status"] = "needs_policy_evidence"
    return report


def write_hf_adapter_promotion_chain(
    report_or_sources: Mapping[str, object] | str | Path | Sequence[str | Path],
    out: str | Path,
    *,
    recursive: bool = True,
    allow_inferred_roots: bool = True,
    select_adapter_id: str | None = None,
    command_artifacts: Sequence[Mapping[str, object] | str | Path] | None = None,
    max_lineage_depth: int | None = None,
    target_eval_loss: float | None = None,
    min_eval_improvement: float | None = None,
    plateau_patience: int = 1,
) -> dict[str, object]:
    report = (
        dict(report_or_sources)
        if isinstance(report_or_sources, Mapping)
        else hf_adapter_promotion_chain_report(
            report_or_sources,
            recursive=recursive,
            allow_inferred_roots=allow_inferred_roots,
            select_adapter_id=select_adapter_id,
            command_artifacts=command_artifacts,
            max_lineage_depth=max_lineage_depth,
            target_eval_loss=target_eval_loss,
            min_eval_improvement=min_eval_improvement,
            plateau_patience=plateau_patience,
        )
    )
    path = Path(out)
    report["report_path"] = str(path.resolve())
    _atomic_write_json(path, report)
    return report


def load_hf_adapter_promotion_chain(value: str | Path) -> dict[str, object]:
    path = Path(value)
    payload, _ = _json_mapping(path)
    if payload.get("schema") != HF_ADAPTER_PROMOTION_CHAIN_SCHEMA:
        raise ValueError(
            "unsupported HF adapter promotion chain schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("row_type") != "hf_adapter_promotion_chain":
        raise ValueError(
            "unsupported HF adapter promotion chain row type: "
            f"{payload.get('row_type')}"
        )
    payload["report_path"] = str(path.resolve())
    return payload


def write_hf_adapter_continuation_policy(
    report_or_chain: Mapping[str, object] | str | Path,
    out: str | Path,
    *,
    max_lineage_depth: int | None = None,
    target_eval_loss: float | None = None,
    min_eval_improvement: float | None = None,
    plateau_patience: int = 1,
) -> dict[str, object]:
    report = (
        dict(report_or_chain)
        if isinstance(report_or_chain, Mapping)
        and report_or_chain.get("row_type") == "hf_adapter_continuation_policy"
        else hf_adapter_continuation_policy_report(
            report_or_chain,
            max_lineage_depth=max_lineage_depth,
            target_eval_loss=target_eval_loss,
            min_eval_improvement=min_eval_improvement,
            plateau_patience=plateau_patience,
        )
    )
    path = Path(out)
    report["report_path"] = str(path.resolve())
    _atomic_write_json(path, report)
    return report


def load_hf_adapter_continuation_policy(value: str | Path) -> dict[str, object]:
    path = Path(value)
    payload, _ = _json_mapping(path)
    if payload.get("schema") != HF_ADAPTER_CONTINUATION_POLICY_SCHEMA:
        raise ValueError(
            "unsupported HF adapter continuation policy schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("row_type") != "hf_adapter_continuation_policy":
        raise ValueError(
            "unsupported HF adapter continuation policy row type: "
            f"{payload.get('row_type')}"
        )
    payload["report_path"] = str(path.resolve())
    return payload


def hf_adapter_continuation_policy_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_policy(report_or_path)
    )
    reason_codes = ",".join(
        str(code) for code in report.get("stop_reason_codes") or []
    )
    lines = [
        (
            "hf_adapter_continuation_policy "
            f"status={report.get('status')} "
            f"allowed={report.get('continuation_allowed')} "
            f"depth={report.get('selected_lineage_depth')} "
            f"next_depth={report.get('next_lineage_depth')} "
            f"eval_loss={report.get('selected_path_eval_end_loss')} "
            f"path_improvement={report.get('selected_path_eval_improvement')} "
            f"plateau={report.get('consecutive_below_min_eval_improvement')}/"
            f"{report.get('plateau_patience')} "
            f"reasons={reason_codes or '-'}"
        )
    ]
    for reason in report.get("stop_reasons") or []:
        if not isinstance(reason, Mapping):
            continue
        lines.append(
            "hf_adapter_continuation_policy_stop "
            f"code={reason.get('code')} "
            f"observed={reason.get('observed')} "
            f"threshold={reason.get('threshold')} "
            f"message={reason.get('message')}"
        )
    for missing in report.get("missing_evidence") or []:
        if not isinstance(missing, Mapping):
            continue
        lines.append(
            "hf_adapter_continuation_policy_missing "
            f"field={missing.get('field')} "
            f"depth={missing.get('lineage_depth')} "
            f"adapter={missing.get('adapter_id')} "
            f"message={missing.get('message')}"
        )
    return lines


def hf_adapter_promotion_chain_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report, _ = (
        (dict(report_or_path), None)
        if isinstance(report_or_path, Mapping)
        else (load_hf_adapter_promotion_chain(report_or_path), None)
    )
    lines = [
        (
            "hf_adapter_promotion_chain "
            f"status={report.get('status')} "
            f"nodes={report.get('node_count')} "
            f"eligible={report.get('eligible_node_count')} "
            f"rejected={report.get('rejected_node_count')} "
            f"transitions={report.get('transition_count')} "
            f"transitions_ready={report.get('ready_transition_count')} "
            f"forks={report.get('fork_count')} "
            f"selection={report.get('selection_status')} "
            f"selected={report.get('selected_adapter_id')} "
            f"depth={report.get('selected_lineage_depth')} "
            f"policy={report.get('continuation_policy_status')} "
            f"allowed={report.get('continuation_allowed')} "
            f"continuation_ready={report.get('continuation_ready')}"
        )
    ]
    policy = report.get("continuation_policy")
    if isinstance(policy, Mapping):
        lines.extend(hf_adapter_continuation_policy_lines(policy))
    for raw_transition in report.get("transitions", []):
        if not isinstance(raw_transition, Mapping):
            continue
        lines.append(
            "hf_adapter_promotion_chain_transition "
            f"status={raw_transition.get('status')} "
            f"depth={raw_transition.get('parent_lineage_depth')}"
            f"->{raw_transition.get('child_lineage_depth')} "
            f"parent={raw_transition.get('parent_adapter_id')} "
            f"child={raw_transition.get('child_adapter_id')} "
            f"selected={raw_transition.get('selected_path')} "
            f"parent_verified={raw_transition.get('parent_fingerprint_verified')} "
            f"weights_changed={raw_transition.get('weights_changed_from_parent')} "
            f"eval_handoff_delta={raw_transition.get('eval_handoff_delta')} "
            f"eval_improvement={raw_transition.get('child_eval_improvement')} "
            f"promotion_ready={raw_transition.get('promotion_ready')} "
            f"probe_process={raw_transition.get('artifact_probe_process_status')} "
            f"probe_pid={raw_transition.get('artifact_probe_process_pid')}"
        )
    for raw_node in report.get("nodes", []):
        if not isinstance(raw_node, Mapping):
            continue
        lines.append(
            "hf_adapter_promotion_chain_node "
            f"status={raw_node.get('status')} "
            f"depth={raw_node.get('lineage_depth')} "
            f"adapter={raw_node.get('adapter_id')} "
            f"parent={raw_node.get('parent_adapter_id')} "
            f"promotion_ready={raw_node.get('promotion_ready')} "
            f"artifact_probe={raw_node.get('artifact_probe_status')} "
            f"probe_tokens={raw_node.get('artifact_probe_new_token_count')} "
            f"probe_process={raw_node.get('artifact_probe_process_status')} "
            f"probe_pid={raw_node.get('artifact_probe_process_pid')} "
            f"eval_regression={raw_node.get('eval_loss_regression')} "
            f"command={raw_node.get('launch_command_source')} "
            f"path={raw_node.get('adapter_path')}"
        )
        for raw_issue in raw_node.get("issues", []):
            if not isinstance(raw_issue, Mapping):
                continue
            lines.append(
                "hf_adapter_promotion_chain_issue "
                f"severity={raw_issue.get('severity')} "
                f"code={raw_issue.get('code')} "
                f"adapter={raw_issue.get('adapter_id')} "
                f"path={raw_issue.get('path')} "
                f"message={raw_issue.get('message')}"
            )
    for raw_issue in report.get("issues", []):
        if not isinstance(raw_issue, Mapping):
            continue
        lines.append(
            "hf_adapter_promotion_chain_issue "
            f"severity={raw_issue.get('severity')} "
            f"code={raw_issue.get('code')} "
            f"adapter={raw_issue.get('adapter_id')} "
            f"path={raw_issue.get('path')} "
            f"message={raw_issue.get('message')}"
        )
    return lines
