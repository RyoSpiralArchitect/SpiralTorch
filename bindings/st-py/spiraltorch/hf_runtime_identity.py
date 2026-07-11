"""Content identity for the model and tokenizer used by an HF runtime load."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "HF_CAUSAL_LM_RUNTIME_IDENTITY_SCHEMA",
    "hf_causal_lm_runtime_identity_lines",
    "hf_causal_lm_runtime_identity_report",
]


HF_CAUSAL_LM_RUNTIME_IDENTITY_SCHEMA = (
    "spiraltorch.hf_causal_lm_runtime_identity.v1"
)
_HF_CAUSAL_LM_RUNTIME_BUNDLE_SCHEMA = (
    "spiraltorch.hf_causal_lm_runtime_bundle.v1"
)
_HF_RUNTIME_LOCAL_PAYLOAD_SCHEMA = "spiraltorch.hf_runtime_local_payload.v1"
_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "sentencepiece.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_MODEL_EXACT_FILENAMES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "quantization_config.json",
}
_MODEL_SUFFIXES = (
    ".bin",
    ".h5",
    ".msgpack",
    ".onnx",
    ".py",
    ".safetensors",
)
_VOLATILE_CONFIG_KEYS = {
    "_commit_hash",
    "_name_or_path",
    "name_or_path",
    "transformers_version",
}


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
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "expected_identity_id must be a lowercase sha256:<64 hex> identity id"
        )
    return identity_id


def _stat_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))


def _stable_file_row(path: Path, *, root: Path) -> tuple[dict[str, object], tuple[int, int, int]]:
    before = _stat_signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = _stat_signature(path)
    if before != after:
        raise RuntimeError(f"runtime payload changed while hashing: {path}")
    return (
        {
            "relative_path": path.relative_to(root).as_posix(),
            "content_sha256": digest.hexdigest(),
            "size_bytes": after[0],
        },
        after,
    )


def _is_tokenizer_file(path: Path) -> bool:
    name = path.name
    return bool(
        name in _TOKENIZER_FILENAMES
        or (
            name.endswith(".jinja")
            and (name.startswith("chat_template") or "chat_templates" in path.parts)
        )
        or name.startswith("tokenizer.")
        or name.startswith("vocab.")
        or name.startswith("merges.")
        or name.startswith("special_tokens_map.")
    )


def _is_model_file(path: Path) -> bool:
    name = path.name
    if _is_tokenizer_file(path) or name.startswith("adapter_model"):
        return False
    return bool(
        name in _MODEL_EXACT_FILENAMES
        or name.endswith(".index.json")
        or name.endswith(_MODEL_SUFFIXES)
    )


def _stable_local_payload(
    path: Path,
    *,
    role: str,
) -> dict[str, object]:
    predicate = _is_model_file if role == "base_model" else _is_tokenizer_file
    before = sorted(
        (entry for entry in path.rglob("*") if entry.is_file() and predicate(entry)),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    rows: list[dict[str, object]] = []
    signatures: list[tuple[int, int, int]] = []
    for entry in before:
        row, signature = _stable_file_row(entry, root=path)
        rows.append(row)
        signatures.append(signature)
    after = sorted(
        (entry for entry in path.rglob("*") if entry.is_file() and predicate(entry)),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    before_names = [entry.relative_to(path).as_posix() for entry in before]
    after_names = [entry.relative_to(path).as_posix() for entry in after]
    if before_names != after_names or any(
        _stat_signature(entry) != signature
        for entry, signature in zip(after, signatures)
    ):
        raise RuntimeError(f"runtime payload changed while hashing: {path}")
    payload = {
        "schema": _HF_RUNTIME_LOCAL_PAYLOAD_SCHEMA,
        "role": role,
        "files": rows,
    }
    return {
        "content_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "files": rows,
    }


def _json_safe(value: object) -> object:
    try:
        return json.loads(json.dumps(value, ensure_ascii=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _config_payload(config: Any) -> dict[str, object]:
    to_dict = getattr(config, "to_dict", None)
    payload: object = None
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
    if not isinstance(payload, Mapping):
        instance = getattr(config, "__dict__", None)
        payload = dict(instance) if isinstance(instance, Mapping) else {}
    return {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if str(key) not in _VOLATILE_CONFIG_KEYS
    }


def _runtime_commit_hash(value: object | None) -> str | None:
    if value is None:
        return None
    commit = str(value).strip().lower()
    if 7 <= len(commit) <= 64 and all(
        character in "0123456789abcdef" for character in commit
    ):
        return commit
    return None


def _config_commit_hash(config: Any, requested_revision: object | None) -> str | None:
    for value in (
        getattr(config, "_commit_hash", None),
        requested_revision,
    ):
        commit = _runtime_commit_hash(value)
        if commit is not None:
            return commit
    return None


def _tokenizer_semantic_payload(tokenizer: Any) -> dict[str, object] | None:
    if tokenizer is None:
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    backend_sha256 = None
    if callable(backend_to_str):
        try:
            backend_sha256 = _sha256_bytes(str(backend_to_str()).encode("utf-8"))
        except Exception:
            backend_sha256 = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab: object = None
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:
            vocab = None
    vocab_payload = dict(vocab) if isinstance(vocab, Mapping) else None
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added_vocab: object = None
    if callable(get_added_vocab):
        try:
            added_vocab = get_added_vocab()
        except Exception:
            added_vocab = None
    added_payload = dict(added_vocab) if isinstance(added_vocab, Mapping) else None
    payload = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "backend_sha256": backend_sha256,
        "vocab_sha256": (
            None
            if vocab_payload is None
            else _sha256_bytes(_canonical_json_bytes(vocab_payload))
        ),
        "vocab_size": None if vocab_payload is None else len(vocab_payload),
        "added_vocab_sha256": (
            None
            if added_payload is None
            else _sha256_bytes(_canonical_json_bytes(added_payload))
        ),
        "added_vocab_size": None if added_payload is None else len(added_payload),
        "special_tokens_map": _json_safe(
            getattr(tokenizer, "special_tokens_map", None)
        ),
        "model_input_names": _json_safe(
            getattr(tokenizer, "model_input_names", None)
        ),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    if backend_sha256 is None and vocab_payload is None:
        return None
    return payload


def _source_directory(value: object | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_dir() else None


def hf_causal_lm_runtime_identity_report(
    *,
    base_model_source: str | Path,
    base_model_revision: object | None,
    tokenizer_source: str | Path | None,
    tokenizer_source_kind: str | None,
    config: Any,
    tokenizer: Any,
    expected_identity_id: str | None = None,
    phase: str = "pre_model_load",
) -> dict[str, object]:
    """Fingerprint the effective model basis and tokenizer, not their location."""

    expected_id = _validated_identity_id(expected_identity_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")
    errors: list[str] = []
    base_source = str(base_model_source).strip()
    if not base_source:
        raise ValueError("base_model_source must not be empty")
    base_directory = _source_directory(base_source)
    config_payload = _config_payload(config)
    config_sha256 = _sha256_bytes(_canonical_json_bytes(config_payload))
    base_commit = _config_commit_hash(config, base_model_revision)
    base_local_payload: dict[str, object] | None = None
    if base_directory is not None:
        try:
            base_local_payload = _stable_local_payload(
                base_directory,
                role="base_model",
            )
        except (OSError, RuntimeError) as exc:
            errors.append(f"base_model: {exc}")
        else:
            if base_local_payload["file_count"] == 0:
                errors.append("base_model: no model config or weight files found")
    elif base_commit is None:
        errors.append("base_model: remote source did not expose a pinned commit")

    tokenizer_text = None if tokenizer_source is None else str(tokenizer_source)
    tokenizer_directory = _source_directory(tokenizer_text)
    tokenizer_local_payload: dict[str, object] | None = None
    if tokenizer_directory is not None:
        try:
            tokenizer_local_payload = _stable_local_payload(
                tokenizer_directory,
                role="tokenizer",
            )
        except (OSError, RuntimeError) as exc:
            errors.append(f"tokenizer: {exc}")
        else:
            if tokenizer is not None and tokenizer_local_payload["file_count"] == 0:
                errors.append("tokenizer: no tokenizer payload files found")
    tokenizer_semantics = _tokenizer_semantic_payload(tokenizer)
    if (
        tokenizer is not None
        and tokenizer_semantics is None
        and tokenizer_local_payload is None
    ):
        errors.append("tokenizer: runtime semantics could not be fingerprinted")
    tokenizer_commit = None
    if tokenizer_text == base_source:
        tokenizer_commit = base_commit
    if tokenizer_commit is None and tokenizer is not None:
        tokenizer_commit = _config_commit_hash(
            tokenizer,
            getattr(tokenizer, "init_kwargs", {}).get("revision")
            if isinstance(getattr(tokenizer, "init_kwargs", None), Mapping)
            else None,
        )

    base_identity = {
        "source_kind": "local" if base_directory is not None else "remote",
        "remote_source": None if base_directory is not None else base_source,
        "observed_commit": base_commit,
        "config_sha256": config_sha256,
        "local_payload_sha256": (
            None
            if base_local_payload is None
            else base_local_payload.get("content_sha256")
        ),
        "local_file_count": (
            None
            if base_local_payload is None
            else base_local_payload.get("file_count")
        ),
        "local_total_bytes": (
            None
            if base_local_payload is None
            else base_local_payload.get("total_bytes")
        ),
    }
    tokenizer_identity = {
        "loaded": tokenizer is not None,
        "source_kind": (
            "local"
            if tokenizer_directory is not None
            else "remote" if tokenizer_text is not None else "none"
        ),
        "source_role": tokenizer_source_kind,
        "remote_source": (
            tokenizer_text
            if tokenizer_text is not None and tokenizer_directory is None
            else None
        ),
        "observed_commit": tokenizer_commit,
        "semantic_sha256": (
            None
            if tokenizer_semantics is None
            else _sha256_bytes(_canonical_json_bytes(tokenizer_semantics))
        ),
        "tokenizer_class": (
            None if tokenizer is None else tokenizer.__class__.__name__
        ),
        "vocab_size": (
            None
            if tokenizer_semantics is None
            else tokenizer_semantics.get("vocab_size")
        ),
        "local_payload_sha256": (
            None
            if tokenizer_local_payload is None
            else tokenizer_local_payload.get("content_sha256")
        ),
        "local_file_count": (
            None
            if tokenizer_local_payload is None
            else tokenizer_local_payload.get("file_count")
        ),
        "local_total_bytes": (
            None
            if tokenizer_local_payload is None
            else tokenizer_local_payload.get("total_bytes")
        ),
    }
    identity_payload = {
        "schema": _HF_CAUSAL_LM_RUNTIME_BUNDLE_SCHEMA,
        "base_model": base_identity,
        "tokenizer": tokenizer_identity,
    }
    observed_id = (
        None
        if errors
        else f"sha256:{_sha256_bytes(_canonical_json_bytes(identity_payload))}"
    )
    if expected_id is not None and observed_id != expected_id:
        errors.append("causal-LM runtime identity does not match expected identity id")
    if expected_id is not None and errors:
        status = "blocked"
    elif errors:
        status = "evidence_incomplete"
    else:
        status = "ready"
    return {
        "row_type": "hf_causal_lm_runtime_identity",
        "schema": HF_CAUSAL_LM_RUNTIME_IDENTITY_SCHEMA,
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
        "base_model": {
            **base_identity,
            "requested_revision": base_model_revision,
            "resolved_path": (
                None if base_directory is None else str(base_directory)
            ),
            "local_payload": base_local_payload,
        },
        "tokenizer": {
            **tokenizer_identity,
            "resolved_path": (
                None if tokenizer_directory is None else str(tokenizer_directory)
            ),
            "local_payload": tokenizer_local_payload,
            "semantics": tokenizer_semantics,
        },
        "identity_payload": identity_payload if observed_id is not None else None,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_causal_lm_runtime_identity_lines(
    report: Mapping[str, object],
) -> list[str]:
    base = report.get("base_model")
    base_payload = dict(base) if isinstance(base, Mapping) else {}
    tokenizer = report.get("tokenizer")
    tokenizer_payload = dict(tokenizer) if isinstance(tokenizer, Mapping) else {}
    return [
        "hf_causal_lm_runtime_identity "
        f"status={report.get('status')} "
        f"phase={report.get('phase')} "
        f"verified={report.get('identity_verified')} "
        f"observed={report.get('observed_identity_id')} "
        f"expected={report.get('expected_identity_id')} "
        f"base_kind={base_payload.get('source_kind')} "
        f"base_commit={base_payload.get('observed_commit')} "
        f"base_files={base_payload.get('local_file_count')} "
        f"tokenizer_kind={tokenizer_payload.get('source_kind')} "
        f"tokenizer_class={tokenizer_payload.get('tokenizer_class')} "
        f"tokenizer_files={tokenizer_payload.get('local_file_count')} "
        f"errors={report.get('error_count')}"
    ]
