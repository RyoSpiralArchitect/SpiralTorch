"""Model-zoo helpers for SpiralTorch Python workflows.

This module gives the Python binding a stable way to:
- enumerate available model-zoo recipes,
- resolve script paths from short names,
- execute recipes with consistent defaults.

The helpers work both in a repository checkout and in environments where the
model-zoo root is provided via ``SPIRALTORCH_MODEL_ZOO_ROOT``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ModelZooEntry",
    "list_models",
    "find_model",
    "suggest_models",
    "recommend_model",
    "resolve_model_script",
    "build_model_command",
    "run_model",
    "model_zoo_summary",
    "main",
]


_MODEL_ZOO_ENV_ROOT = "SPIRALTORCH_MODEL_ZOO_ROOT"

_ENTRY_METADATA: dict[str, dict[str, Any]] = {
    "mlp_regression": {
        "task": "regression",
        "family": "mlp",
        "description": "Minimal regression baseline.",
        "tags": ("starter", "tabular"),
    },
    "zconv_classification": {
        "task": "classification",
        "family": "vision",
        "description": "Compact Z-convolution classifier.",
        "tags": ("vision", "zspace"),
    },
    "vision_conv_pool_classification": {
        "task": "classification",
        "family": "vision",
        "description": "Vision classification with convolution + pooling.",
        "tags": ("vision", "conv", "pooling"),
    },
    "mellin_log_grid_classification": {
        "task": "classification",
        "family": "mellin",
        "description": "Mellin log-grid classifier.",
        "tags": ("mellin", "spectral"),
    },
    "maxwell_simulated_z_classification": {
        "task": "classification",
        "family": "maxwell",
        "description": "Maxwell-inspired simulated Z-space classification.",
        "tags": ("physics", "zspace"),
    },
    "zspace_vae_reconstruction": {
        "task": "reconstruction",
        "family": "vae",
        "description": "Z-space VAE reconstruction recipe.",
        "tags": ("vae", "zspace"),
    },
    "zspace_text_vae": {
        "task": "reconstruction",
        "family": "vae",
        "description": "Text-conditioned Z-space VAE recipe.",
        "tags": ("vae", "text", "zspace"),
    },
    "llm_char_finetune": {
        "task": "language-modeling",
        "family": "llm",
        "description": "Character-level fine-tuning baseline.",
        "tags": ("llm", "text"),
    },
    "llm_char_coherence_scan": {
        "task": "language-modeling",
        "family": "llm",
        "description": "Character-level coherence scan.",
        "tags": ("llm", "text", "coherence"),
    },
    "llm_char_coherence_wave": {
        "task": "language-modeling",
        "family": "llm",
        "description": "Wave-conditioned character coherence run.",
        "tags": ("llm", "text", "wave"),
    },
    "llm_char_wave_rnn_mixer": {
        "task": "language-modeling",
        "family": "llm",
        "description": "WaveRNN + mixer character model.",
        "tags": ("llm", "text", "rnn", "mixer"),
    },
}


@dataclass(frozen=True)
class ModelZooEntry:
    key: str
    script_name: str
    task: str
    family: str
    description: str
    tags: tuple[str, ...]
    script_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "script_name": self.script_name,
            "task": self.task,
            "family": self.family,
            "description": self.description,
            "tags": list(self.tags),
            "script_path": str(self.script_path) if self.script_path is not None else None,
        }


def _normalise_key(value: str) -> str:
    key = value.strip().lower()
    if key.endswith(".py"):
        key = key[:-3]
    return key


def _coerce_root(root: str | os.PathLike[str] | None) -> Path | None:
    if root is None:
        return None
    candidate = Path(root).expanduser().resolve()
    if (candidate / "models" / "python").is_dir():
        return candidate
    if candidate.name == "python" and candidate.parent.name == "models":
        return candidate.parent.parent
    return candidate


def _discover_repo_root(root: str | os.PathLike[str] | None = None) -> Path | None:
    explicit = _coerce_root(root)
    if explicit is not None and (explicit / "models" / "python").is_dir():
        return explicit

    env_root = os.environ.get(_MODEL_ZOO_ENV_ROOT)
    if env_root:
        env_path = _coerce_root(env_root)
        if env_path is not None and (env_path / "models" / "python").is_dir():
            return env_path

    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        if (parent / "models" / "python").is_dir():
            return parent

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "models" / "python").is_dir():
            return parent

    return explicit


def _models_dir(root: str | os.PathLike[str] | None = None) -> Path | None:
    repo_root = _discover_repo_root(root=root)
    if repo_root is None:
        return None
    scripts_dir = repo_root / "models" / "python"
    return scripts_dir if scripts_dir.is_dir() else None


def _infer_metadata(key: str) -> tuple[str, str, str, tuple[str, ...]]:
    predefined = _ENTRY_METADATA.get(key)
    if predefined is not None:
        return (
            str(predefined["task"]),
            str(predefined["family"]),
            str(predefined["description"]),
            tuple(str(tag) for tag in predefined.get("tags", ())),
        )

    if "regression" in key:
        task = "regression"
    elif "classification" in key:
        task = "classification"
    elif "vae" in key:
        task = "reconstruction"
    elif "llm" in key or "char" in key or "text" in key:
        task = "language-modeling"
    else:
        task = "other"

    if "llm" in key:
        family = "llm"
    elif "vae" in key:
        family = "vae"
    elif "vision" in key or "conv" in key:
        family = "vision"
    elif "mellin" in key:
        family = "mellin"
    else:
        family = key.split("_", 1)[0] if "_" in key else "misc"

    description = f"{key.replace('_', ' ')} recipe."
    tags = tuple(part for part in key.split("_") if part)
    return task, family, description, tags


def _normalise_tag_set(values: Sequence[str] | None) -> set[str]:
    return {value.strip().lower() for value in (values or ()) if value.strip()}


def _iter_script_paths(
    root: str | os.PathLike[str] | None = None,
    *,
    include_internal: bool,
) -> list[Path]:
    scripts_dir = _models_dir(root=root)
    if scripts_dir is None:
        return []

    scripts = []
    for path in sorted(scripts_dir.glob("*.py")):
        if not include_internal and path.name.startswith("_"):
            continue
        scripts.append(path)
    return scripts


def list_models(
    *,
    root: str | os.PathLike[str] | None = None,
    include_internal: bool = False,
    available_only: bool = False,
    family: str | None = None,
    task: str | None = None,
    tags: Sequence[str] | None = None,
) -> list[ModelZooEntry]:
    discovered = _iter_script_paths(root=root, include_internal=include_internal)
    by_key: dict[str, Path | None] = {
        _normalise_key(path.stem): path for path in discovered
    }

    if not discovered:
        # Still expose known recipes when running outside a full source tree.
        for key in _ENTRY_METADATA:
            if include_internal or not key.startswith("_"):
                by_key.setdefault(key, None)

    family_filter = family.lower() if family else None
    task_filter = task.lower() if task else None
    required_tags = {tag.lower() for tag in (tags or ())}

    entries: list[ModelZooEntry] = []
    for key in sorted(by_key):
        path = by_key[key]
        entry_task, entry_family, description, entry_tags = _infer_metadata(key)
        entry = ModelZooEntry(
            key=key,
            script_name=f"{key}.py",
            task=entry_task,
            family=entry_family,
            description=description,
            tags=entry_tags,
            script_path=path.resolve() if path is not None else None,
        )
        if family_filter and entry.family.lower() != family_filter:
            continue
        if task_filter and entry.task.lower() != task_filter:
            continue
        if required_tags and not required_tags.issubset({tag.lower() for tag in entry.tags}):
            continue
        if available_only and entry.script_path is None:
            continue
        entries.append(entry)

    return entries


def find_model(
    name: str,
    *,
    root: str | os.PathLike[str] | None = None,
    include_internal: bool = False,
    available_only: bool = False,
) -> ModelZooEntry:
    target = _normalise_key(name)
    entries = list_models(
        root=root,
        include_internal=include_internal,
        available_only=available_only,
    )
    by_key = {entry.key: entry for entry in entries}
    if target in by_key:
        return by_key[target]

    candidates = [entry for entry in entries if entry.key.startswith(target)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        labels = ", ".join(entry.key for entry in candidates[:6])
        raise KeyError(f"model '{name}' is ambiguous; matches: {labels}")

    contains = [entry for entry in entries if target in entry.key]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        labels = ", ".join(entry.key for entry in contains[:6])
        raise KeyError(f"model '{name}' is ambiguous; matches: {labels}")

    known = ", ".join(entry.key for entry in entries[:12])
    raise KeyError(f"unknown model '{name}'. known entries: {known}")


def suggest_models(
    query: str | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    include_internal: bool = False,
    available_only: bool = False,
    task: str | None = None,
    family: str | None = None,
    required_tags: Sequence[str] | None = None,
    prefer_tags: Sequence[str] | None = None,
    avoid_tags: Sequence[str] | None = None,
    limit: int | None = 5,
) -> list[ModelZooEntry]:
    entries = list_models(
        root=root,
        include_internal=include_internal,
        available_only=available_only,
    )
    if not entries:
        return []

    query_key = _normalise_key(query) if query else None
    task_filter = task.lower().strip() if task else None
    family_filter = family.lower().strip() if family else None
    required = _normalise_tag_set(required_tags)
    preferred = _normalise_tag_set(prefer_tags)
    avoided = _normalise_tag_set(avoid_tags)

    scored: list[tuple[int, ModelZooEntry]] = []
    for entry in entries:
        entry_tags = {tag.lower() for tag in entry.tags}
        if required and not required.issubset(entry_tags):
            continue

        score = 0
        if query_key:
            if entry.key == query_key:
                score += 120
            elif entry.key.startswith(query_key):
                score += 90
            elif query_key in entry.key:
                score += 55
            else:
                query_tokens = {token for token in query_key.split("_") if token}
                key_tokens = {token for token in entry.key.split("_") if token}
                overlap = len(query_tokens & key_tokens)
                if overlap == 0:
                    continue
                score += overlap * 10

        if task_filter:
            score += 45 if entry.task.lower() == task_filter else -12
        if family_filter:
            score += 25 if entry.family.lower() == family_filter else -8

        if preferred:
            score += 9 * len(preferred & entry_tags)
        if avoided:
            score -= 12 * len(avoided & entry_tags)

        if entry.script_path is not None:
            score += 5
        if "starter" in entry_tags:
            score += 2

        scored.append((score, entry))

    scored.sort(key=lambda item: (-item[0], item[1].key))
    ranked = [entry for _, entry in scored]
    if limit is not None:
        if limit <= 0:
            return []
        ranked = ranked[:limit]
    return ranked


def recommend_model(
    query: str | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    include_internal: bool = False,
    available_only: bool = False,
    task: str | None = None,
    family: str | None = None,
    required_tags: Sequence[str] | None = None,
    prefer_tags: Sequence[str] | None = None,
    avoid_tags: Sequence[str] | None = None,
) -> ModelZooEntry:
    matches = suggest_models(
        query,
        root=root,
        include_internal=include_internal,
        available_only=available_only,
        task=task,
        family=family,
        required_tags=required_tags,
        prefer_tags=prefer_tags,
        avoid_tags=avoid_tags,
        limit=1,
    )
    if matches:
        return matches[0]

    detail_parts = []
    if query:
        detail_parts.append(f"query={query!r}")
    if task:
        detail_parts.append(f"task={task!r}")
    if family:
        detail_parts.append(f"family={family!r}")
    if required_tags:
        detail_parts.append(f"required_tags={list(required_tags)!r}")
    detail = ", ".join(detail_parts) if detail_parts else "no filter criteria"
    raise KeyError(f"no model recommendation available ({detail})")


def resolve_model_script(
    name: str,
    *,
    root: str | os.PathLike[str] | None = None,
    include_internal: bool = False,
) -> Path:
    entry = find_model(name, root=root, include_internal=include_internal)
    if entry.script_path is None:
        raise FileNotFoundError(
            f"model '{entry.key}' is known but its script is not present. "
            f"Set {_MODEL_ZOO_ENV_ROOT} to a SpiralTorch repository root."
        )
    return entry.script_path


def build_model_command(
    name: str,
    *script_args: str,
    root: str | os.PathLike[str] | None = None,
    python_executable: str | os.PathLike[str] | None = None,
) -> list[str]:
    script = resolve_model_script(name, root=root)
    python_bin = (
        str(python_executable)
        if python_executable is not None
        else (sys.executable or "python3")
    )
    return [python_bin, str(script), *script_args]


def run_model(
    name: str,
    *script_args: str,
    root: str | os.PathLike[str] | None = None,
    python_executable: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[str] | list[str]:
    command = build_model_command(
        name,
        *script_args,
        root=root,
        python_executable=python_executable,
    )

    if dry_run:
        return command

    script_path = Path(command[1]).resolve()
    default_cwd = script_path.parents[2]
    process_cwd = Path(cwd).resolve() if cwd is not None else default_cwd
    merged_env = os.environ.copy()
    if env:
        merged_env.update(dict(env))

    return subprocess.run(
        command,
        cwd=str(process_cwd),
        env=merged_env,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def model_zoo_summary(
    *,
    root: str | os.PathLike[str] | None = None,
    include_internal: bool = False,
    available_only: bool = False,
) -> dict[str, Any]:
    entries = list_models(
        root=root,
        include_internal=include_internal,
        available_only=available_only,
    )
    by_family: dict[str, int] = {}
    by_task: dict[str, int] = {}

    for entry in entries:
        by_family[entry.family] = by_family.get(entry.family, 0) + 1
        by_task[entry.task] = by_task.get(entry.task, 0) + 1

    return {
        "count": len(entries),
        "families": dict(sorted(by_family.items())),
        "tasks": dict(sorted(by_task.items())),
        "models": [entry.to_dict() for entry in entries],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpiralTorch model-zoo helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available model-zoo recipes")
    list_parser.add_argument("--root", default=None, help="Override repository root")
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Include internal scripts prefixed with underscore",
    )
    list_parser.add_argument(
        "--available-only",
        action="store_true",
        help="Show only entries with scripts present on disk",
    )
    list_parser.add_argument("--family", default=None, help="Filter by family")
    list_parser.add_argument("--task", default=None, help="Filter by task")
    list_parser.add_argument("--tag", action="append", default=[], help="Filter by tag")
    list_parser.add_argument("--json", action="store_true", help="Emit JSON")

    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Suggest best-matching model-zoo recipes",
    )
    suggest_parser.add_argument("query", nargs="?", default=None, help="Optional free-text query")
    suggest_parser.add_argument("--root", default=None, help="Override repository root")
    suggest_parser.add_argument(
        "--all",
        action="store_true",
        help="Include internal scripts prefixed with underscore",
    )
    suggest_parser.add_argument(
        "--available-only",
        action="store_true",
        help="Show only entries with scripts present on disk",
    )
    suggest_parser.add_argument("--task", default=None, help="Prefer task")
    suggest_parser.add_argument("--family", default=None, help="Prefer family")
    suggest_parser.add_argument(
        "--require-tag",
        action="append",
        default=[],
        help="Require tag (can be repeated)",
    )
    suggest_parser.add_argument(
        "--prefer-tag",
        action="append",
        default=[],
        help="Prefer tag (can be repeated)",
    )
    suggest_parser.add_argument(
        "--avoid-tag",
        action="append",
        default=[],
        help="Penalize tag (can be repeated)",
    )
    suggest_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of suggestions to return",
    )
    suggest_parser.add_argument("--json", action="store_true", help="Emit JSON")

    path_parser = subparsers.add_parser("path", help="Resolve a model-zoo script path")
    path_parser.add_argument("model", help="Model key or script name")
    path_parser.add_argument("--root", default=None, help="Override repository root")

    run_parser = subparsers.add_parser("run", help="Execute a model-zoo recipe")
    run_parser.add_argument("model", help="Model key or script name")
    run_parser.add_argument("--root", default=None, help="Override repository root")
    run_parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help="Python executable used to launch the recipe",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Print command only")
    run_parser.add_argument(
        "--no-check",
        action="store_true",
        help="Do not fail on non-zero exit code",
    )
    run_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the model script",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.command == "list":
        entries = list_models(
            root=args.root,
            include_internal=bool(args.all),
            available_only=bool(args.available_only),
            family=args.family,
            task=args.task,
            tags=args.tag,
        )
        if args.json:
            print(json.dumps([entry.to_dict() for entry in entries], ensure_ascii=True, indent=2))
            return 0

        for entry in entries:
            path = str(entry.script_path) if entry.script_path else "<missing>"
            print(f"{entry.key:32s} {entry.task:18s} {entry.family:10s} {path}")
        return 0

    if args.command == "path":
        print(resolve_model_script(args.model, root=args.root))
        return 0

    if args.command == "suggest":
        entries = suggest_models(
            args.query,
            root=args.root,
            include_internal=bool(args.all),
            available_only=bool(args.available_only),
            task=args.task,
            family=args.family,
            required_tags=args.require_tag,
            prefer_tags=args.prefer_tag,
            avoid_tags=args.avoid_tag,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps([entry.to_dict() for entry in entries], ensure_ascii=True, indent=2))
            return 0

        for entry in entries:
            path = str(entry.script_path) if entry.script_path else "<missing>"
            tags = ",".join(entry.tags) if entry.tags else "-"
            print(f"{entry.key:32s} {entry.task:18s} {entry.family:10s} {tags:24s} {path}")
        return 0

    if args.command == "run":
        script_args = list(args.script_args)
        if script_args and script_args[0] == "--":
            script_args = script_args[1:]
        result = run_model(
            args.model,
            *script_args,
            root=args.root,
            python_executable=args.python_executable,
            dry_run=bool(args.dry_run),
            check=not bool(args.no_check),
        )
        if isinstance(result, list):
            print(" ".join(result))
            return 0
        return int(result.returncode)

    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
