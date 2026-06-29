#!/usr/bin/env python3
"""Run or inspect promoted char VAE recipe eval commands."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.promoted_recipe_eval_run.v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _summary_path(value: Path) -> Path:
    return value / "summary.json" if value.is_dir() else value


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _seed_run_dir(root: Path, seed: int) -> Path:
    if seed < 0:
        return root / f"seed_neg_{abs(seed):06d}"
    return root / f"seed_{seed:06d}"


def _parse_seed_csv(value: Any) -> list[int]:
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        values = value.split(",")
    else:
        return []
    seeds: list[int] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        seeds.append(int(text))
    return seeds


def _remove_value_flags(command: list[str], flags: set[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(command):
        part = command[idx]
        if part in flags:
            idx += 2
            continue
        out.append(part)
        idx += 1
    return out


def _set_value_flag(command: list[str], flag: str, value: Any) -> list[str]:
    out = list(command)
    text = str(value)
    for idx, part in enumerate(out):
        if part == flag:
            if idx + 1 < len(out):
                out[idx + 1] = text
            else:
                out.append(text)
            return out
    out.extend([flag, text])
    return out


def _append_bool_flag(command: list[str], flag: str) -> list[str]:
    out = [part for part in command if part != flag]
    out.append(flag)
    return out


def _mainline_base_command(mainline: dict[str, Any]) -> list[str]:
    script_command = mainline.get("script_command")
    if isinstance(script_command, list) and script_command:
        return [str(part) for part in script_command]
    shell_command = mainline.get("shell_command")
    if isinstance(shell_command, str) and shell_command:
        parts = shlex.split(shell_command)
        if parts and parts[0].startswith("PYTHONNOUSERSITE="):
            parts = parts[1:]
        return parts
    raise ValueError("mainline_scale_up_command has no script_command")


def _legacy_eval_command(
    mainline: dict[str, Any],
    *,
    seed: int,
    seed_dir: Path,
) -> list[str]:
    command = _mainline_base_command(mainline)
    command = _remove_value_flags(
        command,
        {
            "--seeds",
            "--follow-up-from",
            "--follow-up-fail-on-verdict",
            "--follow-up-used-seeds",
            "--follow-up-confirm-tolerance",
        },
    )
    command = _set_value_flag(command, "--seed", seed)
    command = _set_value_flag(command, "--run-dir", seed_dir / "eval_best")
    command = _set_value_flag(command, "--vae-load", seed_dir / "text_vae_weights.bin")
    command = _set_value_flag(command, "--head-load-dir", seed_dir)
    command = _set_value_flag(command, "--head-load-kind", "best")
    command = _set_value_flag(command, "--epochs", 0)
    command = _set_value_flag(command, "--vae-epochs", 0)
    return _append_bool_flag(command, "--eval-only")


def _recipe_from_mainline(mainline: dict[str, Any]) -> dict[str, Any]:
    best_config = _mapping(mainline.get("best_config"))
    family_focus = _mapping(mainline.get("feature_family_focus"))
    feature = best_config.get("best_feature") or family_focus.get("best_feature")
    feature_family = family_focus.get("family")
    run_dir_raw = mainline.get("default_run_dir")
    seeds = _parse_seed_csv(mainline.get("default_new_seeds"))
    if not isinstance(run_dir_raw, str) or not run_dir_raw:
        raise ValueError("mainline_scale_up_command has no default_run_dir")
    if not seeds:
        raise ValueError("mainline_scale_up_command has no default_new_seeds")
    run_dir = Path(run_dir_raw)
    eval_items: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = _seed_run_dir(run_dir, int(seed))
        command = _legacy_eval_command(mainline, seed=int(seed), seed_dir=seed_dir)
        eval_items.append(
            {
                "schema": "st.llm_char_vae_context.promoted_learning_eval_command.v1",
                "seed": int(seed),
                "run_dir": str(seed_dir / "eval_best"),
                "source_run_dir": str(seed_dir),
                "vae_load": str(seed_dir / "text_vae_weights.bin"),
                "head_load_dir": str(seed_dir),
                "head_load_kind": "best",
                "script_command": command,
                "shell_command": "PYTHONNOUSERSITE=1 " + shlex.join(command),
            }
        )
    return {
        "schema": "st.llm_char_vae_context.promoted_learning_recipe.v1",
        "status": "candidate",
        "synthesized_from": "mainline_scale_up_command",
        "feature": feature,
        "feature_family": feature_family,
        "focused_features": mainline.get("focused_features", []),
        "run_budget": {
            "window_chars": mainline.get("train_window_chars"),
            "epochs": mainline.get("train_epochs"),
            "batches": mainline.get("train_batches"),
            "eval_samples": mainline.get("train_eval_samples"),
            "vae_epochs": mainline.get("train_vae_epochs"),
            "vae_batches": mainline.get("train_vae_batches"),
            "gen": mainline.get("train_gen"),
        },
        "eval_reload_commands": {
            "schema": "st.llm_char_vae_context.promoted_learning_eval_commands.v1",
            "count": len(eval_items),
            "items": eval_items,
        },
    }


def _load_recipe(summary: dict[str, Any]) -> dict[str, Any]:
    mainline = _mapping(summary.get("mainline_scale_up_command"))
    recipe = _mapping(mainline.get("promoted_learning_recipe"))
    if not recipe:
        recipe = _mapping(summary.get("promoted_learning_recipe"))
    if not recipe and mainline:
        recipe = _recipe_from_mainline(mainline)
    if not recipe:
        raise ValueError("summary has no promoted_learning_recipe")
    return recipe


def _eval_commands(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _mapping(recipe.get("eval_reload_commands"))
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _select_commands(
    commands: list[dict[str, Any]],
    *,
    seeds: list[int],
    limit: int | None,
    ready_only: bool,
    complete_only: bool,
    cwd: Path,
) -> list[dict[str, Any]]:
    selected = commands
    if seeds:
        wanted = set(int(seed) for seed in seeds)
        selected = [
            item for item in selected if int(item.get("seed") or -1) in wanted
        ]
    if ready_only:
        selected = [item for item in selected if _command_ready(item, cwd=cwd)]
    if complete_only:
        selected = [item for item in selected if _source_summary_exists(item, cwd=cwd)]
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def _command_tokens(item: dict[str, Any]) -> list[str]:
    raw = item.get("script_command")
    if isinstance(raw, list) and raw:
        return [str(part) for part in raw]
    shell_command = item.get("shell_command")
    if isinstance(shell_command, str) and shell_command:
        return shlex.split(shell_command)
    raise ValueError(f"eval command for seed={item.get('seed')} has no command")


def _path_exists(value: Any, *, cwd: Path) -> bool | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return path.exists()


def _resolve_path(value: Any, *, cwd: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else cwd / path


def _flag_value(command: list[str], flag: str) -> str | None:
    for idx, part in enumerate(command):
        if part == flag and idx + 1 < len(command):
            return command[idx + 1]
    return None


def _features_from_command(command: list[str]) -> list[str]:
    raw = _flag_value(command, "--features")
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _required_head_paths(
    item: dict[str, Any],
    command: list[str],
    *,
    cwd: Path,
) -> list[Path]:
    head_dir = _resolve_path(item.get("head_load_dir"), cwd=cwd)
    if head_dir is None:
        return []
    kind = str(item.get("head_load_kind") or "best")
    suffix = "_best.json" if kind == "best" else ".json"
    return [
        head_dir / f"head_{feature}{suffix}"
        for feature in _features_from_command(command)
    ]


def _source_summary_path(item: dict[str, Any], *, cwd: Path) -> Path | None:
    source_dir = _resolve_path(item.get("source_run_dir"), cwd=cwd)
    return source_dir / "summary.json" if source_dir is not None else None


def _source_summary_exists(item: dict[str, Any], *, cwd: Path) -> bool:
    path = _source_summary_path(item, cwd=cwd)
    return bool(path is not None and path.exists())


def _command_ready(item: dict[str, Any], *, cwd: Path) -> bool:
    command = _command_tokens(item)
    vae_path = _resolve_path(item.get("vae_load"), cwd=cwd)
    head_paths = _required_head_paths(item, command, cwd=cwd)
    return bool(
        vae_path is not None
        and vae_path.exists()
        and head_paths
        and all(path.exists() for path in head_paths)
    )


def _run_command(command: list[str], *, cwd: Path) -> tuple[int, float]:
    started = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    proc = subprocess.run(command, cwd=cwd, env=env)
    return int(proc.returncode), time.time() - started


def _command_result(
    item: dict[str, Any],
    *,
    cwd: Path,
    execute: bool,
) -> dict[str, Any]:
    command = _command_tokens(item)
    required_heads = _required_head_paths(item, command, cwd=cwd)
    missing_heads = [path for path in required_heads if not path.exists()]
    source_summary = _source_summary_path(item, cwd=cwd)
    result: dict[str, Any] = {
        "seed": item.get("seed"),
        "run_dir": item.get("run_dir"),
        "source_run_dir": item.get("source_run_dir"),
        "vae_load": item.get("vae_load"),
        "head_load_dir": item.get("head_load_dir"),
        "head_load_kind": item.get("head_load_kind"),
        "vae_load_exists": _path_exists(item.get("vae_load"), cwd=cwd),
        "head_load_dir_exists": _path_exists(item.get("head_load_dir"), cwd=cwd),
        "required_head_paths": [str(path) for path in required_heads],
        "required_heads_all_exist": bool(required_heads) and not missing_heads,
        "missing_head_paths": [str(path) for path in missing_heads],
        "source_summary_path": (
            str(source_summary) if source_summary is not None else None
        ),
        "source_summary_exists": bool(
            source_summary is not None and source_summary.exists()
        ),
        "command": command,
        "shell_command": " ".join(shlex.quote(part) for part in command),
        "executed": bool(execute),
        "returncode": None,
        "elapsed_seconds": None,
    }
    if execute:
        returncode, elapsed = _run_command(command, cwd=cwd)
        result["returncode"] = returncode
        result["elapsed_seconds"] = elapsed
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Char VAE Promoted Recipe Eval",
        "",
        f"- summary: `{payload.get('summary_path')}`",
        f"- feature: {payload.get('feature') or '-'}",
        f"- feature_family: {payload.get('feature_family') or '-'}",
        f"- execute: {_fmt(payload.get('execute'))}",
        f"- ready_only: {_fmt(payload.get('ready_only'))}",
        f"- complete_only: {_fmt(payload.get('complete_only'))}",
        f"- selected_count: {payload.get('selected_count')}",
        f"- cwd: `{payload.get('cwd')}`",
        "",
        "| seed | executed | returncode | vae | heads | source_summary | missing_heads | run_dir |",
        "| ---: | --- | ---: | --- | --- | --- | ---: | --- |",
    ]
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "| {seed} | {executed} | {returncode} | {vae} | {heads} | {source} | {missing} | `{run_dir}` |".format(
                seed=_fmt(item.get("seed")),
                executed=_fmt(item.get("executed")),
                returncode=_fmt(item.get("returncode")),
                vae=_fmt(item.get("vae_load_exists")),
                heads=_fmt(item.get("required_heads_all_exist")),
                source=_fmt(item.get("source_summary_exists")),
                missing=len(item.get("missing_head_paths", [])),
                run_dir=item.get("run_dir") or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or execute eval-only commands from a promoted char VAE "
            "learning recipe."
        )
    )
    parser.add_argument("summary_or_dir", type=Path)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="select only eval commands whose VAE and requested feature heads exist",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="select only eval commands whose source run summary exists",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute selected eval commands; default only reports the plan",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary_path = _summary_path(args.summary_or_dir)
    summary = _read_json(summary_path)
    recipe = _load_recipe(summary)
    commands = _eval_commands(recipe)
    cwd = args.cwd.resolve()
    selected = _select_commands(
        commands,
        seeds=args.seed,
        limit=args.limit,
        ready_only=bool(args.ready_only),
        complete_only=bool(args.complete_only),
        cwd=cwd,
    )
    if not selected:
        raise ValueError("no promoted recipe eval commands selected")

    results = [
        _command_result(item, cwd=cwd, execute=bool(args.execute))
        for item in selected
    ]
    returncode = max(
        [int(item["returncode"]) for item in results if item["returncode"] is not None]
        or [0]
    )
    payload = {
        "schema": SCHEMA,
        "summary_path": str(summary_path),
        "feature": recipe.get("feature"),
        "feature_family": recipe.get("feature_family"),
        "execute": bool(args.execute),
        "ready_only": bool(args.ready_only),
        "complete_only": bool(args.complete_only),
        "selected_count": len(results),
        "available_count": len(commands),
        "cwd": str(cwd),
        "results": results,
        "returncode": returncode,
    }
    if args.write_report:
        report_path = summary_path.parent / "promoted_recipe_eval_run.json"
        markdown_path = summary_path.parent / "promoted_recipe_eval_run.md"
        payload["report_path"] = str(report_path)
        payload["markdown_path"] = str(markdown_path)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path.write_text(_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_markdown(payload))
    return returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
