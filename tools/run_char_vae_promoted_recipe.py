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


def _load_recipe(summary: dict[str, Any]) -> dict[str, Any]:
    mainline = _mapping(summary.get("mainline_scale_up_command"))
    recipe = _mapping(mainline.get("promoted_learning_recipe"))
    if not recipe:
        recipe = _mapping(summary.get("promoted_learning_recipe"))
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
) -> list[dict[str, Any]]:
    selected = commands
    if seeds:
        wanted = set(int(seed) for seed in seeds)
        selected = [
            item for item in selected if int(item.get("seed") or -1) in wanted
        ]
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
    result: dict[str, Any] = {
        "seed": item.get("seed"),
        "run_dir": item.get("run_dir"),
        "source_run_dir": item.get("source_run_dir"),
        "vae_load": item.get("vae_load"),
        "head_load_dir": item.get("head_load_dir"),
        "head_load_kind": item.get("head_load_kind"),
        "vae_load_exists": _path_exists(item.get("vae_load"), cwd=cwd),
        "head_load_dir_exists": _path_exists(item.get("head_load_dir"), cwd=cwd),
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
        f"- selected_count: {payload.get('selected_count')}",
        f"- cwd: `{payload.get('cwd')}`",
        "",
        "| seed | executed | returncode | vae | heads | run_dir |",
        "| ---: | --- | ---: | --- | --- | --- |",
    ]
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "| {seed} | {executed} | {returncode} | {vae} | {heads} | `{run_dir}` |".format(
                seed=_fmt(item.get("seed")),
                executed=_fmt(item.get("executed")),
                returncode=_fmt(item.get("returncode")),
                vae=_fmt(item.get("vae_load_exists")),
                heads=_fmt(item.get("head_load_dir_exists")),
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
    selected = _select_commands(commands, seeds=args.seed, limit=args.limit)
    if not selected:
        raise ValueError("no promoted recipe eval commands selected")

    cwd = args.cwd.resolve()
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
        "selected_count": len(results),
        "available_count": len(commands),
        "cwd": str(cwd),
        "results": results,
        "returncode": returncode,
    }
    if args.write_report:
        report_path = summary_path.parent / "promoted_recipe_eval_run.json"
        markdown_path = summary_path.parent / "promoted_recipe_eval_run.md"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path.write_text(_markdown(payload), encoding="utf-8")
        payload["report_path"] = str(report_path)
        payload["markdown_path"] = str(markdown_path)
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
