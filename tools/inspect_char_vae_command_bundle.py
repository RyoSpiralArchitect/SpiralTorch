#!/usr/bin/env python3
"""Inspect a generated char VAE command bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.command_bundle_inspection.v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _path_from(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _is_executable(path: Path | None) -> bool:
    return path is not None and path.is_file() and os.access(path, os.X_OK)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _check(
    label: str,
    *,
    path: Path | None,
    ok: bool,
    required: bool,
) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path) if path is not None else None,
        "ok": bool(ok),
        "required": bool(required),
    }


def _declared_output(
    label: str,
    path: Path | None,
) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path) if path is not None else None,
        "exists": _exists(path),
    }


def inspect_bundle(command_dir: Path) -> dict[str, Any]:
    command_dir = command_dir.resolve()
    manifest_path = command_dir / "recommendation.json"
    manifest = _read_json(manifest_path)

    command_scripts = manifest.get("command_scripts")
    command_scripts = command_scripts if isinstance(command_scripts, dict) else {}
    comparison = manifest.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    recommendation = manifest.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, dict) else {}

    comparison_json_path = _path_from(command_scripts.get("comparison_json_path"))
    comparison_markdown_path = _path_from(
        command_scripts.get("comparison_markdown_path")
    )
    readme_path = (
        _path_from(command_scripts.get("readme_path")) or command_dir / "README.md"
    )
    next_path = _path_from(command_scripts.get("next_path"))
    follow_up_path = _path_from(command_scripts.get("follow_up_path"))
    review_path = _path_from(command_scripts.get("review_path"))
    run_json_path = _path_from(command_scripts.get("run_json_path"))
    run_markdown_path = _path_from(command_scripts.get("run_markdown_path"))
    run_history_jsonl_path = _path_from(
        command_scripts.get("run_history_jsonl_path")
    )
    run_history_markdown_path = _path_from(
        command_scripts.get("run_history_markdown_path")
    )
    run_history_summary_path = _path_from(
        command_scripts.get("run_history_summary_path")
    )

    chain_sources = comparison.get("chain_sources")
    if not isinstance(chain_sources, list):
        chain_sources = []
    chain_source_paths = [
        Path(source) for source in chain_sources if isinstance(source, str)
    ]
    missing_chain_sources = [
        str(path) for path in chain_source_paths if not path.exists()
    ]

    checks = [
        _check(
            "manifest",
            path=manifest_path,
            ok=manifest_path.is_file(),
            required=True,
        ),
        _check(
            "comparison_json",
            path=comparison_json_path,
            ok=_exists(comparison_json_path),
            required=True,
        ),
        _check(
            "comparison_markdown",
            path=comparison_markdown_path,
            ok=_exists(comparison_markdown_path),
            required=True,
        ),
        _check("readme", path=readme_path, ok=_exists(readme_path), required=True),
        _check(
            "next_script",
            path=next_path,
            ok=next_path is None or _is_executable(next_path),
            required=False,
        ),
        _check(
            "follow_up_script",
            path=follow_up_path,
            ok=follow_up_path is None or _is_executable(follow_up_path),
            required=False,
        ),
        _check(
            "review_script",
            path=review_path,
            ok=review_path is None or _is_executable(review_path),
            required=False,
        ),
        _check(
            "chain_sources",
            path=None,
            ok=bool(chain_source_paths) and not missing_chain_sources,
            required=True,
        ),
    ]
    missing_required = [
        check["label"] for check in checks if check["required"] and not check["ok"]
    ]
    missing_optional = [
        check["label"] for check in checks if not check["required"] and not check["ok"]
    ]
    declared_outputs = [
        _declared_output("run_json", run_json_path),
        _declared_output("run_markdown", run_markdown_path),
        _declared_output("run_history_jsonl", run_history_jsonl_path),
        _declared_output("run_history_markdown", run_history_markdown_path),
        _declared_output("run_history_summary", run_history_summary_path),
    ]
    return {
        "schema": SCHEMA,
        "command_dir": str(command_dir),
        "manifest_path": str(manifest_path),
        "bundle_ready": not missing_required,
        "strict_ready": not missing_required and not missing_optional,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "checks": checks,
        "action": recommendation.get("action"),
        "next_kind": command_scripts.get("next_kind"),
        "written_count": command_scripts.get("written_count"),
        "comparison_json_path": str(comparison_json_path)
        if comparison_json_path is not None
        else None,
        "comparison_markdown_path": str(comparison_markdown_path)
        if comparison_markdown_path is not None
        else None,
        "declared_outputs": declared_outputs,
        "run_json_path": str(run_json_path) if run_json_path is not None else None,
        "run_markdown_path": (
            str(run_markdown_path) if run_markdown_path is not None else None
        ),
        "run_history_jsonl_path": (
            str(run_history_jsonl_path)
            if run_history_jsonl_path is not None
            else None
        ),
        "run_history_markdown_path": (
            str(run_history_markdown_path)
            if run_history_markdown_path is not None
            else None
        ),
        "run_history_summary_path": (
            str(run_history_summary_path)
            if run_history_summary_path is not None
            else None
        ),
        "chain_source_count": len(chain_source_paths),
        "missing_chain_sources": missing_chain_sources,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Char VAE Command Bundle Inspection",
        "",
        f"- schema: {summary.get('schema')}",
        f"- command_dir: {_fmt(summary.get('command_dir'))}",
        f"- manifest_path: {_fmt(summary.get('manifest_path'))}",
        f"- bundle_ready: {_fmt(summary.get('bundle_ready'))}",
        f"- strict_ready: {_fmt(summary.get('strict_ready'))}",
        f"- action: {_fmt(summary.get('action'))}",
        f"- next_kind: {_fmt(summary.get('next_kind'))}",
        f"- written_count: {_fmt(summary.get('written_count'))}",
        f"- comparison_json_path: {_fmt(summary.get('comparison_json_path'))}",
        f"- comparison_markdown_path: {_fmt(summary.get('comparison_markdown_path'))}",
        f"- run_json_path: {_fmt(summary.get('run_json_path'))}",
        f"- run_markdown_path: {_fmt(summary.get('run_markdown_path'))}",
        f"- run_history_jsonl_path: {_fmt(summary.get('run_history_jsonl_path'))}",
        f"- run_history_markdown_path: {_fmt(summary.get('run_history_markdown_path'))}",
        f"- run_history_summary_path: {_fmt(summary.get('run_history_summary_path'))}",
        f"- inspection_json_path: {_fmt(summary.get('inspection_json_path'))}",
        f"- inspection_markdown_path: {_fmt(summary.get('inspection_markdown_path'))}",
        f"- chain_source_count: {_fmt(summary.get('chain_source_count'))}",
        f"- missing_required: {_fmt(', '.join(summary.get('missing_required') or []))}",
        f"- missing_optional: {_fmt(', '.join(summary.get('missing_optional') or []))}",
        f"- missing_chain_sources: {_fmt(', '.join(summary.get('missing_chain_sources') or []))}",
        "",
        "## Checks",
        "",
        "| label | required | ok | path |",
        "| --- | --- | --- | --- |",
    ]
    checks = summary.get("checks")
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, dict):
            continue
        lines.append(
            "| {label} | {required} | {ok} | {path} |".format(
                label=_fmt(check.get("label")),
                required=_fmt(check.get("required")),
                ok=_fmt(check.get("ok")),
                path=_fmt(check.get("path")),
            )
        )
    lines.extend(
        [
            "",
            "## Declared Run Outputs",
            "",
            "| label | exists | path |",
            "| --- | --- | --- |",
        ]
    )
    outputs = summary.get("declared_outputs")
    for output in outputs if isinstance(outputs, list) else []:
        if not isinstance(output, dict):
            continue
        lines.append(
            "| {label} | {exists} | {path} |".format(
                label=_fmt(output.get("label")),
                exists=_fmt(output.get("exists")),
                path=_fmt(output.get("path")),
            )
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command_dir", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print JSON instead of Markdown",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero when required or optional declared artifacts are missing",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="write inspection.json and inspection.md into the command bundle",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = inspect_bundle(args.command_dir)
    command_dir = Path(summary["command_dir"])
    json_out = args.json_out
    markdown_out = args.markdown_out
    if args.write_report:
        if json_out is None:
            json_out = command_dir / "inspection.json"
        if markdown_out is None:
            markdown_out = command_dir / "inspection.md"
    summary["inspection_json_path"] = str(json_out) if json_out is not None else None
    summary["inspection_markdown_path"] = (
        str(markdown_out) if markdown_out is not None else None
    )
    markdown = render_markdown(summary)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(markdown, end="")
    if args.strict:
        return 0 if summary["strict_ready"] else 1
    return 0 if summary["bundle_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
