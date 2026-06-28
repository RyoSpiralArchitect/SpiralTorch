#!/usr/bin/env python3
"""Run a generated char VAE command bundle after inspection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "st.llm_char_vae_context.command_bundle_run.v1"
TARGET_KEYS = {
    "next": "next_path",
    "follow-up": "follow_up_path",
    "review": "review_path",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _path_from(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _load_inspector() -> Any:
    module_path = Path(__file__).resolve().with_name(
        "inspect_char_vae_command_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_spiraltorch_char_vae_command_bundle_inspector",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load command bundle inspector from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command_scripts(manifest: dict[str, Any]) -> dict[str, Any]:
    command_scripts = manifest.get("command_scripts")
    return command_scripts if isinstance(command_scripts, dict) else {}


def _selected_script(
    command_scripts: dict[str, Any],
    *,
    target: str,
) -> tuple[str, Path | None]:
    key = TARGET_KEYS[target]
    return key, _path_from(command_scripts.get(key))


def _write_inspection_report(
    inspector: Any,
    command_dir: Path,
    inspection: dict[str, Any],
    command_scripts: dict[str, Any],
) -> dict[str, Any]:
    json_out = _path_from(command_scripts.get("inspection_json_path"))
    markdown_out = _path_from(command_scripts.get("inspection_markdown_path"))
    json_out = json_out or command_dir / "inspection.json"
    markdown_out = markdown_out or command_dir / "inspection.md"
    inspection = dict(inspection)
    inspection["inspection_json_path"] = str(json_out)
    inspection["inspection_markdown_path"] = str(markdown_out)
    markdown = inspector.render_markdown(inspection)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown, encoding="utf-8")
    return inspection


def _run_report_paths(
    command_dir: Path,
    command_scripts: dict[str, Any],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
    write_report: bool,
) -> tuple[Path | None, Path | None]:
    if write_report or json_out is not None or markdown_out is not None:
        json_out = json_out or _path_from(command_scripts.get("run_json_path"))
        markdown_out = markdown_out or _path_from(
            command_scripts.get("run_markdown_path")
        )
        json_out = json_out or command_dir / "run.json"
        markdown_out = markdown_out or command_dir / "run.md"
    return json_out, markdown_out


def _runner_summary(
    *,
    command_dir: Path,
    manifest_path: Path,
    target: str,
    script_key: str,
    script_path: Path | None,
    strict: bool,
    dry_run: bool,
    inspection: dict[str, Any],
    returncode: int | None,
    error: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_seconds: float | None = None,
    executed: bool = False,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "command_dir": str(command_dir),
        "manifest_path": str(manifest_path),
        "target": target,
        "script_key": script_key,
        "script_path": str(script_path) if script_path is not None else None,
        "command_argv": ["bash", str(script_path)]
        if script_path is not None
        else None,
        "execution_cwd": str(command_dir),
        "strict": strict,
        "dry_run": dry_run,
        "executed": bool(executed),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "bundle_ready": bool(inspection.get("bundle_ready")),
        "strict_ready": bool(inspection.get("strict_ready")),
        "missing_required": inspection.get("missing_required") or [],
        "missing_optional": inspection.get("missing_optional") or [],
        "inspection": inspection,
        "returncode": returncode,
        "error": error,
        "stdout": stdout,
        "stderr": stderr,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Char VAE Command Bundle Runner",
        "",
        f"- command_dir: {_fmt(summary.get('command_dir'))}",
        f"- manifest_path: {_fmt(summary.get('manifest_path'))}",
        f"- target: {_fmt(summary.get('target'))}",
        f"- script_key: {_fmt(summary.get('script_key'))}",
        f"- script_path: {_fmt(summary.get('script_path'))}",
        f"- command_argv: {_fmt(summary.get('command_argv'))}",
        f"- execution_cwd: {_fmt(summary.get('execution_cwd'))}",
        f"- strict: {_fmt(summary.get('strict'))}",
        f"- dry_run: {_fmt(summary.get('dry_run'))}",
        f"- executed: {_fmt(summary.get('executed'))}",
        f"- started_at: {_fmt(summary.get('started_at'))}",
        f"- finished_at: {_fmt(summary.get('finished_at'))}",
        f"- duration_seconds: {_fmt(summary.get('duration_seconds'))}",
        f"- bundle_ready: {_fmt(summary.get('bundle_ready'))}",
        f"- strict_ready: {_fmt(summary.get('strict_ready'))}",
        f"- missing_required: {_fmt(summary.get('missing_required'))}",
        f"- missing_optional: {_fmt(summary.get('missing_optional'))}",
        f"- run_json_path: {_fmt(summary.get('run_json_path'))}",
        f"- run_markdown_path: {_fmt(summary.get('run_markdown_path'))}",
        f"- returncode: {_fmt(summary.get('returncode'))}",
        f"- error: {_fmt(summary.get('error'))}",
        "",
    ]
    return "\n".join(lines)


def write_run_artifacts(
    summary: dict[str, Any],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
) -> dict[str, Any]:
    summary = dict(summary)
    summary["run_json_path"] = str(json_out) if json_out is not None else None
    summary["run_markdown_path"] = (
        str(markdown_out) if markdown_out is not None else None
    )
    markdown = render_markdown(summary)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    return summary


def run_bundle(
    command_dir: Path,
    *,
    target: str,
    strict: bool,
    dry_run: bool,
    json_mode: bool,
    write_inspection_report: bool,
    write_run_report: bool,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    started_at = _utc_now()
    started_perf = time.perf_counter()

    def timing_fields() -> dict[str, Any]:
        return {
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.perf_counter() - started_perf, 6),
        }

    command_dir = command_dir.resolve()
    manifest_path = command_dir / "recommendation.json"
    manifest = _read_json(manifest_path)
    command_scripts = _command_scripts(manifest)
    script_key, script_path = _selected_script(command_scripts, target=target)
    json_out, markdown_out = _run_report_paths(
        command_dir,
        command_scripts,
        json_out=json_out,
        markdown_out=markdown_out,
        write_report=write_run_report,
    )
    inspector = _load_inspector()
    inspection = inspector.inspect_bundle(command_dir)
    if write_inspection_report:
        inspection = _write_inspection_report(
            inspector,
            command_dir,
            inspection,
            command_scripts,
        )
    required_ready = bool(inspection.get("strict_ready" if strict else "bundle_ready"))
    if not required_ready:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=1,
            error="command bundle did not pass the requested inspection gate",
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
        )
        return 1, summary
    if script_path is None:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=1,
            error=f"manifest does not declare {script_key}",
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
        )
        return 1, summary
    if dry_run:
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=0,
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
        )
        return 0, summary
    if json_mode:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=command_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        summary = _runner_summary(
            command_dir=command_dir,
            manifest_path=manifest_path,
            target=target,
            script_key=script_key,
            script_path=script_path,
            strict=strict,
            dry_run=dry_run,
            inspection=inspection,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            executed=True,
            **timing_fields(),
        )
        summary = write_run_artifacts(
            summary,
            json_out=json_out,
            markdown_out=markdown_out,
        )
        return result.returncode, summary
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=command_dir,
        check=False,
    )
    summary = _runner_summary(
        command_dir=command_dir,
        manifest_path=manifest_path,
        target=target,
        script_key=script_key,
        script_path=script_path,
        strict=strict,
        dry_run=dry_run,
        inspection=inspection,
        returncode=result.returncode,
        executed=True,
        **timing_fields(),
    )
    summary = write_run_artifacts(
        summary,
        json_out=json_out,
        markdown_out=markdown_out,
    )
    return result.returncode, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command_dir", type=Path)
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_KEYS),
        default="next",
        help="which declared command script to run",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="require only bundle_ready instead of strict_ready before running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect and report the selected command without executing it",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a JSON run summary; command stdout/stderr are captured",
    )
    parser.add_argument(
        "--write-inspection-report",
        action="store_true",
        help="rewrite inspection.json and inspection.md before running",
    )
    parser.add_argument(
        "--write-run-report",
        action="store_true",
        help="write run.json and run.md into the command bundle",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        returncode, summary = run_bundle(
            args.command_dir,
            target=args.target,
            strict=not args.no_strict,
            dry_run=bool(args.dry_run),
            json_mode=bool(args.json),
            write_inspection_report=bool(args.write_inspection_report),
            write_run_report=bool(args.write_run_report),
            json_out=args.json_out,
            markdown_out=args.markdown_out,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        summary = {
            "schema": SCHEMA,
            "command_dir": str(args.command_dir),
            "target": args.target,
            "strict": not args.no_strict,
            "dry_run": bool(args.dry_run),
            "returncode": 1,
            "error": str(exc),
        }
        returncode = 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.dry_run or returncode != 0:
        stream = sys.stderr if returncode != 0 else sys.stdout
        print(render_markdown(summary), file=stream)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
