from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from itertools import product
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import spiraltorch as st
import zspace_inference_distortion_probe as probe

MappingLike = dict[str, Any]
SUCCESS_STATUSES = {"ok", "reused", "reported"}


def _float_values(raw: str, *, name: str) -> list[float]:
    values = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite floats")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one float")
    return values


def _label_float(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    provided_flags = _provided_flags(raw_argv)
    parser = argparse.ArgumentParser(
        description=(
            "Run a small Z-space inference-distortion grid and compare local HF "
            "/ API-model probe artifacts."
        )
    )
    parser.add_argument("--prompt", default="Describe SpiralTorch as a Z-space runtime.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/zspace-inference-distortion-sweep"))
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--no-markdown-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--from-probe",
        action="append",
        type=Path,
        default=[],
        help=(
            "Promote an existing probe JSON into this sweep report without "
            "calling local/API models. Repeat to compare several saved probes."
        ),
    )
    parser.add_argument(
        "--from-probe-label",
        action="append",
        default=[],
        help="Optional label for the matching --from-probe path.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing successful per-setting probe artifacts instead of rerunning them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun probe artifacts even when --resume-existing would reuse them.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only rebuild sweep-report.json from existing probe artifacts; never call local/API models.",
    )
    parser.add_argument("--local-model", type=Path, default=None)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--activation-module-name", action="append", default=[])
    parser.add_argument("--activation-name-contains", action="append", default=[])
    parser.add_argument(
        "--api-provider",
        choices=["fake", "openai-responses", "openai-chat", "anthropic"],
        default="fake",
    )
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-max-tokens", type=int, default=160)
    parser.add_argument(
        "--api-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Optional OpenAI Responses reasoning effort for GPT-5-style routes.",
    )
    parser.add_argument(
        "--api-text-verbosity",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional OpenAI Responses visible text verbosity.",
    )
    parser.add_argument("--desire-pressure-values", default="0.45,0.8")
    parser.add_argument("--desire-stability-values", default="0.45")
    parser.add_argument("--psi-total-values", default="0.5,0.75")
    parser.add_argument("--coherence-values", default="0.45")
    parser.add_argument("--distortion-strength-values", default="1.0")
    parser.add_argument("--base-temperature", type=float, default=0.7)
    parser.add_argument("--base-top-p", type=float, default=0.95)
    parser.add_argument("--include-penalties", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(raw_argv)
    args.provided_flags = provided_flags
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if args.api_max_tokens <= 0:
        parser.error("--api-max-tokens must be positive")
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.from_probe and args.dry_run:
        parser.error("--from-probe cannot be used with --dry-run")
    if args.from_probe and args.force:
        parser.error("--from-probe cannot be used with --force")
    if args.force and args.report_only:
        parser.error("--force cannot be used with --report-only")
    if len(args.from_probe_label) > len(args.from_probe):
        parser.error("--from-probe-label cannot be provided more times than --from-probe")
    if args.from_probe:
        args.report_only = True
    try:
        args.desire_pressure_grid = _float_values(
            args.desire_pressure_values,
            name="--desire-pressure-values",
        )
        args.desire_stability_grid = _float_values(
            args.desire_stability_values,
            name="--desire-stability-values",
        )
        args.psi_total_grid = _float_values(
            args.psi_total_values,
            name="--psi-total-values",
        )
        args.coherence_grid = _float_values(
            args.coherence_values,
            name="--coherence-values",
        )
        args.distortion_strength_grid = _float_values(
            args.distortion_strength_values,
            name="--distortion-strength-values",
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _provided_flags(argv: list[str]) -> set[str]:
    flags = set()
    for item in argv:
        if item.startswith("--"):
            flags.add(item.split("=", 1)[0])
    return flags


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _runtime_plan(args: argparse.Namespace) -> MappingLike:
    return {
        "local_model": str(args.local_model) if args.local_model is not None else None,
        "allow_remote": bool(args.allow_remote),
        "trust_remote_code": bool(args.trust_remote_code),
        "max_new_tokens": int(args.max_new_tokens),
        "activation_module_name": list(args.activation_module_name),
        "activation_name_contains": list(args.activation_name_contains),
        "api_provider": args.api_provider,
        "api_model": args.api_model,
        "api_max_tokens": int(args.api_max_tokens),
        "api_reasoning_effort": args.api_reasoning_effort,
        "api_text_verbosity": args.api_text_verbosity,
    }


def _markdown_path(args: argparse.Namespace) -> Path | None:
    if args.no_markdown_report:
        return None
    return args.markdown_out or (args.out_dir / "sweep-report.md")


def _shell_join(parts: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _runtime_cli_args(runtime: MappingLike, *, sweep: bool) -> list[object]:
    args: list[object] = []
    if runtime.get("local_model"):
        args.extend(["--local-model", runtime["local_model"]])
    if runtime.get("allow_remote"):
        args.append("--allow-remote")
    if runtime.get("trust_remote_code"):
        args.append("--trust-remote-code")
    if runtime.get("max_new_tokens") is not None:
        args.extend(["--max-new-tokens", runtime["max_new_tokens"]])
    for name in runtime.get("activation_module_name") or []:
        args.extend(["--activation-module-name", name])
    for needle in runtime.get("activation_name_contains") or []:
        args.extend(["--activation-name-contains", needle])
    if runtime.get("api_provider"):
        args.extend(["--api-provider", runtime["api_provider"]])
    if runtime.get("api_model"):
        args.extend(["--api-model", runtime["api_model"]])
    if runtime.get("api_max_tokens") is not None:
        args.extend(["--api-max-tokens", runtime["api_max_tokens"]])
    if runtime.get("api_reasoning_effort"):
        args.extend(["--api-reasoning-effort", runtime["api_reasoning_effort"]])
    if runtime.get("api_text_verbosity"):
        args.extend(["--api-text-verbosity", runtime["api_text_verbosity"]])
    if sweep:
        args.append("--resume-existing")
    return args


def _recommended_commands(report: MappingLike) -> MappingLike:
    summary = report.get("summary")
    runtime = report.get("runtime")
    if not isinstance(summary, dict) or not isinstance(runtime, dict):
        return {}
    if summary.get("recommended_probe") is None:
        return {}
    prompt = report.get("prompt") or ""
    probe_args = [
        "PYTHONPATH=bindings/st-py",
        "python3",
        "bindings/st-py/examples/zspace_inference_distortion_probe.py",
        "--prompt",
        prompt,
    ]
    probe_args.extend(_runtime_cli_args(runtime, sweep=False))
    probe_args.extend(summary.get("recommended_probe_cli_args") or [])
    sweep_args = [
        "PYTHONPATH=bindings/st-py",
        "python3",
        "bindings/st-py/examples/zspace_inference_distortion_sweep.py",
        "--out-dir",
        Path(str(report.get("report_path") or ".")).parent,
        "--prompt",
        prompt,
    ]
    sweep_args.extend(_runtime_cli_args(runtime, sweep=True))
    sweep_args.extend(summary.get("recommended_sweep_cli_args") or [])
    return {
        "probe": _shell_join(probe_args),
        "sweep": _shell_join(sweep_args),
    }


def _recommendation_from_summary(summary: object) -> MappingLike | None:
    if not isinstance(summary, dict) or summary.get("recommended_probe") is None:
        return None
    return {
        "probe": summary.get("recommended_probe"),
        "reason": summary.get("recommendation_reason"),
        "effect_score": summary.get("recommended_effect_score"),
        "risk_score": summary.get("recommended_risk_score"),
        "probe_path": summary.get("recommended_probe_path"),
        "config": summary.get("recommended_config"),
        "request": summary.get("recommended_request"),
        "processor_kwargs": summary.get("recommended_processor_kwargs"),
        "activation_hook": summary.get("recommended_activation_hook"),
        "probe_cli_args": summary.get("recommended_probe_cli_args"),
        "sweep_cli_args": summary.get("recommended_sweep_cli_args"),
    }


def _markdown_table_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _markdown_report(report: MappingLike) -> str:
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    commands = report.get("recommended_commands")
    commands = commands if isinstance(commands, dict) else {}
    lines = [
        "# Z-Space Inference Distortion Sweep",
        "",
        (
            f"- status: `{report.get('status')}` "
            f"({summary.get('completed_run_count')}/{summary.get('run_count')} complete)"
        ),
        f"- prompt: `{report.get('prompt')}`",
        f"- recommended: `{summary.get('recommended_probe')}`",
        f"- effect/risk: `{summary.get('recommended_effect_score')}` / `{summary.get('recommended_risk_score')}`",
        "",
        "## Recommendation",
        "",
    ]
    if summary.get("recommended_config"):
        lines.extend(
            [
                "```json",
                json.dumps(
                    summary.get("recommended_config"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ]
        )
    if commands.get("probe"):
        lines.extend(["Single-probe replay:", "", "```bash", str(commands["probe"]), "```", ""])
    if commands.get("sweep"):
        lines.extend(["Focused sweep replay:", "", "```bash", str(commands["sweep"]), "```", ""])

    lines.extend(
        [
            "## Top Probes",
            "",
            "| rank | label | effect | risk | changed | top changes | api | energy |",
            "| --- | --- | ---: | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for row in summary.get("top_probes", []) if isinstance(summary, dict) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                _markdown_table_value(value)
                for value in [
                    row.get("rank"),
                    row.get("label"),
                    row.get("effect_score"),
                    row.get("risk_score"),
                    row.get("local_changed"),
                    row.get("generation_control_top_token_changed_count"),
                    row.get("api_provider"),
                    row.get("distortion_energy"),
                ]
            )
            + " |"
        )
    if report.get("summary_lines"):
        lines.extend(["", "## Compact Lines", "", "```text"])
        lines.extend(str(line) for line in report.get("summary_lines", []))
        lines.extend(["```"])
    return "\n".join(lines)


def _matching_mapping(expected: MappingLike, actual: object) -> bool:
    return isinstance(actual, dict) and dict(actual) == dict(expected)


def _execution_plan(args: argparse.Namespace) -> MappingLike:
    return {
        "resume_existing": bool(args.resume_existing),
        "force": bool(args.force),
        "report_only": bool(args.report_only),
        "from_probe_count": len(args.from_probe),
    }


def build_sweep_runs(args: argparse.Namespace) -> list[MappingLike]:
    runs = []
    for index, (pressure, stability, psi_total, coherence, strength) in enumerate(
        product(
            args.desire_pressure_grid,
            args.desire_stability_grid,
            args.psi_total_grid,
            args.coherence_grid,
            args.distortion_strength_grid,
        ),
        start=1,
    ):
        name = (
            f"distort-{index:03d}"
            f"-dp{_label_float(pressure)}"
            f"-ds{_label_float(stability)}"
            f"-psi{_label_float(psi_total)}"
            f"-coh{_label_float(coherence)}"
            f"-str{_label_float(strength)}"
        )
        config = {
            "desire_pressure": pressure,
            "desire_stability": stability,
            "psi_total": psi_total,
            "coherence": coherence,
            "distortion_strength": strength,
            "base_temperature": float(args.base_temperature),
            "base_top_p": float(args.base_top_p),
            "include_penalties": bool(args.include_penalties),
        }
        runs.append(
            {
                "name": name,
                "index": index,
                "config": config,
                "probe_path": str(args.out_dir / f"{name}.json"),
            }
        )
    return runs


def _existing_probe_issues(
    args: argparse.Namespace,
    run: MappingLike,
    report: MappingLike,
) -> list[str]:
    issues: list[str] = []
    if report.get("prompt") != args.prompt:
        issues.append("prompt mismatch")
    if not _matching_mapping(dict(run["config"]), report.get("config")):
        issues.append("config mismatch")

    runtime = report.get("runtime")
    if isinstance(runtime, dict):
        expected_runtime = _runtime_plan(args)
        for key, expected in expected_runtime.items():
            if runtime.get(key) != expected:
                issues.append(f"runtime.{key} mismatch")
    else:
        local = report.get("local_hf")
        if isinstance(local, dict):
            expected_model = (
                str(args.local_model) if args.local_model is not None else None
            )
            actual_model = local.get("model")
            if expected_model is None and actual_model is not None:
                issues.append("local model mismatch")
            if expected_model is not None and actual_model != expected_model:
                issues.append("local model mismatch")
        api = report.get("api")
        if isinstance(api, dict):
            if api.get("provider") != args.api_provider:
                issues.append("api provider mismatch")
            if args.api_model is not None and api.get("model") != args.api_model:
                issues.append("api model mismatch")
    return issues


def _load_existing_probe_run(
    args: argparse.Namespace,
    run: MappingLike,
    *,
    status: str,
) -> MappingLike:
    probe_path = Path(str(run["probe_path"]))
    if not probe_path.is_file():
        raise FileNotFoundError(f"{probe_path} does not exist")
    report = st.load_zspace_inference_distortion_probe(probe_path)
    issues = _existing_probe_issues(args, run, report)
    if issues:
        raise ValueError("; ".join(issues))
    summary = st.summarize_zspace_inference_distortion_probe(probe_path)
    return {
        **run,
        "status": status,
        "summary": summary,
        "reused": status == "reused",
        "reported": status == "reported",
    }


def _run_probe(args: argparse.Namespace, run: MappingLike) -> MappingLike:
    config = dict(run["config"])
    adapter = st.api_llm_zspace_inference_distortion_adapter(
        desire_pressure=config["desire_pressure"],
        desire_stability=config["desire_stability"],
        psi_total=config["psi_total"],
        coherence=config["coherence"],
        distortion_strength=config["distortion_strength"],
        base_temperature=config["base_temperature"],
        base_top_p=config["base_top_p"],
        include_penalties=config["include_penalties"],
        activation_module_names=args.activation_module_name,
        activation_name_contains=args.activation_name_contains,
    )
    report: MappingLike = {
        "row_type": "zspace_inference_distortion_probe",
        "name": run["name"],
        "probe_path": str(run["probe_path"]),
        "config": config,
        "prompt": args.prompt,
        "runtime": _runtime_plan(args),
        "adapter": adapter,
        "local_hf": probe._run_local_hf(args, adapter),
        "api": probe._run_api(args, adapter),
    }
    report["summary"] = st.summarize_zspace_inference_distortion_probe(report)
    report["summary_lines"] = st.summarize_zspace_inference_distortion_probe_lines(
        report
    )
    return report


def _comparison_inputs(rows: list[MappingLike]) -> dict[str, str]:
    return {
        str(row["name"]): str(row["probe_path"])
        for row in rows
        if row.get("status") in SUCCESS_STATUSES
    }


def _build_report(
    args: argparse.Namespace,
    *,
    runs: list[MappingLike],
    failed: list[MappingLike],
    attempted_run_count: int,
    reused_run_count: int,
    reported_run_count: int,
    dry_run: bool = False,
    prompt: object | None = None,
    runtime: MappingLike | None = None,
) -> MappingLike:
    comparison = (
        st.compare_zspace_inference_distortion_probes(
            _comparison_inputs(runs),
            top_n=args.top_n,
        )
        if not dry_run
        else None
    )
    failed_run_count = len(failed)
    completed_run_count = sum(1 for run in runs if run.get("status") in SUCCESS_STATUSES)
    missing_run_count = sum(1 for run in runs if run.get("status") == "missing")
    stale_run_count = sum(1 for run in runs if run.get("status") == "stale")
    if dry_run:
        status = "planned"
    elif failed_run_count:
        status = "partial"
    elif args.report_only:
        status = "reported"
    else:
        status = "complete"
    report: MappingLike = {
        "row_type": "zspace_inference_distortion_sweep",
        "status": status,
        "dry_run": bool(dry_run),
        "prompt": args.prompt if prompt is None else prompt,
        "runtime": _runtime_plan(args) if runtime is None else dict(runtime),
        "execution": _execution_plan(args),
        "run_count": len(runs),
        "attempted_run_count": int(attempted_run_count),
        "completed_run_count": int(completed_run_count),
        "failed_run_count": int(failed_run_count),
        "missing_run_count": int(missing_run_count),
        "stale_run_count": int(stale_run_count),
        "reused_run_count": int(reused_run_count),
        "reported_run_count": int(reported_run_count),
        "skipped_run_count": len(runs) if dry_run else 0,
        "runs": runs,
        "comparison": comparison,
        "summary_lines": (
            []
            if comparison is None
            else st.summarize_zspace_inference_distortion_probe_comparison_lines(
                comparison,
                top_n=args.top_n,
            )
        ),
        "plan_path": str(args.out_dir / "sweep-plan.json"),
        "report_path": str(args.out_dir / "sweep-report.json"),
    }
    report["summary"] = st.summarize_zspace_inference_distortion_sweep(
        report,
        top_n=args.top_n,
    )
    report["summary_lines"] = st.summarize_zspace_inference_distortion_sweep_lines(
        report,
        top_n=args.top_n,
    )
    report["recommendation"] = _recommendation_from_summary(report["summary"])
    report["recommended_commands"] = _recommended_commands(report)
    return report


def _probe_runtime(report: MappingLike) -> MappingLike | None:
    runtime = report.get("runtime")
    if isinstance(runtime, dict):
        return dict(runtime)
    api = report.get("api")
    api = api if isinstance(api, dict) else {}
    local = report.get("local_hf")
    local = local if isinstance(local, dict) else {}
    if not api and not local:
        return None
    return {
        "local_model": local.get("model"),
        "allow_remote": None,
        "trust_remote_code": None,
        "max_new_tokens": None,
        "activation_module_name": [],
        "activation_name_contains": [],
        "api_provider": api.get("provider"),
        "api_model": api.get("model"),
        "api_max_tokens": None,
    }


def _unique_probe_name(existing: set[str], raw: object, *, index: int) -> str:
    base = str(raw or f"probe-{index:03d}").strip() or f"probe-{index:03d}"
    name = base
    suffix = 2
    while name in existing:
        name = f"{base}-{suffix}"
        suffix += 1
    existing.add(name)
    return name


def _import_probe_run(
    args: argparse.Namespace,
    probe_path: Path,
    *,
    index: int,
    existing_names: set[str],
) -> tuple[MappingLike, MappingLike | None]:
    report = st.load_zspace_inference_distortion_probe(probe_path)
    summary = st.summarize_zspace_inference_distortion_probe(probe_path)
    label = (
        args.from_probe_label[index - 1]
        if index - 1 < len(args.from_probe_label)
        else report.get("name") or probe_path.stem
    )
    config = report.get("config")
    run = {
        "name": _unique_probe_name(existing_names, label, index=index),
        "index": index,
        "config": dict(config) if isinstance(config, dict) else {},
        "probe_path": str(probe_path),
        "status": "reported",
        "summary": summary,
        "reported": True,
    }
    return run, report


def _run_from_probe_report(args: argparse.Namespace) -> MappingLike:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed: list[MappingLike] = []
    failed: list[MappingLike] = []
    probe_reports: list[MappingLike] = []
    names: set[str] = set()
    for index, probe_path in enumerate(args.from_probe, start=1):
        try:
            run, report = _import_probe_run(
                args,
                probe_path,
                index=index,
                existing_names=names,
            )
            completed.append(run)
            if isinstance(report, dict):
                probe_reports.append(report)
        except Exception as exc:
            failure = {
                "name": _unique_probe_name(names, probe_path.stem, index=index),
                "index": index,
                "config": {},
                "probe_path": str(probe_path),
                "status": "missing"
                if isinstance(exc, FileNotFoundError)
                else "stale",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            failed.append(failure)
            completed.append(failure)
    prompt = args.prompt
    flags = getattr(args, "provided_flags", set())
    first_report = probe_reports[0] if probe_reports else {}
    if "--prompt" not in flags and first_report.get("prompt") is not None:
        prompt = first_report.get("prompt")
    runtime = _probe_runtime(first_report) if first_report else None
    plan: MappingLike = {
        "row_type": "zspace_inference_distortion_sweep_plan",
        "dry_run": False,
        "prompt": prompt,
        "runtime": _runtime_plan(args) if runtime is None else dict(runtime),
        "execution": _execution_plan(args),
        "run_count": len(completed),
        "runs": completed,
        "source_probe_paths": [str(path) for path in args.from_probe],
    }
    _write_json(args.out_dir / "sweep-plan.json", plan)
    report = _build_report(
        args,
        runs=completed,
        failed=failed,
        attempted_run_count=0,
        reused_run_count=0,
        reported_run_count=sum(
            1 for run in completed if run.get("status") == "reported"
        ),
        prompt=prompt,
        runtime=runtime,
    )
    report["source_probe_paths"] = [str(path) for path in args.from_probe]
    _write_report_outputs(args, report)
    return report


def _write_report_outputs(args: argparse.Namespace, report: MappingLike) -> None:
    markdown_path = _markdown_path(args)
    if markdown_path is not None:
        report["markdown_path"] = str(markdown_path)
        _write_text(markdown_path, _markdown_report(report))
    _write_json(args.out_dir / "sweep-report.json", report)


def run_sweep(args: argparse.Namespace) -> MappingLike:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_probe:
        return _run_from_probe_report(args)
    runs = build_sweep_runs(args)
    plan: MappingLike = {
        "row_type": "zspace_inference_distortion_sweep_plan",
        "dry_run": bool(args.dry_run),
        "prompt": args.prompt,
        "runtime": _runtime_plan(args),
        "execution": _execution_plan(args),
        "run_count": len(runs),
        "runs": runs,
    }
    _write_json(args.out_dir / "sweep-plan.json", plan)
    if args.dry_run:
        report = _build_report(
            args,
            runs=[{**run, "status": "planned"} for run in runs],
            failed=[],
            attempted_run_count=0,
            reused_run_count=0,
            reported_run_count=0,
            dry_run=True,
        )
        _write_report_outputs(args, report)
        return report

    completed = []
    failed = []
    attempted_run_count = 0
    reused_run_count = 0
    reported_run_count = 0
    for run in runs:
        if args.report_only:
            try:
                completed.append(_load_existing_probe_run(args, run, status="reported"))
                reported_run_count += 1
            except Exception as exc:
                status = "missing" if isinstance(exc, FileNotFoundError) else "stale"
                failure = {
                    **run,
                    "status": status,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                failed.append(failure)
                completed.append(failure)
            continue
        if args.resume_existing and not args.force:
            try:
                completed.append(_load_existing_probe_run(args, run, status="reused"))
                reused_run_count += 1
                print(f"distortion_probe_reuse {run['name']}")
                continue
            except Exception:
                pass
        print(f"distortion_probe_run {run['name']}")
        attempted_run_count += 1
        try:
            probe_report = _run_probe(args, run)
            _write_json(Path(run["probe_path"]), probe_report)
            summary = st.summarize_zspace_inference_distortion_probe(
                probe_report,
            )
            completed.append({**run, "status": "ok", "summary": summary})
        except Exception as exc:  # pragma: no cover - defensive CLI surface.
            failure = {
                **run,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            failed.append(failure)
            completed.append(failure)

    report = _build_report(
        args,
        runs=completed,
        failed=failed,
        attempted_run_count=attempted_run_count,
        reused_run_count=reused_run_count,
        reported_run_count=reported_run_count,
    )
    _write_report_outputs(args, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sweep(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
