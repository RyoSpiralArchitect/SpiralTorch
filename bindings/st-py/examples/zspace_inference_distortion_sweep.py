from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(
        description=(
            "Run a small Z-space inference-distortion grid and compare local HF "
            "/ API-model probe artifacts."
        )
    )
    parser.add_argument("--prompt", default="Describe SpiralTorch as a Z-space runtime.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/zspace-inference-distortion-sweep"))
    parser.add_argument("--dry-run", action="store_true")
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
    parser.add_argument("--desire-pressure-values", default="0.45,0.8")
    parser.add_argument("--desire-stability-values", default="0.45")
    parser.add_argument("--psi-total-values", default="0.5,0.75")
    parser.add_argument("--coherence-values", default="0.45")
    parser.add_argument("--distortion-strength-values", default="1.0")
    parser.add_argument("--base-temperature", type=float, default=0.7)
    parser.add_argument("--base-top-p", type=float, default=0.95)
    parser.add_argument("--include-penalties", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if args.api_max_tokens <= 0:
        parser.error("--api-max-tokens must be positive")
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.force and args.report_only:
        parser.error("--force cannot be used with --report-only")
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


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    }


def _matching_mapping(expected: MappingLike, actual: object) -> bool:
    return isinstance(actual, dict) and dict(actual) == dict(expected)


def _execution_plan(args: argparse.Namespace) -> MappingLike:
    return {
        "resume_existing": bool(args.resume_existing),
        "force": bool(args.force),
        "report_only": bool(args.report_only),
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
        "prompt": args.prompt,
        "runtime": _runtime_plan(args),
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
    return report


def run_sweep(args: argparse.Namespace) -> MappingLike:
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
        _write_json(args.out_dir / "sweep-report.json", report)
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
    _write_json(args.out_dir / "sweep-report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sweep(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
