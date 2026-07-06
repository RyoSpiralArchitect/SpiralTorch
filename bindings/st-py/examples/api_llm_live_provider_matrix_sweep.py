# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Run repeated live API LLM provider matrices and compare their reports.

This is a small orchestration layer around ``api_llm_live_provider_matrix.py``:
it runs the same prompt population across multiple OpenAI/Claude token-budget
pairs, writes one ``report.json`` per pair, then writes a sweep-level
``sweep-report.json`` using ``spiraltorch.compare_api_llm_matrix_reports``.

Example:

    PYTHONPATH=bindings/st-py python3 \
        bindings/st-py/examples/api_llm_live_provider_matrix_sweep.py \
        --prompt-limit 12 --repeat 3 --budget-pairs 192:768,256:1024 \
        --out-dir /tmp/spiraltorch-live-matrix-sweep

Use ``--dry-run`` to inspect the planned sweep without making API calls. Use
``--resume-existing`` to reuse completed per-budget ``report.json`` files
instead of calling providers again. API keys are read from provider SDK
environment variables and are never printed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import spiraltorch as st

import api_llm_live_provider_matrix as live_matrix


def _budget_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                "budget pairs must use OPENAI:ANTHROPIC, e.g. 192:768"
            )
        openai_raw, anthropic_raw = part.split(":", 1)
        try:
            openai_tokens = int(openai_raw)
            anthropic_tokens = int(anthropic_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        if openai_tokens <= 0 or anthropic_tokens <= 0:
            raise argparse.ArgumentTypeError("budget pair values must be positive")
        pairs.append((openai_tokens, anthropic_tokens))
    if not pairs:
        raise argparse.ArgumentTypeError("at least one budget pair is required")
    return pairs


def _run_label(index: int, openai_tokens: int, anthropic_tokens: int) -> str:
    return f"{index:02d}-o{openai_tokens}-a{anthropic_tokens}"


def _route_settings(
    providers: dict[str, str],
    models: dict[str, str],
    request_kwargs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        label: {
            "provider": providers.get(label),
            "model": models.get(label),
            "request_kwargs": request_kwargs.get(label, {}),
        }
        for label in providers
    }


def _existing_report_summary(label: str, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{report_path} does not contain a JSON object")
    comparison = report.get("comparison")
    comparison_map = comparison if isinstance(comparison, dict) else {}
    client_errors = report.get("client_errors")
    client_error_count = len(client_errors) if isinstance(client_errors, list) else 0
    return {
        "label": label,
        "report": str(report_path),
        "reused": True,
        "skipped": report.get("skipped", {}),
        "client_error_count": client_error_count,
        "winners": comparison_map.get("winners"),
        "selection_profiles": comparison_map.get("selection_profiles"),
    }


def _run_config(
    *,
    label: str,
    out_dir: Path,
    prompts: list[str],
    z_state: list[float],
    backend: str | None,
    create_session: bool,
    openai_model: str,
    openai_max_output_tokens: int,
    anthropic_models: list[str],
    anthropic_efforts: list[str],
    anthropic_max_tokens: int,
    near_best_tolerance: float,
    resume_existing: bool,
    force: bool,
) -> dict[str, Any]:
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    if resume_existing and report_path.exists() and not force:
        return _existing_report_summary(label, report_path)

    errors: list[dict[str, str]] = []
    invokes, providers, models, request_kwargs, skipped = live_matrix.build_routes(
        openai_model=openai_model,
        openai_max_output_tokens=openai_max_output_tokens,
        anthropic_models=anthropic_models,
        anthropic_efforts=anthropic_efforts,
        anthropic_max_tokens=anthropic_max_tokens,
        errors=errors,
    )
    if not invokes:
        raise SystemExit("No live routes are available; set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    matrix = st.run_api_llm_prompt_suite_matrix(
        prompts,
        invokes,
        z_state=z_state,
        backend=backend,
        providers=providers,
        models=models,
        create_session=create_session,
        jsonl_dir=run_dir / "traces",
        request_kwargs=request_kwargs,
        near_best_tolerance=near_best_tolerance,
    )
    report = {
        "kind": "spiraltorch.api_llm_live_provider_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sweep_label": label,
        "prompt_count": len(prompts),
        "route_count": len(invokes),
        "z_state": z_state,
        "near_best_tolerance": near_best_tolerance,
        "budget": {
            "openai_max_output_tokens": openai_max_output_tokens,
            "anthropic_max_tokens": anthropic_max_tokens,
        },
        "route_settings": _route_settings(providers, models, request_kwargs),
        "skipped": skipped,
        "client_errors": errors,
        "trace_paths": matrix["trace_paths"],
        "comparison": matrix["comparison"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "label": label,
        "report": str(report_path),
        "reused": False,
        "skipped": skipped,
        "client_error_count": len(errors),
        "winners": (matrix["comparison"] or {}).get("winners"),
        "selection_profiles": (matrix["comparison"] or {}).get("selection_profiles"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="api-llm-live-provider-matrix-sweep")
    parser.add_argument("--prompt-limit", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--create-session", action="store_true")
    parser.add_argument(
        "--z-state",
        type=live_matrix._z_state,
        default=[0.12, -0.04, 0.33, -0.11],
    )
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
    )
    parser.add_argument(
        "--anthropic-models",
        default=os.environ.get(
            "ANTHROPIC_MODELS",
            "claude-opus-4-8,claude-fable-5",
        ),
    )
    parser.add_argument("--anthropic-efforts", default="low,high")
    parser.add_argument(
        "--budget-pairs",
        type=_budget_pairs,
        default=_budget_pairs("192:768,256:1024"),
    )
    parser.add_argument("--near-best-tolerance", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="reuse existing per-budget report.json files instead of re-running APIs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even when --resume-existing finds an existing report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = live_matrix._prompts(prompt_limit=args.prompt_limit, repeat=args.repeat)
    anthropic_models = live_matrix._csv(args.anthropic_models)
    anthropic_efforts = live_matrix._csv(args.anthropic_efforts)
    configs = [
        {
            "label": _run_label(index, openai_tokens, anthropic_tokens),
            "openai_max_output_tokens": openai_tokens,
            "anthropic_max_tokens": anthropic_tokens,
        }
        for index, (openai_tokens, anthropic_tokens) in enumerate(args.budget_pairs)
    ]
    if args.dry_run:
        plan = {
            "kind": "spiraltorch.api_llm_live_provider_matrix_sweep_plan",
            "out_dir": str(out_dir),
            "prompt_count": len(prompts),
            "anthropic_models": anthropic_models,
            "anthropic_efforts": anthropic_efforts,
            "resume_existing": args.resume_existing,
            "force": args.force,
            "configs": configs,
        }
        plan_path = out_dir / "sweep-plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    run_summaries: list[dict[str, Any]] = []
    report_paths: dict[str, str] = {}
    for config in configs:
        summary = _run_config(
            label=config["label"],
            out_dir=out_dir,
            prompts=prompts,
            z_state=args.z_state,
            backend=args.backend,
            create_session=args.create_session,
            openai_model=args.openai_model,
            openai_max_output_tokens=config["openai_max_output_tokens"],
            anthropic_models=anthropic_models,
            anthropic_efforts=anthropic_efforts,
            anthropic_max_tokens=config["anthropic_max_tokens"],
            near_best_tolerance=args.near_best_tolerance,
            resume_existing=args.resume_existing,
            force=args.force,
        )
        run_summaries.append(summary)
        report_paths[summary["label"]] = summary["report"]

    comparison = st.compare_api_llm_matrix_reports(report_paths)
    sweep_report = {
        "kind": "spiraltorch.api_llm_live_provider_matrix_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_count": len(prompts),
        "run_count": len(run_summaries),
        "reused_count": sum(1 for row in run_summaries if row.get("reused")),
        "resume_existing": args.resume_existing,
        "force": args.force,
        "configs": configs,
        "reports": report_paths,
        "runs": run_summaries,
        "comparison": comparison,
    }
    report_path = out_dir / "sweep-report.json"
    report_path.write_text(
        json.dumps(sweep_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "prompt_count": len(prompts),
                "run_count": len(run_summaries),
                "reused_count": sweep_report["reused_count"],
                "profile_winners": comparison.get("profile_winners"),
                "top_route": (comparison.get("routes") or [{}])[0].get("label"),
                "recommendations": comparison.get("recommendations"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
