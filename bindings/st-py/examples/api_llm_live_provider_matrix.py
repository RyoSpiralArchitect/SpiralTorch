# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Run a live OpenAI/Claude API-model matrix through SpiralTorch Z-space.

Requires provider SDKs plus keys in the environment.  Missing providers are
skipped, and request failures are recorded as incomplete trace rows so the
remaining routes can still be compared.

Example:

    PYTHONPATH=bindings/st-py python3 \
        bindings/st-py/examples/api_llm_live_provider_matrix.py \
        --prompt-limit 12 --repeat 3 --out-dir /tmp/spiraltorch-live-matrix

The API keys are read by the provider SDKs from the environment and are never
printed by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import spiraltorch as st


SYSTEM_PROMPT = (
    "You are evaluating SpiralTorch as a Z-space runtime beside hosted LLM "
    "inference. Answer concretely, keep the geometry vocabulary explicit, and "
    "avoid inventing measurements that are not in the prompt. Keep visible "
    "answers under 100 words; if code is requested, use at most 8 lines."
)

PROMPTS = [
    "In two sentences, explain how a hosted LLM answer can become a Z-space partial trace.",
    "Give a compact plan for fine-tuning an LLM while auditing Z-space route health.",
    "Compare tokenized transformer inference with tokenless geometric routing in SpiralTorch.",
    "Name three trace-health metrics that should stop a bad API-model route from winning.",
    "Write a tiny Python sketch that imports spiraltorch and compares two API LLM JSONL traces.",
    "Diagnose a run where Claude returns empty text with stop_reason='refusal'.",
    "Explain how provider latency and total token count should affect a route score.",
    "Suggest one way to fuse transformer hidden states into a SpiralTorch Z-space posterior.",
    "Describe a failure mode where a high-confidence response should still be rejected.",
    "Give a one-paragraph roadmap from API-model tracing to robust LLM fine-tuning.",
    "Explain why route-specific request kwargs matter when comparing OpenAI and Claude.",
    "Summarize how WGPU-first runtime evidence could be carried into LLM trace artifacts.",
]


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _z_state(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("z-state must contain at least one float")
    return parsed


def _prompts(*, prompt_limit: int, repeat: int) -> list[str]:
    selected = PROMPTS if prompt_limit <= 0 else PROMPTS[:prompt_limit]
    repeats = max(1, repeat)
    if repeats == 1:
        return list(selected)
    result: list[str] = []
    for run_index in range(repeats):
        for prompt in selected:
            result.append(f"{prompt}\nRepeat index: {run_index + 1}.")
    return result


def _summary_loss(summary: Mapping[str, Any]) -> float | None:
    def _as_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    learning = summary.get("learning")
    if not isinstance(learning, Mapping):
        return None
    for key in ("final_loss", "last_loss"):
        value = _as_float(learning.get(key))
        if value is not None:
            return value
    loss_stats = learning.get("loss")
    if isinstance(loss_stats, Mapping):
        value = _as_float(loss_stats.get("mean"))
        if value is not None:
            return value
    return None


def _compact_wasm_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    runtime = summary.get("runtime")
    runtime_map = runtime if isinstance(runtime, Mapping) else {}
    return {
        "schema": summary.get("schema"),
        "kind": summary.get("kind"),
        "family": summary.get("family"),
        "artifact_path": summary.get("artifact_path"),
        "loss": _summary_loss(summary),
        "webgpu_available": runtime_map.get("webgpu_available"),
        "webgpu_device_ready": runtime_map.get("webgpu_device_ready"),
    }


def _dedupe_paths(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        key = str(Path(path).expanduser())
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _discover_wasm_report_paths(
    wasm_reports: Sequence[str | os.PathLike[str]] | str | os.PathLike[str] | None,
    *,
    wasm_report_globs: Sequence[str] | None = None,
    wasm_report_dirs: Sequence[str | os.PathLike[str]] | None = None,
    recursive: bool = False,
) -> list[str]:
    paths: list[str] = []
    if wasm_reports is None:
        pass
    elif isinstance(wasm_reports, (str, os.PathLike)):
        paths.append(str(wasm_reports))
    else:
        paths.extend(str(path) for path in wasm_reports if str(path))

    for pattern in wasm_report_globs or ():
        paths.extend(sorted(glob.glob(str(pattern), recursive=True)))

    for raw_dir in wasm_report_dirs or ():
        report_dir = Path(raw_dir)
        iterator = report_dir.rglob("*.json") if recursive else report_dir.glob("*.json")
        paths.extend(str(path) for path in sorted(iterator))

    return _dedupe_paths(paths)


def _select_wasm_report_indices(
    summaries: Sequence[Mapping[str, Any]],
    max_reports: int | None,
) -> list[int]:
    if max_reports is None or max_reports <= 0 or max_reports >= len(summaries):
        return list(range(len(summaries)))

    def _rank_key(index: int) -> tuple[bool, float, int]:
        loss = _summary_loss(summaries[index])
        return (loss is None, loss if loss is not None else 0.0, index)

    ranked = sorted(
        range(len(summaries)),
        key=_rank_key,
    )
    return sorted(ranked[:max_reports])


def _compact_wasm_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    def _row(row: Any) -> dict[str, Any] | None:
        if not isinstance(row, Mapping):
            return None
        return {
            "label": row.get("label"),
            "family": row.get("family"),
            "loss": row.get("loss"),
            "stability": row.get("stability"),
        }

    return {
        "kind": comparison.get("kind"),
        "count": comparison.get("count"),
        "families": comparison.get("families"),
        "best_loss": _row(comparison.get("best_loss")),
        "best_stability": _row(comparison.get("best_stability")),
    }


def build_wasm_context(
    wasm_reports: Sequence[str | os.PathLike[str]] | str | os.PathLike[str] | None,
    *,
    gradient_dim: int,
    bundle_weight: float,
    telemetry_prefix: str,
    wasm_report_globs: Sequence[str] | None = None,
    wasm_report_dirs: Sequence[str | os.PathLike[str]] | None = None,
    max_reports: int | None = None,
    recursive: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Load WASM reports and convert them into reusable API LLM context."""

    candidate_paths = _discover_wasm_report_paths(
        wasm_reports,
        wasm_report_globs=wasm_report_globs,
        wasm_report_dirs=wasm_report_dirs,
        recursive=recursive,
    )
    metadata: dict[str, Any] = {
        "candidate_count": len(candidate_paths),
        "report_count": 0,
        "gradient_dim": int(gradient_dim),
        "bundle_weight": float(bundle_weight),
        "telemetry_prefix": telemetry_prefix,
        "selection": {
            "max_reports": max_reports,
            "recursive": bool(recursive),
        },
        "candidate_reports": [],
        "reports": [],
        "context_origins": [],
        "comparison": None,
    }
    if not candidate_paths:
        return [], metadata

    summaries = [st.summarize_wasm_report(path) for path in candidate_paths]
    selected_indices = _select_wasm_report_indices(summaries, max_reports)
    selected_paths = [candidate_paths[index] for index in selected_indices]
    selected_summaries = [summaries[index] for index in selected_indices]
    comparison = st.compare_wasm_reports(
        candidate_paths,
        labels=[Path(path).stem for path in candidate_paths],
    )
    context_partials = st.api_llm_wasm_context_partials(
        selected_paths,
        gradient_dim=gradient_dim,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
    )
    metadata["report_count"] = len(selected_paths)
    metadata["candidate_reports"] = [
        _compact_wasm_summary(summary) for summary in summaries
    ]
    metadata["reports"] = [
        _compact_wasm_summary(summary) for summary in selected_summaries
    ]
    metadata["context_origins"] = [partial.origin for partial in context_partials]
    metadata["comparison"] = _compact_wasm_comparison(comparison)
    return context_partials, metadata


def _client_error_response(
    *,
    model: str,
    prompt: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "model": model,
        "content": [],
        "stop_reason": "incomplete",
        "stop_details": {
            "category": "client_error",
            "explanation": exc.__class__.__name__,
        },
        "usage": {
            "input_tokens": max(1, len(prompt.split())),
            "output_tokens": 0,
        },
    }


def _guarded(
    label: str,
    model: str,
    invoke: Callable[..., Any],
    errors: list[dict[str, str]],
) -> Callable[..., Any]:
    def _invoke(prompt: str, **request: Any) -> Any:
        try:
            return invoke(prompt, **request)
        except Exception as exc:  # pragma: no cover - live provider behavior varies.
            errors.append({"label": label, "error_type": exc.__class__.__name__})
            return _client_error_response(model=model, prompt=prompt, exc=exc)

    return _invoke


def _anthropic_request(*, max_tokens: int, effort: str) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "extra_body": {"output_config": {"effort": effort}},
    }


def build_routes(
    *,
    openai_model: str,
    openai_max_output_tokens: int,
    anthropic_models: list[str],
    anthropic_efforts: list[str],
    anthropic_max_tokens: int,
    errors: list[dict[str, str]],
) -> tuple[
    dict[str, Callable[..., Any]],
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    invokes: dict[str, Callable[..., Any]] = {}
    providers: dict[str, str] = {}
    models: dict[str, str] = {}
    request_kwargs: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    if os.environ.get("OPENAI_API_KEY"):
        openai_invoke = st.make_openai_responses_invoke(model=openai_model)
        for label, token_multiplier, instructions in (
            (
                "openai-compact",
                1,
                SYSTEM_PROMPT + " Prefer compact, directly auditable answers.",
            ),
            (
                "openai-expanded",
                2,
                SYSTEM_PROMPT + " Allow one extra explanatory step when useful.",
            ),
        ):
            invokes[label] = _guarded(label, openai_model, openai_invoke, errors)
            providers[label] = "openai"
            models[label] = openai_model
            request_kwargs[label] = {
                "instructions": instructions,
                "max_output_tokens": max(1, openai_max_output_tokens * token_multiplier),
            }
    else:
        skipped["openai"] = "OPENAI_API_KEY is not set"

    if os.environ.get("ANTHROPIC_API_KEY"):
        for model in anthropic_models:
            anthropic_invoke = st.make_anthropic_messages_invoke(
                model=model,
                system=SYSTEM_PROMPT,
            )
            for effort in anthropic_efforts:
                label = f"{model}-{effort}-effort"
                invokes[label] = _guarded(label, model, anthropic_invoke, errors)
                providers[label] = "anthropic"
                models[label] = model
                request_kwargs[label] = _anthropic_request(
                    max_tokens=anthropic_max_tokens,
                    effort=effort,
                )
    else:
        skipped["anthropic"] = "ANTHROPIC_API_KEY is not set"

    return invokes, providers, models, request_kwargs, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="api-llm-live-provider-matrix")
    parser.add_argument("--prompt-limit", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--create-session", action="store_true")
    parser.add_argument("--z-state", type=_z_state, default=[0.12, -0.04, 0.33, -0.11])
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--openai-max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--anthropic-models",
        default=os.environ.get("ANTHROPIC_MODELS", "claude-opus-4-8,claude-fable-5"),
    )
    parser.add_argument("--anthropic-efforts", default="low,high")
    parser.add_argument("--anthropic-max-tokens", type=int, default=768)
    parser.add_argument("--near-best-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--wasm-report",
        action="append",
        default=[],
        help=(
            "Path to a browser-exported WASM report JSON. Repeat to blend "
            "multiple browser learning signals into every provider route."
        ),
    )
    parser.add_argument("--wasm-gradient-dim", type=int, default=8)
    parser.add_argument("--wasm-bundle-weight", type=float, default=1.0)
    parser.add_argument("--wasm-telemetry-prefix", default="wasm")
    parser.add_argument(
        "--wasm-report-glob",
        action="append",
        default=[],
        help=(
            "Glob pattern for browser-exported WASM report JSON files. "
            "Repeat to collect reports from several run directories."
        ),
    )
    parser.add_argument(
        "--wasm-report-dir",
        action="append",
        default=[],
        help="Directory containing WASM report JSON files.",
    )
    parser.add_argument(
        "--wasm-report-recursive",
        action="store_true",
        help="Search --wasm-report-dir directories recursively.",
    )
    parser.add_argument(
        "--wasm-max-reports",
        type=int,
        default=0,
        help=(
            "Use only the best N discovered WASM reports by loss. "
            "Use 0 to pass all discovered reports."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    invokes, providers, models, request_kwargs, skipped = build_routes(
        openai_model=args.openai_model,
        openai_max_output_tokens=args.openai_max_output_tokens,
        anthropic_models=_csv(args.anthropic_models),
        anthropic_efforts=_csv(args.anthropic_efforts),
        anthropic_max_tokens=args.anthropic_max_tokens,
        errors=errors,
    )
    if not invokes:
        raise SystemExit("No live routes are available; set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    prompts = _prompts(prompt_limit=args.prompt_limit, repeat=args.repeat)
    context_partials, wasm_context = build_wasm_context(
        args.wasm_report,
        gradient_dim=args.wasm_gradient_dim,
        bundle_weight=args.wasm_bundle_weight,
        telemetry_prefix=args.wasm_telemetry_prefix,
        wasm_report_globs=args.wasm_report_glob,
        wasm_report_dirs=args.wasm_report_dir,
        max_reports=args.wasm_max_reports,
        recursive=args.wasm_report_recursive,
    )
    matrix = st.run_api_llm_prompt_suite_matrix(
        prompts,
        invokes,
        z_state=args.z_state,
        backend=args.backend,
        providers=providers,
        models=models,
        create_session=args.create_session,
        jsonl_dir=out_dir / "traces",
        request_kwargs=request_kwargs,
        near_best_tolerance=args.near_best_tolerance,
        context_partials=context_partials,
    )
    route_settings = {
        label: {
            "provider": providers.get(label),
            "model": models.get(label),
            "request_kwargs": request_kwargs.get(label, {}),
        }
        for label in invokes
    }
    report = {
        "kind": "spiraltorch.api_llm_live_provider_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_count": len(prompts),
        "route_count": len(invokes),
        "z_state": args.z_state,
        "near_best_tolerance": args.near_best_tolerance,
        "route_settings": route_settings,
        "wasm_context": wasm_context,
        "skipped": skipped,
        "client_errors": errors,
        "trace_paths": matrix["trace_paths"],
        "comparison": matrix["comparison"],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "prompt_count": report["prompt_count"],
                "routes": matrix["labels"],
                "skipped": skipped,
                "client_error_count": len(errors),
                "wasm_report_count": wasm_context["report_count"],
                "wasm_context_origins": wasm_context["context_origins"],
                "winners": (matrix["comparison"] or {}).get("winners"),
                "selection_profiles": (matrix["comparison"] or {}).get(
                    "selection_profiles"
                ),
                "near_best": (matrix["comparison"] or {}).get("near_best"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
