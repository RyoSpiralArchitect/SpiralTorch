# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Inject WASM geometry probes into OpenAI/GPT Z-space inference.

The default path is keyless and deterministic. For a real node-st-wasm to
OpenAI route, first build ``bindings/st-wasm`` with ``wasm-pack --target
nodejs`` and pass the generated JS glue:

    OPENAI_API_KEY=... python \
        bindings/st-py/examples/openai_api_llm_wasm_geometry_injection.py \
        --wasm-pkg /tmp/spiraltorch-wasm-node-pkg/spiraltorch_wasm.js \
        --live-openai --model gpt-5.5 \
        --reasoning-effort none \
        --trace-dir /tmp/spiraltorch-openai-geometry-traces

The API key is read only by the OpenAI SDK and is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import spiraltorch as st
from anthropic_api_llm_wasm_geometry_injection import (
    DEFAULT_CONDITIONS,
    DEFAULT_PROMPT,
    DEFAULT_Z_STATE,
    _compact_trace,
    _context_for,
    _run_trace_label,
    _safe_trace_label,
    builtin_geometry_probe_sets,
    fake_api_model,
    wasm_geometry_probe_sets_from_node,
)


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
DEFAULT_INSTRUCTIONS = (
    "Return the final answer only, in one compact sentence under 24 words. "
    "Treat SpiralTorch Z-space context as runtime telemetry; do not quote or "
    "enumerate the telemetry block."
)


def run_openai_geometry_injection(
    *,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    z_state: Sequence[float] = DEFAULT_Z_STATE,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    probe_sets: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    wasm_pkg: str | os.PathLike[str] | None = None,
    live_openai: bool = False,
    invoke: Callable[..., Any] | None = None,
    instructions: str | None = DEFAULT_INSTRUCTIONS,
    max_output_tokens: int = 180,
    reasoning_effort: str | None = "none",
    repeat: int = 1,
    gradient_dim: int = 6,
    consensus_weight: float = 1.35,
    full_context: bool = False,
    trace_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run baseline/calm/turbulent OpenAI Z-space geometry comparisons."""

    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if probe_sets is None:
        probe_source = "node-st-wasm" if wasm_pkg else "builtin-wasm-shaped"
        resolved_probe_sets = (
            wasm_geometry_probe_sets_from_node(wasm_pkg)
            if wasm_pkg
            else builtin_geometry_probe_sets()
        )
    else:
        probe_source = "provided"
        resolved_probe_sets = dict(probe_sets)
    if live_openai and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY or omit --live-openai")

    rows: dict[str, Any] = {}
    run_rows: dict[str, list[dict[str, Any]]] = {}
    trace_paths: dict[str, str] = {}
    trace_summaries: dict[str, Any] = {}
    run_trace_paths: dict[str, str] = {}
    run_trace_summaries: dict[str, Any] = {}
    trace_root = Path(trace_dir) if trace_dir is not None else None
    if trace_root is not None:
        trace_root.mkdir(parents=True, exist_ok=True)

    for condition in conditions:
        context = _context_for(
            condition,
            resolved_probe_sets,
            gradient_dim=gradient_dim,
            consensus_weight=consensus_weight,
            consensus_only=not full_context,
        )
        condition_rows: list[dict[str, Any]] = []
        condition_traces: list[st.ApiLLMTrace] = []
        for run_index in range(repeat):
            runtime = st.ApiLLMZSpaceRuntime(
                list(z_state),
                provider="openai" if live_openai else "local-demo",
                model=model,
                create_session=False,
                smoothing=0.32,
            )
            if live_openai:
                request: dict[str, Any] = {
                    "max_output_tokens": max_output_tokens,
                    "context_partials": context,
                    "context_prompt": bool(context),
                    "context_prompt_options": {
                        "max_partials": 5 if full_context else 1,
                        "max_metrics": 8,
                        "max_telemetry": 18,
                    },
                }
                if instructions is not None:
                    request["instructions"] = instructions
                if reasoning_effort:
                    request["reasoning"] = {"effort": reasoning_effort}
                trace = runtime.call_openai_responses(
                    prompt,
                    model=model,
                    **request,
                )
            else:
                trace = runtime.call(
                    invoke or fake_api_model,
                    prompt,
                    provider="local-demo",
                    model=model,
                    context_partials=context,
                    context_prompt=bool(context),
                    context_prompt_options={
                        "max_partials": 5 if full_context else 1,
                        "max_metrics": 8,
                        "max_telemetry": 18,
                    },
                )
            condition_rows.append(_compact_trace(trace, context, runtime))
            condition_traces.append(trace)
            if trace_root is not None and repeat > 1:
                run_label = _run_trace_label(str(condition), run_index)
                run_trace_path = trace_root / f"{run_label}.jsonl"
                run_trace_paths[run_label] = st.write_api_llm_trace_jsonl(
                    [trace],
                    run_trace_path,
                )
                run_trace_summaries[run_label] = st.summarize_api_llm_trace_events(
                    run_trace_path
                )
        rows[condition] = condition_rows[-1]
        run_rows[condition] = condition_rows
        if trace_root is not None:
            trace_path = trace_root / f"{_safe_trace_label(str(condition))}.jsonl"
            trace_paths[str(condition)] = st.write_api_llm_trace_jsonl(
                condition_traces,
                trace_path,
            )
            trace_summaries[str(condition)] = st.summarize_api_llm_trace_events(
                trace_path
            )

    result: dict[str, Any] = {
        "kind": "spiraltorch.openai_wasm_geometry_injection",
        "provider": "openai" if live_openai else "local-demo",
        "model": model,
        "prompt": prompt,
        "probe_source": probe_source,
        "context_mode": "full" if full_context else "consensus-only",
        "wasm_pkg": None if wasm_pkg is None else str(wasm_pkg),
        "conditions": list(conditions),
        "repeat": repeat,
        "reasoning_effort": reasoning_effort,
        "results": rows,
    }
    if repeat > 1:
        result["runs"] = run_rows
    if trace_paths:
        result["trace_paths"] = trace_paths
        result["trace_summaries"] = trace_summaries
        result["trace_comparison"] = st.compare_api_llm_trace_runs(trace_paths)
    if run_trace_paths:
        result["run_trace_paths"] = run_trace_paths
        result["run_trace_summaries"] = run_trace_summaries
        result["run_trace_comparison"] = st.compare_api_llm_trace_runs(
            run_trace_paths
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--wasm-pkg", default=None)
    parser.add_argument("--live-openai", action="store_true")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--max-output-tokens", type=int, default=180)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gradient-dim", type=int, default=6)
    parser.add_argument("--consensus-weight", type=float, default=1.35)
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Send per-probe partials plus consensus instead of only geometry:consensus.",
    )
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Optional directory for per-condition API LLM trace JSONL files.",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_openai_geometry_injection(
        prompt=args.prompt,
        model=args.model,
        conditions=args.condition or DEFAULT_CONDITIONS,
        wasm_pkg=args.wasm_pkg,
        live_openai=args.live_openai,
        instructions=args.instructions,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        repeat=args.repeat,
        gradient_dim=args.gradient_dim,
        consensus_weight=args.consensus_weight,
        full_context=args.full_context,
        trace_dir=args.trace_dir,
    )
    text = json.dumps(result, indent=args.indent, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
