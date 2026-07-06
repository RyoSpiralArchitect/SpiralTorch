# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Blend a browser-exported WASM learning report into API-model Z-space runtime.

This example is intentionally keyless and network-free. It uses a small local
WASM-report-shaped mapping plus a fake hosted-model response so the whole bridge
can be exercised in CI, notebooks, or first-run shells:

1. Browser WASM demo exports a report JSON.
2. Python converts that report into a Z-space context partial.
3. API-model responses are inferred with that browser-side learning context.

Replace ``canvas_wasm_report()`` with ``st.load_wasm_report("report.json")`` and
``fake_api`` with ``runtime.call_openai_responses(...)`` or
``runtime.call_anthropic_messages(...)`` for a real hosted-model route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import spiraltorch as st


def canvas_wasm_report() -> dict[str, object]:
    """Return a compact Canvas Hypertrain report in the browser export shape."""

    return {
        "schema": "spiraltorch.wasm.canvas_hypertrain_report.v1",
        "kind": "canvas-hypertrain-training",
        "createdAt": "2026-07-06T00:00:00.000Z",
        "runtime": {
            "wasm": True,
            "webgpuAvailable": True,
            "webgpuDeviceReady": True,
            "webgpuTrainerReady": True,
        },
        "config": {
            "width": 8,
            "height": 8,
            "mode": "hyper",
            "objective": "target-mse",
        },
        "target": {
            "dims": {"width": 8, "height": 8},
            "stats": {"count": 64, "finiteCount": 64, "rms": 0.24},
        },
        "currentFrame": {
            "width": 8,
            "height": 8,
            "relationStats": {"count": 64, "finiteCount": 64, "mean": 0.02, "rms": 0.18},
            "fieldStats": {"count": 256, "finiteCount": 256, "rms": 0.31},
            "trailStats": {"count": 448, "finiteCount": 448, "rms": 0.37},
            "desire": {
                "balance": 0.56,
                "stability": 0.86,
                "saturation": 0.14,
                "eventsMask": 1,
            },
            "gradients": {
                "hypergradRms": 0.11,
                "realgradRms": 0.07,
                "hypergradCount": 64,
                "realgradCount": 64,
            },
            "learningControl": {
                "hyperLearningRateScale": 0.92,
                "realLearningRateScale": 0.81,
                "operatorMix": 0.38,
                "operatorGain": 0.74,
            },
        },
        "metrics": {
            "step": 6,
            "historyLength": 6,
            "truncated": False,
            "last": {"step": 5, "loss": 0.041, "hyperRms": 0.11, "realRms": 0.07},
            "lossStats": {
                "count": 6,
                "finiteCount": 6,
                "min": 0.041,
                "max": 0.19,
                "mean": 0.082,
                "rms": 0.095,
            },
            "hyperRmsStats": {"count": 6, "finiteCount": 6, "mean": 0.11, "rms": 0.11},
            "realRmsStats": {"count": 6, "finiteCount": 6, "mean": 0.07, "rms": 0.07},
            "lrStats": {"count": 6, "finiteCount": 6, "mean": 0.018, "rms": 0.018},
        },
    }


def fake_api(prompt: str) -> dict[str, object]:
    """Return an OpenAI Responses-shaped local payload."""

    text = (
        "The hosted model reads the browser WASM context as a live Z-space "
        f"prior before answering: {prompt}"
    )
    return {
        "id": "resp-wasm-context-demo",
        "object": "response",
        "model": "local-wasm-context-demo",
        "output_text": text,
        "status": "completed",
        "usage": {
            "input_tokens": max(1, len(prompt.split())),
            "output_tokens": max(1, len(text.split())),
        },
    }


DEFAULT_PROMPTS = (
    "Use the browser-side training signal as context.",
    "Explain one safe next step for runtime fine-tuning.",
)
DEFAULT_Z_STATE = (0.12, -0.04, 0.33, -0.11)


def _parse_z_state(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_Z_STATE)
    values = [part.strip() for part in raw.split(",")]
    parsed = [float(part) for part in values if part]
    if not parsed:
        raise ValueError("--z-state must contain at least one number")
    return parsed


def _load_reports(paths: Sequence[str] | None) -> list[dict[str, Any]]:
    if not paths:
        return [canvas_wasm_report()]
    return [st.load_wasm_report(path) for path in paths]


def _compact_report_summary(summary: dict[str, Any]) -> dict[str, Any]:
    learning = summary.get("learning")
    runtime = summary.get("runtime")
    loss = None
    if isinstance(learning, dict):
        loss = learning.get("final_loss")
        if loss is None:
            loss = learning.get("last_loss")
    return {
        "schema": summary.get("schema"),
        "kind": summary.get("kind"),
        "family": summary.get("family"),
        "artifact_path": summary.get("artifact_path"),
        "loss": loss,
        "webgpu_available": runtime.get("webgpu_available")
        if isinstance(runtime, dict)
        else None,
        "webgpu_device_ready": runtime.get("webgpu_device_ready")
        if isinstance(runtime, dict)
        else None,
    }


def run_demo(
    *,
    wasm_reports: Sequence[str] | None = None,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    z_state: Sequence[float] = DEFAULT_Z_STATE,
    gradient_dim: int = 6,
    provider: str = "local-demo",
    model: str = "local-wasm-context-demo",
    trace_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    reports = _load_reports(wasm_reports)
    report_summaries = [st.summarize_wasm_report(report) for report in reports]
    context_partials = st.api_llm_wasm_context_partials(
        reports,
        gradient_dim=gradient_dim,
    )

    suite = st.run_api_llm_prompt_suite(
        list(prompts),
        fake_api,
        z_state=list(z_state),
        provider=provider,
        model=model,
        create_session=False,
        context_partials=context_partials,
        jsonl_out=trace_jsonl,
    )

    first_inference = suite["traces"][0]["inference"]
    telemetry = first_inference["telemetry"]["payload"] if first_inference else {}
    return {
        "kind": "spiraltorch.api_llm_wasm_context_runtime_demo",
        "report_count": len(reports),
        "reports": [_compact_report_summary(summary) for summary in report_summaries],
        "context_origins": [partial.origin for partial in context_partials],
        "trace_count": suite["count"],
        "trace_jsonl": suite.get("jsonl"),
        "summary": suite["summary"],
        "first_confidence": first_inference["confidence"] if first_inference else None,
        "wasm_context_seen": {
            "family_canvas": telemetry.get("wasm.family_canvas"),
            "family_mellin": telemetry.get("wasm.family_mellin"),
            "webgpu_device_ready": telemetry.get("wasm.webgpu_device_ready"),
            "loss": telemetry.get("wasm.loss"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blend browser-exported SpiralTorch WASM report JSON into a keyless "
            "API LLM Z-space prompt suite."
        )
    )
    parser.add_argument(
        "--wasm-report",
        action="append",
        default=[],
        help=(
            "Path to a WASM report JSON exported by mellin-log-grid or "
            "canvas-hypertrain. Repeat to blend multiple reports. If omitted, "
            "a compact built-in Canvas report is used."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to run through the local fake API. Repeat for a suite.",
    )
    parser.add_argument(
        "--z-state",
        default=None,
        help="Comma-separated Z-state vector. Defaults to a small four-value seed.",
    )
    parser.add_argument("--gradient-dim", type=int, default=6)
    parser.add_argument("--provider", default="local-demo")
    parser.add_argument("--model", default="local-wasm-context-demo")
    parser.add_argument(
        "--trace-jsonl",
        default=None,
        help="Optional path for API LLM trace JSONL output.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for the compact demo summary JSON.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    result = run_demo(
        wasm_reports=args.wasm_report,
        prompts=prompts,
        z_state=_parse_z_state(args.z_state),
        gradient_dim=args.gradient_dim,
        provider=args.provider,
        model=args.model,
        trace_jsonl=args.trace_jsonl,
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
