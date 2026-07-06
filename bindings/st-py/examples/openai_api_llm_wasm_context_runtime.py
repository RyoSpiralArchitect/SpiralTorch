# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""OpenAI Responses API + browser WASM report -> SpiralTorch Z-space runtime.

Requires ``OPENAI_API_KEY`` and the optional ``openai`` Python package:

    PYTHONPATH=bindings/st-py python3 \
        bindings/st-py/examples/openai_api_llm_wasm_context_runtime.py \
        --wasm-report report.json \
        --trace-jsonl /tmp/spiraltorch-openai-wasm-trace.jsonl

The API key is read by the OpenAI SDK from the environment and is never printed.
Use ``--wasm-context-artifact`` to replay a persisted handoff created by
``spiraltorch.write_wasm_report_context_artifact(...)``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import spiraltorch as st


DEFAULT_PROMPTS = (
    "Use the SpiralTorch WASM context telemetry. In three concise bullets, "
    "diagnose the browser-side learning run and choose one next experiment."
)
DEFAULT_Z_STATE = (0.14, -0.05, 0.28, 0.09, -0.12, 0.22)


def _parse_z_state(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_Z_STATE)
    values = [part.strip() for part in raw.split(",")]
    parsed = [float(part) for part in values if part]
    if not parsed:
        raise ValueError("--z-state must contain at least one number")
    return parsed


def _telemetry_payload(trace: Mapping[str, Any] | None) -> dict[str, Any]:
    if not trace:
        return {}
    inference = trace.get("inference")
    candidates: list[Any] = []
    if isinstance(inference, Mapping):
        candidates.append(inference.get("telemetry"))
    candidates.append(trace.get("telemetry"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            payload = candidate.get("payload")
            if isinstance(payload, Mapping):
                return dict(payload)
            return dict(candidate)
    return {}


def _usage_payload(trace: Mapping[str, Any] | None) -> dict[str, Any]:
    if not trace:
        return {}
    usage = trace.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}


def _text_preview(trace: Mapping[str, Any] | None, *, limit: int = 700) -> str:
    if not trace:
        return ""
    text = trace.get("text")
    if not isinstance(text, str):
        inference = trace.get("inference")
        if isinstance(inference, Mapping):
            inferred = inference.get("text")
            text = inferred if isinstance(inferred, str) else ""
        else:
            text = ""
    return text[:limit]


def _context_report_count(metadata: Mapping[str, Any]) -> int:
    reports = metadata.get("reports")
    if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes, bytearray)):
        return len(reports)
    count = metadata.get("report_count")
    return int(count) if isinstance(count, (int, float)) else 0


def load_or_build_wasm_context(
    *,
    wasm_context_artifact: str | Path | None = None,
    wasm_reports: Sequence[str | Path] | None = None,
    wasm_report_globs: Sequence[str] | None = None,
    wasm_report_dirs: Sequence[str | Path] | None = None,
    wasm_report_recursive: bool = False,
    wasm_max_reports: int | None = None,
    gradient_dim: int = 6,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "wasm",
    write_wasm_context_artifact: str | Path | None = None,
) -> tuple[list[Any], dict[str, Any], str | None]:
    """Return context partials, metadata, and an optional artifact path."""

    has_report_inputs = bool(wasm_reports or wasm_report_globs or wasm_report_dirs)
    if wasm_context_artifact is not None:
        if has_report_inputs or write_wasm_context_artifact is not None:
            raise ValueError(
                "--wasm-context-artifact is mutually exclusive with report collection "
                "and --write-wasm-context-artifact"
            )
        context, metadata = st.load_wasm_report_context_artifact(wasm_context_artifact)
        enriched = dict(metadata)
        enriched["context_source"] = "artifact"
        return context, enriched, str(wasm_context_artifact)

    if not has_report_inputs:
        raise ValueError(
            "provide --wasm-report, --wasm-report-glob, --wasm-report-dir, "
            "or --wasm-context-artifact"
        )

    if write_wasm_context_artifact is not None:
        written = st.write_wasm_report_context_artifact(
            write_wasm_context_artifact,
            list(wasm_reports or ()),
            report_globs=list(wasm_report_globs or ()),
            report_dirs=list(wasm_report_dirs or ()),
            max_reports=wasm_max_reports,
            recursive=wasm_report_recursive,
            bundle_weight=bundle_weight,
            telemetry_prefix=telemetry_prefix,
            gradient_dim=gradient_dim,
        )
        context, metadata = st.load_wasm_report_context_artifact(written)
        enriched = dict(metadata)
        enriched["context_source"] = "built-and-written"
        return context, enriched, written

    context, metadata = st.build_wasm_report_context(
        list(wasm_reports or ()),
        report_globs=list(wasm_report_globs or ()),
        report_dirs=list(wasm_report_dirs or ()),
        max_reports=wasm_max_reports,
        recursive=wasm_report_recursive,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )
    enriched = dict(metadata)
    enriched["context_source"] = "built"
    return context, enriched, None


def run_openai_wasm_context(
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    z_state: Sequence[float] = DEFAULT_Z_STATE,
    model: str | None = None,
    backend: str | None = "auto",
    max_output_tokens: int = 180,
    trace_jsonl: str | Path | None = None,
    instructions: str | None = None,
    invoke: Callable[..., Any] | None = None,
    wasm_context_artifact: str | Path | None = None,
    wasm_reports: Sequence[str | Path] | None = None,
    wasm_report_globs: Sequence[str] | None = None,
    wasm_report_dirs: Sequence[str | Path] | None = None,
    wasm_report_recursive: bool = False,
    wasm_max_reports: int | None = None,
    gradient_dim: int = 6,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "wasm",
    write_wasm_context_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Run OpenAI-compatible inference with selected browser WASM context."""

    selected_model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    context_partials, context_metadata, context_artifact_path = load_or_build_wasm_context(
        wasm_context_artifact=wasm_context_artifact,
        wasm_reports=wasm_reports,
        wasm_report_globs=wasm_report_globs,
        wasm_report_dirs=wasm_report_dirs,
        wasm_report_recursive=wasm_report_recursive,
        wasm_max_reports=wasm_max_reports,
        gradient_dim=gradient_dim,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        write_wasm_context_artifact=write_wasm_context_artifact,
    )
    if invoke is None:
        invoke = st.make_openai_responses_invoke(
            model=selected_model,
            max_output_tokens=max_output_tokens,
        )

    request_kwargs: dict[str, Any] = {}
    if instructions is not None:
        request_kwargs["instructions"] = instructions
    suite = st.run_api_llm_prompt_suite(
        list(prompts),
        invoke,
        z_state=list(z_state),
        backend=backend,
        provider="openai",
        model=selected_model,
        create_session=False,
        context_partials=context_partials,
        jsonl_out=trace_jsonl,
        **request_kwargs,
    )

    traces = suite.get("traces") or []
    first_trace = traces[0] if traces and isinstance(traces[0], Mapping) else {}
    telemetry = _telemetry_payload(first_trace)
    return {
        "kind": "spiraltorch.openai_wasm_context_runtime",
        "provider": "openai",
        "model": selected_model,
        "prompt_count": len(prompts),
        "trace_count": suite.get("count", len(traces)),
        "trace_jsonl": suite.get("jsonl"),
        "context_artifact": context_artifact_path,
        "wasm_context": {
            "source": context_metadata.get("context_source"),
            "report_count": _context_report_count(context_metadata),
            "context_origins": context_metadata.get("context_origins"),
            "reports": context_metadata.get("reports"),
            "comparison": context_metadata.get("comparison"),
            "artifact_schema": context_metadata.get("artifact_schema"),
        },
        "wasm_context_seen": {
            "family_canvas": telemetry.get(f"{telemetry_prefix}.family_canvas"),
            "family_mellin": telemetry.get(f"{telemetry_prefix}.family_mellin"),
            "loss": telemetry.get(f"{telemetry_prefix}.loss"),
            "stability_hint": telemetry.get(f"{telemetry_prefix}.stability_hint"),
            "webgpu_available": telemetry.get(f"{telemetry_prefix}.webgpu_available"),
            "webgpu_device_ready": telemetry.get(
                f"{telemetry_prefix}.webgpu_device_ready"
            ),
            "work_units": telemetry.get(f"{telemetry_prefix}.work_units"),
        },
        "first_text_preview": _text_preview(first_trace),
        "first_usage": _usage_payload(first_trace),
        "summary": suite.get("summary"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--max-output-tokens", type=int, default=180)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to send through OpenAI Responses. Repeat for a suite.",
    )
    parser.add_argument(
        "--instructions",
        default=(
            "Answer as a compact ML systems diagnostic. Ground the answer only "
            "in supplied SpiralTorch Z-space context and telemetry."
        ),
    )
    parser.add_argument(
        "--z-state",
        default=None,
        help="Comma-separated Z-state vector. Defaults to a small six-value seed.",
    )
    parser.add_argument(
        "--wasm-context-artifact",
        default=None,
        help="Persisted handoff JSON from write_wasm_report_context_artifact(...).",
    )
    parser.add_argument(
        "--wasm-report",
        action="append",
        default=[],
        help="Path to a browser-exported WASM report JSON. Repeat to blend reports.",
    )
    parser.add_argument(
        "--wasm-report-glob",
        action="append",
        default=[],
        help="Glob for browser-exported WASM report JSON files.",
    )
    parser.add_argument(
        "--wasm-report-dir",
        action="append",
        default=[],
        help="Directory containing browser-exported WASM report JSON files.",
    )
    parser.add_argument("--wasm-report-recursive", action="store_true")
    parser.add_argument("--wasm-max-reports", type=int, default=None)
    parser.add_argument("--gradient-dim", type=int, default=6)
    parser.add_argument("--bundle-weight", type=float, default=1.0)
    parser.add_argument("--telemetry-prefix", default="wasm")
    parser.add_argument(
        "--write-wasm-context-artifact",
        default=None,
        help="Optional path to persist the selected report handoff before inference.",
    )
    parser.add_argument(
        "--trace-jsonl",
        default=None,
        help="Optional path for API LLM trace JSONL output.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for the compact live-run summary JSON.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    result = run_openai_wasm_context(
        prompts=args.prompt or list(DEFAULT_PROMPTS),
        z_state=_parse_z_state(args.z_state),
        model=args.model,
        backend=args.backend,
        max_output_tokens=args.max_output_tokens,
        trace_jsonl=args.trace_jsonl,
        instructions=args.instructions,
        wasm_context_artifact=args.wasm_context_artifact,
        wasm_reports=args.wasm_report,
        wasm_report_globs=args.wasm_report_glob,
        wasm_report_dirs=args.wasm_report_dir,
        wasm_report_recursive=args.wasm_report_recursive,
        wasm_max_reports=args.wasm_max_reports,
        gradient_dim=args.gradient_dim,
        bundle_weight=args.bundle_weight,
        telemetry_prefix=args.telemetry_prefix,
        write_wasm_context_artifact=args.write_wasm_context_artifact,
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
