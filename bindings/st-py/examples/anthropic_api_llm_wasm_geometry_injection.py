# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Inject WASM geometry probes into Anthropic/API-model Z-space inference.

The default path is keyless and deterministic: it uses compact WASM-shaped
geometry probes plus a fake hosted-model response. For a real browser/WASM to
Anthropic route, first build ``bindings/st-wasm`` with ``wasm-pack --target
nodejs`` and pass the generated JS glue:

    ANTHROPIC_API_KEY=... python \
        bindings/st-py/examples/anthropic_api_llm_wasm_geometry_injection.py \
        --wasm-pkg /tmp/spiraltorch-wasm-node-pkg/spiraltorch_wasm.js \
        --live-anthropic --model claude-fable-5 \
        --trace-dir /tmp/spiraltorch-geometry-traces

The API key is read only by the Anthropic SDK and is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import spiraltorch as st


DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-fable-5")
DEFAULT_PROMPT = (
    "Return exactly one sentence under 30 words: how should this Z-space "
    "geometry change decoding? If no telemetry is supplied, say to use neutral "
    "baseline decoding."
)
DEFAULT_SYSTEM = (
    "Answer in one compact sentence. Treat SpiralTorch Z-space context as "
    "runtime telemetry; do not quote or enumerate the telemetry block."
)
DEFAULT_Z_STATE = [0.18, -0.07, 0.31, -0.14, 0.22, -0.05]
DEFAULT_CONDITIONS = ("baseline", "calm", "turbulent")


def _scale_probe(*, density: float, threshold: float) -> dict[str, object]:
    return {
        "kind": "spiraltorch.wasm_scale_stack_probe",
        "source_crate": "st-frac::scale_stack",
        "mode": "scalar",
        "threshold": threshold,
        "sample_count": 3,
        "samples": [
            {"scale": 1.0, "gate_mean": density},
            {"scale": 2.0, "gate_mean": min(1.0, density + 0.2)},
            {"scale": 3.0, "gate_mean": min(1.0, density + 0.35)},
        ],
        "persistence": [{"scale_low": 0.0, "scale_high": 1.0, "mass": density}],
        "interface_density": density,
        "moment_0": 1.0,
        "moment_1": 0.6 + density,
        "moment_2": 0.25 + density,
        "boundary_dimension": 0.5 + 1.5 * density,
        "coherence_profile": [{"level": 0.25, "scale": 1.0}],
    }


def _fractal_probe(*, energy: float, coherence: float, drift: float) -> dict[str, object]:
    return {
        "kind": "spiraltorch.wasm_fractal_field_probe",
        "source_crate": "st-frac::fractal_field",
        "mode": "branching_field",
        "generator": {"octaves": 3, "lacunarity": 2.0, "gain": 0.5, "iterations": 16},
        "log_lattice": {"log_start": -2.0, "log_step": 0.25, "len": 16},
        "sample_count": 16,
        "preview_count": 2,
        "energy": energy,
        "mean_abs": max(0.01, energy * 2.0),
        "max_abs": max(0.02, energy * 5.0),
        "phase_drift": drift,
        "total_variation": max(0.01, drift * 0.02),
        "coherence_score": coherence,
        "samples": [{"index": 0, "re": energy, "im": 0.0, "abs": abs(energy)}],
    }


def _log_z_probe(*, stability: float, energy: float, phase_drift: float) -> dict[str, object]:
    return {
        "kind": "spiraltorch.wasm_log_z_series_probe",
        "source_crate": "st-frac::cosmology",
        "mode": "log_z_series",
        "log_lattice": {"log_start": 0.0, "log_step": 0.25, "len": 8},
        "options": {"window": "hann", "normalisation": "l1"},
        "sample_count": 8,
        "sample_stats": {
            "count": 8,
            "mean": 1.4,
            "min": 1.0,
            "max": 2.0,
            "energy": max(0.1, energy),
        },
        "weight_stats": {"count": 8, "mean": 0.125, "min": 0.0, "max": 0.3, "energy": 0.03},
        "z_count": 2,
        "projection": {
            "count": 2,
            "mean_abs": max(0.01, energy * 0.1),
            "max_abs": max(0.01, energy * 0.2),
            "energy": energy,
            "phase_drift": phase_drift,
            "stability_score": stability,
            "preview_count": 1,
            "preview": [{"index": 0, "re": 1.0, "im": 0.0, "abs": 1.0}],
        },
    }


def builtin_geometry_probe_sets() -> dict[str, dict[str, dict[str, object]]]:
    """Return compact WASM-shaped calm/turbulent geometry probe sets."""

    return {
        "calm": {
            "scale": _scale_probe(density=0.3333333333333333, threshold=0.08),
            "field": _fractal_probe(energy=0.0318, coherence=0.96, drift=3.15),
            "logz": _log_z_probe(stability=0.97, energy=0.0033, phase_drift=0.47),
        },
        "turbulent": {
            "scale": _scale_probe(density=1.0, threshold=0.18),
            "field": _fractal_probe(energy=0.00052, coherence=0.99, drift=2.09),
            "logz": _log_z_probe(stability=0.000009, energy=9.35e9, phase_drift=3.41),
        },
    }


def wasm_geometry_probe_sets_from_node(
    wasm_pkg: str | os.PathLike[str],
) -> dict[str, dict[str, dict[str, object]]]:
    """Generate calm/turbulent probe JSON by calling a nodejs st-wasm package."""

    pkg = str(wasm_pkg)
    script = r"""
const wasm = require(PKG_PLACEHOLDER);
function emit(kind) {
  let scaleField, threshold, fractalArgs, logSamples, zValues;
  if (kind === 'calm') {
    scaleField = [0.01, 0.025, 0.04, 0.055, 0.07, 0.085, 0.1, 0.115, 0.13];
    threshold = 0.08;
    fractalArgs = [2, 1.6, 0.35, 12, -1.5, 0.18, 18, 5];
    logSamples = [1, 1.03, 1.08, 1.15, 1.24, 1.35, 1.48, 1.62];
    zValues = [0.1, 0.2, 0.3, 0.4];
  } else {
    scaleField = [0.02, 0.8, -0.55, 1.2, -1.0, 0.35, 1.55, -1.4, 0.95];
    threshold = 0.18;
    fractalArgs = [5, 2.7, 0.82, 32, -2.75, 0.31, 28, 6];
    logSamples = [1, -1.8, 3.6, -4.2, 6.4, -7.8, 9.5, -11.2, 13.6, -15.0];
    zValues = [0.35, 0.9, 1.4, 2.1, 3.0, 4.4];
  }
  return {
    scale: JSON.parse(wasm.scalarScaleStackProbeJson(
      new Float32Array(scaleField), new Uint32Array([3, 3]),
      new Float32Array([1, 2, 3]), threshold, 2, 3,
      new Float32Array([0.25, 0.5, 0.75]))),
    field: JSON.parse(wasm.fractalFieldProbeJson(...fractalArgs)),
    logz: JSON.parse(wasm.logZSeriesProbeJson(
      0.0, 0.25, new Float32Array(logSamples), 'hann', 'l1',
      new Float32Array(zValues), Math.min(6, zValues.length)))
  };
}
console.log(JSON.stringify({calm: emit('calm'), turbulent: emit('turbulent')}));
""".replace("PKG_PLACEHOLDER", json.dumps(pkg))
    output = subprocess.check_output(["node", "-e", script], text=True)
    payload = json.loads(output)
    if not isinstance(payload, Mapping):
        raise ValueError("node st-wasm probe generator returned a non-object payload")
    return {str(label): dict(probes) for label, probes in payload.items()}


def fake_api_model(prompt: str, **_kwargs: object) -> dict[str, object]:
    """Return a deterministic hosted-model-shaped payload for keyless runs."""

    text = f"Geometry context routed decoding: {prompt[:120]}"
    return {
        "model": "local-wasm-geometry-injection",
        "output_text": text,
        "status": "completed",
        "usage": {
            "input_tokens": max(1, len(prompt.split())),
            "output_tokens": max(1, len(text.split())),
            "total_tokens": max(2, len(prompt.split()) + len(text.split())),
        },
    }


def _context_for(
    condition: str,
    probe_sets: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    gradient_dim: int,
    consensus_weight: float,
    consensus_only: bool,
) -> list[st.ZSpacePartialBundle]:
    if condition == "baseline":
        return []
    probes = probe_sets.get(condition)
    if not isinstance(probes, Mapping):
        raise ValueError(f"unknown geometry condition '{condition}'")
    return st.api_llm_geometry_context_partials(
        probes,
        gradient_dim=gradient_dim,
        include_consensus=True,
        consensus_only=consensus_only,
        consensus_weight=consensus_weight,
    )


def _telemetry_focus(trace: Mapping[str, Any]) -> dict[str, float]:
    inference = trace.get("inference")
    payload: Mapping[str, Any] = {}
    if isinstance(inference, Mapping):
        telemetry = inference.get("telemetry")
        if isinstance(telemetry, Mapping):
            inner = telemetry.get("payload")
            payload = inner if isinstance(inner, Mapping) else telemetry
    keys = (
        "geometry.scale_stack.1.interface_density",
        "geometry.fractal_field.2.energy",
        "geometry.fractal_field.2.coherence_score",
        "geometry.log_z_series.3.projection_stability",
        "geometry.consensus.probe_count",
        "geometry.consensus.family_count",
        "geometry.consensus.metric_count",
        "geometry.consensus.strategy_code",
        "geometry.consensus.scale_stack_interface_density_mean",
        "geometry.consensus.fractal_field_energy_mean",
        "geometry.consensus.fractal_field_coherence_score_mean",
        "geometry.consensus.log_z_series_projection_stability_mean",
    )
    return {key: float(payload[key]) for key in keys if key in payload}


def _compact_trace(
    trace: st.ApiLLMTrace,
    context: Sequence[st.ZSpacePartialBundle],
    runtime: st.ApiLLMZSpaceRuntime,
) -> dict[str, Any]:
    row = trace.as_dict()
    inference = row.get("inference")
    applied = inference.get("applied") if isinstance(inference, Mapping) else {}
    summary = runtime.summary()
    return {
        "text": row.get("text"),
        "finish_reason": row.get("finish_reason"),
        "usage": row.get("usage"),
        "context_origins": [partial.origin for partial in context],
        "trace_metrics": {
            key: row.get("metrics", {}).get(key)
            for key in ("speed", "memory", "stability", "drs", "frac")
        },
        "applied_metrics": {
            key: applied.get(key) if isinstance(applied, Mapping) else None
            for key in ("speed", "memory", "stability", "drs", "frac")
        },
        "confidence": (
            inference.get("confidence") if isinstance(inference, Mapping) else None
        ),
        "telemetry": _telemetry_focus(row),
        "text_quality": summary.get("text_quality"),
    }


def _safe_trace_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in label)
    return cleaned.strip("-_.") or "condition"


def run_geometry_injection(
    *,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    z_state: Sequence[float] = DEFAULT_Z_STATE,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    probe_sets: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    wasm_pkg: str | os.PathLike[str] | None = None,
    live_anthropic: bool = False,
    invoke: Callable[..., Any] | None = None,
    system: str | None = DEFAULT_SYSTEM,
    max_tokens: int = 180,
    gradient_dim: int = 6,
    consensus_weight: float = 1.35,
    full_context: bool = False,
    trace_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run baseline/calm/turbulent Z-space geometry injection comparisons."""

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
    if live_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY or omit --live-anthropic")

    rows: dict[str, Any] = {}
    trace_paths: dict[str, str] = {}
    trace_summaries: dict[str, Any] = {}
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
        runtime = st.ApiLLMZSpaceRuntime(
            list(z_state),
            provider="anthropic" if live_anthropic else "local-demo",
            model=model,
            create_session=False,
            smoothing=0.32,
        )
        if live_anthropic:
            trace = runtime.call_anthropic_messages(
                prompt,
                model=model,
                system=system,
                context_partials=context,
                context_prompt=bool(context),
                context_prompt_options={
                    "max_partials": 5 if full_context else 1,
                    "max_metrics": 8,
                    "max_telemetry": 18,
                },
                max_tokens=max_tokens,
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
        rows[condition] = _compact_trace(trace, context, runtime)
        if trace_root is not None:
            trace_path = trace_root / f"{_safe_trace_label(str(condition))}.jsonl"
            trace_paths[str(condition)] = runtime.write_jsonl(trace_path)
            trace_summaries[str(condition)] = st.summarize_api_llm_trace_events(
                trace_path
            )

    result: dict[str, Any] = {
        "kind": "spiraltorch.anthropic_wasm_geometry_injection",
        "model": model,
        "prompt": prompt,
        "probe_source": probe_source,
        "context_mode": "full" if full_context else "consensus-only",
        "wasm_pkg": None if wasm_pkg is None else str(wasm_pkg),
        "conditions": list(conditions),
        "results": rows,
    }
    if trace_paths:
        result["trace_paths"] = trace_paths
        result["trace_summaries"] = trace_summaries
        result["trace_comparison"] = st.compare_api_llm_trace_runs(trace_paths)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--wasm-pkg", default=None)
    parser.add_argument("--live-anthropic", action="store_true")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-tokens", type=int, default=180)
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
    result = run_geometry_injection(
        prompt=args.prompt,
        model=args.model,
        conditions=args.condition or DEFAULT_CONDITIONS,
        wasm_pkg=args.wasm_pkg,
        live_anthropic=args.live_anthropic,
        system=args.system,
        max_tokens=args.max_tokens,
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
