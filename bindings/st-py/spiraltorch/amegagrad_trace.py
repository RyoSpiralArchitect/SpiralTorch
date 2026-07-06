"""Amegagrad topos-training trace helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "compare_amegagrad_topos_training_traces",
    "trace_amegagrad_topos_training_sweep",
]


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _summary_metric(
    summary: Mapping[str, Any],
    key: str,
    *,
    stat: str = "mean",
) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    source = metrics.get(key)
    if not isinstance(source, Mapping):
        return None
    return _finite_float(source.get(stat))


def _topos_stat(
    summary: Mapping[str, Any],
    key: str,
    *,
    stat: str = "mean",
) -> float | None:
    context = summary.get("topos_context")
    if not isinstance(context, Mapping):
        return None
    source = context.get(key)
    if not isinstance(source, Mapping):
        return None
    samples = _finite_float(source.get("samples"))
    if samples is not None and samples <= 0.0:
        return None
    return _finite_float(source.get(stat))


def _topos_observed_rate(summary: Mapping[str, Any]) -> float:
    context = summary.get("topos_context")
    if not isinstance(context, Mapping):
        return 0.0
    return _finite_float(context.get("observed_rate")) or 0.0


def _trace_entries(
    traces: Mapping[str, str | Path] | Sequence[str | Path],
) -> list[tuple[str, Path]]:
    if isinstance(traces, Mapping):
        return [(str(label), Path(path)) for label, path in traces.items()]
    entries: list[tuple[str, Path]] = []
    for path in traces:
        trace_path = Path(path)
        entries.append((trace_path.stem, trace_path))
    return entries


def _winner(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    higher_is_better: bool = True,
) -> str | None:
    scored: list[tuple[float, str]] = []
    for row in rows:
        value = _finite_float(row.get(key))
        label = row.get("label")
        if value is None or label is None:
            continue
        scored.append((value, str(label)))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=higher_is_better)
    return scored[0][1]


def compare_amegagrad_topos_training_traces(
    traces: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    event_type: str = "TrainerStep",
) -> dict[str, Any]:
    """Compare AmegagradSession trainer traces by optimizer/topos pressure."""

    import spiraltorch as st

    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for label, path in _trace_entries(traces):
        summary = st.summarize_trainer_trace_events(path, event_type=event_type)
        summaries[label] = summary
        rows.append(
            {
                "label": label,
                "trace_jsonl": str(path),
                "count": int(summary.get("count") or 0),
                "realgrad_l2_mean": _summary_metric(summary, "realgrad.l2"),
                "zspace_loss_mean": _summary_metric(summary, "zspace.loss"),
                "hypergrad_learning_rate_last": _summary_metric(
                    summary,
                    "hypergrad.learning_rate",
                    stat="last",
                ),
                "realgrad_learning_rate_last": _summary_metric(
                    summary,
                    "realgrad.learning_rate",
                    stat="last",
                ),
                "topos_context_observed_rate": _topos_observed_rate(summary),
                "topos_closure_pressure_mean": _topos_stat(
                    summary,
                    "closure_pressure",
                ),
                "topos_optimizer_rate_scale_mean": _topos_stat(
                    summary,
                    "optimizer_rate_scale",
                ),
                "topos_optimizer_raw_rate_scale_mean": _topos_stat(
                    summary,
                    "optimizer_raw_rate_scale",
                ),
                "topos_optimizer_effective_gradient_bias_scale_mean": _topos_stat(
                    summary,
                    "optimizer_effective_gradient_bias_scale",
                ),
                "topos_optimizer_effective_momentum_damping_mean": _topos_stat(
                    summary,
                    "optimizer_effective_momentum_damping",
                ),
                "topos_optimizer_hyper_learning_rate_mean": _topos_stat(
                    summary,
                    "optimizer_hyper_learning_rate",
                ),
                "topos_optimizer_real_learning_rate_mean": _topos_stat(
                    summary,
                    "optimizer_real_learning_rate",
                ),
                "topos_runtime_control_energy_mean": _topos_stat(
                    summary,
                    "runtime_profile_control_energy",
                ),
                "topos_runtime_closure_risk_mean": _topos_stat(
                    summary,
                    "runtime_profile_closure_risk",
                ),
                "topos_runtime_exploration_budget_mean": _topos_stat(
                    summary,
                    "runtime_profile_exploration_budget",
                ),
                "topos_runtime_training_rate_scale_mean": _topos_stat(
                    summary,
                    "runtime_profile_training_rate_scale",
                ),
                "topos_runtime_training_gradient_bias_scale_mean": _topos_stat(
                    summary,
                    "runtime_profile_training_gradient_bias_scale",
                ),
                "topos_runtime_learning_inference_balance_mean": _topos_stat(
                    summary,
                    "runtime_profile_learning_inference_balance",
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            _finite_float(row.get("topos_context_observed_rate")) or 0.0,
            -(_finite_float(row.get("realgrad_l2_mean")) or 0.0),
            str(row.get("label") or ""),
        ),
        reverse=True,
    )
    return {
        "kind": "spiraltorch.amegagrad_topos_training_trace_comparison",
        "event_type": event_type,
        "count": len(rows),
        "runs": rows,
        "summaries": summaries,
        "winners": {
            "highest_topos_observed": _winner(rows, "topos_context_observed_rate"),
            "lowest_realgrad_l2": _winner(
                rows,
                "realgrad_l2_mean",
                higher_is_better=False,
            ),
            "lowest_optimizer_rate_scale": _winner(
                rows,
                "topos_optimizer_rate_scale_mean",
                higher_is_better=False,
            ),
            "lowest_optimizer_raw_rate_scale": _winner(
                rows,
                "topos_optimizer_raw_rate_scale_mean",
                higher_is_better=False,
            ),
            "highest_effective_gradient_bias": _winner(
                rows,
                "topos_optimizer_effective_gradient_bias_scale_mean",
            ),
            "highest_runtime_control_energy": _winner(
                rows,
                "topos_runtime_control_energy_mean",
            ),
            "lowest_runtime_closure_risk": _winner(
                rows,
                "topos_runtime_closure_risk_mean",
                higher_is_better=False,
            ),
            "lowest_runtime_training_rate_scale": _winner(
                rows,
                "topos_runtime_training_rate_scale_mean",
                higher_is_better=False,
            ),
        },
    }


def _default_waves(steps: int, total: int) -> list[list[float]]:
    waves: list[list[float]] = []
    for step in range(max(1, int(steps))):
        row: list[float] = []
        for idx in range(total):
            sign = -1.0 if (idx + step) % 2 else 1.0
            row.append(sign * (0.15 + 0.05 * ((idx + step) % 5)))
        waves.append(row)
    return waves


def _profile_mapping(
    profiles: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if profiles is not None:
        return {str(label): dict(config) for label, config in profiles.items()}
    return {
        "guard_only": {
            "topos_control_gain": 0.0,
            "observed_depth": 4,
            "visited_volume": 8,
        },
        "topos_tuned": {
            "topos_control_gain": 1.0,
            "observed_depth": 4,
            "visited_volume": 8,
        },
    }


def _build_topos(st: Any, base: Mapping[str, Any], profile: Mapping[str, Any]) -> Any:
    params = dict(base)
    for key in ("curvature", "tolerance", "saturation", "max_depth", "max_volume"):
        if key in profile:
            params[key] = profile[key]
    guard = st.hypergrad_topos(
        curvature=float(params.get("curvature", -0.9)),
        tolerance=float(params.get("tolerance", 1e-4)),
        saturation=float(params.get("saturation", 2.0)),
        max_depth=int(params.get("max_depth", 8)),
        max_volume=int(params.get("max_volume", 16)),
    )
    porosity = profile.get("porosity")
    if porosity is not None:
        with_porosity = getattr(guard, "with_porosity", None)
        if callable(with_porosity):
            guard = with_porosity(float(porosity))
    return guard


def trace_amegagrad_topos_training_sweep(
    trace_dir: str | Path,
    *,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    steps: int = 3,
    shape: tuple[int, int] = (1, 4),
    curvature: float = -0.9,
    hyper_learning_rate: float = 0.04,
    real_learning_rate: float = 0.02,
    topos: Mapping[str, Any] | None = None,
    waves: Sequence[Sequence[float]] | None = None,
    event_type: str = "TrainerStep",
) -> dict[str, Any]:
    """Run a small AmegagradSession topos sweep and summarize the traces."""

    import spiraltorch as st

    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("shape must contain positive rows and cols")
    step_count = max(1, int(steps))
    total = rows * cols
    wave_rows = (
        [list(map(float, row)) for row in waves]
        if waves is not None
        else _default_waves(step_count, total)
    )
    if not wave_rows:
        raise ValueError("waves must not be empty")
    for row in wave_rows:
        if len(row) != total:
            raise ValueError(f"each wave must contain {total} values")

    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_topos = {
        "curvature": curvature,
        "tolerance": 1e-4,
        "saturation": 2.0,
        "max_depth": 8,
        "max_volume": max(16, total * 4),
    }
    if topos is not None:
        base_topos.update(dict(topos))

    trace_paths: dict[str, str] = {}
    for label, profile in _profile_mapping(profiles).items():
        guard = _build_topos(st, base_topos, profile)
        observed_depth = int(
            profile.get("observed_depth", min(4, int(base_topos["max_depth"])))
        )
        visited_volume = int(
            profile.get("visited_volume", min(total * 2, int(base_topos["max_volume"])))
        )
        session = st.amegagrad_session(
            (rows, cols),
            curvature=float(profile.get("curvature", curvature)),
            hyper_learning_rate=float(
                profile.get("hyper_learning_rate", hyper_learning_rate)
            ),
            real_learning_rate=float(
                profile.get("real_learning_rate", real_learning_rate)
            ),
            topos=guard,
            topos_control_gain=float(profile.get("topos_control_gain", 0.0)),
            topos_observed_depth=observed_depth,
            topos_visited_volume=visited_volume,
            telemetry=False,
            z_lam_frac=float(profile.get("z_lam_frac", 0.0)),
        )
        trace_path = out_dir / f"{label}.jsonl"
        for step_index in range(step_count):
            data = wave_rows[step_index % len(wave_rows)]
            wave = st.Tensor((rows, cols), data=data)
            session.step_wave(
                wave,
                observed_depth=observed_depth,
                visited_volume=visited_volume,
                note=f"{label}:{step_index + 1}",
            )
            session.write_trainer_trace_event(
                trace_path,
                step=step_index + 1,
                event_type=event_type,
                mode="w" if step_index == 0 else "a",
            )
        trace_paths[label] = str(trace_path)

    comparison = compare_amegagrad_topos_training_traces(
        trace_paths,
        event_type=event_type,
    )
    return {
        "kind": "spiraltorch.amegagrad_topos_training_trace_sweep",
        "trace_dir": str(out_dir),
        "trace_paths": trace_paths,
        "comparison": comparison,
        "summaries": comparison["summaries"],
    }
