from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "load_gnn_band_replay_trace",
    "summarize_gnn_band_replays",
    "compare_gnn_band_replay_runs",
]


def load_gnn_band_replay_trace(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the JSON emitted by `gnn_trainer_band_trace_demo.rs`."""

    trace_path = Path(path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GNN band replay trace must be a JSON object")
    return payload


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _stats(values: Iterable[float]) -> dict[str, float] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "count": float(len(finite)),
        "first": finite[0],
        "last": finite[-1],
        "min": min(finite),
        "max": max(finite),
        "mean": sum(finite) / len(finite),
    }


def _sequence_floats(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    values: list[float] = []
    for item in value:
        number = _as_finite_float(item)
        if number is not None:
            values.append(number)
    return values


def _mean_by_index(rows: Iterable[Sequence[float]]) -> list[float]:
    sums: list[float] = []
    counts: list[int] = []
    for row in rows:
        for idx, value in enumerate(row):
            number = _as_finite_float(value)
            if number is None:
                continue
            while len(sums) <= idx:
                sums.append(0.0)
                counts.append(0)
            sums[idx] += number
            counts[idx] += 1
    return [total / count for total, count in zip(sums, counts) if count > 0]


def _entries_from_band_replays(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    band_replays = payload.get("band_replays")
    if not isinstance(band_replays, Mapping):
        return []
    entries: list[dict[str, Any]] = []
    for band, raw_entries in band_replays.items():
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            continue
        for entry in raw_entries:
            if not isinstance(entry, Mapping):
                continue
            copied = dict(entry)
            copied.setdefault("band", str(band))
            entries.append(copied)
    return entries


def _entries_from_reports(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = payload.get("reports")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes, bytearray)):
        return []
    entries: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, Mapping):
            continue
        roundtable = report.get("roundtable")
        if not isinstance(roundtable, Mapping):
            continue
        band_pass = roundtable.get("band_pass")
        aggregation = roundtable.get("aggregation")
        if not isinstance(band_pass, Mapping) or not isinstance(aggregation, Mapping):
            continue
        band = band_pass.get("band")
        if not isinstance(band, str):
            continue
        entries.append(
            {
                "band": band,
                "layer": report.get("layer"),
                "gradient_l1": band_pass.get("gradient_l1"),
                "gradient_l2": band_pass.get("gradient_l2"),
                "gradient_rms": band_pass.get("gradient_rms"),
                "total_flow_energy": report.get("total_flow_energy"),
                "effective_coefficients": aggregation.get("effective_coefficients"),
                "step_scales": aggregation.get("step_scales"),
                "band_pass_scales": aggregation.get("band_pass_scales"),
            }
        )
    return entries


def _band_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = _entries_from_band_replays(payload)
    return entries if entries else _entries_from_reports(payload)


def _summarize_band(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layers = sorted(
        {
            str(layer)
            for entry in entries
            for layer in [entry.get("layer")]
            if isinstance(layer, str) and layer
        }
    )
    scale_rows = [_sequence_floats(entry.get("band_pass_scales")) for entry in entries]
    effective_rows = [
        _sequence_floats(entry.get("effective_coefficients")) for entry in entries
    ]
    scale_deltas = [
        scale - 1.0 for row in scale_rows for scale in row if math.isfinite(scale)
    ]
    effective_values = [
        coeff for row in effective_rows for coeff in row if math.isfinite(coeff)
    ]

    summary: dict[str, Any] = {
        "count": len(entries),
        "layers": layers,
        "gradient_l1": _stats(
            value
            for entry in entries
            for value in [_as_finite_float(entry.get("gradient_l1"))]
            if value is not None
        ),
        "gradient_l2": _stats(
            value
            for entry in entries
            for value in [_as_finite_float(entry.get("gradient_l2"))]
            if value is not None
        ),
        "gradient_rms": _stats(
            value
            for entry in entries
            for value in [_as_finite_float(entry.get("gradient_rms"))]
            if value is not None
        ),
        "total_flow_energy": _stats(
            value
            for entry in entries
            for value in [_as_finite_float(entry.get("total_flow_energy"))]
            if value is not None
        ),
        "band_pass_scales": {
            "mean_by_index": _mean_by_index(scale_rows),
            "last_by_index": scale_rows[-1] if scale_rows else [],
            "delta": _stats(scale_deltas),
            "max_abs_delta": max((abs(delta) for delta in scale_deltas), default=0.0),
        },
        "effective_coefficients": {
            "mean_by_index": _mean_by_index(effective_rows),
            "last_by_index": effective_rows[-1] if effective_rows else [],
            "values": _stats(effective_values),
        },
    }
    return summary


def summarize_gnn_band_replays(
    trace: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Summarise band-specific GNN replay coefficient shifts from a trace JSON."""

    payload = (
        load_gnn_band_replay_trace(trace)
        if isinstance(trace, (str, os.PathLike))
        else dict(trace)
    )
    entries = _band_entries(payload)
    by_band: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        band = entry.get("band")
        if not isinstance(band, str) or not band:
            continue
        by_band.setdefault(band, []).append(entry)

    return {
        "count": len(entries),
        "bands": {band: _summarize_band(items) for band, items in sorted(by_band.items())},
        "trainer": payload.get("trainer") if isinstance(payload.get("trainer"), Mapping) else None,
        "signal": payload.get("signal") if isinstance(payload.get("signal"), Mapping) else None,
    }


def compare_gnn_band_replay_runs(
    traces: Iterable[str | os.PathLike[str] | Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare several GNN band replay traces using per-band summary means."""

    runs: list[dict[str, Any]] = []
    by_band: dict[str, dict[str, list[float]]] = {}
    for trace in traces:
        path_label = str(trace) if isinstance(trace, (str, os.PathLike)) else None
        summary = summarize_gnn_band_replays(trace)
        runs.append({"path": path_label, "summary": summary})
        for band, band_summary in summary.get("bands", {}).items():
            bucket = by_band.setdefault(
                band,
                {
                    "gradient_rms_mean": [],
                    "scale_delta_mean": [],
                    "max_abs_scale_delta": [],
                },
            )
            gradient_rms = band_summary.get("gradient_rms")
            if isinstance(gradient_rms, Mapping):
                value = _as_finite_float(gradient_rms.get("mean"))
                if value is not None:
                    bucket["gradient_rms_mean"].append(value)
            scale_summary = band_summary.get("band_pass_scales")
            if isinstance(scale_summary, Mapping):
                delta = scale_summary.get("delta")
                if isinstance(delta, Mapping):
                    value = _as_finite_float(delta.get("mean"))
                    if value is not None:
                        bucket["scale_delta_mean"].append(value)
                value = _as_finite_float(scale_summary.get("max_abs_delta"))
                if value is not None:
                    bucket["max_abs_scale_delta"].append(value)

    return {
        "count": len(runs),
        "runs": runs,
        "bands": {
            band: {name: _stats(values) for name, values in metrics.items()}
            for band, metrics in sorted(by_band.items())
        },
    }
