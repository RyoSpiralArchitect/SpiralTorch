from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "load_gnn_band_replay_trace",
    "flatten_gnn_band_replay_rows",
    "summarize_gnn_band_replays",
    "compare_gnn_band_replay_runs",
    "write_gnn_band_replay_html",
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


def _trace_payload(trace: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    return (
        load_gnn_band_replay_trace(trace)
        if isinstance(trace, (str, os.PathLike))
        else dict(trace)
    )


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
                "base_coefficients": aggregation.get("base_coefficients"),
                "effective_coefficients": aggregation.get("effective_coefficients"),
                "step_scales": aggregation.get("step_scales"),
                "band_pass_scales": aggregation.get("band_pass_scales"),
            }
        )
    return entries


def _band_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = _entries_from_band_replays(payload)
    return entries if entries else _entries_from_reports(payload)


def flatten_gnn_band_replay_rows(
    trace: str | os.PathLike[str] | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand a GNN band replay trace into band/layer/hop rows."""

    rows: list[dict[str, Any]] = []
    payload = _trace_payload(trace)
    for replay_index, entry in enumerate(_band_entries(payload)):
        band = entry.get("band")
        if not isinstance(band, str) or not band:
            continue
        base = _sequence_floats(entry.get("base_coefficients"))
        step = _sequence_floats(entry.get("step_scales"))
        band_pass = _sequence_floats(entry.get("band_pass_scales"))
        effective = _sequence_floats(entry.get("effective_coefficients"))
        hop_count = max(len(base), len(step), len(band_pass), len(effective))
        for hop_index in range(hop_count):
            step_scale = step[hop_index] if hop_index < len(step) else None
            band_pass_scale = (
                band_pass[hop_index] if hop_index < len(band_pass) else None
            )
            roundtable_step_scale = None
            if (
                step_scale is not None
                and band_pass_scale is not None
                and abs(band_pass_scale) > 1.0e-9
            ):
                roundtable_step_scale = step_scale / band_pass_scale
            rows.append(
                {
                    "band": band,
                    "layer": entry.get("layer"),
                    "replay_index": replay_index,
                    "hop_index": hop_index,
                    "gradient_l1": _as_finite_float(entry.get("gradient_l1")),
                    "gradient_l2": _as_finite_float(entry.get("gradient_l2")),
                    "gradient_rms": _as_finite_float(entry.get("gradient_rms")),
                    "total_flow_energy": _as_finite_float(
                        entry.get("total_flow_energy")
                    ),
                    "base_coefficient": base[hop_index]
                    if hop_index < len(base)
                    else None,
                    "step_scale": step_scale,
                    "roundtable_step_scale": roundtable_step_scale,
                    "band_pass_scale": band_pass_scale,
                    "scale_delta": band_pass_scale - 1.0
                    if band_pass_scale is not None
                    else None,
                    "effective_coefficient": effective[hop_index]
                    if hop_index < len(effective)
                    else None,
                }
            )
    return rows


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

    payload = _trace_payload(trace)
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


def _json_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_gnn_band_replay_html(
    trace: str | os.PathLike[str] | Mapping[str, Any],
    html_path: str | os.PathLike[str] | None = None,
    *,
    title: str = "SpiralTorch GNN Band Replay Trace",
) -> str:
    """Render a self-contained HTML viewer for GNN band replay coefficients."""

    payload = _trace_payload(trace)
    summary = summarize_gnn_band_replays(payload)
    rows = flatten_gnn_band_replay_rows(payload)
    if html_path is None:
        if isinstance(trace, (str, os.PathLike)):
            output_path = Path(trace).with_suffix(".gnn.html")
        else:
            output_path = Path("gnn_band_replay_trace.html")
    else:
        output_path = Path(html_path)

    title_html = html.escape(title)
    summary_json = _json_script(summary)
    rows_json = _json_script(rows)
    payload_json = _json_script(
        {
            "trainer": payload.get("trainer"),
            "signal": payload.get("signal"),
        }
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_html}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07100d;
      --panel: #101b16;
      --panel2: #15251f;
      --text: #eafff5;
      --muted: #9fc3b2;
      --line: rgba(234,255,245,.12);
      --above: #ffc857;
      --here: #6ee7b7;
      --beneath: #76a9ff;
      --danger: #ff7a90;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 16% 8%, rgba(110,231,183,.16), transparent 28rem),
        radial-gradient(circle at 86% 18%, rgba(118,169,255,.15), transparent 26rem),
        linear-gradient(180deg, #07100d, #0b1411 58%, #050807);
      color: var(--text);
    }}
    header {{
      padding: 26px 26px 16px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0; font-size: clamp(22px, 4vw, 38px); letter-spacing: -.04em; }}
    .subtitle {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px; display: grid; gap: 16px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: linear-gradient(180deg, var(--panel2), var(--panel));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 16px 42px rgba(0,0,0,.22);
    }}
    .card h2 {{ margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); }}
    .metric {{ display: flex; justify-content: space-between; gap: 12px; padding: 4px 0; font-size: 13px; }}
    .metric span:last-child {{ color: var(--text); font-variant-numeric: tabular-nums; }}
    .band-above {{ border-color: color-mix(in srgb, var(--above), transparent 56%); }}
    .band-here {{ border-color: color-mix(in srgb, var(--here), transparent 56%); }}
    .band-beneath {{ border-color: color-mix(in srgb, var(--beneath), transparent 56%); }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    select {{
      background: rgba(0,0,0,.25);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 14px;
      font-size: 12px;
    }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(255,255,255,.04); position: sticky; top: 0; }}
    tr:hover {{ background: rgba(255,255,255,.045); }}
    .bar {{
      height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
      min-width: 76px;
    }}
    .bar > i {{ display: block; height: 100%; border-radius: inherit; }}
    pre {{
      white-space: pre-wrap;
      overflow: auto;
      max-height: 340px;
      color: #c9f7df;
      background: rgba(0,0,0,.22);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      font-size: 11px;
    }}
    @media (max-width: 760px) {{
      header {{ padding: 20px 16px 12px; }}
      main {{ padding: 12px; }}
      .table-wrap {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title_html}</h1>
    <div class="subtitle">Roundtable band replay intensity, hop scaling, and effective GNN coefficients.</div>
  </header>
  <main>
    <section class="cards" id="cards"></section>
    <section class="card">
      <h2>Replay Rows</h2>
      <div class="controls">
        <label>Band <select id="bandFilter"><option value="">all</option></select></label>
        <label>Layer <select id="layerFilter"><option value="">all</option></select></label>
      </div>
      <div class="table-wrap"><table id="rows"></table></div>
    </section>
    <section class="card">
      <h2>Trace Header</h2>
      <pre id="payload"></pre>
    </section>
  </main>
  <script>
    const summary = {summary_json};
    const rows = {rows_json};
    const payload = {payload_json};
    const fmt = (v, digits = 4) => Number.isFinite(v) ? v.toFixed(digits) : "n/a";
    const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (ch) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    const bandColor = (band) => band === "above" ? "var(--above)" : band === "beneath" ? "var(--beneath)" : "var(--here)";
    const cards = document.getElementById("cards");
    const bands = summary.bands || {{}};
    for (const [band, data] of Object.entries(bands)) {{
      const scale = data.band_pass_scales || {{}};
      const rms = data.gradient_rms || {{}};
      const eff = data.effective_coefficients || {{}};
      const card = document.createElement("article");
      card.className = `card band-${{String(band).replace(/[^a-z0-9_-]/ig, "")}}`;
      card.innerHTML = `
        <h2>${{esc(band)}}</h2>
        <div class="metric"><span>replays</span><span>${{data.count || 0}}</span></div>
        <div class="metric"><span>gradient_rms mean</span><span>${{fmt(rms.mean)}}</span></div>
        <div class="metric"><span>max |scale_delta|</span><span>${{fmt(scale.max_abs_delta)}}</span></div>
        <div class="metric"><span>coeff mean</span><span>${{fmt(eff.values && eff.values.mean)}}</span></div>
        <div class="metric"><span>layers</span><span>${{(data.layers || []).length}}</span></div>
      `;
      cards.appendChild(card);
    }}
    const bandFilter = document.getElementById("bandFilter");
    const layerFilter = document.getElementById("layerFilter");
    for (const band of [...new Set(rows.map((row) => row.band).filter(Boolean))].sort()) {{
      const option = document.createElement("option");
      option.value = band;
      option.textContent = band;
      bandFilter.appendChild(option);
    }}
    for (const layer of [...new Set(rows.map((row) => row.layer).filter(Boolean))].sort()) {{
      const option = document.createElement("option");
      option.value = layer;
      option.textContent = layer;
      layerFilter.appendChild(option);
    }}
    const table = document.getElementById("rows");
    function renderRows() {{
      const band = bandFilter.value;
      const layer = layerFilter.value;
      const visible = rows.filter((row) => (!band || row.band === band) && (!layer || row.layer === layer));
      const maxDelta = Math.max(...visible.map((row) => Math.abs(row.scale_delta || 0)), 1e-6);
      table.innerHTML = `<thead><tr>
        <th>band</th><th>layer</th><th>hop</th><th>grad_rms</th><th>pass_scale</th>
        <th>delta</th><th>effective</th><th>delta bar</th>
      </tr></thead><tbody></tbody>`;
      const body = table.querySelector("tbody");
      for (const row of visible) {{
        const delta = row.scale_delta;
        const width = Math.min(100, Math.abs(delta || 0) / maxDelta * 100);
        const color = delta < 0 ? "var(--danger)" : bandColor(row.band);
        body.insertAdjacentHTML("beforeend", `<tr>
          <td>${{esc(row.band)}}</td>
          <td>${{esc(row.layer)}}</td>
          <td>${{row.hop_index}}</td>
          <td>${{fmt(row.gradient_rms, 5)}}</td>
          <td>${{fmt(row.band_pass_scale)}}</td>
          <td>${{fmt(delta)}}</td>
          <td>${{fmt(row.effective_coefficient)}}</td>
          <td><div class="bar"><i style="width:${{width}}%;background:${{color}}"></i></div></td>
        </tr>`);
      }}
    }}
    bandFilter.addEventListener("change", renderRows);
    layerFilter.addEventListener("change", renderRows);
    document.getElementById("payload").textContent = JSON.stringify(payload, null, 2);
    renderRows();
  </script>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return str(output_path)
