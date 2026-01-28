from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "load_trainer_trace_events",
    "write_trainer_trace_html",
]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _extract_payload(record: dict[str, Any], *, event_type: str) -> Any | None:
    record_type = record.get("event_type") or record.get("type")
    if record_type == event_type and "payload" in record:
        return record.get("payload")
    if "payload" in record and record_type is None:
        return record.get("payload")
    if len(record) == 1:
        return record
    return None


def load_trainer_trace_events(
    path: str | Path,
    *,
    event_type: str = "TrainerStep",
) -> list[dict[str, Any]]:
    """Load a trainer trace JSONL file recorded via `spiraltorch.plugin.record(...)`."""

    trace_path = Path(path)
    events: list[dict[str, Any]] = []
    for record in _iter_jsonl(trace_path):
        payload = _extract_payload(record, event_type=event_type)
        if not isinstance(payload, dict):
            continue
        event = dict(payload)
        if "ts" in record and "ts" not in event:
            event["ts"] = record["ts"]
        events.append(event)
    return events


def write_trainer_trace_html(
    trace_jsonl: str | Path,
    html_path: str | Path | None = None,
    *,
    title: str = "SpiralTorch Trainer Trace",
    event_type: str = "TrainerStep",
    marker_event_type: str | None = "TrainerPhase",
) -> str:
    """Render a self-contained HTML viewer for a trainer trace JSONL file."""

    trace_jsonl = Path(trace_jsonl)
    events = load_trainer_trace_events(trace_jsonl, event_type=event_type)
    markers: list[dict[str, Any]] = []
    if marker_event_type:
        markers = load_trainer_trace_events(trace_jsonl, event_type=marker_event_type)
    html_path = Path(html_path) if html_path is not None else trace_jsonl.with_suffix(".html")
    payload = json.dumps(events, ensure_ascii=True)
    marker_payload = json.dumps(markers, ensure_ascii=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #121826;
      --text: #e8eefc;
      --muted: #9ab0d0;
      --accent: #6ee7ff;
      --border: rgba(255,255,255,.08);
      --danger: #fb7185;
      --phase0: rgba(148,163,184,.18);
      --phase1: rgba(110,231,255,.14);
      --phase2: rgba(167,139,250,.14);
      --phase3: rgba(251,113,133,.14);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(110,231,255,.08), rgba(0,0,0,0));
    }}
    header h1 {{
      margin: 0;
      font-size: 16px;
      letter-spacing: .2px;
      color: var(--text);
    }}
    header p {{
      margin: 6px 0 0;
      font-size: 12px;
      color: var(--muted);
    }}
    main {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 14px;
      padding: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      overflow: hidden;
    }}
    .row {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .row label {{
      font-size: 12px;
      color: var(--muted);
    }}
    select {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
      color: var(--text);
      outline: none;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    canvas {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
    }}
    pre {{
      margin: 0;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
      overflow: auto;
      max-height: 360px;
      font-size: 11px;
      color: #cfe0ff;
    }}
    .kv {{
      margin-top: 10px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .kv div strong {{
      color: var(--text);
      font-weight: 600;
    }}
    .badge {{
      display: inline-flex;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(110,231,255,.08);
      color: var(--accent);
      font-size: 11px;
    }}
    .badge.danger {{
      background: rgba(251,113,133,.10);
      color: var(--danger);
    }}
    .legend {{
      margin-top: 10px;
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>event_type: <span class="badge" id="event-type">—</span> · steps: <span id="count">0</span> · markers: <span id="marker-count">0</span></p>
  </header>
  <main>
    <section class="panel" style="grid-column: 1;">
      <div class="row" style="justify-content: space-between;">
        <span class="badge" id="epoch">epoch —</span>
        <span class="badge">step <span id="step">—</span></span>
      </div>
      <div class="row" style="justify-content: space-between; margin-top: 10px;">
        <span class="badge" id="phase">phase —</span>
        <span class="badge" id="turnover">turnover —</span>
      </div>
      <div style="margin-top: 12px;">
        <label for="idx">sample index</label>
        <input id="idx" type="range" min="0" max="0" value="0" step="1"/>
      </div>
      <div style="margin-top: 12px;">
        <label for="key">plot key</label>
        <select id="key"></select>
      </div>
      <div class="kv" id="stats"></div>
      <div class="legend" id="phase-legend">
        <div class="legend-item"><span class="swatch" style="background: var(--phase0);"></span><span>0 · Background</span></div>
        <div class="legend-item"><span class="swatch" style="background: var(--phase1);"></span><span>1 · SymmetricPulse</span></div>
        <div class="legend-item"><span class="swatch" style="background: var(--phase2);"></span><span>2 · CascadeImbalance</span></div>
        <div class="legend-item"><span class="swatch" style="background: var(--phase3);"></span><span>3 · DiffuseDrift</span></div>
      </div>
      <div style="margin-top: 12px;">
        <span class="badge">raw sample</span>
        <pre id="raw"></pre>
      </div>
    </section>
    <section class="panel" style="grid-column: 2;">
      <div class="row" style="justify-content: space-between; margin-bottom: 10px;">
        <div class="row">
          <span class="badge">timeseries</span>
          <span style="font-size: 12px; color: var(--muted);">single-key line chart</span>
        </div>
      </div>
      <canvas id="plot" width="960" height="360"></canvas>
    </section>
  </main>

  <script id="trace-meta" type="application/json">{json.dumps({"event_type": event_type, "marker_event_type": marker_event_type}, ensure_ascii=True)}</script>
  <script id="trace-data" type="application/json">{payload}</script>
  <script id="trace-markers" type="application/json">{marker_payload}</script>
  <script>
    const meta = JSON.parse(document.getElementById("trace-meta").textContent || "{{}}");
    const samples = JSON.parse(document.getElementById("trace-data").textContent || "[]");
    const markers = JSON.parse(document.getElementById("trace-markers").textContent || "[]");
    const idx = document.getElementById("idx");
    const count = document.getElementById("count");
    const markerCount = document.getElementById("marker-count");
    const stepEl = document.getElementById("step");
    const epochEl = document.getElementById("epoch");
    const keyEl = document.getElementById("key");
    const rawEl = document.getElementById("raw");
    const statsEl = document.getElementById("stats");
    const plot = document.getElementById("plot");
    const eventTypeEl = document.getElementById("event-type");
    const phaseEl = document.getElementById("phase");
    const turnoverEl = document.getElementById("turnover");

    eventTypeEl.textContent = meta.event_type || "TrainerStep";
    count.textContent = String(samples.length);
    markerCount.textContent = String(markers.length);
    idx.max = Math.max(0, samples.length - 1);

    const phaseMap = new Map([
      [0, {{ name: "Background", css: "var(--phase0)" }}],
      [1, {{ name: "SymmetricPulse", css: "var(--phase1)" }}],
      [2, {{ name: "CascadeImbalance", css: "var(--phase2)" }}],
      [3, {{ name: "DiffuseDrift", css: "var(--phase3)" }}],
    ]);
    const phaseLineColors = new Map([
      [0, "rgba(148,163,184,.55)"],
      [1, "rgba(110,231,255,.68)"],
      [2, "rgba(167,139,250,.68)"],
      [3, "rgba(251,113,133,.68)"],
    ]);

    function isFiniteNumber(v) {{
      return typeof v === "number" && Number.isFinite(v);
    }}

    function extractValue(sample, key) {{
      if (!sample || !key) return null;
      const metrics = sample.metrics || {{}};
      const extra = metrics.extra || {{}};
      if (isFiniteNumber(extra[key])) return extra[key];
      if (isFiniteNumber(metrics[key])) return metrics[key];
      if (isFiniteNumber(sample[key])) return sample[key];
      return null;
    }}

    function phaseFor(sample) {{
      const code = extractValue(sample, "spectral_label");
      if (!isFiniteNumber(code)) return null;
      const rounded = Math.round(code);
      if (!phaseMap.has(rounded)) return null;
      return {{ code: rounded, ...phaseMap.get(rounded) }};
    }}

    function turnoverFor(sample) {{
      const v = extractValue(sample, "spectral_turnover");
      return isFiniteNumber(v) ? v : null;
    }}

    function collectKeys() {{
      const keys = new Set();
      for (const s of samples) {{
        const metrics = (s && s.metrics) ? s.metrics : {{}};
        const extra = (metrics && metrics.extra) ? metrics.extra : {{}};
        for (const k of Object.keys(extra)) keys.add(k);
      }}
      for (const k of ["loss_weighted", "loss_weighted_base", "spectral_label", "spectral_turnover", "spectral_lr_scale", "softlogic_inertia", "softlogic_z", "curvature_value", "curvature_pressure_rel_var"]) {{
        if (samples.some(s => extractValue(s, k) !== null)) keys.add(k);
      }}
      return Array.from(keys).sort();
    }}

    const keys = collectKeys();
    const defaultKey = (keys.includes("loss_weighted") ? "loss_weighted"
      : keys.includes("spectral_label") ? "spectral_label"
      : (keys[0] || ""));

    for (const k of keys) {{
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      keyEl.appendChild(opt);
    }}
    keyEl.value = defaultKey;

    function computeSeries(key) {{
      const ys = [];
      const xs = [];
      for (let i = 0; i < samples.length; i++) {{
        const s = samples[i];
        xs.push(isFiniteNumber(s.step) ? s.step : i);
        ys.push(extractValue(s, key));
      }}
      return {{ xs, ys }};
    }}

    function drawLine(ctx, pts, color) {{
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (const p of pts) {{
        if (p === null) {{
          started = false;
          continue;
        }}
        if (!started) {{
          ctx.moveTo(p.x, p.y);
          started = true;
        }} else {{
          ctx.lineTo(p.x, p.y);
        }}
      }}
      ctx.stroke();
    }}

    function renderPlot() {{
      const key = keyEl.value;
      const {{ xs, ys }} = computeSeries(key);
      const w = plot.width;
      const h = plot.height;
      const padL = 46, padR = 16, padT = 16, padB = 28;
      const ctx = plot.getContext("2d");
      ctx.clearRect(0, 0, w, h);

      const numeric = ys.filter(v => isFiniteNumber(v));
      let min = numeric.length ? Math.min(...numeric) : 0;
      let max = numeric.length ? Math.max(...numeric) : 1;
      if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {{
        const base = Number.isFinite(min) ? min : 0;
        min = base - 1;
        max = base + 1;
      }}
      const span = max - min;
      min = min - span * 0.05;
      max = max + span * 0.05;

      const x0 = xs[0] ?? 0;
      const x1 = xs[xs.length - 1] ?? Math.max(1, xs.length - 1);
      const xSpan = (x1 - x0) || 1;

      const toX = (x) => padL + ((x - x0) / xSpan) * (w - padL - padR);
      const toY = (v) => padT + ((max - v) / (max - min)) * (h - padT - padB);

      // phase background stripes (based on spectral_label)
      let lastPhase = null;
      let segStart = 0;
      for (let i = 0; i <= samples.length; i++) {{
        const p = (i < samples.length) ? phaseFor(samples[i]) : null;
        const code = p ? p.code : null;
        if (i === 0) {{
          lastPhase = code;
          segStart = 0;
          continue;
        }}
        if (code === lastPhase && i < samples.length) continue;
        if (lastPhase !== null) {{
          const left = toX(xs[segStart] ?? segStart);
          const right = (i < xs.length) ? toX(xs[i] ?? i) : (w - padR);
          const width = Math.max(0, right - left);
          const phaseInfo = phaseMap.get(lastPhase);
          if (phaseInfo) {{
            ctx.fillStyle = phaseInfo.css;
            ctx.fillRect(left, padT, width, h - padT - padB);
          }}
        }}
        lastPhase = code;
        segStart = i;
      }}

      // axes
      ctx.strokeStyle = "rgba(255,255,255,.14)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, h - padB);
      ctx.lineTo(w - padR, h - padB);
      ctx.stroke();

      // y labels
      ctx.fillStyle = "rgba(154,176,208,.95)";
      ctx.font = "12px ui-sans-serif, system-ui";
      const ticks = 4;
      for (let i = 0; i <= ticks; i++) {{
        const t = i / ticks;
        const v = max - t * (max - min);
        const y = padT + t * (h - padT - padB);
        ctx.strokeStyle = "rgba(255,255,255,.06)";
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.fillText(v.toFixed(4), 6, y + 4);
      }}

      // series
      const pts = ys.map((v, i) => {{
        if (!isFiniteNumber(v)) return null;
        return {{ x: toX(xs[i]), y: toY(v) }};
      }});
      drawLine(ctx, pts, "rgba(110,231,255,.95)");

      // marker events (TrainerPhase)
      for (const m of markers) {{
        if (!m || !isFiniteNumber(m.step)) continue;
        const x = toX(m.step);
        const kind = String(m.kind || "");
        let color = "rgba(110,231,255,.25)";
        let width = 1.0;
        if (kind === "turnover_spike") {{
          color = "rgba(251,113,133,.9)";
          width = 2.0;
        }} else if (kind === "label_change") {{
          let code = null;
          if (m.to && isFiniteNumber(m.to.code)) code = Math.round(m.to.code);
          if (code === null && isFiniteNumber(m.label_code)) code = Math.round(m.label_code);
          if (code !== null && phaseLineColors.has(code)) {{
            color = phaseLineColors.get(code);
            width = 1.75;
          }}
        }} else if (kind === "loss_spike") {{
          color = "rgba(250,204,21,.88)";
          width = 2.0;
        }} else if (kind === "drift_spike") {{
          color = "rgba(251,146,60,.88)";
          width = 2.0;
        }} else if (kind === "band_shift") {{
          color = "rgba(110,231,255,.55)";
          width = 1.5;
        }}
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, h - padB);
        ctx.stroke();
      }}

      // marker for current idx
      const i = Number(idx.value) || 0;
      const cx = toX(xs[i] ?? i);
      ctx.strokeStyle = "rgba(251,113,133,.9)";
      ctx.beginPath();
      ctx.moveTo(cx, padT);
      ctx.lineTo(cx, h - padB);
      ctx.stroke();

      ctx.fillStyle = "rgba(110,231,255,.95)";
      ctx.fillText(key, padL + 6, padT + 14);
    }}

    function renderSample() {{
      const i = Math.max(0, Math.min(samples.length - 1, Number(idx.value) || 0));
      const s = samples[i] || {{}};
      stepEl.textContent = String(isFiniteNumber(s.step) ? s.step : i);
      epochEl.textContent = "epoch " + String(isFiniteNumber(s.epoch) ? s.epoch : "—");
      const phase = phaseFor(s);
      phaseEl.textContent = phase ? (`phase ${{phase.code}} · ${{phase.name}}`) : "phase —";
      const turnover = turnoverFor(s);
      turnoverEl.textContent = turnover !== null ? (`turnover ${{turnover.toFixed(4)}}`) : "turnover —";
      rawEl.textContent = JSON.stringify(s, null, 2);

      const key = keyEl.value;
      const v = extractValue(s, key);
      const series = computeSeries(key).ys.filter(isFiniteNumber);
      const last = series.length ? series[series.length - 1] : null;
      const min = series.length ? Math.min(...series) : null;
      const max = series.length ? Math.max(...series) : null;
      const mean = series.length ? (series.reduce((a,b)=>a+b,0) / series.length) : null;

      statsEl.innerHTML = "";
      const entries = [
        ["value", v],
        ["last", last],
        ["min", min],
        ["max", max],
        ["mean", mean],
      ];
      for (const [label, value] of entries) {{
        const div = document.createElement("div");
        const formatted = isFiniteNumber(value) ? value.toFixed(6) : "—";
        div.innerHTML = `<strong>${{label}}</strong><br/>${{formatted}}`;
        statsEl.appendChild(div);
      }}
    }}

    idx.addEventListener("input", () => {{
      renderSample();
      renderPlot();
    }});
    keyEl.addEventListener("change", () => {{
      renderSample();
      renderPlot();
    }});

    renderSample();
    renderPlot();
  </script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return str(html_path)
