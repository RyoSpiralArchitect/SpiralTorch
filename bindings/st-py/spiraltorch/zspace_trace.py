from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "load_zspace_trace_events",
    "write_zspace_trace_html",
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


def _normalise_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and len(payload) == 1:
        (kind, body), = payload.items()
        if isinstance(body, dict):
            out: dict[str, Any] = {"kind": str(kind)}
            out.update(body)
            return out
        return {"kind": str(kind), "data": body}
    return {"kind": "Unknown", "data": payload}


def load_zspace_trace_events(path: str | Path, *, event_type: str = "ZSpaceTrace") -> list[dict[str, Any]]:
    """Load a Z-space trace JSONL file (either raw ZSpaceTraceRecorder output or plugin-recorded)."""

    trace_path = Path(path)
    events: list[dict[str, Any]] = []
    for record in _iter_jsonl(trace_path):
        payload = _extract_payload(record, event_type=event_type)
        if payload is None:
            continue
        event = _normalise_payload(payload)
        if "ts" in record and "ts" not in event:
            event["ts"] = record["ts"]
        events.append(event)
    return events


def write_zspace_trace_html(
    trace_jsonl: str | Path,
    html_path: str | Path | None = None,
    *,
    title: str = "SpiralTorch Z-Space Trace",
    event_type: str = "ZSpaceTrace",
) -> str:
    """Render a self-contained HTML viewer for a Z-space trace JSONL file."""

    trace_jsonl = Path(trace_jsonl)
    events = load_zspace_trace_events(trace_jsonl, event_type=event_type)
    html_path = Path(html_path) if html_path is not None else trace_jsonl.with_suffix(".html")
    payload = json.dumps(events, ensure_ascii=True)
    title_json = json.dumps(title, ensure_ascii=True)

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
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
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
      grid-template-columns: 340px 1fr;
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
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Trace events: <span id="count">0</span> · Use the slider to scrub steps.</p>
  </header>
  <main>
    <section class="panel" style="grid-column: 1;">
      <div class="row" style="justify-content: space-between;">
        <span class="badge" id="kind">—</span>
        <span class="badge">step <span id="step">—</span></span>
      </div>
      <div style="margin-top: 12px;">
        <label for="idx">event index</label>
        <input id="idx" type="range" min="0" max="0" value="0" step="1"/>
      </div>
      <div class="kv" id="meta"></div>
    </section>
    <section class="panel" style="grid-column: 2;">
      <div class="row" style="justify-content: space-between; margin-bottom: 10px;">
        <div class="row">
          <span class="badge">coherence</span>
          <span style="font-size: 12px; color: var(--muted);">bars + relation heatmap (outer product)</span>
        </div>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
        <div>
          <canvas id="bars" width="512" height="220"></canvas>
        </div>
        <div>
          <canvas id="heat" width="256" height="256"></canvas>
        </div>
      </div>
      <div style="margin-top: 12px;">
        <span class="badge">raw event</span>
        <pre id="raw"></pre>
      </div>
    </section>
  </main>

  <script id="trace-data" type="application/json">{payload}</script>
  <script>
    const events = JSON.parse(document.getElementById("trace-data").textContent || "[]");
    const idx = document.getElementById("idx");
    const count = document.getElementById("count");
    const kindEl = document.getElementById("kind");
    const stepEl = document.getElementById("step");
    const metaEl = document.getElementById("meta");
    const rawEl = document.getElementById("raw");
    const bars = document.getElementById("bars");
    const heat = document.getElementById("heat");

    count.textContent = String(events.length);
    idx.max = Math.max(0, events.length - 1);

    function clamp01(v) {{
      if (!Number.isFinite(v)) return 0;
      if (v <= 0) return 0;
      if (v >= 1) return 1;
      return v;
    }}

    function extractCoherence(ev) {{
      const coh = ev && Array.isArray(ev.coherence) ? ev.coherence : null;
      if (coh) return coh;
      if (ev && Array.isArray(ev.filtered)) return ev.filtered;
      if (ev && Array.isArray(ev.original)) return ev.original;
      return null;
    }}

    function renderBars(coherence) {{
      const ctx = bars.getContext("2d");
      ctx.clearRect(0, 0, bars.width, bars.height);
      if (!coherence || coherence.length === 0) {{
        ctx.fillStyle = "rgba(154,176,208,.8)";
        ctx.fillText("no coherence in this event", 14, 24);
        return;
      }}

      let max = 1e-6;
      for (const v of coherence) {{
        if (Number.isFinite(v)) max = Math.max(max, v);
      }}

      const n = coherence.length;
      const padding = 14;
      const w = bars.width - padding * 2;
      const h = bars.height - padding * 2;
      const barW = Math.max(1, w / n);

      ctx.fillStyle = "rgba(110,231,255,.18)";
      ctx.fillRect(padding, padding, w, h);

      for (let i = 0; i < n; i++) {{
        const v = clamp01((Number.isFinite(coherence[i]) ? coherence[i] : 0) / max);
        const bh = v * h;
        const x = padding + i * barW;
        const y = padding + (h - bh);
        ctx.fillStyle = "rgba(110,231,255,.75)";
        ctx.fillRect(x, y, Math.max(1, barW - 1), bh);
      }}
    }}

    function renderHeatmap(coherence) {{
      const ctx = heat.getContext("2d");
      ctx.clearRect(0, 0, heat.width, heat.height);
      if (!coherence || coherence.length === 0) {{
        ctx.fillStyle = "rgba(154,176,208,.8)";
        ctx.fillText("no relation", 14, 24);
        return;
      }}
      let max = 1e-6;
      for (const v of coherence) {{
        if (Number.isFinite(v)) max = Math.max(max, v);
      }}

      const n = coherence.length;
      const norm = new Array(n);
      for (let i = 0; i < n; i++) {{
        norm[i] = clamp01((Number.isFinite(coherence[i]) ? coherence[i] : 0) / max);
      }}

      heat.width = n;
      heat.height = n;
      const image = ctx.createImageData(n, n);
      const data = image.data;
      for (let y = 0; y < n; y++) {{
        for (let x = 0; x < n; x++) {{
          const v = norm[y] * norm[x];
          const c = Math.max(0, Math.min(255, Math.floor(v * 255)));
          const i = (y * n + x) * 4;
          data[i] = c;
          data[i + 1] = Math.floor(c * 0.85);
          data[i + 2] = 255 - c;
          data[i + 3] = 255;
        }}
      }}
      ctx.putImageData(image, 0, 0);
    }}

    function renderMeta(ev) {{
      metaEl.innerHTML = "";
      const items = [];
      if (typeof ev.ts === "number") items.push(["ts", ev.ts.toFixed(3)]);
      if (typeof ev.input_shape !== "undefined") items.push(["input_shape", JSON.stringify(ev.input_shape)]);
      if (typeof ev.projected_shape !== "undefined") items.push(["projected_shape", JSON.stringify(ev.projected_shape)]);
      if (typeof ev.aggregated_shape !== "undefined") items.push(["aggregated_shape", JSON.stringify(ev.aggregated_shape)]);
      if (ev.diagnostics && typeof ev.diagnostics === "object") {{
        for (const [k, v] of Object.entries(ev.diagnostics)) {{
          items.push([`diag.${{k}}`, typeof v === "number" ? v.toFixed(6) : JSON.stringify(v)]);
        }}
      }}
      for (const [k, v] of items) {{
        const cell = document.createElement("div");
        cell.innerHTML = `<strong>${{k}}</strong><br/>${{v}}`;
        metaEl.appendChild(cell);
      }}
    }}

    function renderAt(i) {{
      const ev = events[i] || {{}};
      kindEl.textContent = ev.kind || "—";
      stepEl.textContent = typeof ev.step === "number" ? String(ev.step) : "—";
      rawEl.textContent = JSON.stringify(ev, null, 2);
      renderMeta(ev);
      const coherence = extractCoherence(ev);
      renderBars(coherence);
      renderHeatmap(coherence);
    }}

    idx.addEventListener("input", () => renderAt(Number(idx.value)));
    renderAt(0);
  </script>
</body>
</html>
"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html + "\n", encoding="utf-8")
    return str(html_path)
