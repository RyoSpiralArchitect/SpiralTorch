from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "load_kdsl_trace_events",
    "write_kdsl_trace_jsonl",
    "write_kdsl_trace_html",
]


def _iter_jsonl(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_kdsl_trace_events(path: str | Path) -> list[dict[str, Any]]:
    """Load SpiralK (KDSL) trace events from a `.json` or `.jsonl` file."""

    trace_path = Path(path)
    suffix = trace_path.suffix.lower()
    if suffix == ".jsonl":
        out: list[dict[str, Any]] = []
        for item in _iter_jsonl(trace_path):
            if isinstance(item, dict):
                out.append(item)
        return out

    if suffix == ".json":
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            events = payload.get("events", [])
        else:
            events = payload
        if isinstance(events, list):
            return [item for item in events if isinstance(item, dict)]
        raise TypeError("trace JSON must contain an 'events' list or be a list of events")

    raise ValueError("expected a .json or .jsonl trace file")


def write_kdsl_trace_jsonl(trace: Mapping[str, Any] | Sequence[Mapping[str, Any]], path: str | Path) -> str:
    """Write a SpiralK trace as JSONL (one event per line)."""

    events: Any = trace
    if isinstance(trace, Mapping):
        events = trace.get("events", [])
    if not isinstance(events, Sequence):
        raise TypeError("trace must be a mapping with 'events' or a sequence of events")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in events:
            if not isinstance(item, Mapping):
                continue
            handle.write(json.dumps(dict(item), ensure_ascii=True) + "\n")
    return str(out_path)


def write_kdsl_trace_html(
    trace: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    html_path: str | Path | None = None,
    *,
    title: str = "SpiralTorch SpiralK Trace",
) -> str:
    """Render a self-contained HTML viewer for a SpiralK evaluation trace."""

    if isinstance(trace, Mapping):
        meta = {
            "max_events": trace.get("max_events"),
            "dropped_events": trace.get("dropped_events"),
        }
        events: Any = trace.get("events", [])
    else:
        meta = {"max_events": None, "dropped_events": None}
        events = trace

    if not isinstance(events, list):
        events = list(events)
    events = [dict(item) for item in events if isinstance(item, Mapping)]

    payload = json.dumps(events, ensure_ascii=True)
    meta_json = json.dumps(meta, ensure_ascii=True)
    title_json = json.dumps(str(title), ensure_ascii=True)

    html_path = Path(html_path) if html_path is not None else Path.cwd() / "kdsl_trace.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

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
      --good: #4ade80;
      --bad: #fb7185;
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
    input[type="range"] {{
      width: 100%;
    }}
    pre {{
      margin: 0;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
      overflow: auto;
      max-height: 520px;
      font-size: 11px;
      color: #cfe0ff;
    }}
    .row {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
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
    .badge.good {{ color: var(--good); }}
    .badge.bad {{ color: var(--bad); }}
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
  </style>
</head>
<body>
  <header>
    <h1 id="title"></h1>
    <p>
      Events: <span id="count">0</span>
      · max_events: <span id="maxEvents">—</span>
      · dropped: <span id="droppedEvents">—</span>
    </p>
  </header>
  <main>
    <section class="panel" style="grid-column: 1;">
      <div class="row" style="justify-content: space-between;">
        <span class="badge" id="kind">—</span>
        <span class="badge" id="status" style="display:none;">—</span>
      </div>
      <div style="margin-top: 12px;">
        <label for="idx" style="font-size: 12px; color: var(--muted);">event index</label>
        <input id="idx" type="range" min="0" max="0" value="0" step="1"/>
      </div>
      <div class="kv" id="meta"></div>
    </section>
    <section class="panel" style="grid-column: 2;">
      <div class="row" style="justify-content: space-between; margin-bottom: 10px;">
        <span class="badge">raw event</span>
      </div>
      <pre id="raw"></pre>
    </section>
  </main>

  <script>
    document.getElementById("title").textContent = {title_json};
    const meta = {meta_json};
    const events = {payload};
    const idx = document.getElementById("idx");
    const count = document.getElementById("count");
    const kindEl = document.getElementById("kind");
    const statusEl = document.getElementById("status");
    const rawEl = document.getElementById("raw");
    const metaEl = document.getElementById("meta");

    count.textContent = String(events.length);
    idx.max = String(Math.max(0, events.length - 1));

    document.getElementById("maxEvents").textContent = meta.max_events == null ? "—" : String(meta.max_events);
    document.getElementById("droppedEvents").textContent = meta.dropped_events == null ? "—" : String(meta.dropped_events);

    function eventKind(ev) {{
      if (!ev || typeof ev !== "object") return "Unknown";
      const keys = Object.keys(ev);
      if (keys.length === 1) return keys[0];
      return "Unknown";
    }}

    function eventBody(ev) {{
      const kind = eventKind(ev);
      if (kind === "Unknown") return {{}};
      return ev[kind] && typeof ev[kind] === "object" ? ev[kind] : {{}};
    }}

    function renderMeta(ev) {{
      metaEl.innerHTML = "";
      const body = eventBody(ev);
      const items = [];
      for (const [k, v] of Object.entries(body)) {{
        items.push([k, typeof v === "number" ? v.toFixed(6) : JSON.stringify(v)]);
      }}
      for (const [k, v] of items) {{
        const cell = document.createElement("div");
        cell.innerHTML = `<strong>${{k}}</strong><br/>${{v}}`;
        metaEl.appendChild(cell);
      }}
    }}

    function renderAt(i) {{
      const ev = events[i] || {{}};
      const kind = eventKind(ev);
      kindEl.textContent = kind;
      statusEl.style.display = "none";
      statusEl.className = "badge";
      const body = eventBody(ev);
      if (kind === "Soft" && body && typeof body.applied === "boolean") {{
        statusEl.style.display = "inline-flex";
        statusEl.textContent = body.applied ? "applied" : "skipped";
        statusEl.className = body.applied ? "badge good" : "badge bad";
      }}
      rawEl.textContent = JSON.stringify(ev, null, 2);
      renderMeta(ev);
    }}

    idx.addEventListener("input", () => renderAt(Number(idx.value)));
    if (events.length > 0) {{
      idx.value = String(events.length - 1);
      renderAt(events.length - 1);
    }} else {{
      renderAt(0);
    }}
  </script>
</body>
</html>
"""

    html_path.write_text(html + "\n", encoding="utf-8")
    return str(html_path)

