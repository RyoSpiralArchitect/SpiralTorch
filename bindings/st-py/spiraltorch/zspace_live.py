from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse
import webbrowser

__all__ = [
    "serve_zspace_trace",
    "ZSpaceTraceLiveServer",
]


def _normalise_zspace_trace_record(record: Mapping[str, Any], *, event_type: str) -> dict[str, Any] | None:
    if "kind" in record:
        return dict(record)

    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else None
    record_type = record.get("event_type") or record.get("type")

    if payload is None and len(record) == 1:
        (key, value), = record.items()
        if isinstance(value, Mapping):
            payload = {key: value}
        else:
            payload = {key: {"data": value}}
        record_type = event_type

    if payload is None or record_type not in (event_type, None):
        return None
    if len(payload) != 1:
        return None
    (kind, body), = payload.items()
    event: dict[str, Any] = {"kind": str(kind)}
    if isinstance(body, Mapping):
        event.update(body)
    else:
        event["data"] = body
    if "ts" in record:
        event["ts"] = record["ts"]
    return event


def _viewer_html(title: str) -> str:
    title_json = json.dumps(str(title), ensure_ascii=True)
    return f"""<!doctype html>
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
    <h1 id="title"></h1>
    <p>Live stream · Trace events: <span id="count">0</span></p>
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

  <script>
    document.getElementById("title").textContent = {title_json};

    const events = [];
    const idx = document.getElementById("idx");
    const count = document.getElementById("count");
    const kindEl = document.getElementById("kind");
    const stepEl = document.getElementById("step");
    const metaEl = document.getElementById("meta");
    const rawEl = document.getElementById("raw");
    const bars = document.getElementById("bars");
    const heat = document.getElementById("heat");

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

    const source = new EventSource("/events");
    source.onmessage = (msg) => {{
      try {{
        const ev = JSON.parse(msg.data);
        events.push(ev);
        count.textContent = String(events.length);
        idx.max = Math.max(0, events.length - 1);
        idx.value = String(events.length - 1);
        renderAt(events.length - 1);
      }} catch (e) {{
        console.error(e);
      }}
    }};
  </script>
</body>
</html>
"""


class _LiveHub:
    def __init__(
        self,
        *,
        event_type: str,
        maxlen: int,
        poll_interval: float,
        max_batch: int,
        record_jsonl: str | None,
        buffer: int,
    ) -> None:
        self._event_type = str(event_type)
        self._maxlen = max(1, int(maxlen))
        self._poll_interval = max(0.0, float(poll_interval))
        self._max_batch = max(1, int(max_batch))
        self._clients: list[Queue[dict[str, Any]]] = []
        self._clients_lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._buffer_limit = max(1, int(buffer))
        self._stop = threading.Event()
        self._record_path = str(record_jsonl) if record_jsonl is not None else None
        self._record_handle = None
        if self._record_path is not None:
            Path(self._record_path).parent.mkdir(parents=True, exist_ok=True)
            self._record_handle = open(self._record_path, "a", encoding="utf-8")

        self._thread = threading.Thread(target=self._pump, name="spiraltorch-zspace-live", daemon=True)
        self._thread.start()

    @property
    def record_path(self) -> str | None:
        return self._record_path

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=2.0)
        if self._record_handle is not None:
            try:
                self._record_handle.close()
            except Exception:
                pass
            self._record_handle = None

    def _pump(self) -> None:
        import spiraltorch as st

        queue = st.plugin.listen(self._event_type, maxlen=self._maxlen)
        try:
            while not self._stop.is_set():
                drained = queue.drain(self._max_batch)
                if not drained:
                    self._stop.wait(self._poll_interval)
                    continue
                for record in drained:
                    if not isinstance(record, Mapping):
                        continue
                    event = _normalise_zspace_trace_record(record, event_type=self._event_type)
                    if event is None:
                        continue
                    self._buffer.append(event)
                    if len(self._buffer) > self._buffer_limit:
                        overflow = len(self._buffer) - self._buffer_limit
                        del self._buffer[:overflow]
                    if self._record_handle is not None:
                        try:
                            self._record_handle.write(json.dumps(event, ensure_ascii=True) + "\n")
                            self._record_handle.flush()
                        except Exception:
                            pass
                    with self._clients_lock:
                        clients = list(self._clients)
                    for client in clients:
                        try:
                            client.put_nowait(event)
                        except Exception:
                            continue
        finally:
            try:
                queue.close()
            except Exception:
                pass
            if self._record_handle is not None:
                try:
                    self._record_handle.close()
                except Exception:
                    pass
                self._record_handle = None

    def subscribe(self) -> tuple[Queue[dict[str, Any]], list[dict[str, Any]]]:
        q: Queue[dict[str, Any]] = Queue(maxsize=256)
        with self._clients_lock:
            self._clients.append(q)
            snapshot = list(self._buffer)
        return q, snapshot

    def unsubscribe(self, q: Queue[dict[str, Any]]) -> None:
        with self._clients_lock:
            try:
                self._clients.remove(q)
            except ValueError:
                return


class ZSpaceTraceLiveServer:
    def __init__(
        self,
        url: str,
        event_type: str,
        record_jsonl: str | None,
        *,
        hub: _LiveHub,
        server: ThreadingHTTPServer,
        thread: threading.Thread | None,
    ) -> None:
        self.url = url
        self.event_type = event_type
        self.record_jsonl = record_jsonl
        self._hub = hub
        self._server = server
        self._thread = thread

    def close(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass
        try:
            self._hub.close()
        except Exception:
            pass

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)


class _ZSpaceHandler(BaseHTTPRequestHandler):
    hub: _LiveHub
    title: str

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            body = _viewer_html(self.title).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q, bootstrap = self.hub.subscribe()
            try:
                for event in bootstrap:
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=True)}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        event = q.get(timeout=2.0)
                    except Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=True)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.hub.unsubscribe(q)
            return

        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"not found")


def serve_zspace_trace(
    *,
    event_type: str = "ZSpaceTrace",
    host: str = "127.0.0.1",
    port: int = 0,
    title: str = "SpiralTorch Z-Space Trace (Live)",
    maxlen: int = 8192,
    poll_interval: float = 0.05,
    max_batch: int = 256,
    buffer: int = 1024,
    record_jsonl: str | None = None,
    open_browser: bool = True,
    background: bool = True,
) -> ZSpaceTraceLiveServer:
    """Serve a live ZSpaceTrace stream at `http://host:port/` (SSE + Canvas viewer)."""

    hub = _LiveHub(
        event_type=event_type,
        maxlen=maxlen,
        poll_interval=poll_interval,
        max_batch=max_batch,
        record_jsonl=record_jsonl,
        buffer=buffer,
    )

    handler = type("_Handler", (_ZSpaceHandler,), {})
    handler.hub = hub
    handler.title = str(title)
    try:
        server = ThreadingHTTPServer((host, int(port)), handler)
    except Exception:
        hub.close()
        raise

    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"

    def _run() -> None:
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            try:
                server.server_close()
            except Exception:
                pass
            hub.close()

    server_thread: threading.Thread | None = None
    if background:
        server_thread = threading.Thread(target=_run, name="spiraltorch-zspace-http", daemon=True)
        server_thread.start()
    else:
        _run()

    handle = ZSpaceTraceLiveServer(
        url,
        str(event_type),
        hub.record_path,
        hub=hub,
        server=server,
        thread=server_thread,
    )

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return handle
