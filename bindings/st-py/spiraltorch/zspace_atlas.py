from __future__ import annotations

import html
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .zspace_trace import load_zspace_trace_events

__all__ = [
    "zspace_trace_to_atlas_route",
    "zspace_trace_event_to_atlas_frame",
    "write_zspace_atlas_noncollapse_html",
    "trainer_events_to_atlas_route",
    "trainer_step_event_to_atlas_frame",
]


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _as_metric_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _as_float(value)


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _merge_noncollapse_metrics(
    dest: dict[str, float],
    payload: Mapping[str, Any],
) -> None:
    for key, value in payload.items():
        if key == "phase":
            continue
        if key == "band_energy" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            triplet = list(value[:3])
            if len(triplet) == 3:
                above = _as_metric_float(triplet[0])
                here = _as_metric_float(triplet[1])
                beneath = _as_metric_float(triplet[2])
                if above is not None:
                    dest["noncollapse.band_energy_above"] = above
                if here is not None:
                    dest["noncollapse.band_energy_here"] = here
                if beneath is not None:
                    dest["noncollapse.band_energy_beneath"] = beneath
            continue

        metric_name = str(key)
        if key == "pre_discard_preserved_ratio":
            metric_name = "preserved_ratio"
        elif key == "pre_discard_survivor_energy_ratio":
            metric_name = "survivor_energy_ratio"

        metric = _as_metric_float(value)
        if metric is None:
            continue
        dest[f"noncollapse.{metric_name}"] = metric


def _push_noncollapse_overlay(fragment: Any, event: Mapping[str, Any], district: str) -> None:
    metrics: dict[str, float] = {}

    snapshot = event.get("noncollapse")
    if isinstance(snapshot, Mapping):
        metrics["noncollapse.present"] = 1.0
        _merge_noncollapse_metrics(metrics, snapshot)
        phase = snapshot.get("phase")
        if phase is not None:
            fragment.push_note(f"zspace.trace.noncollapse.phase={phase}")

    card = event.get("noncollapse_card")
    if isinstance(card, Mapping):
        metrics["noncollapse.present"] = 1.0
        stage = card.get("stage")
        title = card.get("title")
        summary = card.get("summary")
        if stage is not None:
            fragment.push_note(f"zspace.trace.noncollapse.stage={stage}")
            stage_slug = _slug(stage)
            if stage_slug:
                metrics[f"noncollapse.stage.{stage_slug}"] = 1.0
        if title is not None:
            fragment.push_note(f"zspace.trace.noncollapse.title={title}")
        if summary is not None:
            fragment.push_note(f"zspace.trace.noncollapse.summary={summary}")

        card_metrics = card.get("metrics")
        if isinstance(card_metrics, Mapping):
            _merge_noncollapse_metrics(metrics, card_metrics)

    for name, value in metrics.items():
        fragment.push_metric(name, value, district)


def _normalise_event(obj: Mapping[str, Any], *, event_type: str = "ZSpaceTrace") -> dict[str, Any] | None:
    if "kind" in obj and isinstance(obj.get("payload"), Mapping):
        event = dict(obj["payload"])
        event.setdefault("kind", str(obj["kind"]))
        for key in ("step", "schema", "schema_version", "noncollapse", "ts"):
            if key in obj and key not in event:
                event[key] = obj[key]
        return event
    if "kind" in obj:
        return dict(obj)

    if "payload" in obj:
        record_type = obj.get("event_type") or obj.get("type")
        if record_type not in (None, event_type):
            return None
        payload = obj.get("payload")
        if not isinstance(payload, Mapping):
            return None
        if len(payload) != 1:
            return None
        (kind, body), = payload.items()
        event: dict[str, Any] = {"kind": str(kind)}
        if isinstance(body, Mapping):
            event.update(body)
        else:
            event["data"] = body
        if "ts" in obj:
            event["ts"] = obj["ts"]
        return event

    if len(obj) == 1:
        (kind, body), = obj.items()
        event = {"kind": str(kind)}
        if isinstance(body, Mapping):
            event.update(body)
        else:
            event["data"] = body
        return event

    return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, Mapping):
                yield dict(obj)


def _normalise_trainer_step_event(
    obj: Mapping[str, Any],
    *,
    event_type: str = "TrainerStep",
) -> dict[str, Any] | None:
    record_type = obj.get("event_type") or obj.get("type")
    if record_type not in (None, event_type):
        return None

    if "payload" in obj:
        payload = obj.get("payload")
        if not isinstance(payload, Mapping):
            return None
        event = dict(payload)
        if "ts" in obj and "ts" not in event:
            event["ts"] = obj["ts"]
        return event

    if {"epoch", "step", "metrics"}.issubset(obj.keys()):
        return dict(obj)

    return None


def zspace_trace_event_to_atlas_frame(
    event: Mapping[str, Any],
    *,
    district: str = "Concourse",
    timestamp_base: float | None = None,
    step_seconds: float = 0.001,
) -> Any | None:
    """Convert one ZSpaceTrace event (normalised dict or plugin record) into an AtlasFrame.

    Returns `None` when the event cannot be converted.
    """

    normalised = _normalise_event(event)
    if normalised is None:
        return None

    ts = _as_float(normalised.get("ts"))
    if ts is None:
        base = time.time() if timestamp_base is None else float(timestamp_base)
        step = _as_float(normalised.get("step")) or 0.0
        ts = base + step * max(0.0, float(step_seconds))

    import spiraltorch as st

    fragment = st.telemetry.AtlasFragment(timestamp=ts)
    kind = str(normalised.get("kind") or "ZSpaceTrace")
    fragment.push_note(f"zspace.trace.kind={kind}")

    step_val = _as_float(normalised.get("step"))
    if step_val is not None:
        fragment.push_metric("zspace.trace.step", float(step_val), district)

    coherence = normalised.get("coherence")
    if isinstance(coherence, Sequence):
        coh_values = [_as_float(v) for v in coherence]
        coh = [v for v in coh_values if v is not None]
        fragment.push_metric("coherence_channels", float(len(coh)), district)
        fragment.push_metric("coherence_response_mean", _mean(coh), district)
        fragment.push_metric("coherence_response_peak", max(coh) if coh else 0.0, district)

    diagnostics = normalised.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        mean_coherence = _as_float(diagnostics.get("mean_coherence"))
        entropy = _as_float(diagnostics.get("entropy"))
        energy_ratio = _as_float(diagnostics.get("energy_ratio"))
        fractional_order = _as_float(diagnostics.get("fractional_order"))
        z_bias = _as_float(diagnostics.get("z_bias"))
        preserved = _as_float(diagnostics.get("preserved_channels"))
        discarded = _as_float(diagnostics.get("discarded_channels"))
        dominant = _as_float(diagnostics.get("dominant_channel"))
        label = diagnostics.get("label")

        if mean_coherence is not None:
            fragment.push_metric("coherence_mean", mean_coherence, district)
            fragment.push_metric("speed", math.tanh(mean_coherence), district)
        if entropy is not None:
            fragment.push_metric("coherence_entropy", entropy, district)
            fragment.push_metric("stability", math.tanh(1.0 - entropy), district)
        if energy_ratio is not None:
            fragment.push_metric("coherence_energy_ratio", energy_ratio, district)
            fragment.push_metric("drs", math.tanh(energy_ratio - 0.5), district)
        if fractional_order is not None:
            fragment.push_metric("coherence_fractional_order", fractional_order, district)
            fragment.push_metric("frac", math.tanh(fractional_order), district)
        if z_bias is not None:
            fragment.push_metric("coherence_z_bias", z_bias, district)
            fragment.push_metric("memory", math.tanh(z_bias), district)
        if preserved is not None:
            fragment.push_metric("coherence_preserved", preserved, district)
        if discarded is not None:
            fragment.push_metric("coherence_discarded", discarded, district)
        if dominant is not None:
            fragment.push_metric("coherence_dominant", dominant, district)
        if label is not None:
            fragment.push_note(f"zspace.trace.label={label}")

    _push_noncollapse_overlay(fragment, normalised, district)

    return fragment.to_frame()


def trainer_step_event_to_atlas_frame(
    event: Mapping[str, Any],
    *,
    district: str = "Training",
    timestamp_base: float | None = None,
    step_seconds: float = 0.001,
) -> Any | None:
    """Convert one TrainerStep plugin record (or normalised dict) into an AtlasFrame."""

    normalised = _normalise_trainer_step_event(event)
    if normalised is None:
        return None

    ts = _as_float(normalised.get("ts"))
    if ts is None:
        base = time.time() if timestamp_base is None else float(timestamp_base)
        step = _as_float(normalised.get("step")) or 0.0
        ts = base + step * max(0.0, float(step_seconds))

    import spiraltorch as st

    fragment = st.telemetry.AtlasFragment(timestamp=ts)
    epoch_val = _as_float(normalised.get("epoch"))
    step_val = _as_float(normalised.get("step"))
    if epoch_val is not None:
        fragment.push_metric("epoch", float(epoch_val), district)
    if step_val is not None:
        fragment.push_metric("step", float(step_val), district)

    metrics = normalised.get("metrics")
    if isinstance(metrics, Mapping):
        step_time_ms = _as_float(metrics.get("step_time_ms"))
        if step_time_ms is not None:
            fragment.push_metric("step_time_ms", step_time_ms, district)
        mem_peak_mb = _as_float(metrics.get("mem_peak_mb"))
        if mem_peak_mb is not None:
            fragment.push_metric("mem_peak_mb", mem_peak_mb, district)
        retry_rate = _as_float(metrics.get("retry_rate"))
        if retry_rate is not None:
            fragment.push_metric("retry_rate", retry_rate, district)

        extra = metrics.get("extra")
        if isinstance(extra, Mapping):
            for key, value in extra.items():
                val = _as_float(value)
                if val is None:
                    continue
                fragment.push_metric(str(key), float(val), district)

    return fragment.to_frame()


def zspace_trace_to_atlas_route(
    trace: str | Path | Iterable[Mapping[str, Any]],
    *,
    district: str = "Concourse",
    bound: int = 512,
    event_type: str = "ZSpaceTrace",
    timestamp_base: float | None = None,
    step_seconds: float = 0.001,
) -> Any:
    """Convert a ZSpaceTrace JSONL trace (or iterable of events) into a telemetry.AtlasRoute."""

    import spiraltorch as st

    if isinstance(trace, (str, Path)):
        events: list[dict[str, Any]] = load_zspace_trace_events(trace, event_type=event_type)
    else:
        events = []
        for item in trace:
            if isinstance(item, Mapping):
                normalised = _normalise_event(item, event_type=event_type)
                if normalised is not None:
                    events.append(normalised)

    route = st.telemetry.AtlasRoute()
    base = time.time() if timestamp_base is None else float(timestamp_base)
    for idx, event in enumerate(events):
        frame = zspace_trace_event_to_atlas_frame(
            event,
            district=district,
            timestamp_base=base + float(idx) * max(0.0, float(step_seconds)),
            step_seconds=step_seconds,
        )
        if frame is None:
            continue
        route.push_bounded(frame, bound=int(bound))
    return route


def _coerce_zspace_atlas_route(
    trace_or_route: str | Path | Iterable[Mapping[str, Any]] | Any,
    *,
    district: str,
    bound: int,
    event_type: str,
    timestamp_base: float | None,
    step_seconds: float,
) -> Any:
    if hasattr(trace_or_route, "perspective_for") and hasattr(trace_or_route, "summary"):
        return trace_or_route
    return zspace_trace_to_atlas_route(
        trace_or_route,
        district=district,
        bound=bound,
        event_type=event_type,
        timestamp_base=timestamp_base,
        step_seconds=step_seconds,
    )


def _normalise_related_links(related_links: Mapping[str, str] | None) -> list[dict[str, str]]:
    if not isinstance(related_links, Mapping):
        return []
    items: list[dict[str, str]] = []
    for label, href in related_links.items():
        if not href:
            continue
        items.append({"label": str(label), "href": str(href)})
    return items


def write_zspace_atlas_noncollapse_html(
    trace_or_route: str | Path | Iterable[Mapping[str, Any]] | Any,
    html_path: str | Path | None = None,
    *,
    title: str = "SpiralTorch Atlas Non-Collapse",
    district: str = "Concourse",
    bound: int = 512,
    event_type: str = "ZSpaceTrace",
    timestamp_base: float | None = None,
    step_seconds: float = 0.001,
    top_k: int = 12,
    related_links: Mapping[str, str] | None = None,
) -> str:
    route = _coerce_zspace_atlas_route(
        trace_or_route,
        district=district,
        bound=bound,
        event_type=event_type,
        timestamp_base=timestamp_base,
        step_seconds=step_seconds,
    )

    perspective = route.perspective_for(district, ["noncollapse."])
    perspective_payload = dict(perspective) if isinstance(perspective, Mapping) else {}
    focus_payload = perspective_payload.get("focus", [])
    focus_items = [dict(item) for item in focus_payload if isinstance(item, Mapping)]
    stage_focus = [
        {
            **item,
            "stage": item["name"].removeprefix("noncollapse.stage."),
        }
        for item in focus_items
        if str(item.get("name", "")).startswith("noncollapse.stage.")
    ]
    metric_focus = [
        item
        for item in focus_items
        if not str(item.get("name", "")).startswith("noncollapse.stage.")
    ]
    if top_k > 0:
        metric_focus = metric_focus[: int(top_k)]

    summary = route.summary() if hasattr(route, "summary") else {}
    summary_payload = dict(summary) if isinstance(summary, Mapping) else {}
    payload = {
        "title": title,
        "district": district,
        "perspective": {
            "coverage": int(perspective_payload.get("coverage", 0)),
            "mean": float(perspective_payload.get("mean", 0.0)),
            "latest": float(perspective_payload.get("latest", 0.0)),
            "delta": float(perspective_payload.get("delta", 0.0)),
            "momentum": float(perspective_payload.get("momentum", 0.0)),
            "volatility": float(perspective_payload.get("volatility", 0.0)),
            "stability": float(perspective_payload.get("stability", 0.0)),
            "guidance": str(perspective_payload.get("guidance", "")),
        },
        "stage_focus": stage_focus,
        "metric_focus": metric_focus,
        "summary": {
            "frames": int(summary_payload.get("frames", 0)),
            "total_notes": int(summary_payload.get("total_notes", 0)),
            "latest_notes": [
                str(note)
                for note in summary_payload.get("latest_notes", [])
                if isinstance(note, str)
            ][:8],
        },
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    related_links_json = json.dumps(_normalise_related_links(related_links), ensure_ascii=False)

    if html_path is None:
        if isinstance(trace_or_route, (str, Path)):
            html_path = Path(trace_or_route).with_suffix(".atlas_noncollapse.html")
        else:
            html_path = Path.cwd() / "zspace_atlas_noncollapse.html"
    html_path = Path(html_path)

    title_html = html.escape(title)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_html}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #081018;
      --panel: #0f1a26;
      --panel-2: #132131;
      --line: rgba(170, 198, 236, 0.16);
      --text: #ebf2ff;
      --muted: #9cb5d7;
      --accent: #78d9ff;
      --warm: #ffd479;
      --good: #83e3b1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 "SF Mono", "IBM Plex Mono", ui-monospace, monospace;
      background:
        radial-gradient(circle at top, rgba(120,217,255,.14), transparent 34%),
        linear-gradient(180deg, #081018 0%, #0b1320 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 20px 44px;
    }}
    .hero {{ margin-bottom: 18px; }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.2;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 72ch;
    }}
    .nav {{
      margin-top: 12px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .nav a {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--accent);
      text-decoration: none;
      background: rgba(120,217,255,.10);
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin: 18px 0;
    }}
    .stat, .panel {{
      background: linear-gradient(180deg, rgba(19,33,49,.96), rgba(10,19,30,.96));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
    }}
    .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .value {{
      margin-top: 8px;
      font-size: 22px;
      color: var(--accent);
    }}
    .guidance {{
      color: var(--warm);
      font-size: 13px;
      white-space: pre-wrap;
    }}
    .stage-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .stage-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(8,16,24,.48);
    }}
    .stage-card strong {{
      display: block;
      margin-bottom: 8px;
      color: var(--good);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    td.metric-name {{
      color: var(--text);
      font-weight: 600;
    }}
    ul.notes {{
      margin: 10px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    @media (max-width: 720px) {{
      main {{ padding: 20px 14px 32px; }}
      th:nth-child(5), td:nth-child(5),
      th:nth-child(6), td:nth-child(6) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{title_html}</h1>
      <div class="subtitle">Atlas perspective for <code>perspective_for("{html.escape(district)}", ["noncollapse."])</code>. This view keeps the focus on shared non-collapse metrics so different event kinds stay comparable.</div>
      <div class="nav" id="related-links"></div>
    </section>

    <section class="grid" id="summary-grid"></section>

    <section class="panel">
      <div class="label">Guidance</div>
      <div class="guidance" id="guidance">no guidance available</div>
    </section>

    <section class="panel">
      <div class="label">Stage Comparison</div>
      <div class="stage-grid" id="stage-grid"></div>
    </section>

    <section class="panel">
      <div class="label">Metric Comparison</div>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Latest</th>
            <th>Mean</th>
            <th>Delta</th>
            <th>Momentum</th>
            <th>Std Dev</th>
          </tr>
        </thead>
        <tbody id="metric-body"></tbody>
      </table>
    </section>

    <section class="panel">
      <div class="label">Latest Notes</div>
      <ul class="notes" id="notes"></ul>
    </section>
  </main>

  <script id="atlas-data" type="application/json">{payload_json}</script>
  <script id="atlas-related-links" type="application/json">{related_links_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("atlas-data").textContent || "{{}}");
    const relatedLinks = JSON.parse(document.getElementById("atlas-related-links").textContent || "[]");

    function formatMetricValue(value) {{
      if (typeof value !== "number" || !Number.isFinite(value)) return "—";
      if (Number.isInteger(value)) return String(value);
      const abs = Math.abs(value);
      if (abs >= 100) return value.toFixed(2);
      if (abs >= 10) return value.toFixed(3);
      return value.toFixed(6);
    }}

    function appendSummaryCard(parent, label, value) {{
      const card = document.createElement("div");
      card.className = "stat";
      const labelEl = document.createElement("div");
      labelEl.className = "label";
      labelEl.textContent = label;
      const valueEl = document.createElement("div");
      valueEl.className = "value";
      valueEl.textContent = value;
      card.appendChild(labelEl);
      card.appendChild(valueEl);
      parent.appendChild(card);
    }}

    const perspective = payload.perspective || {{}};
    const summary = payload.summary || {{}};
    const relatedLinksEl = document.getElementById("related-links");
    if (Array.isArray(relatedLinks) && relatedLinks.length > 0) {{
      for (const item of relatedLinks) {{
        if (!item || typeof item.href !== "string" || !item.href) continue;
        const link = document.createElement("a");
        link.href = item.href;
        link.textContent = String(item.label || item.href);
        relatedLinksEl.appendChild(link);
      }}
    }} else {{
      relatedLinksEl.style.display = "none";
    }}
    const summaryGrid = document.getElementById("summary-grid");
    appendSummaryCard(summaryGrid, "District", payload.district || "Concourse");
    appendSummaryCard(summaryGrid, "Coverage", String(perspective.coverage || 0));
    appendSummaryCard(summaryGrid, "Frames", String(summary.frames || 0));
    appendSummaryCard(summaryGrid, "Latest", formatMetricValue(perspective.latest));
    appendSummaryCard(summaryGrid, "Delta", formatMetricValue(perspective.delta));
    appendSummaryCard(summaryGrid, "Stability", formatMetricValue(perspective.stability));

    document.getElementById("guidance").textContent = perspective.guidance || "no guidance available";

    const stageGrid = document.getElementById("stage-grid");
    const stageFocus = Array.isArray(payload.stage_focus) ? payload.stage_focus : [];
    if (stageFocus.length === 0) {{
      const empty = document.createElement("div");
      empty.className = "stage-card";
      empty.textContent = "No stage metrics available for this route.";
      stageGrid.appendChild(empty);
    }} else {{
      for (const item of stageFocus) {{
        const card = document.createElement("div");
        card.className = "stage-card";
        card.innerHTML = `<strong>${{String(item.stage || item.name || "stage")}}</strong>`
          + `latest: ${{formatMetricValue(item.latest)}}<br/>`
          + `mean: ${{formatMetricValue(item.mean)}}<br/>`
          + `delta: ${{formatMetricValue(item.delta)}}`;
        stageGrid.appendChild(card);
      }}
    }}

    const metricBody = document.getElementById("metric-body");
    const metricFocus = Array.isArray(payload.metric_focus) ? payload.metric_focus : [];
    for (const item of metricFocus) {{
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="metric-name">${{String(item.name || "metric")}}</td>
        <td>${{formatMetricValue(item.latest)}}</td>
        <td>${{formatMetricValue(item.mean)}}</td>
        <td>${{formatMetricValue(item.delta)}}</td>
        <td>${{formatMetricValue(item.momentum)}}</td>
        <td>${{formatMetricValue(item.std_dev)}}</td>
      `;
      metricBody.appendChild(tr);
    }}
    if (metricFocus.length === 0) {{
      const tr = document.createElement("tr");
      tr.innerHTML = '<td colspan="6">No non-collapse focus metrics available.</td>';
      metricBody.appendChild(tr);
    }}

    const notesEl = document.getElementById("notes");
    const notes = Array.isArray(summary.latest_notes) ? summary.latest_notes : [];
    if (notes.length === 0) {{
      const li = document.createElement("li");
      li.textContent = "No recent notes available.";
      notesEl.appendChild(li);
    }} else {{
      for (const note of notes) {{
        const li = document.createElement("li");
        li.textContent = String(note);
        notesEl.appendChild(li);
      }}
    }}
  </script>
</body>
</html>
"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc + "\n", encoding="utf-8")
    return str(html_path)


def trainer_events_to_atlas_route(
    trace: str | Path | Iterable[Mapping[str, Any]],
    *,
    district: str = "Training",
    bound: int = 512,
    event_type: str = "TrainerStep",
    timestamp_base: float | None = None,
    step_seconds: float = 0.001,
) -> Any:
    """Convert a plugin-recorded JSONL (or iterable) into a telemetry.AtlasRoute.

    This adapter targets `TrainerStep` events emitted by `nn.ModuleTrainer`.
    """

    import spiraltorch as st

    if isinstance(trace, (str, Path)):
        events: list[dict[str, Any]] = []
        for item in _iter_jsonl(Path(trace)):
            normalised = _normalise_trainer_step_event(item, event_type=event_type)
            if normalised is not None:
                events.append(normalised)
    else:
        events = []
        for item in trace:
            if isinstance(item, Mapping):
                normalised = _normalise_trainer_step_event(item, event_type=event_type)
                if normalised is not None:
                    events.append(normalised)

    route = st.telemetry.AtlasRoute()
    base = time.time() if timestamp_base is None else float(timestamp_base)
    for idx, event in enumerate(events):
        frame = trainer_step_event_to_atlas_frame(
            event,
            district=district,
            timestamp_base=base + float(idx) * max(0.0, float(step_seconds)),
            step_seconds=step_seconds,
        )
        if frame is None:
            continue
        route.push_bounded(frame, bound=int(bound))
    return route
