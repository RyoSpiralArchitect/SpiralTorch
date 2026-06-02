from __future__ import annotations

import html
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "load_zspace_artifact_manifest",
    "build_zspace_planner_snapshot",
    "write_zspace_experiment_artifacts",
    "summarize_zspace_experiment_manifest",
    "write_zspace_experiment_cockpit_html",
    "summarize_zspace_experiment_index",
    "write_zspace_experiment_index_html",
    "compare_zspace_experiment_manifests",
    "write_zspace_experiment_comparison_html",
    "build_zspace_downstream_hook",
    "build_desire_adapter_from_downstream_hook",
    "desire_step_from_downstream_hook",
    "run_desire_geometry_bias_validation",
]


def load_zspace_artifact_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("artifact manifest must be a JSON object")
    payload.setdefault("artifact_manifest", str(manifest_path))
    return payload


def _coerce_manifest(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest_or_path, Mapping):
        return dict(manifest_or_path)
    return load_zspace_artifact_manifest(manifest_or_path)


def _coerce_focus_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    as_dict = getattr(value, "to_dict", None)
    if callable(as_dict):
        try:
            return _json_ready(as_dict())
        except Exception:
            pass

    payload: dict[str, Any] = {}
    for name in (
        "kind",
        "backend",
        "requested_backend",
        "effective_backend",
        "rows",
        "cols",
        "k",
        "workgroup",
        "lanes",
        "tile",
        "score",
        "algorithm",
        "strategy",
    ):
        if hasattr(value, name):
            try:
                payload[name] = _json_ready(getattr(value, name))
            except Exception:
                continue
    if payload:
        return payload

    if hasattr(value, "__dict__"):
        public = {
            key: item
            for key, item in vars(value).items()
            if isinstance(key, str) and not key.startswith("_")
        }
        if public:
            return _json_ready(public)

    return str(value)


def _runtime_callable(name: str) -> Any | None:
    module = sys.modules.get("spiraltorch")
    if module is None:
        return None
    candidate = getattr(module, name, None)
    return candidate if callable(candidate) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def build_zspace_planner_snapshot(
    *,
    backend: str | None = "auto",
    rows: int | None = None,
    cols: int | None = None,
    k: int | None = None,
    describe_device: Any | None = None,
    plan_topk: Any | None = None,
    device_report: Mapping[str, Any] | None = None,
    rank_plan: Any | None = None,
    runtime_matrix: Mapping[str, Any] | None = None,
    capture_runtime_matrix: bool = False,
    trace_runtime_matrix: Any | None = None,
    runtime_matrix_backends: Any | None = None,
) -> dict[str, Any]:
    """Capture the planner/device decision that frames a Z-space experiment."""

    backend_label = "auto" if backend is None else str(backend).strip().lower() or "auto"
    rows_i = _optional_int(rows)
    cols_i = _optional_int(cols)
    k_i = _optional_int(k)
    errors: list[dict[str, str]] = []

    if describe_device is None:
        describe_device = _runtime_callable("describe_device")
    if plan_topk is None:
        plan_topk = _runtime_callable("plan_topk")
    if trace_runtime_matrix is None:
        trace_runtime_matrix = _runtime_callable("trace_wgpu_first_runtime_matrix")

    if device_report is None and callable(describe_device):
        try:
            candidate = describe_device(backend_label)
            if isinstance(candidate, Mapping):
                device_report = candidate
        except Exception as exc:  # noqa: BLE001 - snapshots should preserve partial evidence.
            errors.append(
                {
                    "stage": "describe_device",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    if (
        rank_plan is None
        and callable(plan_topk)
        and rows_i is not None
        and cols_i is not None
        and k_i is not None
    ):
        try:
            rank_plan = plan_topk(rows_i, cols_i, k_i, backend=backend_label)
        except Exception as exc:  # noqa: BLE001 - snapshots should preserve partial evidence.
            errors.append(
                {
                    "stage": "plan_topk",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    if runtime_matrix is None and capture_runtime_matrix and callable(trace_runtime_matrix):
        matrix_kwargs: dict[str, int] = {}
        if rows_i is not None:
            matrix_kwargs["rows"] = rows_i
        if cols_i is not None:
            matrix_kwargs["cols"] = cols_i
        if k_i is not None:
            matrix_kwargs["k"] = k_i
        try:
            if runtime_matrix_backends is None:
                candidate = trace_runtime_matrix(**matrix_kwargs)
            else:
                candidate = trace_runtime_matrix(runtime_matrix_backends, **matrix_kwargs)
            if isinstance(candidate, Mapping):
                runtime_matrix = candidate
        except Exception as exc:  # noqa: BLE001 - snapshots should preserve partial evidence.
            errors.append(
                {
                    "stage": "trace_wgpu_first_runtime_matrix",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    snapshot: dict[str, Any] = {
        "kind": "spiraltorch.zspace_planner_snapshot",
        "backend": backend_label,
        "available": bool(device_report is not None or rank_plan is not None),
    }
    if rows_i is not None or cols_i is not None or k_i is not None:
        snapshot["shape"] = {
            "rows": rows_i,
            "cols": cols_i,
            "k": k_i,
        }
    if device_report is not None:
        snapshot["device_report"] = _json_ready(device_report)
    if rank_plan is not None:
        snapshot["rank_plan"] = _json_ready(rank_plan)
    if runtime_matrix is not None:
        snapshot["runtime_matrix"] = _json_ready(runtime_matrix)
    if errors:
        snapshot["errors"] = errors
    return snapshot


def _relative_link(from_path: Path, target: Path) -> str:
    return Path(os.path.relpath(target, start=from_path.parent)).as_posix()


def write_zspace_experiment_artifacts(
    trace_jsonl: str | Path,
    *,
    trace_html: str | Path | None = None,
    atlas_html: str | Path | None = None,
    manifest: str | Path | None = None,
    title: str = "SpiralTorch Z-Space Experiment",
    district: str = "Concourse",
    event_type: str = "ZSpaceTrace",
    bound: int = 256,
    top_k: int = 12,
    metadata: Mapping[str, Any] | None = None,
    capture_planner: bool = True,
    planner_backend: str | None = "auto",
    planner_rows: int | None = None,
    planner_cols: int | None = None,
    planner_k: int | None = None,
    planner_snapshot: Mapping[str, Any] | None = None,
    describe_device: Any | None = None,
    plan_topk: Any | None = None,
    device_report: Mapping[str, Any] | None = None,
    rank_plan: Any | None = None,
    runtime_matrix: Mapping[str, Any] | None = None,
    capture_runtime_matrix: bool = False,
    trace_runtime_matrix: Any | None = None,
    runtime_matrix_backends: Any | None = None,
) -> dict[str, Any]:
    """Write trace, Atlas, and manifest artifacts for one observable Z-space run."""

    from .zspace_atlas import zspace_trace_to_atlas_route, write_zspace_atlas_noncollapse_html
    from .zspace_trace import write_zspace_trace_html

    trace_jsonl_path = Path(trace_jsonl)
    trace_html_path = Path(trace_html) if trace_html is not None else trace_jsonl_path.with_suffix(".html")
    atlas_html_path = (
        Path(atlas_html)
        if atlas_html is not None
        else trace_jsonl_path.with_suffix(".atlas_noncollapse.html")
    )
    manifest_path = (
        Path(manifest)
        if manifest is not None
        else trace_jsonl_path.with_suffix(".artifacts.json")
    )

    for path in (trace_html_path, atlas_html_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    trace_related_links = {
        "Atlas view": _relative_link(trace_html_path, atlas_html_path),
        "Artifact manifest": _relative_link(trace_html_path, manifest_path),
        "Trace JSONL": _relative_link(trace_html_path, trace_jsonl_path),
    }
    atlas_related_links = {
        "Trace viewer": _relative_link(atlas_html_path, trace_html_path),
        "Artifact manifest": _relative_link(atlas_html_path, manifest_path),
        "Trace JSONL": _relative_link(atlas_html_path, trace_jsonl_path),
    }

    trace_html_out = write_zspace_trace_html(
        trace_jsonl_path,
        trace_html_path,
        title=title,
        event_type=event_type,
        related_links=trace_related_links,
    )
    route = zspace_trace_to_atlas_route(
        trace_jsonl_path,
        district=district,
        bound=bound,
        event_type=event_type,
    )
    atlas_html_out = write_zspace_atlas_noncollapse_html(
        route,
        atlas_html_path,
        title=f"{title} Atlas Non-Collapse",
        district=district,
        top_k=top_k,
        related_links=atlas_related_links,
    )

    summary = route.summary()
    perspective = route.perspective_for(district, focus_prefixes=["noncollapse."])
    planner_payload = dict(planner_snapshot) if isinstance(planner_snapshot, Mapping) else None
    if planner_payload is None and capture_planner:
        planner_payload = build_zspace_planner_snapshot(
            backend=planner_backend,
            rows=planner_rows,
            cols=planner_cols,
            k=planner_k,
            describe_device=describe_device,
            plan_topk=plan_topk,
            device_report=device_report,
            rank_plan=rank_plan,
            runtime_matrix=runtime_matrix,
            capture_runtime_matrix=capture_runtime_matrix,
            trace_runtime_matrix=trace_runtime_matrix,
            runtime_matrix_backends=runtime_matrix_backends,
        )
    elif planner_payload is not None and runtime_matrix is not None:
        planner_payload.setdefault("runtime_matrix", _json_ready(runtime_matrix))

    manifest_payload: dict[str, Any] = {
        "kind": "spiraltorch.zspace_experiment_manifest",
        "schema": "spiraltorch.zspace_experiment",
        "schema_version": 1,
        "title": str(title),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace_jsonl": str(trace_jsonl_path),
        "trace_html": str(trace_html_out),
        "atlas_noncollapse_html": str(atlas_html_out),
        "artifact_manifest": str(manifest_path),
        "district": str(district),
        "event_type": str(event_type),
        "summary": summary,
        "noncollapse_perspective": perspective,
        "views": {
            "trace_jsonl": str(trace_jsonl_path),
            "trace_html": str(trace_html_out),
            "atlas_noncollapse_html": str(atlas_html_out),
            "artifact_manifest": str(manifest_path),
        },
        "metadata": _json_ready(dict(metadata)) if isinstance(metadata, Mapping) else {},
    }
    if planner_payload is not None:
        manifest_payload["planner_snapshot"] = _json_ready(planner_payload)
    manifest_payload["downstream_hook"] = build_zspace_downstream_hook(manifest_payload)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_payload


def _phase_hint(stage_focus: Sequence[Mapping[str, Any]]) -> str | None:
    ranked = []
    for item in stage_focus:
        latest = item.get("latest")
        try:
            latest_value = float(latest)
        except (TypeError, ValueError):
            latest_value = 0.0
        ranked.append((latest_value, str(item.get("stage") or "")))
    ranked.sort(reverse=True)
    for _, stage in ranked:
        if stage:
            return stage
    return None


def build_zspace_downstream_hook(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    top_k: int = 6,
) -> dict[str, Any]:
    manifest = _coerce_manifest(manifest_or_path)
    perspective = manifest.get("noncollapse_perspective")
    perspective_payload = dict(perspective) if isinstance(perspective, Mapping) else {}
    focus_items = _coerce_focus_items(perspective_payload.get("focus", []))
    stage_focus = []
    metric_focus = []
    for item in focus_items:
        name = str(item.get("name", ""))
        if name.startswith("noncollapse.stage."):
            enriched = dict(item)
            enriched.setdefault("stage", name.removeprefix("noncollapse.stage."))
            stage_focus.append(enriched)
        else:
            metric_focus.append(dict(item))
    if top_k > 0:
        metric_focus = metric_focus[: int(top_k)]

    summary = manifest.get("summary")
    summary_payload = dict(summary) if isinstance(summary, Mapping) else {}
    guidance = str(perspective_payload.get("guidance", ""))
    top_metric = metric_focus[0]["name"] if metric_focus else None
    phase_hint = _phase_hint(stage_focus)

    return {
        "kind": "spiraltorch.zspace_artifact_hook",
        "views": {
            "trace_jsonl": manifest.get("trace_jsonl"),
            "trace_html": manifest.get("trace_html"),
            "atlas_noncollapse_html": manifest.get("atlas_noncollapse_html"),
            "artifact_manifest": manifest.get("artifact_manifest"),
        },
        "summary": {
            "frames": int(summary_payload.get("frames", 0) or 0),
            "total_notes": int(summary_payload.get("total_notes", 0) or 0),
            "guidance": guidance,
        },
        "signals": {
            "coverage": int(perspective_payload.get("coverage", 0) or 0),
            "mean": float(perspective_payload.get("mean", 0.0) or 0.0),
            "latest": float(perspective_payload.get("latest", 0.0) or 0.0),
            "delta": float(perspective_payload.get("delta", 0.0) or 0.0),
            "momentum": float(perspective_payload.get("momentum", 0.0) or 0.0),
            "volatility": float(perspective_payload.get("volatility", 0.0) or 0.0),
            "stability": float(perspective_payload.get("stability", 0.0) or 0.0),
        },
        "stage_focus": stage_focus,
        "top_focus": metric_focus,
        "desire_candidate": {
            "experimental": True,
            "stability_signal": float(perspective_payload.get("stability", 0.0) or 0.0),
            "momentum_signal": float(perspective_payload.get("momentum", 0.0) or 0.0),
            "delta_signal": float(perspective_payload.get("delta", 0.0) or 0.0),
            "phase_hint": phase_hint,
            "focus_metric": top_metric,
            "guidance": guidance,
        },
    }


def _coerce_downstream_hook(hook_or_manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    payload = _coerce_manifest(hook_or_manifest)
    if payload.get("kind") == "spiraltorch.zspace_artifact_hook":
        return payload
    downstream = payload.get("downstream_hook")
    if isinstance(downstream, Mapping):
        return dict(downstream)
    return build_zspace_downstream_hook(payload)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _views_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    views = _coerce_mapping(manifest.get("views"))
    for key in (
        "trace_jsonl",
        "trace_html",
        "atlas_noncollapse_html",
        "artifact_manifest",
    ):
        if key in manifest and key not in views:
            views[key] = manifest.get(key)
    return views


def _planner_story(manifest: Mapping[str, Any]) -> dict[str, Any]:
    planner = _coerce_mapping(manifest.get("planner_snapshot"))
    device_report = _coerce_mapping(planner.get("device_report"))
    rank_plan = _coerce_mapping(planner.get("rank_plan"))
    shape = _coerce_mapping(planner.get("shape"))
    runtime_matrix = _runtime_matrix_story(planner)
    requested_backend = str(
        rank_plan.get("requested_backend")
        or planner.get("backend")
        or device_report.get("backend")
        or "unknown"
    )
    effective_backend = str(
        rank_plan.get("effective_backend")
        or device_report.get("planner_surrogate_backend")
        or device_report.get("backend")
        or "unknown"
    )
    route = str(
        device_report.get("planner_route")
        or rank_plan.get("strategy")
        or effective_backend
    )
    errors = _coerce_focus_items(planner.get("errors", []))

    return {
        "available": bool(planner.get("available", bool(device_report or rank_plan))),
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
        "route": route,
        "shape": {
            "rows": shape.get("rows"),
            "cols": shape.get("cols"),
            "k": shape.get("k"),
        },
        "device_report": device_report,
        "rank_plan": rank_plan,
        "runtime_matrix": runtime_matrix,
        "errors": errors,
    }


def _runtime_matrix_story(planner: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _coerce_mapping(planner.get("runtime_matrix"))
    summary = _coerce_mapping(matrix.get("summary"))
    requested = matrix.get("requested_backends", [])
    if isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)):
        requested_backends = [str(item) for item in requested]
    elif requested:
        requested_backends = [str(requested)]
    else:
        requested_backends = []
    return {
        "available": bool(matrix),
        "kind": matrix.get("kind"),
        "artifact_path": matrix.get("artifact_path"),
        "requested_backends": requested_backends,
        "summary": summary,
        "effective_backends": _coerce_mapping(summary.get("effective_backends")),
        "errors": _coerce_focus_items(matrix.get("errors", [])),
        "runs": _coerce_focus_items(matrix.get("runs", [])),
        "raw": matrix,
    }


def _trim_focus(items: Sequence[Mapping[str, Any]], top_k: int) -> list[dict[str, Any]]:
    limit = max(0, int(top_k))
    focus = [dict(item) for item in items]
    return focus if limit == 0 else focus[:limit]


def _story_card(kind: str, title: str, body: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "title": title,
        "body": body,
    }
    payload.update(extra)
    return payload


def summarize_zspace_experiment_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    top_k: int = 6,
) -> dict[str, Any]:
    """Summarize a Z-space experiment manifest into one portable story packet."""

    manifest = _coerce_manifest(manifest_or_path)
    hook = _coerce_downstream_hook(manifest)
    summary = _coerce_mapping(manifest.get("summary"))
    signals = _coerce_mapping(hook.get("signals"))
    downstream_summary = _coerce_mapping(hook.get("summary"))
    planner = _planner_story(manifest)
    top_focus = _trim_focus(_coerce_focus_items(hook.get("top_focus", [])), top_k)
    stage_focus = _trim_focus(_coerce_focus_items(hook.get("stage_focus", [])), top_k)
    candidate = _coerce_mapping(hook.get("desire_candidate"))
    views = _views_from_manifest(manifest)

    frames = _coerce_int(summary.get("frames", downstream_summary.get("frames", 0)))
    total_notes = _coerce_int(summary.get("total_notes", downstream_summary.get("total_notes", 0)))
    guidance = str(
        downstream_summary.get("guidance")
        or candidate.get("guidance")
        or _coerce_mapping(manifest.get("noncollapse_perspective")).get("guidance")
        or ""
    )
    phase_hint = candidate.get("phase_hint")
    focus_metric = candidate.get("focus_metric") or (top_focus[0].get("name") if top_focus else None)
    stability = _coerce_float(signals.get("stability", candidate.get("stability_signal", 0.0)))
    momentum = _coerce_float(signals.get("momentum", candidate.get("momentum_signal", 0.0)))
    delta = _coerce_float(signals.get("delta", candidate.get("delta_signal", 0.0)))

    title = str(manifest.get("title") or "SpiralTorch Z-Space Experiment")
    story = [
        _story_card(
            "run",
            "Run",
            f"{title}: {frames} trace frame(s), {total_notes} Atlas note(s).",
            frames=frames,
            total_notes=total_notes,
        ),
        _story_card(
            "planner",
            "Planner",
            (
                f"{planner['requested_backend']} routed as "
                f"{planner['effective_backend']} via {planner['route']}."
            ),
            requested_backend=planner["requested_backend"],
            effective_backend=planner["effective_backend"],
            route=planner["route"],
            available=planner["available"],
        ),
        _story_card(
            "noncollapse",
            "Non-Collapse",
            (
                f"stability {stability:.3f}, momentum {momentum:+.3f}, "
                f"delta {delta:+.3f}."
            ),
            stability=stability,
            momentum=momentum,
            delta=delta,
            phase_hint=phase_hint,
            focus_metric=focus_metric,
        ),
    ]
    runtime_matrix = _coerce_mapping(planner.get("runtime_matrix"))
    runtime_summary = _coerce_mapping(runtime_matrix.get("summary"))
    if runtime_matrix.get("available"):
        story.append(
            _story_card(
                "runtime_matrix",
                "Runtime Matrix",
                (
                    f"{_coerce_int(runtime_summary.get('ok', 0))} ok, "
                    f"{_coerce_int(runtime_summary.get('partial', 0))} partial, "
                    f"{_coerce_int(runtime_summary.get('errors', 0))} error(s); "
                    "effective "
                    f"{_format_counts(_coerce_mapping(runtime_summary.get('effective_backends')))}."
                ),
                summary=runtime_summary,
                effective_backends=_coerce_mapping(
                    runtime_summary.get("effective_backends")
                ),
            )
        )
    if guidance:
        story.append(
            _story_card(
                "guidance",
                "Guidance",
                guidance,
            )
        )
    if focus_metric:
        story.append(
            _story_card(
                "focus",
                "Focus",
                f"primary focus metric: {focus_metric}",
                focus_metric=focus_metric,
            )
        )

    return {
        "kind": "spiraltorch.zspace_experiment_story",
        "schema": "spiraltorch.zspace_experiment_story",
        "schema_version": 1,
        "title": title,
        "created_at": manifest.get("created_at"),
        "artifact_manifest": manifest.get("artifact_manifest"),
        "views": views,
        "summary": {
            "frames": frames,
            "total_notes": total_notes,
            "guidance": guidance,
        },
        "planner": planner,
        "noncollapse": {
            "signals": {
                "coverage": _coerce_int(signals.get("coverage", 0)),
                "mean": _coerce_float(signals.get("mean", 0.0)),
                "latest": _coerce_float(signals.get("latest", 0.0)),
                "delta": delta,
                "momentum": momentum,
                "volatility": _coerce_float(signals.get("volatility", 0.0)),
                "stability": stability,
            },
            "stage_focus": stage_focus,
            "top_focus": top_focus,
            "phase_hint": phase_hint,
            "focus_metric": focus_metric,
        },
        "downstream_hook": hook,
        "story": story,
    }


def _html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _format_focus_value(item: Mapping[str, Any]) -> str:
    latest = item.get("latest")
    try:
        return f"{float(latest):.3f}"
    except (TypeError, ValueError):
        return _html_escape(latest)


def _count_strings(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label = str(value or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _merge_count_mappings(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mapping in values:
        for label, count in mapping.items():
            key = str(label or "unknown")
            counts[key] = counts.get(key, 0) + _coerce_int(count, 0)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _mean_or_none(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _experiment_run_digest(story: Mapping[str, Any]) -> dict[str, Any]:
    summary = _coerce_mapping(story.get("summary"))
    planner = _coerce_mapping(story.get("planner"))
    noncollapse = _coerce_mapping(story.get("noncollapse"))
    signals = _coerce_mapping(noncollapse.get("signals"))
    runtime_matrix = _coerce_mapping(planner.get("runtime_matrix"))
    runtime_summary = _coerce_mapping(runtime_matrix.get("summary"))
    views = _coerce_mapping(story.get("views"))

    return {
        "title": story.get("title") or "SpiralTorch Z-Space Experiment",
        "created_at": story.get("created_at"),
        "artifact_manifest": story.get("artifact_manifest") or views.get("artifact_manifest"),
        "views": views,
        "summary": {
            "frames": _coerce_int(summary.get("frames", 0)),
            "total_notes": _coerce_int(summary.get("total_notes", 0)),
            "guidance": str(summary.get("guidance") or ""),
        },
        "planner": {
            "available": bool(planner.get("available", False)),
            "requested_backend": planner.get("requested_backend") or "unknown",
            "effective_backend": planner.get("effective_backend") or "unknown",
            "route": planner.get("route") or "unknown",
            "shape": _coerce_mapping(planner.get("shape")),
        },
        "runtime_matrix": {
            "available": bool(runtime_matrix.get("available", False)),
            "summary": runtime_summary,
            "effective_backends": _coerce_mapping(
                runtime_summary.get("effective_backends")
            ),
            "errors": _coerce_focus_items(runtime_matrix.get("errors", [])),
        },
        "noncollapse": {
            "stability": _coerce_float(signals.get("stability", 0.0)),
            "momentum": _coerce_float(signals.get("momentum", 0.0)),
            "delta": _coerce_float(signals.get("delta", 0.0)),
            "phase_hint": noncollapse.get("phase_hint"),
            "focus_metric": noncollapse.get("focus_metric") or "unknown",
            "top_focus": _coerce_focus_items(noncollapse.get("top_focus", [])),
        },
    }


def summarize_zspace_experiment_index(
    manifests: Sequence[Mapping[str, Any] | str | Path],
    *,
    top_k: int = 6,
    title: str = "SpiralTorch Z-Space Experiment Index",
) -> dict[str, Any]:
    """Summarize multiple Z-space experiment manifests into a comparison index."""

    if isinstance(manifests, (str, bytes, Path)):
        raise TypeError("manifests must be a sequence of manifest mappings or paths")

    stories = [
        summarize_zspace_experiment_manifest(manifest, top_k=top_k)
        for manifest in manifests
    ]
    runs = [_experiment_run_digest(story) for story in stories]
    total_frames = sum(_coerce_int(run["summary"].get("frames", 0)) for run in runs)
    total_notes = sum(_coerce_int(run["summary"].get("total_notes", 0)) for run in runs)
    stability_values = [
        _coerce_float(run["noncollapse"].get("stability", 0.0))
        for run in runs
    ]

    return {
        "kind": "spiraltorch.zspace_experiment_index",
        "schema": "spiraltorch.zspace_experiment_index",
        "schema_version": 1,
        "title": str(title),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "runs": len(runs),
            "total_frames": total_frames,
            "total_notes": total_notes,
            "planner_backends": _count_strings(
                [run["planner"].get("effective_backend") for run in runs]
            ),
            "planner_routes": _count_strings(
                [run["planner"].get("route") for run in runs]
            ),
            "runtime_matrix_effective_backends": _merge_count_mappings(
                [
                    _coerce_mapping(
                        _coerce_mapping(run.get("runtime_matrix")).get(
                            "effective_backends"
                        )
                    )
                    for run in runs
                    if _coerce_mapping(run.get("runtime_matrix")).get("available")
                ]
            ),
            "focus_metrics": _count_strings(
                [run["noncollapse"].get("focus_metric") for run in runs]
            ),
            "mean_stability": _mean_or_none(stability_values),
            "latest_stability": stability_values[-1] if stability_values else None,
        },
        "runs": runs,
    }


def _comparison_check(
    name: str,
    status: str,
    message: str,
    *,
    baseline: Any,
    candidate: Any,
    delta: Any | None = None,
    threshold: Any | None = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "status": status,
        "message": message,
        "baseline": baseline,
        "candidate": candidate,
    }
    if delta is not None:
        payload["delta"] = delta
    if threshold is not None:
        payload["threshold"] = threshold
    return payload


def _overall_status(checks: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status", "pass")) for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _comparison_guidance(status: str) -> str:
    if status == "fail":
        return "candidate regressed beyond the configured comparison thresholds"
    if status == "warn":
        return "candidate is usable but changed planner, focus, or run-shape signals"
    return "candidate stayed within the configured comparison thresholds"


def compare_zspace_experiment_manifests(
    baseline: Mapping[str, Any] | str | Path,
    candidate: Mapping[str, Any] | str | Path,
    *,
    top_k: int = 6,
    title: str | None = None,
    warn_stability_drop: float = 0.03,
    fail_stability_drop: float = 0.10,
    min_frame_ratio: float = 0.80,
    warn_on_planner_change: bool = True,
    warn_on_focus_change: bool = True,
) -> dict[str, Any]:
    """Compare a candidate Z-space experiment manifest against a baseline."""

    baseline_story = summarize_zspace_experiment_manifest(baseline, top_k=top_k)
    candidate_story = summarize_zspace_experiment_manifest(candidate, top_k=top_k)
    baseline_run = _experiment_run_digest(baseline_story)
    candidate_run = _experiment_run_digest(candidate_story)

    baseline_summary = _coerce_mapping(baseline_run.get("summary"))
    candidate_summary = _coerce_mapping(candidate_run.get("summary"))
    baseline_planner = _coerce_mapping(baseline_run.get("planner"))
    candidate_planner = _coerce_mapping(candidate_run.get("planner"))
    baseline_noncollapse = _coerce_mapping(baseline_run.get("noncollapse"))
    candidate_noncollapse = _coerce_mapping(candidate_run.get("noncollapse"))

    baseline_frames = _coerce_int(baseline_summary.get("frames", 0))
    candidate_frames = _coerce_int(candidate_summary.get("frames", 0))
    baseline_notes = _coerce_int(baseline_summary.get("total_notes", 0))
    candidate_notes = _coerce_int(candidate_summary.get("total_notes", 0))
    baseline_stability = _coerce_float(baseline_noncollapse.get("stability", 0.0))
    candidate_stability = _coerce_float(candidate_noncollapse.get("stability", 0.0))
    stability_delta = candidate_stability - baseline_stability
    momentum_delta = (
        _coerce_float(candidate_noncollapse.get("momentum", 0.0))
        - _coerce_float(baseline_noncollapse.get("momentum", 0.0))
    )
    signal_delta_delta = (
        _coerce_float(candidate_noncollapse.get("delta", 0.0))
        - _coerce_float(baseline_noncollapse.get("delta", 0.0))
    )

    checks: list[dict[str, Any]] = []
    fail_drop = abs(float(fail_stability_drop))
    warn_drop = abs(float(warn_stability_drop))
    if stability_delta <= -fail_drop:
        stability_status = "fail"
        stability_message = "stability dropped beyond the fail threshold"
    elif stability_delta <= -warn_drop:
        stability_status = "warn"
        stability_message = "stability dropped beyond the warn threshold"
    else:
        stability_status = "pass"
        stability_message = "stability stayed within threshold"
    checks.append(
        _comparison_check(
            "stability",
            stability_status,
            stability_message,
            baseline=baseline_stability,
            candidate=candidate_stability,
            delta=stability_delta,
            threshold={"warn_drop": warn_drop, "fail_drop": fail_drop},
        )
    )

    frame_delta = candidate_frames - baseline_frames
    frame_ratio = (
        candidate_frames / baseline_frames if baseline_frames > 0 else None
    )
    if baseline_frames > 0 and candidate_frames < baseline_frames * float(min_frame_ratio):
        frame_status = "fail"
        frame_message = "candidate emitted too few frames for this baseline"
    elif candidate_frames < baseline_frames:
        frame_status = "warn"
        frame_message = "candidate emitted fewer frames than the baseline"
    else:
        frame_status = "pass"
        frame_message = "candidate frame count stayed within threshold"
    checks.append(
        _comparison_check(
            "frames",
            frame_status,
            frame_message,
            baseline=baseline_frames,
            candidate=candidate_frames,
            delta=frame_delta,
            threshold={"min_frame_ratio": float(min_frame_ratio)},
        )
    )

    note_delta = candidate_notes - baseline_notes
    checks.append(
        _comparison_check(
            "total_notes",
            "warn" if candidate_notes < baseline_notes else "pass",
            (
                "candidate emitted fewer Atlas notes than the baseline"
                if candidate_notes < baseline_notes
                else "candidate note count stayed within threshold"
            ),
            baseline=baseline_notes,
            candidate=candidate_notes,
            delta=note_delta,
        )
    )

    baseline_backend = str(baseline_planner.get("effective_backend") or "unknown")
    candidate_backend = str(candidate_planner.get("effective_backend") or "unknown")
    baseline_route = str(baseline_planner.get("route") or "unknown")
    candidate_route = str(candidate_planner.get("route") or "unknown")
    planner_changed = (
        baseline_backend != candidate_backend or baseline_route != candidate_route
    )
    checks.append(
        _comparison_check(
            "planner",
            "warn" if planner_changed and warn_on_planner_change else "pass",
            (
                "candidate changed planner backend or route"
                if planner_changed
                else "candidate planner route matched the baseline"
            ),
            baseline={"backend": baseline_backend, "route": baseline_route},
            candidate={"backend": candidate_backend, "route": candidate_route},
        )
    )

    baseline_focus = str(baseline_noncollapse.get("focus_metric") or "unknown")
    candidate_focus = str(candidate_noncollapse.get("focus_metric") or "unknown")
    focus_changed = baseline_focus != candidate_focus
    checks.append(
        _comparison_check(
            "focus_metric",
            "warn" if focus_changed and warn_on_focus_change else "pass",
            (
                "candidate primary focus metric changed"
                if focus_changed
                else "candidate primary focus metric matched the baseline"
            ),
            baseline=baseline_focus,
            candidate=candidate_focus,
        )
    )

    status = _overall_status(checks)
    page_title = title or (
        f"{baseline_run.get('title')} -> {candidate_run.get('title')}"
    )
    return {
        "kind": "spiraltorch.zspace_experiment_comparison",
        "schema": "spiraltorch.zspace_experiment_comparison",
        "schema_version": 1,
        "title": str(page_title),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "status": status,
            "guidance": _comparison_guidance(status),
            "baseline_title": baseline_run.get("title"),
            "candidate_title": candidate_run.get("title"),
            "stability_delta": stability_delta,
            "momentum_delta": momentum_delta,
            "signal_delta_delta": signal_delta_delta,
            "frames_delta": frame_delta,
            "frame_ratio": frame_ratio,
            "total_notes_delta": note_delta,
            "planner_changed": planner_changed,
            "focus_metric_changed": focus_changed,
        },
        "thresholds": {
            "warn_stability_drop": warn_drop,
            "fail_stability_drop": fail_drop,
            "min_frame_ratio": float(min_frame_ratio),
            "warn_on_planner_change": bool(warn_on_planner_change),
            "warn_on_focus_change": bool(warn_on_focus_change),
        },
        "baseline": baseline_run,
        "candidate": candidate_run,
        "checks": checks,
    }


def write_zspace_experiment_cockpit_html(
    manifest_or_path: Mapping[str, Any] | str | Path,
    html_path: str | Path | None = None,
    *,
    title: str | None = None,
    top_k: int = 6,
) -> str:
    """Render a compact HTML cockpit for a Z-space experiment manifest."""

    story = summarize_zspace_experiment_manifest(manifest_or_path, top_k=top_k)
    manifest_path = story.get("artifact_manifest")
    if html_path is None:
        if manifest_path:
            html_path = Path(str(manifest_path)).with_suffix(".cockpit.html")
        else:
            html_path = Path("zspace_experiment.cockpit.html")
    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page_title = str(title or story.get("title") or "SpiralTorch Z-Space Experiment")
    story_json = json.dumps(story, ensure_ascii=True)
    links = [
        (label, href)
        for label, href in (
            ("Trace JSONL", story["views"].get("trace_jsonl")),
            ("Trace Viewer", story["views"].get("trace_html")),
            ("Atlas Non-Collapse", story["views"].get("atlas_noncollapse_html")),
            ("Artifact Manifest", story["views"].get("artifact_manifest")),
        )
        if href
    ]
    link_html = "\n".join(
        f'<a href="{_html_escape(href)}">{_html_escape(label)}</a>' for label, href in links
    )
    card_html = "\n".join(
        "<section class=\"card\">"
        f"<p>{_html_escape(card.get('title'))}</p>"
        f"<strong>{_html_escape(card.get('body'))}</strong>"
        "</section>"
        for card in story.get("story", [])
        if isinstance(card, Mapping)
    )
    focus_html = "\n".join(
        "<tr>"
        f"<td>{_html_escape(item.get('name'))}</td>"
        f"<td>{_format_focus_value(item)}</td>"
        f"<td>{_html_escape(item.get('coverage', ''))}</td>"
        "</tr>"
        for item in story["noncollapse"].get("top_focus", [])
        if isinstance(item, Mapping)
    )
    stage_html = "\n".join(
        "<tr>"
        f"<td>{_html_escape(item.get('stage', item.get('name')))}</td>"
        f"<td>{_format_focus_value(item)}</td>"
        f"<td>{_html_escape(item.get('coverage', ''))}</td>"
        "</tr>"
        for item in story["noncollapse"].get("stage_focus", [])
        if isinstance(item, Mapping)
    )
    planner = story["planner"]
    planner_shape = planner.get("shape", {})
    device_report = planner.get("device_report", {})
    rank_plan = planner.get("rank_plan", {})
    runtime_matrix = _coerce_mapping(planner.get("runtime_matrix"))
    runtime_summary = _coerce_mapping(runtime_matrix.get("summary"))
    runtime_matrix_section = ""
    if runtime_matrix.get("available"):
        runtime_rows = "\n".join(
            "<tr>"
            f"<td>{_html_escape(run.get('requested_backend'))}</td>"
            f"<td>{_html_escape(run.get('effective_backend'))}</td>"
            f"<td>{_html_escape(run.get('matrix_status'))}</td>"
            f"<td>{_html_escape(_coerce_mapping(run.get('tensor_operation')).get('ok'))}</td>"
            "</tr>"
            for run in runtime_matrix.get("runs", [])
            if isinstance(run, Mapping)
        )
        runtime_matrix_section = f"""
      <section>
        <h2>Runtime Matrix</h2>
        <div class="grid">
          <div class="kv"><span>ok</span><strong>{_html_escape(runtime_summary.get("ok", 0))}</strong></div>
          <div class="kv"><span>partial</span><strong>{_html_escape(runtime_summary.get("partial", 0))}</strong></div>
          <div class="kv"><span>errors</span><strong>{_html_escape(runtime_summary.get("errors", 0))}</strong></div>
          <div class="kv"><span>effective</span><strong>{_html_escape(_format_counts(_coerce_mapping(runtime_summary.get("effective_backends"))))}</strong></div>
        </div>
        <table>
          <thead><tr><th>requested</th><th>effective</th><th>status</th><th>tensor ok</th></tr></thead>
          <tbody>{runtime_rows}</tbody>
        </table>
      </section>
"""

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html_escape(page_title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #131923;
      --panel-2: #192231;
      --text: #edf4ff;
      --muted: #9fb2c8;
      --accent: #6ee7b7;
      --border: rgba(255,255,255,.1);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(110,231,183,.09), rgba(0,0,0,0));
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    header p, .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    nav a {{
      color: var(--accent);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 9px;
      text-decoration: none;
      background: rgba(110,231,183,.08);
      font-size: 12px;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(260px, 360px) 1fr;
      gap: 14px;
      padding: 14px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    .cards {{
      display: grid;
      gap: 8px;
    }}
    .card {{
      background: var(--panel-2);
      border-radius: 8px;
      padding: 10px;
    }}
    .card p {{
      margin: 0 0 4px;
      color: var(--muted);
      font-size: 11px;
    }}
    .card strong {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      line-height: 1.35;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .kv {{
      background: var(--panel-2);
      border-radius: 8px;
      padding: 8px;
      min-width: 0;
    }}
    .kv span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
    }}
    .kv strong {{
      display: block;
      overflow-wrap: anywhere;
      font-size: 13px;
      margin-top: 3px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 7px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 340px;
      overflow: auto;
      background: var(--panel-2);
      border-radius: 8px;
      padding: 10px;
      font-size: 11px;
      color: #d8e6fb;
    }}
    @media (max-width: 820px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{_html_escape(page_title)}</h1>
    <p>{_html_escape(story.get("created_at") or "manifest story")}</p>
    <nav>{link_html}</nav>
  </header>
  <main>
    <div class="cards">
      {card_html}
      <section>
        <h2>Planner Snapshot</h2>
        <div class="grid">
          <div class="kv"><span>requested</span><strong>{_html_escape(planner.get("requested_backend"))}</strong></div>
          <div class="kv"><span>effective</span><strong>{_html_escape(planner.get("effective_backend"))}</strong></div>
          <div class="kv"><span>route</span><strong>{_html_escape(planner.get("route"))}</strong></div>
          <div class="kv"><span>shape</span><strong>{_html_escape(planner_shape)}</strong></div>
        </div>
      </section>
      {runtime_matrix_section}
    </div>
    <div class="cards">
      <section>
        <h2>Top Focus</h2>
        <table>
          <thead><tr><th>metric</th><th>latest</th><th>coverage</th></tr></thead>
          <tbody>{focus_html}</tbody>
        </table>
      </section>
      <section>
        <h2>Stage Focus</h2>
        <table>
          <thead><tr><th>stage</th><th>latest</th><th>coverage</th></tr></thead>
          <tbody>{stage_html}</tbody>
        </table>
      </section>
      <section>
        <h2>Raw Planner Payload</h2>
        <pre>{_html_escape(json.dumps({"device_report": device_report, "rank_plan": rank_plan, "runtime_matrix": runtime_matrix.get("raw")}, indent=2, ensure_ascii=False))}</pre>
      </section>
      <section>
        <h2>Story Packet</h2>
        <pre id="story-json">{_html_escape(json.dumps(story, indent=2, ensure_ascii=False))}</pre>
      </section>
    </div>
  </main>
  <script type="application/json" id="zspace-story">{_html_escape(story_json)}</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return str(out_path)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _index_run_links(run: Mapping[str, Any]) -> str:
    views = _coerce_mapping(run.get("views"))
    links = [
        ("Cockpit", views.get("cockpit_html") or views.get("experiment_cockpit_html")),
        ("Trace Viewer", views.get("trace_html")),
        ("Atlas Non-Collapse", views.get("atlas_noncollapse_html")),
        ("Manifest", run.get("artifact_manifest") or views.get("artifact_manifest")),
    ]
    return " ".join(
        f'<a href="{_html_escape(href)}">{_html_escape(label)}</a>'
        for label, href in links
        if href
    )


def _format_counts(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(
        f"{label} ({_coerce_int(count)})"
        for label, count in counts.items()
    )


def write_zspace_experiment_index_html(
    manifests: Sequence[Mapping[str, Any] | str | Path],
    html_path: str | Path | None = None,
    *,
    title: str = "SpiralTorch Z-Space Experiment Index",
    top_k: int = 6,
) -> str:
    """Render a static HTML index comparing multiple Z-space experiment manifests."""

    index = summarize_zspace_experiment_index(manifests, top_k=top_k, title=title)
    out_path = Path(html_path) if html_path is not None else Path("zspace_experiment.index.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page_title = str(index.get("title") or title)
    summary = _coerce_mapping(index.get("summary"))
    index_json = json.dumps(index, ensure_ascii=True)
    stat_html = "\n".join(
        "<section class=\"stat\">"
        f"<span>{_html_escape(label)}</span>"
        f"<strong>{_html_escape(value)}</strong>"
        "</section>"
        for label, value in (
            ("runs", summary.get("runs", 0)),
            ("frames", summary.get("total_frames", 0)),
            ("notes", summary.get("total_notes", 0)),
            ("mean stability", _format_optional_float(summary.get("mean_stability"))),
            ("latest stability", _format_optional_float(summary.get("latest_stability"))),
            ("backends", _format_counts(_coerce_mapping(summary.get("planner_backends")))),
            (
                "runtime backends",
                _format_counts(
                    _coerce_mapping(summary.get("runtime_matrix_effective_backends"))
                ),
            ),
        )
    )
    rows_html = "\n".join(
        "<tr>"
        f"<td><strong>{_html_escape(run.get('title'))}</strong>"
        f"<span>{_html_escape(run.get('created_at') or '')}</span></td>"
        f"<td>{_html_escape(_coerce_mapping(run.get('planner')).get('effective_backend'))}"
        f"<span>{_html_escape(_coerce_mapping(run.get('planner')).get('route'))}</span></td>"
        f"<td>{_html_escape(_format_counts(_coerce_mapping(_coerce_mapping(run.get('runtime_matrix')).get('effective_backends'))))}</td>"
        f"<td>{_html_escape(_coerce_mapping(run.get('summary')).get('frames'))}</td>"
        f"<td>{_html_escape(_coerce_mapping(run.get('summary')).get('total_notes'))}</td>"
        f"<td>{_html_escape(_format_optional_float(_coerce_mapping(run.get('noncollapse')).get('stability')))}</td>"
        f"<td>{_html_escape(_coerce_mapping(run.get('noncollapse')).get('focus_metric'))}</td>"
        f"<td class=\"links\">{_index_run_links(run)}</td>"
        "</tr>"
        for run in index.get("runs", [])
        if isinstance(run, Mapping)
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html_escape(page_title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #131923;
      --panel-2: #192231;
      --text: #edf4ff;
      --muted: #9fb2c8;
      --accent: #6ee7b7;
      --border: rgba(255,255,255,.1);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(110,231,183,.09), rgba(0,0,0,0));
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    header p {{
      color: var(--muted);
      font-size: 12px;
      margin: 6px 0 0;
    }}
    main {{
      padding: 14px;
      display: grid;
      gap: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(7, minmax(120px, 1fr));
      gap: 8px;
    }}
    .stat, section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .stat span, td span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-top: 3px;
    }}
    .stat strong {{
      display: block;
      overflow-wrap: anywhere;
      font-size: 14px;
      margin-top: 4px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .links {{
      min-width: 190px;
    }}
    .links a {{
      display: inline-block;
      margin: 0 8px 6px 0;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 360px;
      overflow: auto;
      background: var(--panel-2);
      border-radius: 8px;
      padding: 10px;
      font-size: 11px;
      color: #d8e6fb;
    }}
    @media (max-width: 980px) {{
      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 680px) {{
      .stats {{
        grid-template-columns: 1fr;
      }}
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{_html_escape(page_title)}</h1>
    <p>{_html_escape(index.get("created_at"))}</p>
  </header>
  <main>
    <div class="stats">{stat_html}</div>
    <section>
      <h2>Experiment Runs</h2>
      <table>
        <thead>
          <tr>
            <th>run</th><th>backend</th><th>runtime</th><th>frames</th><th>notes</th>
            <th>stability</th><th>focus</th><th>links</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </section>
    <section>
      <h2>Index Packet</h2>
      <pre>{_html_escape(json.dumps(index, indent=2, ensure_ascii=False))}</pre>
    </section>
  </main>
  <script type="application/json" id="zspace-index">{_html_escape(index_json)}</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return str(out_path)


def write_zspace_experiment_comparison_html(
    baseline: Mapping[str, Any] | str | Path,
    candidate: Mapping[str, Any] | str | Path,
    html_path: str | Path | None = None,
    *,
    title: str | None = None,
    top_k: int = 6,
    warn_stability_drop: float = 0.03,
    fail_stability_drop: float = 0.10,
    min_frame_ratio: float = 0.80,
    warn_on_planner_change: bool = True,
    warn_on_focus_change: bool = True,
) -> str:
    """Render a static HTML comparison between baseline and candidate manifests."""

    comparison = compare_zspace_experiment_manifests(
        baseline,
        candidate,
        top_k=top_k,
        title=title,
        warn_stability_drop=warn_stability_drop,
        fail_stability_drop=fail_stability_drop,
        min_frame_ratio=min_frame_ratio,
        warn_on_planner_change=warn_on_planner_change,
        warn_on_focus_change=warn_on_focus_change,
    )
    candidate_run = _coerce_mapping(comparison.get("candidate"))
    candidate_manifest = candidate_run.get("artifact_manifest")
    if html_path is None:
        if candidate_manifest:
            html_path = Path(str(candidate_manifest)).with_suffix(".comparison.html")
        else:
            html_path = Path("zspace_experiment.comparison.html")
    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page_title = str(comparison.get("title") or "SpiralTorch Z-Space Comparison")
    summary = _coerce_mapping(comparison.get("summary"))
    baseline_run = _coerce_mapping(comparison.get("baseline"))
    comparison_json = json.dumps(comparison, ensure_ascii=True)
    stat_html = "\n".join(
        "<section class=\"stat\">"
        f"<span>{_html_escape(label)}</span>"
        f"<strong>{_html_escape(value)}</strong>"
        "</section>"
        for label, value in (
            ("status", comparison.get("status")),
            ("stability delta", _format_optional_float(summary.get("stability_delta"))),
            ("frames delta", summary.get("frames_delta")),
            ("notes delta", summary.get("total_notes_delta")),
            ("planner changed", summary.get("planner_changed")),
            ("focus changed", summary.get("focus_metric_changed")),
        )
    )
    run_html = "\n".join(
        "<section>"
        f"<h2>{_html_escape(label)}</h2>"
        f"<p><strong>{_html_escape(run.get('title'))}</strong></p>"
        f"<p class=\"muted\">{_html_escape(run.get('created_at') or '')}</p>"
        f"<p class=\"links\">{_index_run_links(run)}</p>"
        "</section>"
        for label, run in (("Baseline", baseline_run), ("Candidate", candidate_run))
    )
    checks_html = "\n".join(
        "<tr>"
        f"<td>{_html_escape(check.get('name'))}</td>"
        f"<td><strong>{_html_escape(check.get('status'))}</strong></td>"
        f"<td>{_html_escape(check.get('baseline'))}</td>"
        f"<td>{_html_escape(check.get('candidate'))}</td>"
        f"<td>{_html_escape(_format_optional_float(check.get('delta')) if 'delta' in check else '')}</td>"
        f"<td>{_html_escape(check.get('message'))}</td>"
        "</tr>"
        for check in comparison.get("checks", [])
        if isinstance(check, Mapping)
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html_escape(page_title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #131923;
      --panel-2: #192231;
      --text: #edf4ff;
      --muted: #9fb2c8;
      --accent: #6ee7b7;
      --border: rgba(255,255,255,.1);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(110,231,183,.09), rgba(0,0,0,0));
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    header p, .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    main {{
      padding: 14px;
      display: grid;
      gap: 14px;
    }}
    .stats, .runs {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .stats {{
      grid-template-columns: repeat(6, minmax(120px, 1fr));
    }}
    .stat, section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
    }}
    .stat strong {{
      display: block;
      overflow-wrap: anywhere;
      font-size: 14px;
      margin-top: 4px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .links a {{
      display: inline-block;
      margin: 0 8px 6px 0;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 360px;
      overflow: auto;
      background: var(--panel-2);
      border-radius: 8px;
      padding: 10px;
      font-size: 11px;
      color: #d8e6fb;
    }}
    @media (max-width: 980px) {{
      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 680px) {{
      .stats, .runs {{
        grid-template-columns: 1fr;
      }}
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{_html_escape(page_title)}</h1>
    <p>{_html_escape(summary.get("guidance"))}</p>
  </header>
  <main>
    <div class="stats">{stat_html}</div>
    <div class="runs">{run_html}</div>
    <section>
      <h2>Comparison Checks</h2>
      <table>
        <thead>
          <tr>
            <th>check</th><th>status</th><th>baseline</th><th>candidate</th>
            <th>delta</th><th>message</th>
          </tr>
        </thead>
        <tbody>{checks_html}</tbody>
      </table>
    </section>
    <section>
      <h2>Comparison Packet</h2>
      <pre>{_html_escape(json.dumps(comparison, indent=2, ensure_ascii=False))}</pre>
    </section>
  </main>
  <script type="application/json" id="zspace-comparison">{_html_escape(comparison_json)}</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return str(out_path)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_desire_adapter_from_downstream_hook(
    hook_or_manifest: Mapping[str, Any] | str | Path,
    *,
    base_gain: float = 1.0,
    min_gain: float = 0.45,
    max_gain: float = 1.8,
    stability_weight: float = 0.45,
    momentum_weight: float = 0.2,
    delta_weight: float = 0.35,
    phase_bias: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    hook = _coerce_downstream_hook(hook_or_manifest)
    candidate = hook.get("desire_candidate")
    candidate_payload = dict(candidate) if isinstance(candidate, Mapping) else {}
    phase_bias_payload = {
        "observation": -0.18,
        "injection": 0.0,
        "integration": 0.16,
    }
    if isinstance(phase_bias, Mapping):
        for key, value in phase_bias.items():
            try:
                phase_bias_payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    stability = _clamp(float(candidate_payload.get("stability_signal", 0.0) or 0.0), 0.0, 1.0)
    momentum_raw = float(candidate_payload.get("momentum_signal", 0.0) or 0.0)
    delta_raw = max(0.0, float(candidate_payload.get("delta_signal", 0.0) or 0.0))
    phase_hint = candidate_payload.get("phase_hint")
    phase_hint_str = str(phase_hint) if phase_hint is not None else None
    momentum = math.tanh(momentum_raw)
    delta = math.tanh(delta_raw)
    phase_term = phase_bias_payload.get(phase_hint_str or "", 0.0)

    gain_factor = (
        1.0
        + stability_weight * ((stability - 0.5) * 2.0)
        + momentum_weight * momentum
        - delta_weight * delta
        + phase_term
    )
    gain = _clamp(float(base_gain) * gain_factor, float(min_gain), float(max_gain))
    temperature_scale = _clamp(1.0 / gain if gain > 1e-6 else float(max_gain), 0.6, 1.8)

    top_focus = hook.get("top_focus")
    top_focus_items = _coerce_focus_items(top_focus)
    top_metric = top_focus_items[0]["name"] if top_focus_items else candidate_payload.get("focus_metric")

    return {
        "kind": "spiraltorch.desire_adapter",
        "gain": gain,
        "temperature_scale": temperature_scale,
        "phase_hint": phase_hint_str,
        "stability_signal": stability,
        "momentum_signal": momentum_raw,
        "delta_signal": delta_raw,
        "focus_metric": top_metric,
        "geometry_bias_signal": [
            stability,
            momentum,
            delta,
            gain / float(max_gain) if float(max_gain) > 1e-6 else gain,
        ],
        "guidance": candidate_payload.get("guidance") or hook.get("summary", {}).get("guidance"),
    }


def desire_step_from_downstream_hook(
    pipeline: Any,
    logits: Sequence[float],
    previous_token: int,
    hook_or_manifest: Mapping[str, Any] | str | Path,
    *,
    concept: Sequence[float] | None = None,
    window: Sequence[tuple[int, float]] | None = None,
    base_gain: float = 1.0,
    min_gain: float = 0.45,
    max_gain: float = 1.8,
    stability_weight: float = 0.45,
    momentum_weight: float = 0.2,
    delta_weight: float = 0.35,
    phase_bias: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    adapter = build_desire_adapter_from_downstream_hook(
        hook_or_manifest,
        base_gain=base_gain,
        min_gain=min_gain,
        max_gain=max_gain,
        stability_weight=stability_weight,
        momentum_weight=momentum_weight,
        delta_weight=delta_weight,
        phase_bias=phase_bias,
    )
    ingest_geometry_bias = getattr(pipeline, "ingest_geometry_bias", None)
    geometry_bias_ingested = False
    if callable(ingest_geometry_bias):
        ingest_geometry_bias(adapter["geometry_bias_signal"], source="zspace")
        geometry_bias_ingested = True
    scaled_logits = [float(value) * float(adapter["gain"]) for value in logits]
    result = pipeline.step(
        scaled_logits,
        int(previous_token),
        concept=None if concept is None else list(concept),
        window=None if window is None else list(window),
    )
    payload = dict(result) if isinstance(result, Mapping) else {"result": result}
    payload["downstream_adapter"] = adapter
    payload["scaled_logits_gain"] = adapter["gain"]
    payload["geometry_bias_ingested"] = geometry_bias_ingested
    geometry_bias_metrics = getattr(pipeline, "geometry_bias_metrics", None)
    if callable(geometry_bias_metrics) and "geometry_bias_metrics" not in payload:
        payload["geometry_bias_metrics"] = geometry_bias_metrics()
    geometry_bias_coherence = getattr(pipeline, "geometry_bias_coherence", None)
    if callable(geometry_bias_coherence) and "geometry_bias_coherence" not in payload:
        payload["geometry_bias_coherence"] = geometry_bias_coherence()
    return payload


def _coerce_modes(modes: Sequence[str] | None) -> list[str]:
    if modes is None:
        return ["inference", "training"]
    out = [str(mode) for mode in modes if str(mode)]
    return out or ["inference"]


def _is_finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _non_negative_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) >= 0.0


def _validate_step_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    adapter = payload.get("downstream_adapter")
    adapter_payload = dict(adapter) if isinstance(adapter, Mapping) else {}
    signal = adapter_payload.get("geometry_bias_signal")
    signal_values = list(signal) if isinstance(signal, Sequence) and not isinstance(signal, (str, bytes)) else []

    checks = {
        "adapter_kind": adapter_payload.get("kind") == "spiraltorch.desire_adapter",
        "gain_finite": _is_finite_number(adapter_payload.get("gain")),
        "temperature_scale_finite": _is_finite_number(adapter_payload.get("temperature_scale")),
        "geometry_bias_signal_finite": bool(signal_values)
        and all(_is_finite_number(value) for value in signal_values),
        "geometry_bias_ingested": payload.get("geometry_bias_ingested") is True,
    }

    if "geometry_bias_metrics" in payload:
        metrics = payload.get("geometry_bias_metrics")
        metrics_payload = dict(metrics) if isinstance(metrics, Mapping) else {}
        checks["geometry_bias_metrics_available"] = bool(metrics_payload)
        checks["geometry_bias_metrics_non_negative"] = (
            _non_negative_number(metrics_payload.get("accuracy_mean"))
            and _non_negative_number(metrics_payload.get("fairness_mean"))
        )

    if "geometry_bias_coherence" in payload:
        coherence = payload.get("geometry_bias_coherence")
        coherence_payload = dict(coherence) if isinstance(coherence, Mapping) else {}
        checks["geometry_bias_coherence_available"] = bool(coherence_payload)
        checks["geometry_bias_coherence_non_negative"] = (
            _non_negative_number(coherence_payload.get("composite_energy"))
            and _non_negative_number(coherence_payload.get("z_energy"))
        )

    return checks


def run_desire_geometry_bias_validation(
    pipeline_factory: Any,
    logits: Sequence[float],
    previous_token: int,
    hook_or_manifest: Mapping[str, Any] | str | Path,
    *,
    concept: Sequence[float] | None = None,
    window: Sequence[tuple[int, float]] | None = None,
    modes: Sequence[str] = ("inference", "training"),
    base_gain: float = 1.0,
    min_gain: float = 0.45,
    max_gain: float = 1.8,
    stability_weight: float = 0.45,
    momentum_weight: float = 0.2,
    delta_weight: float = 0.35,
    phase_bias: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not callable(pipeline_factory):
        raise TypeError("pipeline_factory must be callable")

    mode_list = _coerce_modes(modes)
    adapter = build_desire_adapter_from_downstream_hook(
        hook_or_manifest,
        base_gain=base_gain,
        min_gain=min_gain,
        max_gain=max_gain,
        stability_weight=stability_weight,
        momentum_weight=momentum_weight,
        delta_weight=delta_weight,
        phase_bias=phase_bias,
    )

    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for mode in mode_list:
        pipeline = pipeline_factory()
        set_bias_context = getattr(pipeline, "set_bias_context", None)
        bias_context_applied = False
        if callable(set_bias_context):
            set_bias_context(mode)
            bias_context_applied = True

        try:
            payload = desire_step_from_downstream_hook(
                pipeline,
                logits,
                previous_token,
                hook_or_manifest,
                concept=concept,
                window=window,
                base_gain=base_gain,
                min_gain=min_gain,
                max_gain=max_gain,
                stability_weight=stability_weight,
                momentum_weight=momentum_weight,
                delta_weight=delta_weight,
                phase_bias=phase_bias,
            )
        except Exception as exc:  # noqa: BLE001 - return a structured validation report.
            failure = {
                "mode": mode,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            failures.append(failure)
            results[mode] = {
                "ok": False,
                "passed": False,
                "bias_context_applied": bias_context_applied,
                "error": failure["error"],
            }
            continue

        checks = _validate_step_payload(payload)
        ok = all(checks.values())
        if not ok:
            failures.append(
                {
                    "mode": mode,
                    "failed_checks": [
                        name for name, passed in sorted(checks.items()) if not passed
                    ],
                }
            )
        results[mode] = {
            "ok": ok,
            "passed": ok,
            "bias_context_applied": bias_context_applied,
            "checks": checks,
            "result": payload,
        }

    ok = not failures
    return {
        "kind": "spiraltorch.desire_geometry_bias_validation",
        "ok": ok,
        "passed": ok,
        "modes": mode_list,
        "adapter": adapter,
        "results": results,
        "failures": failures,
        "summary": {
            "total": len(mode_list),
            "passed": sum(1 for result in results.values() if result.get("ok") is True),
            "failed": len(failures),
        },
    }
