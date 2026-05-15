from __future__ import annotations

import json

import pytest
import spiraltorch as st

from spiraltorch.zspace_atlas import (
    write_zspace_atlas_noncollapse_html,
    zspace_trace_event_to_atlas_frame,
    zspace_trace_to_atlas_route,
)
from spiraltorch.zspace_artifacts import (
    build_desire_adapter_from_downstream_hook,
    compare_zspace_experiment_manifests,
    build_zspace_planner_snapshot,
    build_zspace_downstream_hook,
    desire_step_from_downstream_hook,
    load_zspace_artifact_manifest,
    run_desire_geometry_bias_validation,
    summarize_zspace_experiment_index,
    summarize_zspace_experiment_manifest,
    write_zspace_experiment_cockpit_html,
    write_zspace_experiment_comparison_html,
    write_zspace_experiment_index_html,
    write_zspace_experiment_artifacts,
)
from spiraltorch.zspace_trace import load_zspace_trace_events, write_zspace_trace_html


def _require_native_nn() -> None:
    if not hasattr(st, "nn") or not hasattr(st.nn, "DesirePipeline"):
        pytest.skip("native SpiralTorch nn bindings unavailable")


def test_load_zspace_trace_events_flattens_stable_records(tmp_path) -> None:
    trace_path = tmp_path / "zspace_trace.jsonl"
    record = {
        "schema": "spiraltorch.zspace_trace",
        "schema_version": 1,
        "kind": "Aggregated",
        "step": 7,
        "noncollapse": {
            "coherence_entropy": 0.42,
            "preserved_channels": 4,
            "discarded_channels": 12,
        },
        "payload": {
            "aggregated_shape": [1, 64],
            "coherence": [0.7, 0.2, 0.1],
            "noncollapse_card": {
                "stage": "aggregated",
                "title": "Aggregated non-collapse",
                "summary": "4 preserved / 12 discarded · label cascade_imbalance",
                "metrics": {
                    "coherence_entropy": 0.42,
                    "preserved_channels": 4,
                    "discarded_channels": 12,
                },
            },
            "diagnostics": {
                "label": "cascade_imbalance",
                "energy_ratio": 0.7,
            },
        },
    }
    trace_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    events = load_zspace_trace_events(trace_path)

    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "Aggregated"
    assert event["step"] == 7
    assert event["schema"] == "spiraltorch.zspace_trace"
    assert event["schema_version"] == 1
    assert event["aggregated_shape"] == [1, 64]
    assert event["diagnostics"]["label"] == "cascade_imbalance"
    assert event["noncollapse"]["coherence_entropy"] == 0.42
    assert event["noncollapse_card"]["stage"] == "aggregated"
    assert event["noncollapse_card"]["metrics"]["preserved_channels"] == 4


def test_write_zspace_trace_html_contains_noncollapse_card_container(tmp_path) -> None:
    trace_path = tmp_path / "zspace_trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "PreDiscardApplied",
                "step": 2,
                "payload": {
                    "coherence": [0.9, 0.1],
                    "noncollapse_card": {
                        "stage": "pre_discard",
                        "title": "Pre-discard retention",
                        "summary": "2 preserved / 6 discarded",
                        "metrics": {"preserved_ratio": 0.25, "used_fallback": False},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    html_path = write_zspace_trace_html(
        trace_path,
        tmp_path / "trace.html",
        related_links={
            "Atlas view": "trace.atlas_noncollapse.html",
            "Artifact manifest": "trace.artifacts.json",
        },
    )
    html = (tmp_path / "trace.html").read_text(encoding="utf-8")

    assert html_path.endswith("trace.html")
    assert 'id="nc-card"' in html
    assert "renderNonCollapseCard" in html
    assert "Atlas view" in html
    assert "trace.artifacts.json" in html


@pytest.mark.parametrize(
    ("event", "expected_metric", "expected_value", "expected_stage"),
    [
        (
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "Aggregated",
                "step": 7,
                "noncollapse": {
                    "coherence_entropy": 0.42,
                    "preserved_channels": 4,
                    "discarded_channels": 12,
                },
                "payload": {
                    "noncollapse_card": {
                        "stage": "aggregated",
                        "title": "Aggregated non-collapse",
                        "summary": "4 preserved / 12 discarded · label cascade_imbalance",
                        "metrics": {
                            "coherence_entropy": 0.42,
                            "preserved_channels": 4,
                            "discarded_channels": 12,
                        },
                    }
                },
            },
            "noncollapse.coherence_entropy",
            0.42,
            "aggregated",
        ),
        (
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "LanguageBridged",
                "step": 8,
                "noncollapse": {
                    "z_bias": 0.31,
                    "band_energy": [0.2, 0.5, 0.3],
                },
                "payload": {
                    "noncollapse_card": {
                        "stage": "language_bridged",
                        "title": "Language pulse non-collapse",
                        "summary": "z_bias +0.310 · bands 0.200/0.500/0.300",
                        "metrics": {
                            "z_bias": 0.31,
                            "band_energy_above": 0.2,
                            "band_energy_here": 0.5,
                            "band_energy_beneath": 0.3,
                        },
                    }
                },
            },
            "noncollapse.band_energy_here",
            0.5,
            "language_bridged",
        ),
        (
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "PreDiscardApplied",
                "step": 9,
                "noncollapse": {
                    "preserved_channels": 2,
                    "discarded_channels": 6,
                    "pre_discard_preserved_ratio": 0.25,
                    "pre_discard_survivor_energy_ratio": 0.81,
                },
                "payload": {
                    "noncollapse_card": {
                        "stage": "pre_discard",
                        "title": "Pre-discard retention",
                        "summary": "2 preserved / 6 discarded",
                        "metrics": {
                            "preserved_channels": 2,
                            "discarded_channels": 6,
                            "preserved_ratio": 0.25,
                            "survivor_energy_ratio": 0.81,
                            "used_fallback": False,
                        },
                    }
                },
            },
            "noncollapse.preserved_ratio",
            0.25,
            "pre_discard",
        ),
    ],
)
def test_zspace_trace_event_to_atlas_frame_surfaces_noncollapse_metrics(
    event,
    expected_metric,
    expected_value,
    expected_stage,
) -> None:
    frame = zspace_trace_event_to_atlas_frame(event)

    assert frame is not None
    assert frame.metric_value("noncollapse.present") == pytest.approx(1.0)
    assert frame.metric_value(expected_metric) == pytest.approx(expected_value)
    assert frame.metric_value(f"noncollapse.stage.{expected_stage}") == pytest.approx(1.0)
    assert f"zspace.trace.noncollapse.stage={expected_stage}" in frame.notes()


def test_zspace_trace_to_atlas_route_exposes_noncollapse_focus_metrics() -> None:
    route = zspace_trace_to_atlas_route(
        [
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "Aggregated",
                "step": 1,
                "payload": {
                    "noncollapse_card": {
                        "stage": "aggregated",
                        "title": "Aggregated non-collapse",
                        "summary": "4 preserved / 12 discarded",
                        "metrics": {
                            "coherence_entropy": 0.42,
                            "preserved_channels": 4,
                        },
                    }
                },
            },
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "LanguageBridged",
                "step": 2,
                "payload": {
                    "noncollapse_card": {
                        "stage": "language_bridged",
                        "title": "Language pulse non-collapse",
                        "summary": "z_bias +0.310",
                        "metrics": {
                            "band_energy_here": 0.5,
                            "z_bias": 0.31,
                        },
                    }
                },
            },
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "PreDiscardApplied",
                "step": 3,
                "payload": {
                    "noncollapse_card": {
                        "stage": "pre_discard",
                        "title": "Pre-discard retention",
                        "summary": "2 preserved / 6 discarded",
                        "metrics": {
                            "preserved_ratio": 0.25,
                            "survivor_energy_ratio": 0.81,
                        },
                    }
                },
            },
        ]
    )

    perspective = route.perspective_for("Concourse", ["noncollapse."])

    assert perspective is not None
    focus_names = {item["name"] for item in perspective["focus"]}
    assert focus_names
    assert all(name.startswith("noncollapse.") for name in focus_names)
    assert "noncollapse.present" in focus_names
    assert "noncollapse.band_energy_here" in focus_names
    assert "noncollapse.stage.pre_discard" in focus_names
    assert any(
        name in focus_names
        for name in ("noncollapse.preserved_channels", "noncollapse.preserved_ratio")
    )


def test_write_zspace_atlas_noncollapse_html_renders_focus_view(tmp_path) -> None:
    route = zspace_trace_to_atlas_route(
        [
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "Aggregated",
                "step": 1,
                "payload": {
                    "noncollapse_card": {
                        "stage": "aggregated",
                        "title": "Aggregated non-collapse",
                        "summary": "4 preserved / 12 discarded",
                        "metrics": {
                            "coherence_entropy": 0.42,
                            "preserved_channels": 4,
                        },
                    }
                },
            },
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "LanguageBridged",
                "step": 2,
                "payload": {
                    "noncollapse_card": {
                        "stage": "language_bridged",
                        "title": "Language pulse non-collapse",
                        "summary": "z_bias +0.310",
                        "metrics": {
                            "band_energy_here": 0.5,
                            "z_bias": 0.31,
                        },
                    }
                },
            },
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "PreDiscardApplied",
                "step": 3,
                "payload": {
                    "noncollapse_card": {
                        "stage": "pre_discard",
                        "title": "Pre-discard retention",
                        "summary": "2 preserved / 6 discarded",
                        "metrics": {
                            "preserved_ratio": 0.25,
                            "survivor_energy_ratio": 0.81,
                        },
                    }
                },
            },
        ]
    )

    html_path = write_zspace_atlas_noncollapse_html(
        route,
        tmp_path / "atlas_noncollapse.html",
        related_links={
            "Trace viewer": "trace.html",
            "Artifact manifest": "trace.artifacts.json",
        },
    )
    html = (tmp_path / "atlas_noncollapse.html").read_text(encoding="utf-8")

    assert html_path.endswith("atlas_noncollapse.html")
    assert 'perspective_for("Concourse", ["noncollapse."])' in html
    assert "Stage Comparison" in html
    assert "Metric Comparison" in html
    assert "noncollapse.band_energy_here" in html
    assert "pre_discard" in html
    assert "Trace viewer" in html
    assert "trace.artifacts.json" in html


def test_build_zspace_downstream_hook_extracts_desire_candidate(tmp_path) -> None:
    manifest_path = tmp_path / "trace.artifacts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trace_jsonl": "/tmp/trace.jsonl",
                "trace_html": "/tmp/trace.html",
                "atlas_noncollapse_html": "/tmp/trace.atlas_noncollapse.html",
                "artifact_manifest": str(manifest_path),
                "summary": {"frames": 3, "total_notes": 5},
                "noncollapse_perspective": {
                    "coverage": 3,
                    "mean": 0.4,
                    "latest": 0.7,
                    "delta": 0.2,
                    "momentum": 0.15,
                    "volatility": 0.05,
                    "stability": 0.92,
                    "guidance": "keep preserved channels alive",
                    "focus": [
                        {
                            "name": "noncollapse.stage.pre_discard",
                            "coverage": 3,
                            "mean": 0.3,
                            "latest": 1.0,
                            "delta": 0.0,
                            "momentum": 0.0,
                            "std_dev": 0.0,
                        },
                        {
                            "name": "noncollapse.preserved_ratio",
                            "coverage": 3,
                            "mean": 0.25,
                            "latest": 0.25,
                            "delta": 0.05,
                            "momentum": 0.02,
                            "std_dev": 0.01,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_zspace_artifact_manifest(manifest_path)
    hook = build_zspace_downstream_hook(manifest_path)

    assert manifest["artifact_manifest"] == str(manifest_path)
    assert hook["kind"] == "spiraltorch.zspace_artifact_hook"
    assert hook["views"]["trace_html"] == "/tmp/trace.html"
    assert hook["signals"]["stability"] == pytest.approx(0.92)
    assert hook["stage_focus"][0]["stage"] == "pre_discard"
    assert hook["top_focus"][0]["name"] == "noncollapse.preserved_ratio"
    assert hook["desire_candidate"]["experimental"] is True
    assert hook["desire_candidate"]["phase_hint"] == "pre_discard"
    assert hook["desire_candidate"]["focus_metric"] == "noncollapse.preserved_ratio"


def test_build_zspace_planner_snapshot_captures_device_and_plan() -> None:
    class FakePlan:
        kind = "topk"
        requested_backend = "auto"
        effective_backend = "wgpu"
        rows = 8
        cols = 128
        k = 4
        workgroup = 128
        lanes = 32

    def describe_device(backend: str):
        assert backend == "auto"
        return {
            "backend": "wgpu",
            "lane_width": 32,
            "planner_route": "wgpu",
        }

    def plan_topk(rows: int, cols: int, k: int, *, backend: str):
        assert (rows, cols, k, backend) == (8, 128, 4, "auto")
        return FakePlan()

    snapshot = build_zspace_planner_snapshot(
        backend="auto",
        rows=8,
        cols=128,
        k=4,
        describe_device=describe_device,
        plan_topk=plan_topk,
    )

    assert snapshot["kind"] == "spiraltorch.zspace_planner_snapshot"
    assert snapshot["available"] is True
    assert snapshot["backend"] == "auto"
    assert snapshot["shape"] == {"rows": 8, "cols": 128, "k": 4}
    assert snapshot["device_report"]["backend"] == "wgpu"
    assert snapshot["rank_plan"]["kind"] == "topk"
    assert snapshot["rank_plan"]["effective_backend"] == "wgpu"
    assert snapshot["rank_plan"]["workgroup"] == 128
    assert "errors" not in snapshot


def test_write_zspace_experiment_artifacts_writes_manifest_with_planner_snapshot(
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema": "spiraltorch.zspace_trace",
                "schema_version": 1,
                "kind": "PreDiscardApplied",
                "step": 3,
                "payload": {
                    "noncollapse_card": {
                        "stage": "pre_discard",
                        "title": "Pre-discard retention",
                        "summary": "2 preserved / 6 discarded",
                        "metrics": {
                            "preserved_ratio": 0.25,
                            "survivor_energy_ratio": 0.81,
                        },
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def describe_device(_backend: str):
        return {"backend": "wgpu", "lane_width": 32}

    def plan_topk(rows: int, cols: int, k: int, *, backend: str):
        return {
            "kind": "topk",
            "requested_backend": backend,
            "effective_backend": "wgpu",
            "rows": rows,
            "cols": cols,
            "k": k,
        }

    manifest = write_zspace_experiment_artifacts(
        trace_path,
        trace_html=tmp_path / "trace.html",
        atlas_html=tmp_path / "trace.atlas_noncollapse.html",
        manifest=tmp_path / "trace.artifacts.json",
        title="Experiment packet",
        planner_backend="auto",
        planner_rows=4,
        planner_cols=64,
        planner_k=2,
        metadata={"run_id": "demo-run"},
        describe_device=describe_device,
        plan_topk=plan_topk,
    )
    loaded = load_zspace_artifact_manifest(tmp_path / "trace.artifacts.json")

    assert (tmp_path / "trace.html").is_file()
    assert (tmp_path / "trace.atlas_noncollapse.html").is_file()
    assert manifest["kind"] == "spiraltorch.zspace_experiment_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["metadata"]["run_id"] == "demo-run"
    assert manifest["views"]["trace_jsonl"] == str(trace_path)
    assert manifest["planner_snapshot"]["available"] is True
    assert manifest["planner_snapshot"]["rank_plan"]["cols"] == 64
    assert manifest["downstream_hook"]["kind"] == "spiraltorch.zspace_artifact_hook"
    assert loaded["planner_snapshot"]["device_report"]["backend"] == "wgpu"
    assert loaded["downstream_hook"]["views"]["artifact_manifest"] == str(
        tmp_path / "trace.artifacts.json"
    )


def test_summarize_zspace_experiment_manifest_builds_story() -> None:
    manifest = {
        "kind": "spiraltorch.zspace_experiment_manifest",
        "title": "Planner-backed trace",
        "created_at": "2026-05-05T00:00:00+00:00",
        "views": {
            "trace_jsonl": "/tmp/trace.jsonl",
            "trace_html": "/tmp/trace.html",
            "atlas_noncollapse_html": "/tmp/trace.atlas_noncollapse.html",
            "artifact_manifest": "/tmp/trace.artifacts.json",
        },
        "summary": {"frames": 3, "total_notes": 7},
        "planner_snapshot": {
            "kind": "spiraltorch.zspace_planner_snapshot",
            "backend": "auto",
            "available": True,
            "shape": {"rows": 4, "cols": 64, "k": 2},
            "device_report": {
                "backend": "wgpu",
                "planner_route": "metal-via-wgpu",
                "lane_width": 32,
            },
            "rank_plan": {
                "kind": "topk",
                "requested_backend": "auto",
                "effective_backend": "wgpu",
                "workgroup": 128,
            },
        },
        "noncollapse_perspective": {
            "coverage": 3,
            "mean": 0.61,
            "latest": 0.88,
            "delta": 0.06,
            "momentum": 0.24,
            "volatility": 0.03,
            "stability": 0.94,
            "guidance": "lean into integration when stability is high",
            "focus": [
                {"name": "noncollapse.stage.integration", "latest": 1.0, "coverage": 3},
                {"name": "noncollapse.z_bias", "latest": 0.38, "coverage": 3},
                {"name": "noncollapse.preserved_ratio", "latest": 0.25, "coverage": 2},
            ],
        },
    }

    story = summarize_zspace_experiment_manifest(manifest, top_k=1)

    assert story["kind"] == "spiraltorch.zspace_experiment_story"
    assert story["title"] == "Planner-backed trace"
    assert story["summary"]["frames"] == 3
    assert story["planner"]["requested_backend"] == "auto"
    assert story["planner"]["effective_backend"] == "wgpu"
    assert story["planner"]["route"] == "metal-via-wgpu"
    assert story["noncollapse"]["signals"]["stability"] == pytest.approx(0.94)
    assert story["noncollapse"]["phase_hint"] == "integration"
    assert story["noncollapse"]["focus_metric"] == "noncollapse.z_bias"
    assert len(story["noncollapse"]["top_focus"]) == 1
    assert story["noncollapse"]["top_focus"][0]["name"] == "noncollapse.z_bias"
    assert any(card["kind"] == "guidance" for card in story["story"])


def test_write_zspace_experiment_cockpit_html_renders_story_and_links(tmp_path) -> None:
    manifest_path = tmp_path / "trace.artifacts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "spiraltorch.zspace_experiment_manifest",
                "title": "Cockpit packet",
                "trace_jsonl": "trace.jsonl",
                "trace_html": "trace.html",
                "atlas_noncollapse_html": "trace.atlas_noncollapse.html",
                "artifact_manifest": str(manifest_path),
                "summary": {"frames": 2, "total_notes": 5},
                "planner_snapshot": {
                    "backend": "auto",
                    "available": True,
                    "shape": {"rows": 4, "cols": 64, "k": 2},
                    "device_report": {"backend": "wgpu", "planner_route": "wgpu"},
                    "rank_plan": {"effective_backend": "wgpu", "workgroup": 128},
                },
                "noncollapse_perspective": {
                    "coverage": 2,
                    "stability": 0.91,
                    "momentum": 0.12,
                    "delta": 0.03,
                    "guidance": "keep the trace readable",
                    "focus": [
                        {"name": "noncollapse.stage.pre_discard", "latest": 1.0},
                        {"name": "noncollapse.preserved_ratio", "latest": 0.25},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    html_path = write_zspace_experiment_cockpit_html(
        manifest_path,
        tmp_path / "trace.cockpit.html",
    )
    html = (tmp_path / "trace.cockpit.html").read_text(encoding="utf-8")

    assert html_path.endswith("trace.cockpit.html")
    assert "Cockpit packet" in html
    assert "Planner Snapshot" in html
    assert "Top Focus" in html
    assert "noncollapse.preserved_ratio" in html
    assert "Trace Viewer" in html
    assert "trace.html" in html
    assert 'id="zspace-story"' in html


def _experiment_index_manifest(
    title: str,
    *,
    backend: str,
    route: str,
    frames: int,
    total_notes: int,
    stability: float,
    momentum: float,
    focus_metric: str,
    artifact_manifest: str = "/tmp/trace.artifacts.json",
) -> dict[str, object]:
    return {
        "kind": "spiraltorch.zspace_experiment_manifest",
        "title": title,
        "created_at": "2026-05-05T00:00:00+00:00",
        "trace_jsonl": f"{title}.jsonl",
        "trace_html": f"{title}.html",
        "atlas_noncollapse_html": f"{title}.atlas_noncollapse.html",
        "artifact_manifest": artifact_manifest,
        "summary": {"frames": frames, "total_notes": total_notes},
        "planner_snapshot": {
            "backend": "auto",
            "available": True,
            "shape": {"rows": 4, "cols": 64, "k": 2},
            "device_report": {"backend": backend, "planner_route": route},
            "rank_plan": {"effective_backend": backend, "workgroup": 128},
        },
        "noncollapse_perspective": {
            "coverage": frames,
            "stability": stability,
            "momentum": momentum,
            "delta": 0.03,
            "guidance": "compare this run against the index",
            "focus": [
                {"name": "noncollapse.stage.integration", "latest": 1.0},
                {"name": focus_metric, "latest": 0.5, "coverage": frames},
            ],
        },
    }


def test_summarize_zspace_experiment_index_compares_runs() -> None:
    index = summarize_zspace_experiment_index(
        [
            _experiment_index_manifest(
                "WGPU run",
                backend="wgpu",
                route="metal-via-wgpu",
                frames=3,
                total_notes=7,
                stability=0.9,
                momentum=0.2,
                focus_metric="noncollapse.z_bias",
            ),
            _experiment_index_manifest(
                "CPU run",
                backend="cpu",
                route="cpu",
                frames=2,
                total_notes=5,
                stability=0.7,
                momentum=-0.1,
                focus_metric="noncollapse.preserved_ratio",
            ),
        ],
        title="Experiment Index",
    )

    assert index["kind"] == "spiraltorch.zspace_experiment_index"
    assert index["title"] == "Experiment Index"
    assert index["summary"]["runs"] == 2
    assert index["summary"]["total_frames"] == 5
    assert index["summary"]["total_notes"] == 12
    assert index["summary"]["planner_backends"] == {"cpu": 1, "wgpu": 1}
    assert index["summary"]["focus_metrics"] == {
        "noncollapse.preserved_ratio": 1,
        "noncollapse.z_bias": 1,
    }
    assert index["summary"]["mean_stability"] == pytest.approx(0.8)
    assert index["summary"]["latest_stability"] == pytest.approx(0.7)
    assert [run["title"] for run in index["runs"]] == ["WGPU run", "CPU run"]
    assert index["runs"][0]["planner"]["route"] == "metal-via-wgpu"


def test_write_zspace_experiment_index_html_renders_runs_and_links(tmp_path) -> None:
    manifest_a = tmp_path / "wgpu.artifacts.json"
    manifest_b = tmp_path / "cpu.artifacts.json"
    manifest_a.write_text(
        json.dumps(
            _experiment_index_manifest(
                "WGPU run",
                backend="wgpu",
                route="metal-via-wgpu",
                frames=3,
                total_notes=7,
                stability=0.9,
                momentum=0.2,
                focus_metric="noncollapse.z_bias",
                artifact_manifest=str(manifest_a),
            )
        ),
        encoding="utf-8",
    )
    manifest_b.write_text(
        json.dumps(
            _experiment_index_manifest(
                "CPU run",
                backend="cpu",
                route="cpu",
                frames=2,
                total_notes=5,
                stability=0.7,
                momentum=-0.1,
                focus_metric="noncollapse.preserved_ratio",
                artifact_manifest=str(manifest_b),
            )
        ),
        encoding="utf-8",
    )

    html_path = write_zspace_experiment_index_html(
        [manifest_a, manifest_b],
        tmp_path / "experiments.index.html",
        title="Experiment Index",
    )
    html = (tmp_path / "experiments.index.html").read_text(encoding="utf-8")

    assert html_path.endswith("experiments.index.html")
    assert "Experiment Index" in html
    assert "WGPU run" in html
    assert "CPU run" in html
    assert "Trace Viewer" in html
    assert "metal-via-wgpu" in html
    assert "noncollapse.preserved_ratio" in html
    assert 'id="zspace-index"' in html


def test_compare_zspace_experiment_manifests_flags_regression() -> None:
    comparison = compare_zspace_experiment_manifests(
        _experiment_index_manifest(
            "Baseline run",
            backend="wgpu",
            route="metal-via-wgpu",
            frames=10,
            total_notes=20,
            stability=0.95,
            momentum=0.3,
            focus_metric="noncollapse.z_bias",
        ),
        _experiment_index_manifest(
            "Candidate run",
            backend="cpu",
            route="cpu",
            frames=6,
            total_notes=12,
            stability=0.78,
            momentum=0.1,
            focus_metric="noncollapse.preserved_ratio",
        ),
        title="Candidate versus baseline",
    )

    checks = {check["name"]: check for check in comparison["checks"]}

    assert comparison["kind"] == "spiraltorch.zspace_experiment_comparison"
    assert comparison["title"] == "Candidate versus baseline"
    assert comparison["status"] == "fail"
    assert comparison["summary"]["stability_delta"] == pytest.approx(-0.17)
    assert comparison["summary"]["frames_delta"] == -4
    assert comparison["summary"]["planner_changed"] is True
    assert comparison["summary"]["focus_metric_changed"] is True
    assert checks["stability"]["status"] == "fail"
    assert checks["frames"]["status"] == "fail"
    assert checks["planner"]["status"] == "warn"
    assert checks["focus_metric"]["status"] == "warn"


def test_write_zspace_experiment_comparison_html_renders_checks_and_links(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.artifacts.json"
    candidate_path = tmp_path / "candidate.artifacts.json"
    baseline_path.write_text(
        json.dumps(
            _experiment_index_manifest(
                "Baseline run",
                backend="wgpu",
                route="metal-via-wgpu",
                frames=10,
                total_notes=20,
                stability=0.95,
                momentum=0.3,
                focus_metric="noncollapse.z_bias",
                artifact_manifest=str(baseline_path),
            )
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            _experiment_index_manifest(
                "Candidate run",
                backend="cpu",
                route="cpu",
                frames=6,
                total_notes=12,
                stability=0.78,
                momentum=0.1,
                focus_metric="noncollapse.preserved_ratio",
                artifact_manifest=str(candidate_path),
            )
        ),
        encoding="utf-8",
    )

    html_path = write_zspace_experiment_comparison_html(
        baseline_path,
        candidate_path,
        tmp_path / "candidate.comparison.html",
        title="Candidate versus baseline",
    )
    html = (tmp_path / "candidate.comparison.html").read_text(encoding="utf-8")

    assert html_path.endswith("candidate.comparison.html")
    assert "Candidate versus baseline" in html
    assert "Baseline run" in html
    assert "Candidate run" in html
    assert "stability dropped beyond the fail threshold" in html
    assert "Trace Viewer" in html
    assert "noncollapse.preserved_ratio" in html
    assert 'id="zspace-comparison"' in html


def test_build_desire_adapter_from_downstream_hook_maps_gain_and_bias(tmp_path) -> None:
    manifest_path = tmp_path / "trace.artifacts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trace_jsonl": "/tmp/trace.jsonl",
                "summary": {"frames": 4, "total_notes": 8},
                "noncollapse_perspective": {
                    "coverage": 4,
                    "mean": 0.61,
                    "latest": 0.88,
                    "delta": 0.06,
                    "momentum": 0.24,
                    "volatility": 0.03,
                    "stability": 0.94,
                    "guidance": "lean into integration when stability is high",
                    "focus": [
                        {
                            "name": "noncollapse.stage.integration",
                            "latest": 1.0,
                        },
                        {
                            "name": "noncollapse.z_bias",
                            "latest": 0.38,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    adapter = build_desire_adapter_from_downstream_hook(manifest_path)

    assert adapter["kind"] == "spiraltorch.desire_adapter"
    assert adapter["phase_hint"] == "integration"
    assert adapter["gain"] > 1.0
    assert adapter["temperature_scale"] < 1.0
    assert adapter["focus_metric"] == "noncollapse.z_bias"
    assert len(adapter["geometry_bias_signal"]) == 4
    assert adapter["geometry_bias_signal"][0] == pytest.approx(0.94)
    assert adapter["guidance"] == "lean into integration when stability is high"


def test_desire_step_from_downstream_hook_scales_logits_and_returns_adapter() -> None:
    _require_native_nn()
    hook = {
        "kind": "spiraltorch.zspace_artifact_hook",
        "summary": {"guidance": "keep ambiguity alive"},
        "top_focus": [{"name": "noncollapse.preserved_ratio"}],
        "desire_candidate": {
            "stability_signal": 0.87,
            "momentum_signal": 0.18,
            "delta_signal": 0.04,
            "phase_hint": "integration",
            "focus_metric": "noncollapse.preserved_ratio",
            "guidance": "keep ambiguity alive",
        },
    }

    pipeline = st.nn.DesirePipeline(vocab_size=4)
    result = desire_step_from_downstream_hook(
        pipeline,
        [0.1, 0.2, 0.3, 0.4],
        1,
        hook,
    )

    assert result["phase"] in {"observation", "injection", "integration"}
    assert len(result["probabilities"]) == 4
    assert "noncollapse_snapshot" in result
    assert result["downstream_adapter"]["kind"] == "spiraltorch.desire_adapter"
    assert result["downstream_adapter"]["gain"] == pytest.approx(result["scaled_logits_gain"])
    assert result["downstream_adapter"]["phase_hint"] == "integration"
    assert result["scaled_logits_gain"] > 1.0
    assert result["downstream_adapter"]["focus_metric"] == "noncollapse.preserved_ratio"
    assert result["geometry_bias_ingested"] is True
    assert result["geometry_bias_metrics"]["accuracy_mean"] >= 0.0
    assert result["geometry_bias_metrics"]["fairness_mean"] >= 0.0
    assert result["geometry_bias_coherence"]["composite_energy"] >= 0.0
    assert result["geometry_bias_coherence"]["z_energy"] >= 0.0


def test_run_desire_geometry_bias_validation_reports_modes() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.bias_context = "inference"
            self.ingested: list[float] | None = None

        def set_bias_context(self, context: str) -> None:
            self.bias_context = context

        def ingest_geometry_bias(self, signal, source: str = "zspace") -> None:
            assert source == "zspace"
            self.ingested = [float(value) for value in signal]

        def step(self, logits, previous_token, concept=None, window=None):
            assert self.ingested is not None
            return {
                "phase": self.bias_context,
                "probabilities": list(logits),
                "previous_token": previous_token,
            }

        def geometry_bias_metrics(self):
            return {"accuracy_mean": 0.5, "fairness_mean": 0.75, "window": 1}

        def geometry_bias_coherence(self):
            return {"composite_energy": 0.25, "z_energy": 0.125}

    hook = {
        "kind": "spiraltorch.zspace_artifact_hook",
        "top_focus": [{"name": "noncollapse.preserved_ratio"}],
        "desire_candidate": {
            "stability_signal": 0.9,
            "momentum_signal": 0.1,
            "delta_signal": 0.02,
            "phase_hint": "integration",
        },
    }

    report = run_desire_geometry_bias_validation(
        FakePipeline,
        [0.1, 0.2, 0.3],
        1,
        hook,
        modes=("inference", "training"),
    )

    assert report["kind"] == "spiraltorch.desire_geometry_bias_validation"
    assert report["ok"] is True
    assert report["summary"]["passed"] == 2
    assert set(report["results"]) == {"inference", "training"}
    assert report["results"]["training"]["result"]["phase"] == "training"


def test_desire_pipeline_geometry_bias_ingest_surfaces_metrics() -> None:
    _require_native_nn()
    pipeline = st.nn.DesirePipeline(vocab_size=4)

    assert pipeline.bias_context == "inference"
    assert pipeline.geometry_bias_metrics() is None
    assert pipeline.geometry_bias_coherence() is None

    pipeline.ingest_geometry_bias([0.94, 0.18, 0.04, 0.84], source="zspace")
    result = pipeline.step([0.1, 0.2, 0.3, 0.4], 1)

    assert result["geometry_bias_metrics"]["window"] >= 1
    assert result["geometry_bias_metrics"]["latest"]["accuracy"] >= 0.0
    assert result["geometry_bias_coherence"]["composite_energy"] > 0.0
    assert result["geometry_bias_coherence"]["timestamp_ms"] > 0
