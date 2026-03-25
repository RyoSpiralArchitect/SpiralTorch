from __future__ import annotations

import json

import pytest

from spiraltorch.zspace_atlas import (
    write_zspace_atlas_noncollapse_html,
    zspace_trace_event_to_atlas_frame,
    zspace_trace_to_atlas_route,
)
from spiraltorch.zspace_artifacts import (
    build_zspace_downstream_hook,
    load_zspace_artifact_manifest,
)
from spiraltorch.zspace_trace import load_zspace_trace_events, write_zspace_trace_html


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
