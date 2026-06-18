from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def _write_trace(path: Path, *, scale: float = 1.0) -> None:
    payload = {
        "run": {"backend": "cpu", "nodes": 4, "features": 2},
        "trainer": {"batches": 1, "average_loss": 0.125 * scale},
        "signal": {
            "above": 0.8,
            "here": 0.4,
            "beneath": 0.2,
            "drift": 0.1,
        },
        "readout": {
            "prediction_shape": [2, 2],
            "trace": {
                "readout": "mean",
                "graph_count": 2,
                "total_rows": 8,
                "output_shape": [2, 2],
                "entries": [
                    {
                        "graph_index": 0,
                        "row_start": 0,
                        "row_end": 4,
                        "node_l2": 0.25 * scale,
                        "prediction_l2": 0.1 * scale,
                    },
                    {
                        "graph_index": 1,
                        "row_start": 4,
                        "row_end": 8,
                        "node_l2": 0.5 * scale,
                        "prediction_l2": 0.2 * scale,
                    },
                ],
            },
            "error": {
                "graph_count": 2,
                "output_shape": [2, 2],
                "mean_squared_error": 0.05 * scale,
                "entries": [
                    {
                        "graph_index": 0,
                        "prediction_l2": 0.1 * scale,
                        "target_l2": 0.3 * scale,
                        "residual_l2": 0.2 * scale,
                        "mean_squared_error": 0.04 * scale,
                    },
                    {
                        "graph_index": 1,
                        "prediction_l2": 0.2 * scale,
                        "target_l2": 0.4 * scale,
                        "residual_l2": 0.3 * scale,
                        "mean_squared_error": 0.06 * scale,
                    },
                ],
            },
        },
        "validation_readout": {
            "batch_count": 2,
            "graph_count": 4,
            "total_rows": 16,
            "mean_squared_error": 0.075 * scale,
            "batches": [
                {
                    "batch_index": 0,
                    "trace": {"graph_count": 2, "total_rows": 8},
                    "error": {"graph_count": 2, "mean_squared_error": 0.05 * scale},
                },
                {
                    "batch_index": 1,
                    "trace": {"graph_count": 2, "total_rows": 8},
                    "error": {"graph_count": 2, "mean_squared_error": 0.10 * scale},
                },
            ],
        },
        "band_replays": {
            "above": [
                {
                    "layer": "trainer_band_trace::layer0",
                    "gradient_l1": 0.6 * scale,
                    "gradient_l2": 0.3 * scale,
                    "gradient_rms": 0.15 * scale,
                    "base_coefficients": [1.0, 0.5, 0.25],
                    "step_scales": [0.9, 1.44, 0.8464],
                    "band_pass_scales": [1.0, 1.2, 0.92],
                    "effective_coefficients": [0.8, 0.72, 0.23],
                    "total_flow_energy": 1.5,
                },
                {
                    "layer": "trainer_band_trace::layer1",
                    "gradient_l1": 0.4 * scale,
                    "gradient_l2": 0.2 * scale,
                    "gradient_rms": 0.1 * scale,
                    "base_coefficients": [1.0, 0.5],
                    "step_scales": [0.88, 1.276],
                    "band_pass_scales": [1.0, 1.16],
                    "effective_coefficients": [0.7, 0.58],
                    "total_flow_energy": 1.1,
                },
            ],
            "here": [
                {
                    "layer": "trainer_band_trace::layer0",
                    "gradient_l1": 0.2 * scale,
                    "gradient_l2": 0.1 * scale,
                    "gradient_rms": 0.05 * scale,
                    "base_coefficients": [1.0, 0.5, 0.25],
                    "step_scales": [1.08, 0.846, 1.0],
                    "band_pass_scales": [1.08, 0.94, 1.0],
                    "effective_coefficients": [0.86, 0.47, 0.25],
                    "total_flow_energy": 0.9,
                }
            ],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_summarize_gnn_band_replays_collects_band_coefficients(tmp_path) -> None:
    _ensure_torch_stub()
    import spiraltorch as st

    trace_path = tmp_path / "gnn_band_trace.json"
    _write_trace(trace_path)

    payload = st.load_gnn_band_replay_trace(trace_path)
    summary = st.summarize_gnn_band_replays(payload)

    assert summary["count"] == 3
    assert summary["run"]["backend"] == "cpu"
    assert summary["trainer"]["average_loss"] == pytest.approx(0.125)
    assert summary["readout"]["trace"]["graph_count"] == 2
    assert summary["readout"]["trace"]["entries"][0]["row_start"] == 0
    assert summary["readout"]["trace"]["entries"][1]["row_end"] == 8
    assert summary["readout"]["error"]["mean_squared_error"] == pytest.approx(0.05)
    assert summary["readout"]["error"]["entries"][1]["mean_squared_error"] == pytest.approx(
        0.06
    )
    assert summary["validation_readout"]["graph_count"] == 4
    assert summary["validation_readout"]["total_rows"] == 16
    assert summary["validation_readout"]["mean_squared_error"] == pytest.approx(0.075)
    above = summary["bands"]["above"]
    assert above["count"] == 2
    assert above["layers"] == [
        "trainer_band_trace::layer0",
        "trainer_band_trace::layer1",
    ]
    assert above["gradient_rms"]["mean"] == pytest.approx(0.125)
    assert above["band_pass_scales"]["mean_by_index"] == pytest.approx(
        [1.0, 1.18, 0.92]
    )
    assert above["band_pass_scales"]["max_abs_delta"] == pytest.approx(0.2)
    assert above["effective_coefficients"]["mean_by_index"] == pytest.approx(
        [0.75, 0.65, 0.23]
    )


def test_flatten_gnn_band_replay_rows_expands_hops(tmp_path) -> None:
    _ensure_torch_stub()
    import spiraltorch as st

    trace_path = tmp_path / "gnn_band_trace.json"
    _write_trace(trace_path)

    rows = st.flatten_gnn_band_replay_rows(trace_path)

    assert len(rows) == 8
    first = rows[0]
    assert first["band"] == "above"
    assert first["layer"] == "trainer_band_trace::layer0"
    assert first["hop_index"] == 0
    assert first["band_pass_scale"] == pytest.approx(1.0)
    assert first["scale_delta"] == pytest.approx(0.0)
    assert first["effective_coefficient"] == pytest.approx(0.8)
    assert rows[1]["roundtable_step_scale"] == pytest.approx(
        rows[1]["step_scale"] / rows[1]["band_pass_scale"]
    )


def test_compare_gnn_band_replay_runs_tracks_per_band_drift(tmp_path) -> None:
    _ensure_torch_stub()
    import spiraltorch as st

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_trace(first, scale=1.0)
    _write_trace(second, scale=2.0)

    comparison = st.compare_gnn_band_replay_runs([first, second])

    assert comparison["count"] == 2
    assert comparison["bands"]["above"]["gradient_rms_mean"]["mean"] == pytest.approx(
        0.1875
    )
    assert comparison["bands"]["above"]["max_abs_scale_delta"]["max"] == pytest.approx(
        0.2
    )
    assert comparison["bands"]["here"]["scale_delta_mean"]["count"] == pytest.approx(2.0)


def test_write_gnn_band_replay_html_renders_dashboard(tmp_path) -> None:
    _ensure_torch_stub()
    import spiraltorch as st

    trace_path = tmp_path / "gnn_band_trace.json"
    html_path = tmp_path / "gnn_band_trace.html"
    _write_trace(trace_path)

    rendered = Path(st.write_gnn_band_replay_html(trace_path, html_path))

    assert rendered == html_path
    html = rendered.read_text(encoding="utf-8")
    assert "SpiralTorch GNN Band Replay Trace" in html
    assert "trainer_band_trace::layer0" in html
    assert "band_pass_scale" in html
    assert "scale_delta" in html
