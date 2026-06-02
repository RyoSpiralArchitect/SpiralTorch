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
        "trainer": {"batches": 1, "average_loss": 0.125 * scale},
        "signal": {
            "above": 0.8,
            "here": 0.4,
            "beneath": 0.2,
            "drift": 0.1,
        },
        "band_replays": {
            "above": [
                {
                    "layer": "trainer_band_trace::layer0",
                    "gradient_l1": 0.6 * scale,
                    "gradient_l2": 0.3 * scale,
                    "gradient_rms": 0.15 * scale,
                    "band_pass_scales": [1.0, 1.2, 0.92],
                    "effective_coefficients": [0.8, 0.72, 0.23],
                    "total_flow_energy": 1.5,
                },
                {
                    "layer": "trainer_band_trace::layer1",
                    "gradient_l1": 0.4 * scale,
                    "gradient_l2": 0.2 * scale,
                    "gradient_rms": 0.1 * scale,
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
    assert summary["trainer"]["average_loss"] == pytest.approx(0.125)
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
