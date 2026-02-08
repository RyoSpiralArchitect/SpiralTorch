from __future__ import annotations

import importlib
import sys
import types

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def _load_native() -> types.ModuleType | None:
    _ensure_torch_stub()
    try:
        module = importlib.import_module("spiraltorch")
    except Exception:
        return None

    for native_name in (
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    ):
        try:
            importlib.import_module(native_name)
        except Exception:
            continue
        return module
    return None


def test_vision_stream_frame_and_chrono_snapshot_bridge() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

    assert hasattr(st, "vision")
    assert hasattr(st.vision, "ChronoSnapshot")
    assert hasattr(st.vision, "ZSpaceStreamFrame")
    assert hasattr(st.vision, "StreamedVolume")
    assert hasattr(st, "ChronoSnapshot")
    assert hasattr(st, "ZSpaceStreamFrame")
    assert hasattr(st, "StreamedVolume")

    snapshot = st.vision.ChronoSnapshot.from_values(
        frames=3,
        duration=0.3,
        latest_timestamp=1.2,
        mean_drift=0.01,
        mean_abs_drift=0.02,
        drift_std=0.005,
        mean_energy=0.75,
        energy_std=0.1,
        mean_decay=0.03,
        min_energy=0.5,
        max_energy=1.1,
        dt=0.1,
    )
    assert snapshot.dt == pytest.approx(0.1)
    summary = snapshot.summary()
    assert int(summary["frames"]) == 3

    slice_a = st.Tensor.rand(2, 2, seed=300)
    slice_b = st.Tensor.rand(2, 2, seed=301)
    frame = st.vision.ZSpaceStreamFrame(
        [slice_a, slice_b],
        chrono_snapshot=snapshot,
    )

    assert frame.depth == 2
    assert frame.slice_shape == (2, 2)
    assert len(frame.slices()) == 2

    profile = frame.profile()
    assert int(profile["depth"]) == 2
    assert len(profile["means"]) == 2

    report = frame.telemetry_report()
    assert int(report["depth"]) == 2
    assert "total_energy" in report

    atlas = st.telemetry.AtlasFrame.from_metrics({"stream.metric": 1.25}, timestamp=1.0)
    frame.with_atlas(atlas)
    atlas_out = frame.atlas_frame
    assert atlas_out is not None
    assert atlas_out.metric_value("stream.metric") == pytest.approx(1.25)

    snapshot_out = frame.chrono_snapshot
    assert snapshot_out is not None
    assert snapshot_out.dt == pytest.approx(0.1)

    payload = frame.to_dict()
    assert int(payload["depth"]) == 2
    assert tuple(payload["slice_shape"]) == (2, 2)

    streamed = frame.to_streamed_volume()
    assert streamed.depth == 2
    assert streamed.slice_shape == (2, 2)
    assert len(streamed.slices()) == 2

    streamed_profile = streamed.profile()
    assert int(streamed_profile["depth"]) == 2
    streamed_report = streamed.telemetry_report()
    assert int(streamed_report["depth"]) == 2

    streamed_payload = streamed.to_dict()
    assert int(streamed_payload["depth"]) == 2
    assert tuple(streamed_payload["slice_shape"]) == (2, 2)
