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
        try:
            _ = module.telemetry.AtlasFrame
        except Exception:
            return None
        return module
    return None


def test_text_narrator_exposes_telemetry_helpers() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

    assert hasattr(st, "text")
    assert hasattr(st, "TextResonator")
    assert hasattr(st, "RealtimeNarrator")
    assert hasattr(st, "ResonanceNarrative")
    assert hasattr(st, "LanguageWave")

    atlas = st.telemetry.AtlasFrame.from_metrics({"focus": 0.83, "energy.total": 1.7}, timestamp=1.2)

    resonator = st.TextResonator(curvature=-1.0, temperature=0.6)
    narrative = resonator.describe_atlas(atlas)
    assert isinstance(narrative, st.ResonanceNarrative)
    assert narrative.summary
    assert isinstance(narrative.highlights, list)
    assert narrative.highlights

    snapshot = st.ChronoSnapshot.from_values(
        frames=12,
        duration=1.2,
        latest_timestamp=1.2,
        mean_drift=0.08,
        mean_abs_drift=0.11,
        drift_std=0.05,
        mean_energy=1.7,
        energy_std=0.4,
        mean_decay=0.02,
        min_energy=1.1,
        max_energy=2.3,
        dt=0.1,
    )

    chrono_story = resonator.describe_chrono_snapshot(snapshot)
    assert chrono_story.summary
    assert "Chrono summary" in chrono_story.summary

    wave = resonator.wave_from_atlas(atlas)
    assert isinstance(wave, st.LanguageWave)
    assert wave.amplitude
    audio = wave.to_audio_samples(sample_rate=64)
    assert len(audio) >= len(wave.amplitude)

    realtime = st.RealtimeNarrator(curvature=-1.0, temperature=0.6, sample_rate=64)
    audio2 = realtime.narrate_atlas(atlas)
    assert audio2
    audio3 = realtime.narrate_chrono_summary(snapshot.summary())
    assert audio3
