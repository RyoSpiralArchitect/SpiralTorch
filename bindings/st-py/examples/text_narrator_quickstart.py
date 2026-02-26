"""Quickstart: narrate telemetry into language + audio-friendly samples."""

from __future__ import annotations

import spiraltorch as st


def main() -> None:
    atlas = st.telemetry.AtlasFrame.from_metrics(
        {
            "energy.total": 1.7,
            "drift.mean": 0.08,
            "loop.support": 0.42,
            "focus": 0.83,
        },
        timestamp=1.2,
    )

    resonator = st.TextResonator(curvature=-1.0, temperature=0.6)
    narrative = resonator.describe_atlas(atlas)
    print("Atlas narrative:", narrative.summary)
    for line in narrative.highlights[:5]:
        print("  -", line)

    wave = resonator.wave_from_atlas(atlas)
    audio = wave.to_audio_samples(sample_rate=256)
    print(f"\nWave amplitude length={len(wave.amplitude)} audio samples={len(audio)}")

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
    print("\nChrono summary:", chrono_story.summary)

    realtime = st.RealtimeNarrator(curvature=-1.0, temperature=0.6, sample_rate=256)
    audio2 = realtime.narrate_atlas(atlas)
    print(f"Realtime narrator: audio samples={len(audio2)}")


if __name__ == "__main__":
    main()

