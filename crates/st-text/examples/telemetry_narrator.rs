// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal demo narrating atlas + chrono summaries into language + audio-friendly samples.

use st_core::telemetry::atlas::{AtlasFrame, AtlasMetric};
use st_core::telemetry::chrono::ChronoSummary;
use st_tensor::PureResult;
use st_text::{RealtimeNarrator, TextResonator};

fn main() -> PureResult<()> {
    let narrator = TextResonator::new(-1.0, 0.6)?;

    let chrono_summary = ChronoSummary {
        frames: 12,
        duration: 1.2,
        latest_timestamp: 1.2,
        mean_drift: 0.08,
        mean_abs_drift: 0.11,
        drift_std: 0.05,
        mean_energy: 1.7,
        energy_std: 0.4,
        mean_decay: 0.02,
        min_energy: 1.1,
        max_energy: 2.3,
    };

    let story = narrator.describe_summary(&chrono_summary);
    println!("Chrono summary narrative: {}", story.summary);
    for highlight in story.highlights.iter().take(5) {
        println!("  - {highlight}");
    }

    let mut atlas = AtlasFrame::new(chrono_summary.latest_timestamp);
    atlas.loop_support = 0.42;
    atlas.suggested_pressure = Some(0.9);
    atlas
        .notes
        .push("demo: atlas stitched from local loop".to_string());
    if let Some(metric) = AtlasMetric::new("focus", 0.83) {
        atlas.metrics.push(metric);
    }

    let atlas_story = narrator.describe_atlas(&atlas);
    println!("\nAtlas narrative: {}", atlas_story.summary);
    for highlight in atlas_story.highlights.iter().take(5) {
        println!("  - {highlight}");
    }

    let wave = narrator.synthesize_wave(&atlas_story)?;
    println!(
        "\nSynthesized language wave: {} samples",
        wave.amplitude.len()
    );

    let realtime = RealtimeNarrator::from_resonator(narrator, 256);
    let audio = realtime.narrate_atlas(&atlas)?;
    println!("Audio-ready samples: {}", audio.len());
    Ok(())
}
