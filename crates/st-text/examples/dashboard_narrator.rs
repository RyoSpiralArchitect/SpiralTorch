// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal demo narrating dashboard telemetry into language + audio-friendly samples.

use st_core::telemetry::atlas::{AtlasFrame, AtlasMetric, ConceptAnnotation, ConceptSense};
use st_core::telemetry::dashboard::{
    DashboardEvent, DashboardFrame, DashboardMetric, EventSeverity,
};
use st_tensor::PureResult;
use st_text::{RealtimeNarrator, TextResonator};

fn main() -> PureResult<()> {
    let narrator = TextResonator::new(-1.0, 0.6)?;
    let realtime = RealtimeNarrator::from_resonator(narrator.clone(), 256);

    let mut dashboard = DashboardFrame::new(std::time::SystemTime::now());
    dashboard.push_metric(
        DashboardMetric::new("loss", 0.084)
            .with_unit("")
            .with_trend(-0.021),
    );
    dashboard.push_metric(DashboardMetric::new("throughput", 912.0).with_unit("tok/s"));
    dashboard.push_metric(
        DashboardMetric::new("gpu_util", 0.73)
            .with_unit("")
            .with_trend(0.04),
    );
    dashboard.push_event(DashboardEvent {
        message: "gradient noise rising".to_string(),
        severity: EventSeverity::Warning,
    });
    dashboard.push_event(DashboardEvent {
        message: "checkpoint rotated".to_string(),
        severity: EventSeverity::Info,
    });

    let story = narrator.describe_dashboard_frame(&dashboard);
    println!("Dashboard narrative: {}", story.summary);
    for highlight in story.highlights.iter().take(8) {
        println!("  - {highlight}");
    }

    let audio = realtime.narrate_dashboard_frame(&dashboard)?;
    println!("Audio-ready samples: {}", audio.len());

    let mut atlas = AtlasFrame::new(1.2);
    atlas.loop_support = 0.41;
    atlas.concepts.push(ConceptAnnotation::with_rationale(
        "qualia",
        ConceptSense::QualiaNagelSubjectivity,
        "Keep 'what-it-is-like' talk scoped to subjective report, not ontological claims.",
    ));
    if let Some(metric) = AtlasMetric::new("focus", 0.83) {
        atlas.metrics.push(metric);
    }
    atlas
        .notes
        .push("demo: concept annotations included".to_string());

    let atlas_story = narrator.describe_atlas(&atlas);
    println!("\nAtlas narrative: {}", atlas_story.summary);
    for highlight in atlas_story.highlights.iter().take(8) {
        println!("  - {highlight}");
    }

    let atlas_audio = realtime.narrate_atlas(&atlas)?;
    println!("Atlas audio-ready samples: {}", atlas_audio.len());

    Ok(())
}
