use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_core::telemetry::atlas::{AtlasFrame, AtlasMetric};
use st_core::telemetry::chrono::ChronoSummary;
use st_text::TextResonator;

fn demo_atlas() -> AtlasFrame {
    let mut frame = AtlasFrame::new(1.2);
    frame.loop_support = 0.42;
    frame.suggested_pressure = Some(0.9);
    frame.notes.push("bench: atlas baseline".to_string());
    if let Some(metric) = AtlasMetric::new("focus", 0.83) {
        frame.metrics.push(metric);
    }
    if let Some(metric) = AtlasMetric::with_district("energy.total", 1.7, "bench") {
        frame.metrics.push(metric);
    }
    frame
}

fn demo_summary() -> ChronoSummary {
    ChronoSummary {
        frames: 256,
        duration: 25.6,
        latest_timestamp: 25.6,
        mean_drift: 0.08,
        mean_abs_drift: 0.11,
        drift_std: 0.05,
        mean_energy: 1.7,
        energy_std: 0.4,
        mean_decay: 0.02,
        min_energy: 1.1,
        max_energy: 2.3,
    }
}

fn bench_text_narrator(c: &mut Criterion) {
    let resonator = TextResonator::new(-1.0, 0.6).expect("resonator init");
    let atlas = demo_atlas();
    let summary = demo_summary();

    c.bench_function("text.describe_atlas", |b| {
        b.iter(|| black_box(resonator.describe_atlas(black_box(&atlas))))
    });

    c.bench_function("text.describe_summary", |b| {
        b.iter(|| black_box(resonator.describe_summary(black_box(&summary))))
    });

    let story = resonator.describe_atlas(&atlas);
    c.bench_function("text.synthesize_wave", |b| {
        b.iter(|| {
            black_box(
                resonator
                    .synthesize_wave(black_box(&story))
                    .expect("wave synth"),
            )
        })
    });
}

criterion_group!(benches, bench_text_narrator);
criterion_main!(benches);
