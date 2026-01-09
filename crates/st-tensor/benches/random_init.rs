use criterion::AxisScale;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Criterion, PlotConfiguration,
};
use plotters::prelude::*;
use st_tensor::Tensor;

#[track_caller]
fn unwrap_ok<T, E: core::fmt::Debug>(context: &str, result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error:?}"),
    }
}

fn render_histogram(samples: &[f32]) {
    let root = BitMapBackend::new("target/criterion/random_init/histogram.png", (640, 480))
        .into_drawing_area();
    unwrap_ok("failed to clear canvas", root.fill(&WHITE));
    let mut chart = unwrap_ok(
        "failed to build chart",
        ChartBuilder::on(&root)
            .margin(20)
            .caption("Normal initialiser snapshot", ("sans-serif", 20))
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-3f32..3f32, 0..25),
    );
    unwrap_ok(
        "failed to draw mesh",
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("σ buckets")
            .y_desc("count")
            .draw(),
    );

    let buckets = 24usize;
    let width = (6.0 / buckets as f32).max(0.25);
    let chunk_size = samples.len().max(1).div_ceil(buckets);
    unwrap_ok(
        "failed to draw histogram",
        chart.draw_series(
            samples
                .chunks(chunk_size)
                .enumerate()
                .map(|(index, chunk)| {
                    let center = -3.0 + width / 2.0 + width * index as f32;
                    let height = chunk.len() as i32;
                    Rectangle::new(
                        [(center - width / 2.0, 0), (center + width / 2.0, height)],
                        BLUE.mix(0.35).filled(),
                    )
                }),
        ),
    );
    unwrap_ok("failed to flush histogram", root.present());
}

fn bench_random_initialisers(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_random_init");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    group.bench_function("uniform_128x128", |b| {
        b.iter_batched(
            || (128, 128),
            |(rows, cols)| {
                black_box(unwrap_ok(
                    "uniform initialiser failed",
                    Tensor::random_uniform(rows, cols, -1.0, 1.0, Some(7)),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("normal_128x128", |b| {
        b.iter_batched(
            || (128, 128),
            |(rows, cols)| {
                black_box(unwrap_ok(
                    "normal initialiser failed",
                    Tensor::random_normal(rows, cols, 0.0, 0.5, Some(13)),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();

    let snapshot = unwrap_ok(
        "normal snapshot initialiser failed",
        Tensor::random_normal(256, 1, 0.0, 1.0, Some(5)),
    );
    render_histogram(snapshot.data());
}

criterion_group!(benches, bench_random_initialisers);
criterion_main!(benches);
