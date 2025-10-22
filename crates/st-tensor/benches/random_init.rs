use criterion::AxisScale;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Criterion, PlotConfiguration,
};
use plotters::prelude::*;
use st_tensor::Tensor;

fn render_histogram(samples: &[f32]) {
    let root = BitMapBackend::new("target/criterion/random_init/histogram.png", (640, 480))
        .into_drawing_area();
    root.fill(&WHITE).expect("failed to clear canvas");
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Normal initialiser snapshot", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-3f32..3f32, 0..25)
        .expect("failed to build chart");
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("σ buckets")
        .y_desc("count")
        .draw()
        .expect("failed to draw mesh");

    let buckets = 24usize;
    let width = (6.0 / buckets as f32).max(0.25);
    let chunk_size = ((samples.len().max(1) + buckets - 1) / buckets).max(1);
    chart
        .draw_series(
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
        )
        .expect("failed to draw histogram");
    root.present().expect("failed to flush histogram");
}

fn bench_random_initialisers(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_random_init");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    group.bench_function("uniform_128x128", |b| {
        b.iter_batched(
            || (128, 128),
            |(rows, cols)| {
                black_box(Tensor::random_uniform(rows, cols, -1.0, 1.0, Some(7)).unwrap());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("normal_128x128", |b| {
        b.iter_batched(
            || (128, 128),
            |(rows, cols)| {
                black_box(Tensor::random_normal(rows, cols, 0.0, 0.5, Some(13)).unwrap());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();

    let snapshot = Tensor::random_normal(256, 1, 0.0, 1.0, Some(5)).unwrap();
    render_histogram(snapshot.data());
}

criterion_group!(benches, bench_random_initialisers);
criterion_main!(benches);
