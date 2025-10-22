use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use julia_ffi_poc::{rust_latency_score, tempo_latency_score};

fn bench_latency_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("tempo_score");
    group.bench_function("rust_latency_score", |b| {
        b.iter(|| {
            let score = rust_latency_score(512, 192);
            criterion::black_box(score)
        });
    });

    group.bench_function("julia_latency_score_poc", |b| {
        b.iter_batched(
            || (512u32, 192u32),
            |(tile, slack)| {
                let score = tempo_latency_score(tile, slack).expect("score should be available");
                criterion::black_box(score)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_latency_score);
criterion_main!(benches);
