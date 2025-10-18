use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_bench::simulate_linear_epoch;

fn bench_coop_linear_epoch(c: &mut Criterion) {
    c.bench_function("coop_linear_epoch", |b| {
        b.iter(|| {
            let reward = simulate_linear_epoch(black_box(42), black_box(16));
            black_box(reward);
        });
    });
}

criterion_group!(benches, bench_coop_linear_epoch);
criterion_main!(benches);
