use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use st_bench::{benchmark_backends_with_trials, canonical_backend_probes, simulate_linear_epoch};

fn bench_coop_linear_epoch(c: &mut Criterion) {
    let mut group = c.benchmark_group("coop_linear_epoch");
    for &steps in &[8usize, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &steps| {
            b.iter(|| {
                let reward = simulate_linear_epoch(black_box(42), black_box(steps));
                black_box(reward);
            });
        });
    }
    group.finish();
}

fn bench_backend_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_report");
    for &(steps, trials) in &[(8usize, 1usize), (16, 4), (32, 8), (64, 16)] {
        let id = BenchmarkId::from_parameter(format!("{steps}steps_{trials}trials"));
        group.bench_with_input(id, &(steps, trials), |b, &(steps, trials)| {
            let probes = canonical_backend_probes();
            b.iter(|| {
                let report = benchmark_backends_with_trials(
                    black_box(&probes[..]),
                    black_box(steps),
                    black_box(trials),
                );
                black_box(report.best_backend());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_coop_linear_epoch, bench_backend_report);
criterion_main!(benches);
