use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use st_backend_wgpu::transform::{
    CenterCropConfig, ColorJitterConfig, HorizontalFlipConfig, ResizeConfig, TransformDispatcher,
};

fn random_image(channels: usize, height: usize, width: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(channels * height * width);
    for _ in 0..channels * height * width {
        data.push(rng.gen::<f32>());
    }
    data
}

fn bench_resize(c: &mut Criterion) {
    let input = random_image(3, 512, 512, 42);
    let config = ResizeConfig {
        channels: 3,
        src_height: 512,
        src_width: 512,
        dst_height: 224,
        dst_width: 224,
    };
    let cpu = TransformDispatcher::cpu();
    c.bench_with_input(BenchmarkId::new("resize", "cpu"), &config, |b, cfg| {
        b.iter(|| {
            let out = cpu.resize(criterion::black_box(&input), *cfg).unwrap();
            criterion::black_box(out)
        })
    });
    if let Ok(gpu) = TransformDispatcher::new_default_gpu() {
        c.bench_with_input(BenchmarkId::new("resize", "gpu"), &config, |b, cfg| {
            b.iter(|| {
                let out = gpu.resize(criterion::black_box(&input), *cfg).unwrap();
                criterion::black_box(out)
            })
        });
    }
}

fn bench_color_jitter(c: &mut Criterion) {
    let input = random_image(3, 256, 256, 7);
    let config = ColorJitterConfig {
        channels: 3,
        height: 256,
        width: 256,
        brightness: 1.1,
        contrast: 0.9,
        saturation: 1.2,
        hue: 0.05,
    };
    let cpu = TransformDispatcher::cpu();
    c.bench_with_input(
        BenchmarkId::new("color_jitter", "cpu"),
        &config,
        |b, cfg| {
            b.iter(|| {
                let out = cpu
                    .color_jitter(criterion::black_box(&input), *cfg)
                    .unwrap();
                criterion::black_box(out)
            })
        },
    );
    if let Ok(gpu) = TransformDispatcher::new_default_gpu() {
        c.bench_with_input(
            BenchmarkId::new("color_jitter", "gpu"),
            &config,
            |b, cfg| {
                b.iter(|| {
                    let out = gpu
                        .color_jitter(criterion::black_box(&input), *cfg)
                        .unwrap();
                    criterion::black_box(out)
                })
            },
        );
    }
}

fn bench_geometric(c: &mut Criterion) {
    let input = random_image(3, 256, 256, 99);
    let crop = CenterCropConfig {
        channels: 3,
        src_height: 256,
        src_width: 256,
        crop_height: 224,
        crop_width: 224,
    };
    let flip = HorizontalFlipConfig {
        channels: 3,
        height: 256,
        width: 256,
        apply: true,
    };
    let cpu = TransformDispatcher::cpu();
    c.bench_with_input(BenchmarkId::new("center_crop", "cpu"), &crop, |b, cfg| {
        b.iter(|| {
            let out = cpu.center_crop(criterion::black_box(&input), *cfg).unwrap();
            criterion::black_box(out)
        })
    });
    c.bench_with_input(
        BenchmarkId::new("horizontal_flip", "cpu"),
        &flip,
        |b, cfg| {
            b.iter(|| {
                let out = cpu
                    .horizontal_flip(criterion::black_box(&input), *cfg)
                    .unwrap();
                criterion::black_box(out)
            })
        },
    );
    if let Ok(gpu) = TransformDispatcher::new_default_gpu() {
        c.bench_with_input(BenchmarkId::new("center_crop", "gpu"), &crop, |b, cfg| {
            b.iter(|| {
                let out = gpu.center_crop(criterion::black_box(&input), *cfg).unwrap();
                criterion::black_box(out)
            })
        });
        c.bench_with_input(
            BenchmarkId::new("horizontal_flip", "gpu"),
            &flip,
            |b, cfg| {
                b.iter(|| {
                    let out = gpu
                        .horizontal_flip(criterion::black_box(&input), *cfg)
                        .unwrap();
                    criterion::black_box(out)
                })
            },
        );
    }
}

fn transforms_benchmark(c: &mut Criterion) {
    bench_resize(c);
    bench_color_jitter(c);
    bench_geometric(c);
}

criterion_group!(name = benches; config = Criterion::default().sample_size(20); targets = transforms_benchmark);
criterion_main!(benches);
