use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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

fn bench_resize(c: &mut Criterion, cpu: &TransformDispatcher, gpu: Option<&TransformDispatcher>) {
    let mut group = c.benchmark_group("resize");
    let cases = [
        (3usize, 512usize, 512usize, 224usize, 224usize, 42u64),
        (3, 1024, 1024, 384, 384, 1337u64),
    ];

    for &(channels, src_h, src_w, dst_h, dst_w, seed) in &cases {
        let input_cpu = random_image(channels, src_h, src_w, seed);
        let input_gpu = input_cpu.clone();
        let label = format!("{channels}x{src_h}x{src_w}→{channels}x{dst_h}x{dst_w}");
        let config = ResizeConfig {
            channels,
            src_height: src_h,
            src_width: src_w,
            dst_height: dst_h,
            dst_width: dst_w,
        };
        let cpu_config = config;
        let gpu_config = config;

        let cpu_label = label.clone();
        group.bench_with_input(
            BenchmarkId::new("cpu", cpu_label),
            &cpu_config,
            move |b, cfg| {
                b.iter(|| {
                    let out = cpu
                        .resize(black_box(&input_cpu), *cfg)
                        .expect("cpu resize succeeds");
                    black_box(out)
                })
            },
        );

        if let Some(gpu_dispatcher) = gpu {
            let gpu_label = label.clone();
            group.bench_with_input(
                BenchmarkId::new("gpu", gpu_label),
                &gpu_config,
                move |b, cfg| {
                    b.iter(|| {
                        let out = gpu_dispatcher
                            .resize(black_box(&input_gpu), *cfg)
                            .expect("gpu resize succeeds");
                        black_box(out)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_color_jitter(
    c: &mut Criterion,
    cpu: &TransformDispatcher,
    gpu: Option<&TransformDispatcher>,
) {
    let mut group = c.benchmark_group("color_jitter");
    let cases = [
        (
            "3x256x256_default".to_string(),
            ColorJitterConfig {
                channels: 3,
                height: 256,
                width: 256,
                brightness: 1.1,
                contrast: 0.9,
                saturation: 1.2,
                hue: 0.05,
            },
            7u64,
        ),
        (
            "3x512x512_vivid".to_string(),
            ColorJitterConfig {
                channels: 3,
                height: 512,
                width: 512,
                brightness: 1.35,
                contrast: 0.85,
                saturation: 1.4,
                hue: 0.08,
            },
            91u64,
        ),
    ];

    for (label, config, seed) in cases {
        let input_cpu = random_image(config.channels, config.height, config.width, seed);
        let input_gpu = input_cpu.clone();
        let cpu_config = config;
        let gpu_config = config;

        let cpu_label = label.clone();
        group.bench_with_input(
            BenchmarkId::new("cpu", cpu_label),
            &cpu_config,
            move |b, cfg| {
                b.iter(|| {
                    let out = cpu
                        .color_jitter(black_box(&input_cpu), *cfg)
                        .expect("cpu jitter succeeds");
                    black_box(out)
                })
            },
        );

        if let Some(gpu_dispatcher) = gpu {
            let gpu_label = label.clone();
            group.bench_with_input(
                BenchmarkId::new("gpu", gpu_label),
                &gpu_config,
                move |b, cfg| {
                    b.iter(|| {
                        let out = gpu_dispatcher
                            .color_jitter(black_box(&input_gpu), *cfg)
                            .expect("gpu jitter succeeds");
                        black_box(out)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_geometric(
    c: &mut Criterion,
    cpu: &TransformDispatcher,
    gpu: Option<&TransformDispatcher>,
) {
    let mut crop_group = c.benchmark_group("center_crop");
    let crop_cases = [
        (3usize, 256usize, 256usize, 224usize, 224usize, 99u64),
        (3, 512, 512, 320, 320, 133u64),
    ];

    for &(channels, src_h, src_w, crop_h, crop_w, seed) in &crop_cases {
        let input_cpu = random_image(channels, src_h, src_w, seed);
        let input_gpu = input_cpu.clone();
        let label = format!("{channels}x{src_h}x{src_w}→{channels}x{crop_h}x{crop_w}");
        let config = CenterCropConfig {
            channels,
            src_height: src_h,
            src_width: src_w,
            crop_height: crop_h,
            crop_width: crop_w,
        };
        let cpu_config = config;
        let gpu_config = config;

        let cpu_label = label.clone();
        crop_group.bench_with_input(
            BenchmarkId::new("cpu", cpu_label),
            &cpu_config,
            move |b, cfg| {
                b.iter(|| {
                    let out = cpu
                        .center_crop(black_box(&input_cpu), *cfg)
                        .expect("cpu crop succeeds");
                    black_box(out)
                })
            },
        );

        if let Some(gpu_dispatcher) = gpu {
            let gpu_label = label.clone();
            crop_group.bench_with_input(
                BenchmarkId::new("gpu", gpu_label),
                &gpu_config,
                move |b, cfg| {
                    b.iter(|| {
                        let out = gpu_dispatcher
                            .center_crop(black_box(&input_gpu), *cfg)
                            .expect("gpu crop succeeds");
                        black_box(out)
                    })
                },
            );
        }
    }
    crop_group.finish();

    let mut flip_group = c.benchmark_group("horizontal_flip");
    let flip_cases = [
        (3usize, 256usize, 256usize, true, 17u64),
        (3, 512, 512, false, 21u64),
    ];

    for &(channels, height, width, apply, seed) in &flip_cases {
        let input_cpu = random_image(channels, height, width, seed);
        let input_gpu = input_cpu.clone();
        let label = format!(
            "{channels}x{height}x{width}_{}",
            if apply { "flip" } else { "no-op" }
        );
        let config = HorizontalFlipConfig {
            channels,
            height,
            width,
            apply,
        };
        let cpu_config = config;
        let gpu_config = config;

        let cpu_label = label.clone();
        flip_group.bench_with_input(
            BenchmarkId::new("cpu", cpu_label),
            &cpu_config,
            move |b, cfg| {
                b.iter(|| {
                    let out = cpu
                        .horizontal_flip(black_box(&input_cpu), *cfg)
                        .expect("cpu flip succeeds");
                    black_box(out)
                })
            },
        );

        if let Some(gpu_dispatcher) = gpu {
            let gpu_label = label.clone();
            flip_group.bench_with_input(
                BenchmarkId::new("gpu", gpu_label),
                &gpu_config,
                move |b, cfg| {
                    b.iter(|| {
                        let out = gpu_dispatcher
                            .horizontal_flip(black_box(&input_gpu), *cfg)
                            .expect("gpu flip succeeds");
                        black_box(out)
                    })
                },
            );
        }
    }

    flip_group.finish();
}

fn bench_pipeline(c: &mut Criterion, cpu: &TransformDispatcher, gpu: Option<&TransformDispatcher>) {
    let mut group = c.benchmark_group("classification_pipeline");
    let resize_cfg = ResizeConfig {
        channels: 3,
        src_height: 512,
        src_width: 512,
        dst_height: 256,
        dst_width: 256,
    };
    let crop_cfg = CenterCropConfig {
        channels: 3,
        src_height: 256,
        src_width: 256,
        crop_height: 224,
        crop_width: 224,
    };
    let jitter_cfg = ColorJitterConfig {
        channels: 3,
        height: 224,
        width: 224,
        brightness: 1.05,
        contrast: 0.95,
        saturation: 1.1,
        hue: 0.04,
    };
    let flip_cfg = HorizontalFlipConfig {
        channels: 3,
        height: 224,
        width: 224,
        apply: true,
    };

    let cpu_resize = resize_cfg;
    let cpu_crop = crop_cfg;
    let cpu_jitter = jitter_cfg;
    let cpu_flip = flip_cfg;
    let gpu_resize = resize_cfg;
    let gpu_crop = crop_cfg;
    let gpu_jitter = jitter_cfg;
    let gpu_flip = flip_cfg;

    let pipeline_input_cpu = random_image(3, 512, 512, 2048);
    let pipeline_input_gpu = pipeline_input_cpu.clone();

    group.bench_function("cpu", move |b| {
        b.iter(|| {
            let resized = cpu
                .resize(black_box(&pipeline_input_cpu), cpu_resize)
                .expect("cpu resize succeeds");
            let cropped = cpu
                .center_crop(black_box(&resized), cpu_crop)
                .expect("cpu crop succeeds");
            let jittered = cpu
                .color_jitter(black_box(&cropped), cpu_jitter)
                .expect("cpu jitter succeeds");
            let flipped = cpu
                .horizontal_flip(black_box(&jittered), cpu_flip)
                .expect("cpu flip succeeds");
            black_box(flipped);
        });
    });

    if let Some(gpu_dispatcher) = gpu {
        group.bench_function("gpu", move |b| {
            b.iter(|| {
                let resized = gpu_dispatcher
                    .resize(black_box(&pipeline_input_gpu), gpu_resize)
                    .expect("gpu resize succeeds");
                let cropped = gpu_dispatcher
                    .center_crop(black_box(&resized), gpu_crop)
                    .expect("gpu crop succeeds");
                let jittered = gpu_dispatcher
                    .color_jitter(black_box(&cropped), gpu_jitter)
                    .expect("gpu jitter succeeds");
                let flipped = gpu_dispatcher
                    .horizontal_flip(black_box(&jittered), gpu_flip)
                    .expect("gpu flip succeeds");
                black_box(flipped);
            });
        });
    }

    group.finish();
}

fn transforms_benchmark(c: &mut Criterion) {
    let cpu = TransformDispatcher::cpu();
    let gpu = TransformDispatcher::new_default_gpu().ok();
    let gpu_ref = gpu.as_ref();

    bench_resize(c, &cpu, gpu_ref);
    bench_color_jitter(c, &cpu, gpu_ref);
    bench_geometric(c, &cpu, gpu_ref);
    bench_pipeline(c, &cpu, gpu_ref);
}

criterion_group!(name = benches; config = Criterion::default().sample_size(20); targets = transforms_benchmark);
criterion_main!(benches);
