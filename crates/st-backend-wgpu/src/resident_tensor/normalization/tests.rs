use super::*;

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.layer_norm.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(TensorDevice::new(runtime).unwrap())
}

fn read(tensor: &ResidentTensor) -> Vec<f32> {
    tensor.snapshot().unwrap().read().unwrap()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (f64::from(a) - f64::from(b)).abs() <= 2e-5 * (1.0 + f64::from(b).abs()),
            "index {i}: {a} != {b}"
        );
    }
}

// Independent centered-f64 reference, retaining full precision until requested
// final gradients are stored, as in the existing CPU Tensor contract.
fn reference(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    seed: &[f32],
    epsilon: f32,
    scale: f32,
) -> [Vec<f32>; 4] {
    let cols = gamma.len();
    let mut y = vec![];
    let mut dx = vec![];
    let mut dg = vec![0f64; cols];
    let mut db = vec![0f64; cols];
    for (row, seed) in x.chunks_exact(cols).zip(seed.chunks_exact(cols)) {
        let origin = f64::from(row[0]);
        let mean = row.iter().map(|&v| f64::from(v) - origin).sum::<f64>() / cols as f64;
        let centered: Vec<_> = row.iter().map(|&v| f64::from(v) - origin - mean).collect();
        let denominator =
            (centered.iter().map(|v| v * v).sum::<f64>() / cols as f64 + f64::from(epsilon)).sqrt();
        let normed: Vec<_> = centered.iter().map(|v| v / denominator).collect();
        let weighted_origin = f64::from(seed[0]) * f64::from(gamma[0]);
        let weighted: Vec<_> = seed
            .iter()
            .zip(gamma)
            .map(|(&s, &g)| f64::from(s) * f64::from(g) - weighted_origin)
            .collect();
        let mean = weighted.iter().sum::<f64>() / cols as f64;
        let projection = weighted
            .iter()
            .zip(&normed)
            .map(|(g, n)| g * n)
            .sum::<f64>()
            / cols as f64;
        for c in 0..cols {
            y.push(normed[c] as f32 * gamma[c] + beta[c]);
            dx.push(((weighted[c] - mean - normed[c] * projection) / denominator) as f32);
            dg[c] += f64::from(seed[c]) * normed[c];
            db[c] += f64::from(seed[c]);
        }
    }
    [
        y,
        dx,
        dg.iter().map(|v| (v * f64::from(scale)) as f32).collect(),
        db.iter().map(|v| (v * f64::from(scale)) as f32).collect(),
    ]
}

#[test]
fn layer_norm_shaders_validate_without_adapter() {
    for template in [
        include_str!("../shaders/layer_norm.wgsl"),
        include_str!("../shaders/layer_norm_backward.wgsl"),
    ] {
        let text = source(template);
        let module = naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    }
    assert_eq!(std::mem::size_of::<Params>(), 32);
    let shape = LayerNormShape::new(&[1, 3], &[3]).unwrap();
    let mut limits = wgpu::Limits {
        max_compute_workgroup_storage_size: 8291,
        ..Default::default()
    };
    assert!(matches!(
        preflight(shape, &limits),
        Err(TensorError::Limit("LayerNorm pipeline"))
    ));
    limits.max_compute_workgroup_storage_size = 8292;
    assert!(preflight(shape, &limits).is_ok());
}

#[test]
fn layer_norm_ordered_sum_preserves_sum_and_residual_bits() {
    let Some(tensor_device) = device() else {
        return;
    };
    let ctx = tensor_device.0.runtime.context();
    let device = ctx.device();
    let mut pairs = Vec::new();
    let boundary = [
        0u32, 1, 0x7fffff, 0x800000, 0x33800000, 0x3f000000, 0x3f800000, 0x3f800001, 0x40000000,
    ];
    for a in boundary {
        for b in boundary {
            for sign_a in [0, 0x80000000] {
                for sign_b in [0, 0x80000000] {
                    pairs.push([a | sign_a, b | sign_b]);
                }
            }
        }
    }
    let mut state = 17u32;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Finite, non-overflowing significand pairs, including subnormal bits.
        (state & 0x807fffff) | (((state >> 24) % 140) << 23)
    };
    for _ in 0..16384 {
        pairs.push([next(), next()]);
    }
    let input = runtime::upload_slice(
        device,
        "layer_norm.sum.inputs",
        &pairs,
        wgpu::BufferUsages::STORAGE,
    )
    .unwrap();
    let output = runtime::empty_buffer::<[u32; 2]>(
        device,
        "layer_norm.sum.output",
        pairs.len(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )
    .unwrap();
    let shader = source(
        r#"
ROUNDED_ADD
WIDE_ARITHMETIC
@group(0) @binding(0) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&pairs)) {
        output[id.x] = bitcast<vec2<u32>>(two_sum(bitcast<f32>(pairs[id.x].x), bitcast<f32>(pairs[id.x].y)));
    }
}
"#,
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("layer_norm.sum.shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("layer_norm.sum.pipeline"),
        layout: None,
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("layer_norm.sum.group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((pairs.len() as u32).div_ceil(64), 1, 1);
    }
    ctx.queue().submit(Some(encoder.finish()));
    let actual = runtime::read_buffer::<[u32; 2]>(
        device,
        ctx.queue(),
        &output,
        pairs.len(),
        "layer_norm.sum.read",
    )
    .unwrap();
    for ([a, b], actual) in pairs.into_iter().zip(actual) {
        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
        // Independent, unordered TwoSum on the CPU. Do not enable fast-math.
        let sum = a + b;
        let bv = sum - a;
        let residual = (a - (sum - bv)) + (b - bv);
        for (bits, expected) in actual.into_iter().zip([sum, residual]) {
            if expected == 0. {
                assert_eq!(bits & 0x7fffffff, 0);
            } else {
                assert_eq!(bits, expected.to_bits(), "a={a:e}, b={b:e}");
            }
        }
    }
}

#[test]
fn layer_norm_resident_requested_vjps_and_nd_views() {
    let Some(device) = device() else { return };
    let input = device
        .upload(
            &[3, 4, 5],
            &(0..60).map(|v| (v as f32 - 19.) / 16.).collect::<Vec<_>>(),
        )
        .unwrap()
        .narrow(1, 1, 2)
        .unwrap()
        .permute(&[2, 0, 1])
        .unwrap();
    let x = read(&input);
    let shape = input.layout.shape();
    let gamma = device.upload(&[1, 2], &[1.25, -0.75]).unwrap();
    let beta = device.upload(&[1, 2], &[0.1, -0.2]).unwrap();
    let seed = device
        .upload(&[], &[0.25])
        .unwrap()
        .broadcast_to(shape)
        .unwrap();
    let tape = input.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
    let expected = reference(
        &x,
        &[1.25, -0.75],
        &[0.1, -0.2],
        &vec![0.25; x.len()],
        1e-5,
        -0.5,
    );
    close(&read(tape.value()), &expected[0]);
    let mut held = None;
    for mask in 0..8 {
        let requested = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
        let output = tape.backward(&seed, -0.5, requested).unwrap();
        for i in 0..3 {
            assert_eq!(output[i].is_some(), requested[i]);
            if let Some(value) = &output[i] {
                close(&read(value), &expected[i + 1]);
            }
        }
        if mask == 7 {
            held = Some(output);
        }
    }
    let again = tape.backward(&seed, 1.0, [true; 3]).unwrap();
    close(&read(again[0].as_ref().unwrap()), &expected[1]);
    for _ in 0..4 {
        drop(
            input
                .layer_norm_affine(&gamma, &beta, 0.5)
                .unwrap()
                .backward(&seed, 0.0, [true; 3])
                .unwrap(),
        );
    }
    drop((input, gamma, beta, seed, tape, again, device));
    for (i, value) in held.unwrap().iter().enumerate() {
        close(&read(value.as_ref().unwrap()), &expected[i + 1]);
    }
}

#[test]
fn layer_norm_resident_extreme_finite_cases() {
    let Some(device) = device() else { return };
    let tiny = f32::from_bits(1);
    for (rows, cols, x, g, seed, epsilon, scale) in [
        (
            1,
            2,
            vec![0., 1e-20],
            vec![f32::MAX; 2],
            vec![1.; 2],
            f32::MAX,
            1.,
        ),
        (
            1,
            3,
            vec![-1e30, 0., 1e30],
            vec![1e20; 3],
            vec![1e20, -1e20, 1e20],
            1e-5,
            1.,
        ),
        (
            3,
            1,
            vec![1., 2., 3.],
            vec![f32::MAX],
            vec![f32::MAX, f32::MAX, -f32::MAX],
            1e-5,
            1.,
        ),
        (1, 2, vec![0., tiny], vec![1.; 2], vec![1.; 2], 0., 1.),
        (1, 2, vec![0., tiny], vec![1.; 2], vec![0.25, -0.75], 0., 1.),
        (
            1,
            4,
            vec![1e-30, 2e-30, 3e-30, 4e-30],
            vec![1.; 4],
            vec![1.; 4],
            0.,
            1.,
        ),
        (
            1,
            2,
            vec![f32::MAX, -f32::MAX],
            vec![1.; 2],
            vec![1., -1.],
            0.,
            1.,
        ),
        (1, 2, vec![f32::MAX; 2], vec![1.; 2], vec![1.; 2], tiny, 1.),
        (1, 3, vec![7.; 3], vec![1.; 3], vec![1.; 3], tiny, 1.),
        (
            1,
            2,
            vec![0., tiny],
            vec![1.; 2],
            vec![f32::MAX; 2],
            f32::MAX,
            1.,
        ),
        (2, 1, vec![0.; 2], vec![1.], vec![f32::MAX; 2], 1e-5, 0.25),
    ] {
        let beta = vec![0.; cols];
        let expected = reference(&x, &g, &beta, &seed, epsilon, scale);
        let input = device.upload(&[rows, cols], &x).unwrap();
        let gamma = device.upload(&[cols], &g).unwrap();
        let beta = device.upload(&[cols], &beta).unwrap();
        let seed = device.upload(&[rows, cols], &seed).unwrap();
        let tape = input.layer_norm_affine(&gamma, &beta, epsilon).unwrap();
        close(&read(tape.value()), &expected[0]);
        let output = tape.backward(&seed, scale, [true; 3]).unwrap();
        for i in 0..3 {
            close(&read(output[i].as_ref().unwrap()), &expected[i + 1]);
        }
    }
}

#[test]
fn layer_norm_resident_widths_and_offsets() {
    let Some(device) = device() else { return };
    for cols in [1, 3, 255, 256, 257, 513, 1025, 8193] {
        for offset in [0., 10000., 1e7, -1e7] {
            let x: Vec<_> = (0..2 * cols).map(|i| offset + (i % 7) as f32).collect();
            let g: Vec<_> = (0..cols).map(|i| 0.5 + (i % 13) as f32 / 16.).collect();
            let b = vec![0.125; cols];
            let seed: Vec<_> = (0..x.len()).map(|i| ((i % 11) as f32 - 5.) / 16.).collect();
            let expected = reference(&x, &g, &b, &seed, 1e-5, 0.5);
            let input = device.upload(&[2, cols], &x).unwrap();
            let gamma = device.upload(&[cols], &g).unwrap();
            let beta = device.upload(&[cols], &b).unwrap();
            let seed = device.upload(&[2, cols], &seed).unwrap();
            let tape = input.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
            close(&read(tape.value()), &expected[0]);
            let output = tape.backward(&seed, 0.5, [true; 3]).unwrap();
            for i in 0..3 {
                close(&read(output[i].as_ref().unwrap()), &expected[i + 1]);
            }
        }
    }
}

#[test]
fn layer_norm_affine_reduction_workgroup_boundaries() {
    for (rows, expected) in [
        (0, 2),
        (1, 2),
        (32, 2),
        (33, 0),
        (64, 0),
        (65, 1),
        (128, 1),
        (129, 2),
        (256, 2),
        (257, 2),
    ] {
        assert_eq!(affine_pipeline_index(rows), expected);
    }
    let Some(device) = device() else { return };
    let cols = 7;
    let gamma = [0.5, -1., 1.25, 0.75, -0.5, 1.5, 0.25];
    let beta = [0.125; 7];
    for rows in [1, 32, 33, 64, 65, 128, 129, 256, 257] {
        let x: Vec<_> = (0..rows * cols)
            .map(|i| ((i * 17 + 3) % 53) as f32 / 16. - 1.)
            .collect();
        let seed: Vec<_> = (0..x.len())
            .map(|i| ((i * 11 + 5) % 37) as f32 / 32. - 0.5)
            .collect();
        let expected = reference(&x, &gamma, &beta, &seed, 1e-5, 0.5);
        let input = device.upload(&[rows, cols], &x).unwrap();
        let gpu_gamma = device.upload(&[cols], &gamma).unwrap();
        let gpu_beta = device.upload(&[cols], &beta).unwrap();
        let upstream = device.upload(&[rows, cols], &seed).unwrap();
        let tape = input
            .layer_norm_affine(&gpu_gamma, &gpu_beta, 1e-5)
            .unwrap();
        close(&read(tape.value()), &expected[0]);
        let gradients = tape.backward(&upstream, 0.5, [true; 3]).unwrap();
        for i in 0..3 {
            close(&read(gradients[i].as_ref().unwrap()), &expected[i + 1]);
        }
    }
}

#[test]
fn layer_norm_tiled_affine_matches_f64_at_shape_boundaries() {
    let Some(device) = device() else { return };
    let limits = device.runtime().context().device().limits();
    for (rows, cols, tiled) in [
        (31, 256, false),
        (32, 255, false),
        (32, 256, true),
        (32, 257, true),
        (64, 256, true),
        (65, 257, true),
        (128, 1025, true),
        (129, 256, false),
    ] {
        let shape = LayerNormShape::new(&[rows, cols], &[cols]).unwrap();
        let (grid, pipeline) = affine_schedule(shape, &limits).unwrap();
        assert_eq!(pipeline == 3, tiled, "rows={rows} cols={cols}");
        assert_eq!(
            grid[0] as usize * grid[1] as usize,
            if tiled { cols.div_ceil(8) } else { cols }
        );
        let x: Vec<_> = (0..rows * cols)
            .map(|i| ((i * 17 + 3) % 53) as f32 / 16. - 1.)
            .collect();
        let gamma: Vec<_> = (0..cols).map(|i| 0.5 + (i % 13) as f32 / 16.).collect();
        let beta = vec![0.125; cols];
        let seed: Vec<_> = (0..x.len())
            .map(|i| ((i * 11 + 5) % 37) as f32 / 32. - 0.5)
            .collect();
        let expected = reference(&x, &gamma, &beta, &seed, 1e-5, 0.5);
        let input = device.upload(&[rows, cols], &x).unwrap();
        let gain = device.upload(&[cols], &gamma).unwrap();
        let bias = device.upload(&[cols], &beta).unwrap();
        let upstream = device.upload(&[rows, cols], &seed).unwrap();
        let tape = input.layer_norm_affine(&gain, &bias, 1e-5).unwrap();
        close(&read(tape.value()), &expected[0]);
        let gradients = tape.backward(&upstream, 0.5, [true; 3]).unwrap();
        for i in 0..3 {
            close(&read(gradients[i].as_ref().unwrap()), &expected[i + 1]);
        }
    }
}

#[test]
fn layer_norm_tiled_affine_preserves_cancelling_gradients() {
    let Some(device) = device() else { return };
    let cols = 257;
    let gamma: Vec<_> = (0..cols)
        .map(|col| if col % 2 == 0 { 1e5 } else { -1e5 })
        .collect();
    let beta = vec![0.; cols];
    for rows in [64, 128] {
        let x: Vec<_> = (0..rows * cols)
            .map(|i| {
                let sign = if (i / cols) % 2 == 0 { 1. } else { -1. };
                sign * ((i % cols) as f32 - 128.) * 1e-20
            })
            .collect();
        let seed: Vec<_> = (0..x.len())
            .map(|i| {
                let sign = if (i / cols + i % cols) % 2 == 0 {
                    1.
                } else {
                    -1.
                };
                sign * ((i % 5 + 1) as f32 * 1e5)
            })
            .collect();
        let expected = reference(&x, &gamma, &beta, &seed, 1e-20, 0.5);
        let input = device.upload(&[rows, cols], &x).unwrap();
        let gpu_gamma = device.upload(&[cols], &gamma).unwrap();
        let gpu_beta = device.upload(&[cols], &beta).unwrap();
        let upstream = device.upload(&[rows, cols], &seed).unwrap();
        let tape = input
            .layer_norm_affine(&gpu_gamma, &gpu_beta, 1e-20)
            .unwrap();
        close(&read(tape.value()), &expected[0]);
        for mask in 0..8 {
            let requested = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
            let gradients = tape.backward(&upstream, 0.5, requested).unwrap();
            for i in 0..3 {
                assert_eq!(gradients[i].is_some(), requested[i]);
                if let Some(value) = &gradients[i] {
                    close(&read(value), &expected[i + 1]);
                }
            }
        }
    }
}

#[test]
fn layer_norm_affine_small_workgroups_preserve_cancelling_gradients() {
    let Some(device) = device() else { return };
    let cols = 7;
    let gamma = [1e5, -1e5, 0.5e5, -0.5e5, 1.5e5, -1.5e5, 1e5];
    let beta = [0.; 7];
    for rows in [64, 128] {
        let x: Vec<_> = (0..rows * cols)
            .map(|i| {
                let sign = if (i / cols) % 2 == 0 { 1. } else { -1. };
                sign * ((i % cols) as f32 - 3.) * 1e-20
            })
            .collect();
        let seed: Vec<_> = (0..x.len())
            .map(|i| {
                let sign = if (i / cols + i % cols) % 2 == 0 {
                    1.
                } else {
                    -1.
                };
                sign * ((i % 5 + 1) as f32 * 1e5)
            })
            .collect();
        let expected = reference(&x, &gamma, &beta, &seed, 1e-20, 0.5);
        let input = device.upload(&[rows, cols], &x).unwrap();
        let gpu_gamma = device.upload(&[cols], &gamma).unwrap();
        let gpu_beta = device.upload(&[cols], &beta).unwrap();
        let upstream = device.upload(&[rows, cols], &seed).unwrap();
        let tape = input
            .layer_norm_affine(&gpu_gamma, &gpu_beta, 1e-20)
            .unwrap();
        close(&read(tape.value()), &expected[0]);
        for mask in 0..8 {
            let requested = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
            let gradients = tape.backward(&upstream, 0.5, requested).unwrap();
            for i in 0..3 {
                assert_eq!(gradients[i].is_some(), requested[i]);
                if let Some(value) = &gradients[i] {
                    close(&read(value), &expected[i + 1]);
                }
            }
        }
    }
}

#[test]
fn layer_norm_resident_training_matches_cpu_including_slow_convergence() {
    let Some(device) = device() else { return };
    for (x, target_g, target_b, learning_rate, meets_target) in [
        // Frozen exploratory case: it misses 1e-4 at 400 steps on both the
        // resident implementation and CPU/PyTorch. Retain this negative result.
        (
            vec![0., 1., 2., 2., 0., -1.],
            [0.5, 1.5, -0.75],
            [0.2, -0.3, 0.4],
            0.05,
            false,
        ),
        // Exact established st-tensor/layer_norm_autograd.rs learning fixture.
        (
            vec![0.4, -0.8, 1.2, -0.3, 0.9, -1.1, 0.7, 0.1, -0.2],
            [1.7, 0.5, -0.8],
            [0.2, -0.3, 0.6],
            0.1,
            true,
        ),
    ] {
        let target_values =
            reference(&x, &target_g, &target_b, &vec![0.; x.len()], 1e-5, 1.)[0].clone();
        let mut cpu_g = vec![1.; 3];
        let mut cpu_b = vec![0.; 3];
        let mut cpu_losses = Vec::new();
        for _ in 0..400 {
            let y = reference(&x, &cpu_g, &cpu_b, &vec![0.; x.len()], 1e-5, 1.)[0].clone();
            let seed: Vec<_> = y
                .iter()
                .zip(&target_values)
                .map(|(a, b)| (a - b) * (2. / x.len() as f32))
                .collect();
            cpu_losses.push(
                y.iter()
                    .zip(&target_values)
                    .map(|(a, b)| (a - b) * (a - b) / x.len() as f32)
                    .sum::<f32>(),
            );
            let gradients = reference(&x, &cpu_g, &cpu_b, &seed, 1e-5, 1.);
            for col in 0..3 {
                cpu_g[col] += -learning_rate * gradients[2][col];
                cpu_b[col] += -learning_rate * gradients[3][col];
            }
        }
        let input = device.upload(&[x.len() / 3, 3], &x).unwrap();
        let target_gamma = device.upload(&[3], &target_g).unwrap();
        let target_beta = device.upload(&[3], &target_b).unwrap();
        let target = input
            .layer_norm_affine(&target_gamma, &target_beta, 1e-5)
            .unwrap();
        let mut gamma = device.upload(&[3], &[1.; 3]).unwrap();
        let mut beta = device.upload(&[3], &[0.; 3]).unwrap();
        let rate = device.upload(&[], &[-learning_rate]).unwrap();
        let mut first = None;
        let mut last = None;
        // No per-step host statistics, cotangents, gradients or parameter reads.
        for _ in 0..400 {
            let tape = input.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
            let loss = tape.value().mean_squared_error(target.value()).unwrap();
            let [_, dg, db] = tape
                .backward(loss.prediction_gradient(), 1.0, [false, true, true])
                .unwrap();
            gamma = gamma.add(&dg.unwrap().mul(&rate).unwrap()).unwrap();
            beta = beta.add(&db.unwrap().mul(&rate).unwrap()).unwrap();
            if first.is_none() {
                first = Some(loss.clone());
            }
            last = Some(loss);
        }
        let first = read(first.as_ref().unwrap().value())[0];
        let last = read(last.as_ref().unwrap().value())[0];
        close(&[first, last], &[cpu_losses[0], cpu_losses[399]]);
        close(&read(&gamma), &cpu_g);
        close(&read(&beta), &cpu_b);
        assert_eq!(cpu_losses[399] < cpu_losses[0] * 1e-4, meets_target);
        assert_eq!(last < first * 1e-4, meets_target, "{first} -> {last}");
        eprintln!("LayerNorm 400-step training: lr={learning_rate}, first={first}, last={last}, cpu_last={}, meets_1e_4={meets_target}", cpu_losses[399]);
    }
}

#[test]
fn layer_norm_resident_rejections_empty_batches_and_guards() {
    let Some(device) = device() else { return };
    let input = device.upload(&[2, 1], &[1.; 2]).unwrap();
    let gamma = device.upload(&[1], &[f32::MAX]).unwrap();
    let beta = device.upload(&[1], &[0.]).unwrap();
    let seed = device.upload(&[2, 1], &[f32::MAX; 2]).unwrap();
    assert!(input.layer_norm_affine(&gamma, &beta, -1.).is_err());
    assert!(input.layer_norm_affine(&gamma, &seed, 1e-5).is_err());
    assert!(device.0.normalization.get().is_none());
    let tape = input.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
    assert!(tape.backward(&seed, f32::INFINITY, [true; 3]).is_err());
    assert!(tape.backward(&gamma, 1., [true; 3]).is_err());
    let [dx, _, _] = tape.backward(&seed, 1., [true, false, false]).unwrap();
    close(&read(&dx.unwrap()), &[0.; 2]);
    for value in tape
        .backward(&seed, 1., [true; 3])
        .unwrap()
        .iter()
        .flatten()
    {
        assert!(matches!(
            value.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let constant = input.layer_norm_affine(&gamma, &beta, 0.).unwrap();
    assert!(matches!(
        constant.value().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let poisoned = seed.add(&seed).unwrap();
    let masked = poisoned.mul(&device.upload(&[], &[0.]).unwrap()).unwrap();
    let inherited = tape.backward(&masked, 0., [false, false, true]).unwrap()[2]
        .clone()
        .unwrap();
    assert!(matches!(
        inherited.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let invalid_input = masked.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
    assert!(matches!(
        invalid_input.value().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let empty = device.upload(&[2, 0, 1], &[]).unwrap();
    let tape = empty.layer_norm_affine(&gamma, &beta, 0.).unwrap();
    assert_eq!(read(tape.value()), Vec::<f32>::new());
    let grads = tape.backward(&empty, -0.5, [true; 3]).unwrap();
    assert_eq!(read(grads[0].as_ref().unwrap()), Vec::<f32>::new());
    close(&read(grads[1].as_ref().unwrap()), &[0.]);
    close(&read(grads[2].as_ref().unwrap()), &[0.]);
}

#[test]
fn layer_norm_statistics_tape_does_not_inherit_unused_forward_overflow() {
    let Some(device) = device() else { return };
    let input = device.upload(&[1, 2], &[-1., 1.]).unwrap();
    let gamma = device.upload(&[2], &[f32::MAX; 2]).unwrap();
    let beta = device.upload(&[2], &[f32::MAX; 2]).unwrap();
    let seed = device.upload(&[1, 2], &[1., 1.]).unwrap();
    let forward = input.layer_norm_affine(&gamma, &beta, 0.).unwrap();
    assert!(matches!(
        forward.value().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    assert!(matches!(
        forward.backward(&seed, 1., [true; 3]).unwrap()[0]
            .as_ref()
            .unwrap()
            .snapshot()
            .unwrap()
            .read(),
        Err(TensorError::NonFinite)
    ));

    let tape = input.layer_norm_vjp_tape(&gamma, 0.).unwrap();
    let gradients = tape.backward(&seed, 1., [true; 3]).unwrap();
    let pending = device
        .snapshot_many(&gradients.iter().flatten().collect::<Vec<_>>())
        .unwrap();
    assert_eq!(pending.staging_buffer_count(), 1);
    let values = pending.read().unwrap();
    close(&values[0], &[0., 0.]);
    close(&values[1], &[-1., 1.]);
    close(&values[2], &[1., 1.]);

    let huge_seed = device.upload(&[1, 2], &[f32::MAX; 2]).unwrap();
    let bad_seed = huge_seed.add(&huge_seed).unwrap();
    let poisoned = tape.backward(&bad_seed, 1., [true, false, false]).unwrap();
    assert!(matches!(
        poisoned[0].as_ref().unwrap().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let huge_input = device.upload(&[1, 2], &[f32::MAX; 2]).unwrap();
    let bad_input = huge_input.add(&huge_input).unwrap();
    assert!(matches!(
        bad_input.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let bad_input_tape = bad_input.layer_norm_vjp_tape(&gamma, 1e-5).unwrap();
    let poisoned = bad_input_tape
        .backward(&seed, 1., [true, false, false])
        .unwrap();
    assert!(matches!(
        poisoned[0].as_ref().unwrap().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let bad_gamma = gamma.add(&gamma).unwrap();
    assert!(matches!(
        bad_gamma.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let bad_gamma_tape = input.layer_norm_vjp_tape(&bad_gamma, 0.).unwrap();
    let poisoned = bad_gamma_tape
        .backward(&seed, 1., [false, true, false])
        .unwrap();
    assert!(matches!(
        poisoned[1].as_ref().unwrap().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    let constant = device.upload(&[1, 2], &[1., 1.]).unwrap();
    let degenerate = constant.layer_norm_vjp_tape(&gamma, 0.).unwrap();
    let gradients = degenerate
        .backward(&seed, 1., [true, false, false])
        .unwrap();
    assert!(matches!(
        gradients[0].as_ref().unwrap().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));

    let empty = device.upload(&[0, 2], &[]).unwrap();
    let empty_tape = empty.layer_norm_vjp_tape(&gamma, 0.).unwrap();
    let gradients = empty_tape.backward(&empty, 1., [true; 3]).unwrap();
    assert_eq!(read(gradients[0].as_ref().unwrap()), Vec::<f32>::new());
    close(&read(gradients[1].as_ref().unwrap()), &[0., 0.]);
    close(&read(gradients[2].as_ref().unwrap()), &[0., 0.]);
}

#[test]
fn layer_norm_zero_epsilon_scale_direction_is_null_at_tiny_variance() {
    let Some(device) = device() else { return };
    for scale in [1., 1e-10, 1e-20, 1e-30, f32::from_bits(1)] {
        let input = device.upload(&[1, 3], &[-scale, 0., scale]).unwrap();
        let gamma = device.upload(&[3], &[1.; 3]).unwrap();
        let beta = device.upload(&[3], &[0.; 3]).unwrap();
        let seed = device.upload(&[1, 3], &[-1., 0., 1.]).unwrap();
        let tape = input.layer_norm_affine(&gamma, &beta, 0.).unwrap();
        let [dx, _, _] = tape.backward(&seed, 1., [true, false, false]).unwrap();
        let actual = read(dx.as_ref().unwrap());
        eprintln!("scale-direction nullspace: scale={scale}, dx={actual:?}");
        close(&actual, &[0.; 3]);
    }
}

#[test]
fn layer_norm_wide_rows_input_vjp_matches_f64_with_and_without_cancellation() {
    let Some(device) = device() else { return };
    for cols in [256, 1025] {
        let rows = 2;
        let x: Vec<_> = (0..rows * cols)
            .map(|i| (((i * 37 + 17) % 257) as f32 - 128.) / 64.)
            .collect();
        let beta = vec![0.; cols];
        for scale_direction in [false, true] {
            let gamma: Vec<_> = (0..cols)
                .map(|i| {
                    if scale_direction {
                        1.
                    } else {
                        ((i * 17 % 23) as f32 - 11.) / 8.
                    }
                })
                .collect();
            let seed: Vec<_> = (0..rows * cols)
                .map(|i| {
                    if scale_direction {
                        x[i] * 1e8
                    } else {
                        ((i * 11 % 31) as f32 - 15.) / 16.
                    }
                })
                .collect();
            for epsilon in [1e-5, 1e-9] {
                let expected = reference(&x, &gamma, &beta, &seed, epsilon, 1.);
                let input = device.upload(&[rows, cols], &x).unwrap();
                let gain = device.upload(&[cols], &gamma).unwrap();
                let bias = device.upload(&[cols], &beta).unwrap();
                let upstream = device.upload(&[rows, cols], &seed).unwrap();
                let tape = input.layer_norm_affine(&gain, &bias, epsilon).unwrap();
                let gradients = tape.backward(&upstream, 1., [true; 3]).unwrap();
                for (actual, expected) in gradients.iter().zip(&expected[1..]) {
                    close(&read(actual.as_ref().unwrap()), expected);
                }
            }
        }
    }
}

#[test]
fn layer_norm_wide_rows_input_vjp_stays_stable_near_fast_guard_boundary() {
    let Some(device) = device() else { return };
    for cols in [256, 1025] {
        for offset in [0., 1024.] {
            let rows = 2;
            let x: Vec<_> = (0..rows * cols)
                .map(|i| offset + (((i * 37 + 17) % 257) as f32 - 128.) / 64.)
                .collect();
            let gamma = vec![1.; cols];
            let beta = vec![0.; cols];
            let input = device.upload(&[rows, cols], &x).unwrap();
            let gain = device.upload(&[cols], &gamma).unwrap();
            let bias = device.upload(&[cols], &beta).unwrap();
            let tape = input.layer_norm_affine(&gain, &bias, 1e-5).unwrap();
            for noise_scale in [1e5, 5e5, 1e6, 2e6, 5e6, 1e7] {
                let seed: Vec<_> = (0..rows * cols)
                    .map(|i| {
                        (x[i] - offset) * 1e8
                            + (((i * 11 + 3) % 31) as f32 - 15.) / 16. * noise_scale
                    })
                    .collect();
                let expected = reference(&x, &gamma, &beta, &seed, 1e-5, 1.);
                let upstream = device.upload(&[rows, cols], &seed).unwrap();
                let [dx, _, _] = tape.backward(&upstream, 1., [true, false, false]).unwrap();
                close(&read(dx.as_ref().unwrap()), &expected[1]);
            }
        }
    }
}

#[test]
fn layer_norm_wide_rows_preserve_amplified_subnormal_cotangents() {
    let Some(device) = device() else { return };
    let cols = 256;
    let epsilon = 1e-5f32;
    let tiny = f32::from_bits(1_000_000);
    let seed: Vec<_> = (0..cols)
        .map(|col| if col % 2 == 0 { tiny } else { -tiny })
        .collect();
    let input = device.upload(&[1, cols], &vec![0.; cols]).unwrap();
    let gamma = device.upload(&[cols], &vec![1.; cols]).unwrap();
    let beta = device.upload(&[cols], &vec![0.; cols]).unwrap();
    let upstream = device.upload(&[1, cols], &seed).unwrap();
    let tape = input.layer_norm_affine(&gamma, &beta, epsilon).unwrap();
    let [dx, _, _] = tape.backward(&upstream, 1., [true, false, false]).unwrap();
    let expected = f64::from(tiny) / f64::from(epsilon).sqrt();
    assert!(expected > f64::from(f32::MIN_POSITIVE));
    for (col, actual) in read(dx.as_ref().unwrap()).into_iter().enumerate() {
        let target = if col % 2 == 0 { expected } else { -expected };
        assert!(
            actual.is_finite() && ((f64::from(actual) - target) / target).abs() < 0.01,
            "subnormal cotangent at col {col}: {actual} != {target}"
        );
    }
}

#[test]
fn layer_norm_retains_small_epsilon_after_scale_direction_cancellation() {
    let Some(device) = device() else { return };
    let epsilon = f32::from_bits(1);
    for (scale, cotangent) in [(1e-10f32, f32::MAX), (1e-20, 1.)] {
        let input = device.upload(&[1, 3], &[-scale, 0., scale]).unwrap();
        let gamma = device.upload(&[3], &[1.; 3]).unwrap();
        let beta = device.upload(&[3], &[0.; 3]).unwrap();
        let seed = device
            .upload(&[1, 3], &[-cotangent, 0., cotangent])
            .unwrap();
        let tape = input.layer_norm_affine(&gamma, &beta, epsilon).unwrap();
        let [dx, _, _] = tape.backward(&seed, 1., [true, false, false]).unwrap();
        // Closed-form derivative for this symmetric scale direction. Form the
        // small epsilon term directly, not by subtracting two rounded projections.
        let variance = 2. * f64::from(scale).powi(2) / 3.;
        let magnitude = (f64::from(cotangent) * f64::from(epsilon)
            / (variance + f64::from(epsilon)).powf(1.5)) as f32;
        let actual = read(dx.as_ref().unwrap());
        eprintln!("epsilon scale-direction: scale={scale}, seed={cotangent}, dx={actual:?}, expected={magnitude}");
        close(&actual, &[-magnitude, 0., magnitude]);
    }
}

#[test]
fn layer_norm_preserves_subnormal_epsilon_relative_to_large_variance() {
    let Some(device) = device() else { return };
    let epsilon = f32::from_bits(1);
    for scale in [0.1f32, 1., 2., 10.] {
        assert_scale_direction_epsilon(&device, scale, epsilon);
    }
}

#[test]
fn layer_norm_scale_direction_keeps_epsilon_across_combined_boundary() {
    let Some(device) = device() else { return };
    for exponent in -30..=0 {
        let epsilon = 2f32.powi(exponent);
        for scale in [1f32, 10.] {
            assert_scale_direction_epsilon(&device, scale, epsilon);
        }
    }
}

fn assert_scale_direction_epsilon(device: &TensorDevice, scale: f32, epsilon: f32) {
    let input = device.upload(&[1, 3], &[-scale, 0., scale]).unwrap();
    let gamma = device.upload(&[3], &[1.; 3]).unwrap();
    let beta = device.upload(&[3], &[0.; 3]).unwrap();
    let seed = device.upload(&[1, 3], &[-f32::MAX, 0., f32::MAX]).unwrap();
    let tape = input.layer_norm_affine(&gamma, &beta, epsilon).unwrap();
    let [dx, _, _] = tape.backward(&seed, 1., [true, false, false]).unwrap();
    let actual = read(dx.as_ref().unwrap());
    let variance = 2. * f64::from(scale).powi(2) / 3.;
    let expected = (f64::from(f32::MAX) * f64::from(epsilon)
        / (variance + f64::from(epsilon)).powf(1.5)) as f32;
    assert!(expected.is_finite() && expected > 0.);
    for (got, want) in actual.into_iter().zip([-expected, 0., expected]) {
        assert!(
            (got - want).abs() <= expected.abs() * 0.002,
            "scale={scale}, epsilon={epsilon}: {got} != {want}"
        );
    }
}

#[test]
fn layer_norm_review_large_dynamic_range_vjp_matches_f64() {
    let Some(device) = device() else { return };
    let x = [0., -1e10, 1192.0929, 0.];
    let g = [1., 1e20, 1., 1.];
    let seed = [1.; 4];
    let expected = reference(&x, &g, &[0.; 4], &seed, 1e-5, 1.);
    let input = device.upload(&[1, 4], &x).unwrap();
    let gamma = device.upload(&[4], &g).unwrap();
    let beta = device.upload(&[4], &[0.; 4]).unwrap();
    let seed = device.upload(&[1, 4], &seed).unwrap();
    let tape = input.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
    close(&read(tape.value()), &expected[0]);
    let grads = tape.backward(&seed, 1., [true; 3]).unwrap();
    eprintln!(
        "review dynamic-range: dx={:?}, reference={:?}",
        read(grads[0].as_ref().unwrap()),
        expected[1]
    );
    for i in 0..3 {
        close(&read(grads[i].as_ref().unwrap()), &expected[i + 1]);
    }
}

#[test]
fn layer_norm_dynamic_range_vjp_is_permutation_and_seed_scale_equivariant() {
    let Some(device) = device() else { return };
    // 100-digit Decimal oracle on exact f32 inputs, reproduced by
    // benchmarks/layer-norm-resident/decimal_probe.py. The ordinary f64
    // projection itself loses enough bits to be unsuitable for scaled variants.
    let expected = [
        -917.6735572182624,
        0.00021879039951844405,
        1835.3468956461253,
        -917.6735572182624,
    ];
    for shift in 0..4 {
        let mut x = [0., -1e10, 1192.0929, 0.];
        let mut g = [1., 1e20, 1., 1.];
        x.rotate_left(shift);
        g.rotate_left(shift);
        let input = device.upload(&[1, 4], &x).unwrap();
        let gamma = device.upload(&[4], &g).unwrap();
        let beta = device.upload(&[4], &[0.; 4]).unwrap();
        let tape = input.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
        for scale in [1. / 1024., -0.125, 1., 16., 1024.] {
            let seed = device.upload(&[1, 4], &[scale; 4]).unwrap();
            let grads = tape.backward(&seed, 1., [true, false, false]).unwrap();
            assert!(grads[1].is_none() && grads[2].is_none());
            let mut expected = expected.map(|v| (v * f64::from(scale)) as f32);
            expected.rotate_left(shift);
            close(&read(grads[0].as_ref().unwrap()), &expected);
        }
    }
}
