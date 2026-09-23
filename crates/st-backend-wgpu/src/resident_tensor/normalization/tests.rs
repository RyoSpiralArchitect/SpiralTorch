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
