use st_tensor::{LayerNormBackend, Tensor, TensorError};

fn check(input: &[f32], residual: Option<&[f32]>, epsilon: f32, backend: LayerNormBackend) {
    let cols = input.len();
    let gamma: Vec<_> = (0..cols).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
    let beta: Vec<_> = (0..cols).map(|i| (i % 3) as f32 * 0.125).collect();
    let values: Vec<f64> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| f64::from(x + residual.map_or(0.0, |r| r[i])))
        .collect();
    let mean = values.iter().sum::<f64>() / cols as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / cols as f64;
    let denominator = (variance + f64::from(epsilon)).sqrt();
    let expected: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(i, x)| ((x - mean) / denominator) as f32 * gamma[i] + beta[i])
        .collect();
    let tensor = Tensor::from_vec(1, cols, input.to_vec()).unwrap();
    let gamma = Tensor::from_vec(1, cols, gamma).unwrap();
    let beta = Tensor::from_vec(1, cols, beta).unwrap();
    let output = match residual {
        Some(residual) => tensor.layer_norm_affine_add_with_backend(
            &Tensor::from_vec(1, cols, residual.to_vec()).unwrap(),
            &gamma,
            &beta,
            epsilon,
            backend,
        ),
        None => tensor.layer_norm_affine_with_backend(&gamma, &beta, epsilon, backend),
    }
    .unwrap_or_else(|error| {
        panic!(
            "{backend:?}, cols={cols}, first={}, epsilon={epsilon}: {error}",
            input[0]
        )
    });
    for (i, (&actual, expected)) in output.data().iter().zip(expected).enumerate() {
        let tolerance = 3.0e-5 * (1.0 + expected.abs());
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{backend:?}, cols={cols}, index={i}: {actual} != {expected}"
        );
    }
}

fn cases(backend: LayerNormBackend) {
    for offset in [0.0, 1.0e4, 1.0e7, -1.0e7] {
        check(
            &[offset, offset + 1.0, offset + 2.0, offset + 3.0],
            None,
            1.0e-5,
            backend,
        );
    }
    for cols in [1, 3, 255, 256, 257, 513, 1025, 8193] {
        let input: Vec<_> = (0..cols).map(|i| 1.0e7 + (i % 7) as f32).collect();
        check(&input, None, 1.0e-5, backend);
        let mut outlier = vec![1.0e7; cols];
        outlier[cols - 1] += 2.0;
        check(&outlier, None, 1.0e-5, backend);
    }
    check(&[f32::MAX, -f32::MAX], None, 0.0, backend);
    check(&[f32::MAX, f32::MAX], None, 1.0e-5, backend);
    check(
        &[f32::MAX, -f32::MAX, 0.0, f32::MAX / 4.0],
        None,
        f32::MAX,
        backend,
    );
    check(&[1e-30, 2e-30, 3e-30, 4e-30], None, 0.0, backend);
    check(&[7.0, 7.0, 7.0], None, f32::from_bits(1), backend);
    check(
        &[1e7, 1e7, 1e7, 1e7],
        Some(&[0.0, 1.0, 2.0, 3.0]),
        1e-5,
        backend,
    );
}

#[test]
fn cpu_layer_norm_preserves_variance_across_offsets_widths_and_scales() {
    cases(LayerNormBackend::Cpu);
}

#[test]
fn training_stats_share_forward_moments_and_logical_layout() {
    let input = Tensor::from_vec(
        2,
        4,
        vec![1e7, 1e7 + 1.0, 1e7 + 2.0, 1e7 + 3.0, 4.0, 3.0, 2.0, 1.0],
    )
    .unwrap();
    let gamma = Tensor::from_vec(1, 4, vec![1.0; 4]).unwrap();
    let beta = Tensor::zeros(1, 4).unwrap();
    let forward = input
        .layer_norm_affine_with_backend(&gamma, &beta, 1e-5, LayerNormBackend::Cpu)
        .unwrap();
    for layout in [st_tensor::Layout::RowMajor, st_tensor::Layout::ColMajor] {
        let (normalized, inverse_std) = input
            .to_layout(layout)
            .unwrap()
            .layer_norm_stats(1e-5)
            .unwrap();
        assert_eq!(normalized.data(), forward.data());
        assert_eq!(inverse_std.shape(), (2, 1));
        for value in inverse_std.data() {
            assert!((value - (1.25001_f64.sqrt().recip() as f32)).abs() < 1e-7);
        }
    }
    let (normalized, inverse_std) = Tensor::zeros(0, 4).unwrap().layer_norm_stats(1e-5).unwrap();
    assert_eq!(normalized.shape(), (0, 4));
    assert_eq!(inverse_std.shape(), (0, 1));
    let tiny = f32::from_bits(1);
    let input = Tensor::from_vec(1, 2, vec![-tiny, tiny]).unwrap();
    assert!(
        input.layer_norm_stats(0.0).is_err(),
        "unrepresentable inverse std must not escape"
    );
}

#[test]
fn layer_norm_rejects_degenerate_and_overflowed_fused_values() {
    let gamma = Tensor::from_vec(1, 2, vec![1.0, 1.0]).unwrap();
    let beta = Tensor::zeros(1, 2).unwrap();
    for backend in [LayerNormBackend::Cpu, LayerNormBackend::GpuWgpu] {
        let constant = Tensor::from_vec(1, 2, vec![2.0, 2.0]).unwrap();
        assert!(matches!(
            constant.layer_norm_affine_with_backend(&gamma, &beta, 0.0, backend),
            Err(TensorError::InvalidValue {
                label: "layer_norm_zero_variance_without_epsilon"
            })
        ));
        let large = Tensor::from_vec(1, 2, vec![f32::MAX, -f32::MAX]).unwrap();
        assert!(matches!(
            large.layer_norm_affine_add_with_backend(&large, &gamma, &beta, 1e-5, backend),
            Err(TensorError::NonFiniteValue {
                label: "layer_norm_value",
                value,
            }) if value.is_infinite()
        ));
        for epsilon in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                large.layer_norm_affine_with_backend(&gamma, &beta, epsilon, backend),
                Err(TensorError::NonFiniteValue {
                    label: "layernorm_epsilon",
                    ..
                })
            ));
        }
    }
}

#[cfg(feature = "wgpu_dense")]
#[test]
fn wgpu_layer_norm_preserves_variance_without_cpu_fallback() {
    use std::sync::{Arc, Mutex};
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        eprintln!("set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 to exercise the real WGPU kernel");
        return;
    }
    assert!(st_tensor::wgpu_dense::is_available());
    let receipts = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&receipts);
    let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(
        move |event: &st_tensor::TensorOpMetaEvent| {
            if event.op_name == "layer_norm" && event.data.get("execution_receipt").is_some() {
                captured
                    .lock()
                    .unwrap()
                    .push(event.data["execution_receipt"].clone());
            }
        },
    )));
    cases(LayerNormBackend::GpuWgpu);
    st_tensor::set_thread_meta_observer(previous);
    let receipts = receipts.lock().unwrap();
    assert_eq!(receipts.len(), 26);
    for receipt in receipts.iter() {
        assert_eq!(receipt["executed_backend"], "wgpu");
        assert_eq!(receipt["route_status"], "direct");
    }
}
