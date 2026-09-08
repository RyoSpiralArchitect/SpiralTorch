use super::*;

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
            "{label}[{i}]: {a} != {b}"
        );
    }
}

#[test]
fn transposed_vjps_cover_rectangular_tiles_and_partial_edges_when_enabled() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("training.transpose").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    for (rows, inner, cols) in [(3, 5, 7), (17, 31, 19), (65, 29, 11)] {
        let input: Vec<_> = (0..rows * inner)
            .map(|i| (i % 11) as f32 / 16. - 0.3125)
            .collect();
        let target: Vec<_> = (0..rows * cols)
            .map(|i| (i % 7) as f32 / 8. - 0.375)
            .collect();
        let layer = DenseLayer {
            inner,
            cols,
            weights: (0..inner * cols)
                .map(|i| (i % 17) as f32 / 32. - 0.25)
                .collect(),
            bias: (0..cols).map(|i| (i % 3) as f32 / 32.).collect(),
            activation: DenseActivation::None,
        };
        let prediction: Vec<_> = (0..rows * cols)
            .map(|i| {
                (0..inner)
                    .map(|k| input[i / cols * inner + k] * layer.weights[k * cols + i % cols])
                    .sum::<f32>()
                    + layer.bias[i % cols]
            })
            .collect();
        let delta: Vec<_> = prediction
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t) * (2. / target.len() as f32))
            .collect();
        let dw: Vec<_> = (0..inner * cols)
            .map(|i| {
                (0..rows)
                    .map(|r| input[r * inner + i / cols] * delta[r * cols + i % cols])
                    .sum::<f32>()
            })
            .collect();
        let db: Vec<_> = (0..cols)
            .map(|c| (0..rows).map(|r| delta[r * cols + c]).sum::<f32>())
            .collect();
        let dx: Vec<_> = (0..rows * inner)
            .map(|i| {
                (0..cols)
                    .map(|c| delta[i / inner * cols + c] * layer.weights[i % inner * cols + c])
                    .sum::<f32>()
            })
            .collect();
        let updated_w: Vec<_> = layer
            .weights
            .iter()
            .zip(&dw)
            .map(|(w, g)| w - 0.03125 * g)
            .collect();
        let updated_b: Vec<_> = layer
            .bias
            .iter()
            .zip(&db)
            .map(|(b, g)| b - 0.03125 * g)
            .collect();
        for [m, n, k] in [[8, 8, 16], [4, 16, 8], [16, 4, 32], [8, 32, 16]] {
            for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
                for accumulation in [
                    MatmulAccumulation::Sequential,
                    MatmulAccumulation::Tiled,
                    MatmulAccumulation::Compensated,
                ] {
                    let mut gpu = ResidentDenseTraining::new(
                        runtime.clone(),
                        NdLayout::contiguous(&[1, rows, inner]).unwrap(),
                        std::slice::from_ref(&layer),
                        MatmulTile::new(m, n, k).unwrap(),
                        kernel,
                        accumulation,
                    )
                    .unwrap();
                    gpu.upload_batch(&input, &target).unwrap();
                    gpu.step(0.03125).unwrap();
                    let state = gpu.state_snapshot().unwrap().read().unwrap();
                    close(&state.prediction, &prediction, "prediction");
                    close(&state.input_gradient, &dx, "dInput");
                    close(&state.parameter_gradients[0].weights, &dw, "dWeight");
                    close(&state.parameter_gradients[0].bias, &db, "dBias");
                    close(&state.parameters[0].weights, &updated_w, "weight after SGD");
                    close(&state.parameters[0].bias, &updated_b, "bias after SGD");
                }
            }
        }
    }
}

#[test]
fn forward_specializations_are_reused_without_changing_mixed_activation_semantics() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("training.stages").unwrap();
    let plain = DenseLayer {
        inner: 1,
        cols: 1,
        weights: vec![1.],
        bias: vec![0.],
        activation: DenseActivation::None,
    };
    let mut gelu = plain.clone();
    gelu.activation = DenseActivation::Gelu;
    for layers in [
        vec![plain.clone()],
        vec![gelu.clone()],
        vec![plain.clone(), gelu.clone(), plain, gelu],
    ] {
        let mut gpu = ResidentDenseTraining::new(
            runtime.clone(),
            NdLayout::contiguous(&[1, 2, 1]).unwrap(),
            &layers,
            MatmulTile::default(),
            MatmulKernel::Register2x2,
            MatmulAccumulation::Sequential,
        )
        .unwrap();
        if layers.len() == 4 {
            assert!(Shared::ptr_eq(
                &gpu.passes[0].pipeline,
                &gpu.passes[2].pipeline
            ));
            assert!(Shared::ptr_eq(
                &gpu.passes[1].pipeline,
                &gpu.passes[3].pipeline
            ));
            assert!(!Shared::ptr_eq(
                &gpu.passes[0].pipeline,
                &gpu.passes[1].pipeline
            ));
        }
        let mut expected = [-1f32, 1f32];
        for layer in &layers {
            if layer.activation == DenseActivation::Gelu {
                for x in &mut expected {
                    let inner = 0.7978846 * (*x + 0.044715 * *x * *x * *x);
                    *x = 0.5 * *x * (1. + inner.tanh());
                }
            }
        }
        gpu.upload_batch(&[-1., 1.], &[0., 0.]).unwrap();
        gpu.step(0.).unwrap();
        close(
            &gpu.state_snapshot().unwrap().read().unwrap().prediction,
            &expected,
            "mixed activations",
        );
    }
}
