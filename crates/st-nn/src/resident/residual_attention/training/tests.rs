use super::*;

fn row(width: usize, value: f32) -> Tensor {
    Tensor::from_vec(1, width, vec![value; width]).unwrap()
}

fn plan() -> ResidualAttentionPlan {
    let layout = NdLayout::contiguous(&[2, 3, 4]).unwrap();
    let norm = || InferenceOp::LayerNorm {
        gain: row(4, 1.),
        bias: row(4, 0.),
        epsilon: 1e-5,
    };
    let pre = InferencePlan::from_operations(layout.clone(), vec![norm()]).unwrap();
    let weight = Tensor::from_vec(4, 4, vec![0.1; 16]).unwrap();
    let bias = row(4, 0.);
    let attention = AttentionInferencePlan::from_parameters(
        layout.clone(),
        2,
        AttentionMask::Causal { query_offset: 0 },
        [(&weight, &bias); 4],
    )
    .unwrap();
    let feed = InferencePlan::from_operations(
        layout,
        vec![
            norm(),
            InferenceOp::Linear {
                weight: Tensor::from_vec(4, 6, vec![0.2; 24]).unwrap(),
                bias: row(6, 0.),
            },
            InferenceOp::Gelu,
            InferenceOp::ToposResonator {
                gate: row(6, 0.8),
                kernel: ToposResonatorKernel::new(0.2, 0.12, 0.3, 4).unwrap(),
                max_volume: 36,
            },
            InferenceOp::Linear {
                weight: Tensor::from_vec(6, 4, vec![0.15; 24]).unwrap(),
                bias: row(4, 0.),
            },
        ],
    )
    .unwrap();
    ResidualAttentionPlan::from_plans(&pre, &attention, &feed).unwrap()
}

fn read(t: &ResidentTensor) -> Vec<f32> {
    t.snapshot().unwrap().read().unwrap()
}

#[test]
fn one_candidate_overflow_in_any_parameter_rejects_the_entire_block() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("residual.attention.atomicity")
            .unwrap();
    assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let plan = plan();
    let input = device
        .upload(
            &[2, 3, 4],
            &(0..24).map(|i| i as f32 * 0.03).collect::<Vec<_>>(),
        )
        .unwrap();
    let seed = device.upload(&[2, 3, 4], &[1.; 24]).unwrap();
    for failing in 0..13 {
        let mut model = plan.compile_training_wgpu(runtime.clone()).unwrap();
        let forward = model.forward(&input, None, None).unwrap();
        let original = read(forward.prediction());
        let mut gradients = model.backward(&forward, &seed).unwrap();
        let before = model.parameter_snapshot();
        assert_eq!(before.values().len(), 13);
        // Fault injection at the private binding boundary isolates one candidate
        // overflow; all other finite candidates would change their parameter.
        gradients.parameters = before
            .values()
            .iter()
            .enumerate()
            .map(|(i, value)| {
                device
                    .upload(
                        value.layout().shape(),
                        &vec![if i == failing { 2. } else { 0.25 }; value.layout().len()],
                    )
                    .unwrap()
            })
            .collect();
        gradients.bound = before.bind_gradients(gradients.parameters.clone()).unwrap();
        let rejected = model.sgd(&gradients, f32::MAX).unwrap();
        assert!(matches!(
            rejected.snapshot().unwrap().read(),
            Err(TrainingError::Rejected { .. })
        ));
        assert_eq!(model.parameter_snapshot().revision(), 1);
        for (i, (a, b)) in before
            .values()
            .iter()
            .zip(model.parameter_snapshot().values())
            .enumerate()
        {
            let old = read(a);
            let candidate: Vec<_> = old
                .iter()
                .map(|p| p - f32::MAX * if i == failing { 2. } else { 0.25 })
                .collect();
            assert_eq!(candidate.iter().any(|p| !p.is_finite()), i == failing);
            assert!(candidate
                .iter()
                .zip(&old)
                .any(|(a, b)| a.to_bits() != b.to_bits()));
            assert!(
                old.iter()
                    .map(|v| v.to_bits())
                    .eq(read(b).iter().map(|v| v.to_bits())),
                "parameter {i}, fault {failing}"
            );
        }
        assert!(matches!(
            model.backward(&forward, &seed),
            Err(InferenceError::Training(TrainingError::StaleForward))
        ));
        assert!(matches!(
            model.sgd(&gradients, 0.),
            Err(InferenceError::Training(TrainingError::ParameterVersion))
        ));
        let recovered = model.forward(&input, None, None).unwrap();
        assert_eq!(read(recovered.prediction()), original);
        let valid = model.backward(&recovered, &seed).unwrap();
        assert_eq!(
            model
                .sgd(&valid, 0.)
                .unwrap()
                .snapshot()
                .unwrap()
                .read()
                .unwrap(),
            2
        );
    }
}

#[test]
fn parameter_free_branches_preserve_identity_residual_and_rebind_after_update() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
        "residual.attention.parameter_free",
    )
    .unwrap();
    assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
    let layout = NdLayout::contiguous(&[1, 2, 2]).unwrap();
    let branch = InferencePlan::from_operations(layout.clone(), vec![InferenceOp::Relu]).unwrap();
    let weight = Tensor::zeros(2, 2).unwrap();
    let bias = row(2, 0.);
    let attention = AttentionInferencePlan::from_parameters(
        layout,
        1,
        AttentionMask::None,
        [(&weight, &bias); 4],
    )
    .unwrap();
    let plan = ResidualAttentionPlan::from_plans(&branch, &attention, &branch).unwrap();
    let mut model = plan.compile_training_wgpu(runtime).unwrap();
    assert_eq!(model.parameter_snapshot().values().len(), 4);
    let device = model.tensor_device();
    let input = device.upload(&[1, 2, 2], &[-1., 2., 3., -4.]).unwrap();
    let seed = device.upload(&[1, 2, 2], &[1., 2., 3., 4.]).unwrap();
    let forward = model.forward(&input, None, None).unwrap();
    assert_eq!(read(forward.prediction()), [-1., 4., 6., -4.]);
    let gradients = model.backward(&forward, &seed).unwrap();
    assert_eq!(read(gradients.input_gradient()), [1., 4., 6., 4.]);
    assert_eq!(read(&gradients.parameter_gradients()[3]), [7., 8.]);
    assert_eq!(
        model
            .sgd(&gradients, 0.25)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        1
    );
    let forward = model.forward(&input, None, None).unwrap();
    assert_eq!(read(forward.prediction()), [-2.75, 0., 2.5, -6.]);
}

fn overflow_model(runtime: WgpuRuntime, output_failure: bool) -> ResidentResidualAttentionTraining {
    let layout = NdLayout::contiguous(&[1, 1, 1]).unwrap();
    let linear = |weight, bias| InferenceOp::Linear {
        weight: row(1, weight),
        bias: row(1, bias),
    };
    let pre = InferencePlan::from_operations(layout.clone(), vec![linear(1., 0.)]).unwrap();
    let zero = row(1, 0.);
    let one = row(1, 1.);
    let huge = row(1, if output_failure { 2e38 } else { 0. });
    let attention = AttentionInferencePlan::from_parameters(
        layout.clone(),
        1,
        AttentionMask::None,
        [(&zero, &zero), (&zero, &zero), (&one, &zero), (&one, &huge)],
    )
    .unwrap();
    let feed = InferencePlan::from_operations(
        layout,
        vec![linear(0., if output_failure { 2e38 } else { 0. })],
    )
    .unwrap();
    ResidualAttentionPlan::from_plans(&pre, &attention, &feed)
        .unwrap()
        .compile_training_wgpu(runtime)
        .unwrap()
}

fn terminal_failure(output_failure: bool) {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
        "residual.attention.terminal_failure",
    )
    .unwrap();
    assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
    let mut model = overflow_model(runtime, output_failure);
    let input = model.tensor_device().upload(&[1, 1, 1], &[0.]).unwrap();
    let seed = model
        .tensor_device()
        .upload(&[1, 1, 1], &[if output_failure { 1. } else { 2e38 }])
        .unwrap();
    let before = model.parameter_snapshot();
    let z = model.tensor_device().upload(&[1, 1, 1], &[0.]).unwrap();
    let pair = model.tensor_device().upload(&[1, 1, 1, 1], &[0.]).unwrap();
    let forward = model.forward(&input, Some(&z), Some(&pair)).unwrap();
    let prediction = forward.prediction().snapshot().unwrap().read();
    assert_eq!(prediction.is_err(), output_failure);
    let gradients = model.backward(&forward, &seed).unwrap();
    assert!(matches!(
        gradients.input_gradient().snapshot().unwrap().read(),
        Err(st_backend_wgpu::resident_tensor::TensorError::NonFinite)
    ));
    let rejected = model.sgd(&gradients, 0.01).unwrap();
    assert!(matches!(
        rejected.snapshot().unwrap().read(),
        Err(TrainingError::Rejected { .. })
    ));
    assert!(gradients.z_bias_gradient().is_some());
    assert!(gradients.pair_bias_gradient().is_some());
    for gradient in gradients
        .parameter_gradients()
        .iter()
        .chain(gradients.z_bias_gradient())
        .chain(gradients.pair_bias_gradient())
    {
        assert!(matches!(
            gradient.snapshot().unwrap().read(),
            Err(st_backend_wgpu::resident_tensor::TensorError::NonFinite)
        ));
    }
    for (before, after) in before
        .values()
        .iter()
        .zip(model.parameter_snapshot().values())
    {
        assert!(read(before)
            .iter()
            .map(|v| v.to_bits())
            .eq(read(after).iter().map(|v| v.to_bits())));
    }
}

#[test]
fn terminal_prediction_overflow_invalidates_vjp_and_update() {
    terminal_failure(true);
}

#[test]
fn terminal_input_gradient_overflow_invalidates_vjp_and_update() {
    terminal_failure(false);
}
