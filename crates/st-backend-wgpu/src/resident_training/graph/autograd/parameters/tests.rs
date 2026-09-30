use super::*;

#[test]
fn resident_parameter_rebinding_invalidates_tape_and_preserves_guards() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.parameters").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[1, 2]).unwrap(),
        vec![GraphStage::Linear {
            weight: 0,
            bias: 1,
            gelu: false,
        }],
        vec![
            GraphParameter {
                role: ParameterRole::Weight,
                shape: vec![2, 2],
                values: vec![1., 0., 0., 1.],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![2],
                values: vec![0.; 2],
            },
        ],
    )
    .unwrap();
    let mut graph = ResidentGraphAutograd::new(
        runtime,
        definition,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap();
    graph.upload(&[1., 2.]).unwrap();
    let device = graph.tensor_device().clone();
    let seed = device.upload(&[1, 2], &[1., 1.]).unwrap();
    let original = graph.forward().unwrap();
    assert!(matches!(
        graph.set_parameter_tensors(&[]),
        Err(TrainingError::ParameterLayout)
    ));
    graph.backward(&original, &seed).unwrap();
    let replacement = device.upload(&[2, 2], &[2.0; 4]).unwrap();
    assert!(matches!(
        graph.set_parameter_tensors(&[
            replacement.clone(),
            device.upload(&[1, 2], &[0.0; 2]).unwrap(),
        ]),
        Err(TrainingError::ParameterLayout)
    ));
    let other = TensorDevice::new(
        pollster::block_on(WgpuRuntime::request_headless("graph.parameters.other")).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        graph.set_parameter_tensors(&[replacement, other.upload(&[2], &[0.0; 2]).unwrap()]),
        Err(TrainingError::Tensor(
            crate::resident_tensor::TensorError::DeviceMismatch
        ))
    ));
    let preserved = graph.backward(&original, &seed).unwrap();
    assert_eq!(
        preserved
            .input_gradient()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![1.0; 2]
    );
    let weight = device
        .upload(&[2, 2], &[2., 4., 3., 5.])
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    let bias = device
        .upload(&[3], &[99., 1., -1.])
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
    graph
        .set_parameter_tensors(&[weight.clone(), bias.clone()])
        .unwrap();
    assert!(matches!(
        graph.backward(&original, &seed),
        Err(TrainingError::StaleForward)
    ));
    let forward = graph.forward().unwrap();
    assert_eq!(
        forward.prediction().snapshot().unwrap().read().unwrap(),
        vec![11., 12.]
    );
    let gradients = graph.backward(&forward, &seed).unwrap();
    assert_eq!(
        gradients
            .input_gradient()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![5., 9.]
    );
    assert_eq!(
        original.prediction().snapshot().unwrap().read().unwrap(),
        vec![1., 2.]
    );
    let bad = device
        .upload(&[2], &[-f32::MAX; 2])
        .unwrap()
        .mul(&device.upload(&[2], &[2.; 2]).unwrap())
        .unwrap()
        .relu()
        .unwrap();
    graph.set_parameter_tensors(&[weight.clone(), bad]).unwrap();
    let invalid = graph.forward().unwrap();
    assert!(invalid.prediction().snapshot().unwrap().read().is_err());
    let bad_gradients = graph.backward(&invalid, &seed).unwrap();
    for tensor in
        std::iter::once(bad_gradients.input_gradient()).chain(bad_gradients.parameter_gradients())
    {
        assert!(tensor.snapshot().unwrap().read().is_err());
    }
    graph.set_parameter_tensors(&[weight, bias]).unwrap();
    let recovered = graph.forward().unwrap();
    assert_eq!(
        recovered.prediction().snapshot().unwrap().read().unwrap(),
        vec![11., 12.]
    );
    assert_eq!(
        gradients
            .input_gradient()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![5., 9.]
    );
}

#[test]
fn resident_parameter_guards_survive_layer_norm_and_recover() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.parameters.norm").unwrap();
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[1, 2]).unwrap(),
        vec![GraphStage::LayerNorm {
            gain: 0,
            bias: 1,
            epsilon: 1e-3,
        }],
        vec![
            GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![2],
                values: vec![1.0; 2],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![2],
                values: vec![0.0; 2],
            },
        ],
    )
    .unwrap();
    let mut graph = ResidentGraphAutograd::new(
        runtime,
        definition,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap();
    graph.upload(&[1.0, 2.0]).unwrap();
    let device = graph.tensor_device().clone();
    let seed = device.upload(&[1, 2], &[0.5, 1.0]).unwrap();
    let gain = device.upload(&[2], &[2.0; 2]).unwrap();
    let bad_bias = device
        .upload(&[2], &[-f32::MAX; 2])
        .unwrap()
        .mul(&gain)
        .unwrap()
        .relu()
        .unwrap();
    graph
        .set_parameter_tensors(&[gain.clone(), bad_bias])
        .unwrap();
    let bad = graph.forward().unwrap();
    assert!(bad.prediction().snapshot().unwrap().read().is_err());
    let vjp = graph.backward(&bad, &seed).unwrap();
    for tensor in std::iter::once(vjp.input_gradient()).chain(vjp.parameter_gradients()) {
        assert!(tensor.snapshot().unwrap().read().is_err());
    }
    graph
        .set_parameter_tensors(&[gain, device.upload(&[2], &[0.0; 2]).unwrap()])
        .unwrap();
    let good = graph.forward().unwrap();
    assert!(good.prediction().snapshot().unwrap().read().is_ok());
    let vjp = graph.backward(&good, &seed).unwrap();
    for tensor in std::iter::once(vjp.input_gradient()).chain(vjp.parameter_gradients()) {
        assert!(tensor.snapshot().unwrap().read().is_ok());
    }
}
