use super::*;
use crate::resident_graph::{GraphInferenceError, ResidentGraph};
use st_kernel_contracts::pointwise::{PointwiseChain, PointwiseStep};
use st_kernel_contracts::{gradient_clip::GlobalNormClip, momentum::EmaMomentum};

fn runtime() -> Option<WgpuRuntime> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.layer_norm.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(runtime)
}

fn definition(epsilon: f32) -> GraphDefinition {
    GraphDefinition::new(
        NdLayout::contiguous(&[2, 3]).unwrap(),
        vec![GraphStage::LayerNorm {
            gain: 0,
            bias: 1,
            epsilon,
        }],
        vec![
            GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![3],
                values: vec![1.0, 0.5, 1.5],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![3],
                values: vec![0.25, -0.5, 0.75],
            },
        ],
    )
    .unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "index={index} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn layer_norm_graph_matches_standalone_forward_vjp_and_sgd() {
    let Some(runtime) = runtime() else { return };
    let definition = definition(1e-5);
    let values = [1.0, 2.0, 4.0, 2.0, 0.0, -1.0];
    let target = [0.0; 6];
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let x = device.upload(&[2, 3], &values).unwrap();
    let gain = device
        .upload(&[3], &definition.parameters()[0].values)
        .unwrap();
    let bias = device
        .upload(&[3], &definition.parameters()[1].values)
        .unwrap();
    let standalone = x.layer_norm_affine(&gain, &bias, 1e-5).unwrap();
    let expected = standalone.value().snapshot().unwrap().read().unwrap();

    let mut inference = ResidentGraph::new(
        runtime.clone(),
        definition.clone(),
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    inference.upload(&values).unwrap();
    inference.dispatch().unwrap();
    assert_close(
        &inference.snapshot().unwrap().read().unwrap(),
        &expected,
        1e-6,
    );
    let owned = inference.forward_tensor(&x).unwrap();
    assert_close(&owned.snapshot().unwrap().read().unwrap(), &expected, 1e-6);

    let mut autograd = ResidentGraphAutograd::new(
        runtime.clone(),
        definition.clone(),
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    autograd.upload(&values).unwrap();
    let forward = autograd.forward().unwrap();
    assert_close(
        &forward.prediction().snapshot().unwrap().read().unwrap(),
        &expected,
        1e-6,
    );
    let seed: Vec<_> = expected.iter().map(|&value| value / 3.0).collect();
    let cotangent = device.upload(&[2, 3], &seed).unwrap();
    let graph_vjp = autograd.backward(&forward, &cotangent).unwrap();
    let reference_vjp = standalone.backward(&cotangent, 1.0, [true; 3]).unwrap();
    let input_vjp = reference_vjp[0]
        .as_ref()
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    let gain_vjp = reference_vjp[1]
        .as_ref()
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    let bias_vjp = reference_vjp[2]
        .as_ref()
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert_close(
        &graph_vjp
            .input_gradient()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &input_vjp,
        1e-5,
    );
    assert_close(
        &graph_vjp.parameter_gradients()[0]
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &gain_vjp,
        1e-5,
    );
    assert_close(
        &graph_vjp.parameter_gradients()[1]
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &bias_vjp,
        1e-5,
    );
    let held = graph_vjp.input_gradient().snapshot().unwrap();
    let zero = device.upload(&[2, 3], &[0.0; 6]).unwrap();
    autograd.backward(&forward, &zero).unwrap();
    assert_close(&held.read().unwrap(), &input_vjp, 1e-5);

    let mut training = ResidentGraphTraining::new(
        runtime,
        definition.clone(),
        GraphGradientPolicy::Exact,
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    training.upload_batch(&values, &target).unwrap();
    assert_eq!(training.step(0.1).unwrap(), 1);
    let state = training.state_snapshot().unwrap().read().unwrap();
    assert_close(&state.prediction, &expected, 1e-6);
    assert_close(&state.input_gradient, &input_vjp, 1e-5);
    assert_close(&state.raw_gradients[0], &gain_vjp, 1e-5);
    assert_close(&state.raw_gradients[1], &bias_vjp, 1e-5);
    for ((before, gradient), after) in definition
        .parameters()
        .iter()
        .zip(&state.raw_gradients)
        .zip(state.graph.parameters())
    {
        let expected: Vec<_> = before
            .values
            .iter()
            .zip(gradient)
            .map(|(&value, &gradient)| value - 0.1 * gradient)
            .collect();
        assert_close(&after.values, &expected, 1e-5);
    }

    let mut compatible = ResidentGraphTraining::new(
        device.runtime().clone(),
        definition,
        GraphGradientPolicy::ModuleCompatible,
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    compatible.upload_batch(&values, &target).unwrap();
    compatible.step(0.1).unwrap();
    let compatible = compatible.state_snapshot().unwrap().read().unwrap();
    assert_close(&compatible.raw_gradients[0], &gain_vjp, 1e-5);
    assert_close(
        &compatible.effective_gradients[0],
        &gain_vjp.iter().map(|&v| v / 2.).collect::<Vec<_>>(),
        1e-5,
    );
    assert_close(
        &compatible.effective_gradients[1],
        &bias_vjp.iter().map(|&v| v / 2.).collect::<Vec<_>>(),
        1e-5,
    );
}

#[test]
fn layer_norm_graph_guard_reports_the_correct_stage() {
    let Some(runtime) = runtime() else { return };
    let original = definition(0.0);
    let mut graph = ResidentGraph::new(
        runtime,
        GraphDefinition::new(
            original.input_layout().clone(),
            vec![
                GraphStage::Pointwise {
                    chain: PointwiseChain::new(
                        1,
                        vec![PointwiseStep::named("relu", None).unwrap()],
                    )
                    .unwrap(),
                    parameters: vec![],
                },
                original.stages()[0].clone(),
            ],
            original.parameters().to_vec(),
        )
        .unwrap(),
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    graph.upload(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
    graph.dispatch().unwrap();
    assert!(matches!(
        graph.snapshot().unwrap().read(),
        Err(GraphInferenceError::NonFinite { stage: 1, .. })
    ));
}

#[test]
fn dense_then_layer_norm_keeps_forward_and_training_resident() {
    let Some(runtime) = runtime() else { return };
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[2, 3]).unwrap(),
        vec![
            GraphStage::Linear {
                weight: 0,
                bias: 1,
                gelu: false,
            },
            GraphStage::LayerNorm {
                gain: 2,
                bias: 3,
                epsilon: 1e-5,
            },
        ],
        vec![
            GraphParameter {
                role: ParameterRole::Weight,
                shape: vec![3, 4],
                values: vec![1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1.],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![4],
                values: vec![0.; 4],
            },
            GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![4],
                values: vec![1.; 4],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![4],
                values: vec![0.; 4],
            },
        ],
    )
    .unwrap();
    let input = [1., 2., 3., 4., 5., 6.];
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let linear = device
        .upload(&[2, 4], &[1., 2., 3., 6., 4., 5., 6., 15.])
        .unwrap();
    let gain = device.upload(&[4], &[1.; 4]).unwrap();
    let bias = device.upload(&[4], &[0.; 4]).unwrap();
    let expected = linear
        .layer_norm_affine(&gain, &bias, 1e-5)
        .unwrap()
        .value()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();

    let mut inference = ResidentGraph::new(
        runtime.clone(),
        definition.clone(),
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    inference.upload(&input).unwrap();
    inference.dispatch().unwrap();
    assert_close(
        &inference.snapshot().unwrap().read().unwrap(),
        &expected,
        1e-5,
    );
    let direct = inference
        .forward_tensor(&device.upload(&[2, 3], &input).unwrap())
        .unwrap();
    assert_close(&direct.snapshot().unwrap().read().unwrap(), &expected, 1e-5);

    let mut training = ResidentGraphTraining::new(
        runtime,
        definition,
        GraphGradientPolicy::Exact,
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    training.upload_batch(&input, &[0.; 8]).unwrap();
    training.step(0.01).unwrap();
    let first = training.state_snapshot().unwrap().read().unwrap();
    assert_close(&first.prediction, &expected, 1e-5);
    training.step(0.01).unwrap();
    let second = training.state_snapshot().unwrap().read().unwrap();
    assert!(second.loss < first.loss);
}

#[test]
fn stacked_layer_norm_preserves_nd_forward_and_all_vjps() {
    let Some(runtime) = runtime() else { return };
    let shape = [2, 2, 257];
    let input_values: Vec<_> = (0..shape.iter().product::<usize>())
        .map(|index| ((index * 37 % 251) as f32 - 125.) / 64.)
        .collect();
    let gain_values: Vec<_> = (0..257).map(|i| 0.75 + (i % 5) as f32 / 10.).collect();
    let bias_values: Vec<_> = (0..257).map(|i| (i % 7) as f32 / 20. - 0.15).collect();
    let parameters = [
        GraphParameter {
            role: ParameterRole::Gain,
            shape: vec![257],
            values: gain_values.clone(),
        },
        GraphParameter {
            role: ParameterRole::Bias,
            shape: vec![257],
            values: bias_values.clone(),
        },
        GraphParameter {
            role: ParameterRole::Gain,
            shape: vec![257],
            values: gain_values.clone(),
        },
        GraphParameter {
            role: ParameterRole::Bias,
            shape: vec![257],
            values: bias_values.clone(),
        },
    ];
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&shape).unwrap(),
        vec![
            GraphStage::LayerNorm {
                gain: 0,
                bias: 1,
                epsilon: 1e-5,
            },
            GraphStage::LayerNorm {
                gain: 2,
                bias: 3,
                epsilon: 1e-5,
            },
        ],
        parameters.to_vec(),
    )
    .unwrap();
    let device = TensorDevice::new(runtime.clone()).unwrap();
    let input = device.upload(&shape, &input_values).unwrap();
    let gain = device.upload(&[257], &gain_values).unwrap();
    let bias = device.upload(&[257], &bias_values).unwrap();
    let first = input.layer_norm_affine(&gain, &bias, 1e-5).unwrap();
    let second = first.value().layer_norm_affine(&gain, &bias, 1e-5).unwrap();
    let expected = second.value().snapshot().unwrap().read().unwrap();
    let cotangent = device
        .upload(
            &shape,
            &expected
                .iter()
                .map(|&value| value / 100.)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let second_vjp = second.backward(&cotangent, 1., [true; 3]).unwrap();
    let first_vjp = first
        .backward(second_vjp[0].as_ref().unwrap(), 1., [true; 3])
        .unwrap();
    let references = [
        first_vjp[0].as_ref().unwrap(),
        first_vjp[1].as_ref().unwrap(),
        first_vjp[2].as_ref().unwrap(),
        second_vjp[1].as_ref().unwrap(),
        second_vjp[2].as_ref().unwrap(),
    ];
    let mut graph = ResidentGraphAutograd::new(
        runtime,
        definition,
        MatmulTile::default(),
        MatmulKernel::Scalar,
        MatmulAccumulation::Sequential,
    )
    .unwrap();
    graph.upload(&input_values).unwrap();
    let forward = graph.forward().unwrap();
    assert_close(
        &forward.prediction().snapshot().unwrap().read().unwrap(),
        &expected,
        1e-5,
    );
    let vjp = graph.backward(&forward, &cotangent).unwrap();
    for (actual, reference) in std::iter::once(vjp.input_gradient())
        .chain(vjp.parameter_gradients().iter())
        .zip(references)
    {
        assert_close(
            &actual.snapshot().unwrap().read().unwrap(),
            &reference.snapshot().unwrap().read().unwrap(),
            1e-4,
        );
    }
}

#[test]
fn layer_norm_learner_averages_bias_for_clipping_and_momentum() {
    let Some(runtime) = runtime() else { return };
    let definition = definition(1e-5);
    let values = [1., 2., 4., 2., 0., -1.];
    for (clipping, momentum) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut learner = ResidentGraphLearner::new(
            runtime.clone(),
            definition.clone(),
            GraphGradientPolicy::ModuleCompatible,
            MatmulTile::default(),
            MatmulKernel::Scalar,
            MatmulAccumulation::Sequential,
        )
        .unwrap();
        if clipping {
            learner.set_grad_clip_max_norm(0.25).unwrap();
        }
        if momentum {
            learner.set_momentum_damping(0.5).unwrap();
        }
        learner.upload(&values).unwrap();
        let forward = learner.forward().unwrap();
        let prediction = forward.prediction().snapshot().unwrap().read().unwrap();
        let seed = learner
            .tensor_device()
            .upload(
                &[2, 3],
                &prediction
                    .iter()
                    .map(|&value| value / 3.)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let gradients = learner.backward(&forward, &seed).unwrap();
        let mut expected: Vec<Vec<f32>> = gradients
            .parameter_gradients()
            .iter()
            .map(|tensor| tensor.snapshot().unwrap().read().unwrap())
            .collect();
        for value in expected.iter_mut().flatten() {
            *value *= 0.5;
        }
        if clipping {
            let norm_squared = expected
                .iter()
                .flatten()
                .map(|&value| f64::from(value).powi(2))
                .sum();
            let factors = GlobalNormClip::new(0.25)
                .unwrap()
                .factors(norm_squared)
                .unwrap();
            assert!(!factors.as_slice().is_empty());
            for value in expected.iter_mut().flatten() {
                for &factor in factors.as_slice() {
                    *value *= factor;
                }
            }
        }
        if momentum {
            let ema = EmaMomentum::new(0.5).unwrap();
            for value in expected.iter_mut().flatten() {
                *value = ema.transition(*value, 0.).unwrap();
            }
        }
        learner.sgd(&gradients, 0.1).unwrap();
        learner.update_snapshot().unwrap().read().unwrap();
        let updated = learner.parameter_snapshot().unwrap().read().unwrap();
        for (id, parameter) in updated.parameters().iter().enumerate() {
            let baseline = &definition.parameters()[id].values;
            let next: Vec<_> = baseline
                .iter()
                .zip(&expected[id])
                .map(|(&value, &gradient)| value - 0.1 * gradient)
                .collect();
            assert_close(&parameter.values, &next, 1e-5);
        }
        if momentum {
            for (actual, expected) in learner.momentum_tensors().unwrap().iter().zip(&expected) {
                assert_close(&actual.snapshot().unwrap().read().unwrap(), expected, 1e-5);
            }
        }
    }
}
