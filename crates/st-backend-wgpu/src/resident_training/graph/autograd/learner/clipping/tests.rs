use super::*;
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    pointwise::{PointwiseChain, PointwiseStep},
};

fn learner(
    runtime: WgpuRuntime,
    rows: usize,
    cols: usize,
    policy: GraphGradientPolicy,
) -> ResidentGraphLearner {
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[1, rows, cols]).unwrap(),
        vec![GraphStage::Pointwise {
            chain: PointwiseChain::new(
                2,
                vec![PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(1),
                }],
            )
            .unwrap(),
            parameters: vec![0],
        }],
        vec![GraphParameter {
            role: ParameterRole::Gain,
            shape: vec![cols],
            values: vec![0.; cols],
        }],
    )
    .unwrap();
    ResidentGraphLearner::new(
        runtime,
        definition,
        policy,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap()
}

fn gradient(l: &mut ResidentGraphLearner, cotangent: &[f32]) -> GraphGradients {
    l.upload(&vec![1.; l.input_layout().len()]).unwrap();
    let forward = l.forward().unwrap();
    let seed = l
        .tensor_device()
        .upload(l.output_layout().shape(), cotangent)
        .unwrap();
    l.backward(&forward, &seed).unwrap()
}

#[test]
fn global_clip_gpu_wide_norm_tiny_scale_policy_and_disabled_path() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("learner.clip").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    for (rows, cols, magnitude, limit) in [
        (2, 7, 4., 0.5),
        (1, 513, 1e38, 1.),
        (1, 513, 1e38, 1e-20),
        (2, 3, 1e-9, 1e-30),
        (1, 7, 0., 1.),
        (1, 257, 1., 100.),
    ] {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            let mut l = learner(runtime.clone(), rows, cols, policy);
            assert_eq!(l.grad_clip_max_norm(), None);
            l.set_grad_clip_max_norm(limit).unwrap();
            for bad in [0., -1., f32::NAN, f32::INFINITY] {
                assert!(l.set_grad_clip_max_norm(bad).is_err());
                assert_eq!(l.grad_clip_max_norm(), Some(limit));
            }
            let values: Vec<_> = (0..rows * cols)
                .map(|i| {
                    if i % cols % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect();
            let gradients = gradient(&mut l, &values);
            let mut raw = vec![0.; cols];
            for row in values.chunks(cols) {
                for (a, &b) in raw.iter_mut().zip(row) {
                    *a += b;
                }
            }
            if policy == GraphGradientPolicy::ModuleCompatible {
                for x in &mut raw {
                    *x *= 1. / rows as f32;
                }
            }
            let factors = GlobalNormClip::new(limit)
                .unwrap()
                .factors(raw.iter().map(|&v| f64::from(v).powi(2)).sum())
                .unwrap();
            for x in &mut raw {
                for &factor in factors.as_slice() {
                    *x *= factor;
                }
            }
            l.sgd(&gradients, 1.).unwrap();
            assert_eq!(l.update_snapshot().unwrap().read().unwrap(), 1);
            let result = l.parameter_snapshot().unwrap().read().unwrap();
            for (&actual, &expected) in result.parameters()[0].values.iter().zip(&raw) {
                let tolerance = 1e-5 * expected.abs().max(f32::MIN_POSITIVE);
                assert!(
                    (actual + expected).abs() <= tolerance,
                    "{rows} {cols} {magnitude} {limit}: {actual} vs {}",
                    -expected
                );
            }
            l.clear_grad_clip();
            assert_eq!(l.grad_clip_max_norm(), None);
            let gradients = gradient(&mut l, &vec![0.; rows * cols]);
            l.sgd(&gradients, 0.).unwrap();
            assert_eq!(l.update_snapshot().unwrap().read().unwrap(), 2);
        }
    }
}

#[test]
fn global_clip_gpu_accumulator_guard_is_not_sanitized_and_recovers() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("learner.clip.guard").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let mut l = learner(runtime, 2, 3, GraphGradientPolicy::Exact);
    l.set_grad_clip_max_norm(0.5).unwrap();
    let mut sum = l.gradient_accumulator().unwrap();
    let bad = gradient(&mut l, &[f32::MAX; 6]); // row reduction overflows before clipping.
    l.accumulate(&mut sum, &bad, 0.).unwrap();
    l.sgd_accumulated(&sum, 0.).unwrap();
    assert!(l.update_snapshot().unwrap().read().is_err());
    assert_eq!(
        l.parameter_snapshot().unwrap().read().unwrap().parameters()[0].values,
        vec![0.; 3]
    );
    l.zero_accumulator(&mut sum).unwrap();
    for magnitude in [3., -1.] {
        let gradients = gradient(&mut l, &[magnitude; 6]);
        l.accumulate(&mut sum, &gradients, 0.5).unwrap();
    }
    let held = sum.parameter_gradients().unwrap();
    l.sgd_accumulated(&sum, 1.).unwrap();
    assert_eq!(l.update_snapshot().unwrap().read().unwrap(), 2);
    let result = l.parameter_snapshot().unwrap().read().unwrap();
    for &x in &result.parameters()[0].values {
        assert!((x + 0.5 / 3f32.sqrt()).abs() < 1e-6);
    }
    assert_eq!(held[0].snapshot().unwrap().read().unwrap(), vec![2.; 3]);
    assert!(matches!(
        l.sgd_accumulated(&sum, 0.),
        Err(TrainingError::AccumulatorState)
    ));
}

#[test]
fn global_clip_gpu_weighted_sum_is_global_after_mixed_parameter_policy() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("learner.clip.weighted").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    for policy in [
        GraphGradientPolicy::Exact,
        GraphGradientPolicy::ModuleCompatible,
    ] {
        let definition = GraphDefinition::new(
            NdLayout::contiguous(&[1, 2, 2]).unwrap(),
            vec![
                GraphStage::Linear {
                    weight: 0,
                    bias: 1,
                    gelu: false,
                },
                GraphStage::Pointwise {
                    chain: PointwiseChain::new(
                        2,
                        vec![PointwiseStep {
                            op: ElementwiseOp::Multiply,
                            rhs: Some(1),
                        }],
                    )
                    .unwrap(),
                    parameters: vec![2],
                },
            ],
            vec![
                GraphParameter {
                    role: ParameterRole::Weight,
                    shape: vec![2, 3],
                    values: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![3],
                    values: vec![0.; 3],
                },
                GraphParameter {
                    role: ParameterRole::Gain,
                    shape: vec![3],
                    values: vec![1.; 3],
                },
            ],
        )
        .unwrap();
        let mut l = ResidentGraphLearner::new(
            runtime.clone(),
            definition.clone(),
            policy,
            Default::default(),
            MatmulKernel::Scalar,
            Default::default(),
        )
        .unwrap();
        l.set_grad_clip_max_norm(0.5).unwrap();
        l.upload(&[1., 2., 3., 4.]).unwrap();
        let forward = l.forward().unwrap();
        let a = l
            .tensor_device()
            .upload(&[1, 2, 3], &[1., 2., 3., -4., 5., 6.])
            .unwrap();
        let b = l
            .tensor_device()
            .upload(&[1, 2, 3], &[-1., -2., -3., 4., -5., -6.])
            .unwrap();
        let ga = l.backward(&forward, &a).unwrap();
        let gb = l.backward(&forward, &b).unwrap();
        let mut effective: Vec<Vec<f32>> = ga
            .parameter_gradients()
            .iter()
            .map(|t| t.snapshot().unwrap().read().unwrap())
            .collect();
        for (id, values) in effective.iter_mut().enumerate() {
            for v in values {
                *v *= 2.5;
                if id == 2 && policy == GraphGradientPolicy::ModuleCompatible {
                    *v *= 0.5;
                }
            }
        }
        let squared = effective
            .iter()
            .flatten()
            .map(|&x| f64::from(x).powi(2))
            .sum();
        let factors = GlobalNormClip::new(0.5).unwrap().factors(squared).unwrap();
        for v in effective.iter_mut().flatten() {
            for &s in factors.as_slice() {
                *v *= s;
            }
        }
        let mut batch = GraphGradientBatch::new();
        batch.add(&ga, 2.).unwrap();
        batch.add(&gb, -0.5).unwrap();
        l.sgd_batch(&batch, 0.1).unwrap();
        l.update_snapshot().unwrap().read().unwrap();
        let result = l.parameter_snapshot().unwrap().read().unwrap();
        for ((actual, initial), g) in result
            .parameters()
            .iter()
            .zip(definition.parameters())
            .zip(effective)
        {
            for ((&a, &b), g) in actual.values.iter().zip(&initial.values).zip(g) {
                assert!((a - (b - 0.1 * g)).abs() < 2e-6);
            }
        }
    }
}
