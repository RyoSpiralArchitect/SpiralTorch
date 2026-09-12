use super::*;
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    pointwise::{PointwiseChain, PointwiseStep},
};

fn gain(runtime: WgpuRuntime) -> ResidentGraphLearner {
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[1, 1, 2]).unwrap(),
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
            shape: vec![2],
            values: vec![0.; 2],
        }],
    )
    .unwrap();
    ResidentGraphLearner::new(
        runtime,
        definition,
        GraphGradientPolicy::Exact,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap()
}
fn gradient(l: &mut ResidentGraphLearner, values: &[f32]) -> GraphGradients {
    l.upload(&vec![1.; l.input_layout().len()]).unwrap();
    let forward = l.forward().unwrap();
    let seed = l
        .tensor_device()
        .upload(l.output_layout().shape(), values)
        .unwrap();
    l.backward(&forward, &seed).unwrap()
}
fn states(l: &ResidentGraphLearner) -> Vec<Vec<f32>> {
    l.momentum_tensors()
        .unwrap()
        .into_iter()
        .map(|t| t.snapshot().unwrap().read().unwrap())
        .collect()
}
fn weights(l: &ResidentGraphLearner) -> Vec<Vec<f32>> {
    l.parameter_snapshot()
        .unwrap()
        .read()
        .unwrap()
        .parameters()
        .iter()
        .map(|p| p.values.clone())
        .collect()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-6 * b.abs().max(1.),
            "{a} vs {b}"
        );
    }
}

#[test]
fn momentum_gpu_shared_ema_switching_reset_and_owning_snapshots() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("learner.momentum").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let mut l = gain(runtime);
    let mut previous = vec![0.; 2];
    let mut parameters = vec![0.; 2];
    let mut held = Vec::new();
    assert!(matches!(
        l.momentum_tensors(),
        Err(TrainingError::MissingMomentum)
    ));
    assert!(matches!(
        l.reset_momentum(),
        Err(TrainingError::MissingMomentum)
    ));
    for i in 0..16 {
        let damping = [0.6, 0.85, 0., 0.3][i % 4];
        l.set_momentum_damping(damping).unwrap();
        for bad in [-1., 0.86, f32::NAN, f32::INFINITY] {
            assert!(l.set_momentum_damping(bad).is_err());
            assert_eq!(l.momentum_damping(), Some(damping));
        }
        if i % 3 == 0 {
            l.set_grad_clip_max_norm(0.5).unwrap();
        } else {
            l.clear_grad_clip();
        }
        if i == 9 {
            l.reset_momentum().unwrap();
            previous.fill(0.);
        }
        let raw = [i as f32 - 5., 3. - i as f32];
        let gradients = gradient(&mut l, &raw);
        let mut effective = raw;
        if i % 3 == 0 {
            let factors = GlobalNormClip::new(0.5)
                .unwrap()
                .factors(raw.iter().map(|&v| f64::from(v).powi(2)).sum())
                .unwrap();
            for v in &mut effective {
                for &s in factors.as_slice() {
                    *v *= s;
                }
            }
        }
        let rate = if i % 5 == 0 { 0. } else { 0.1 };
        l.sgd(&gradients, rate).unwrap();
        l.update_snapshot().unwrap().read().unwrap();
        if rate != 0. {
            for j in 0..2 {
                previous[j] = EmaMomentum::new(damping)
                    .unwrap()
                    .transition(effective[j], previous[j])
                    .unwrap();
                parameters[j] -= rate * previous[j];
            }
        }
        close(&weights(&l)[0], &parameters);
        close(&states(&l)[0], &previous);
        held.push((
            l.momentum_tensors().unwrap().pop().unwrap(),
            previous.clone(),
        ));
    }
    l.clear_momentum();
    assert_eq!(l.momentum_damping(), None);
    assert!(matches!(
        l.momentum_tensors(),
        Err(TrainingError::MissingMomentum)
    ));
    let g = gradient(&mut l, &[1., 2.]);
    l.sgd(&g, 0.1).unwrap();
    l.update_snapshot().unwrap().read().unwrap();
    l.set_momentum_damping(0.5).unwrap();
    assert_eq!(states(&l), vec![vec![0.; 2]]);
    drop(l);
    for (t, expected) in held {
        close(&t.snapshot().unwrap().read().unwrap(), &expected);
    }
}

#[test]
fn momentum_gpu_rejected_or_zero_rate_transaction_preserves_all_history() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("learner.momentum.atomic").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[1, 1]).unwrap(),
        vec![GraphStage::Linear {
            weight: 0,
            bias: 1,
            gelu: false,
        }],
        vec![
            GraphParameter {
                role: ParameterRole::Weight,
                shape: vec![1, 2],
                values: vec![0.; 2],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![2],
                values: vec![0.; 2],
            },
        ],
    )
    .unwrap();
    let mut l = ResidentGraphLearner::new(
        runtime,
        definition,
        GraphGradientPolicy::Exact,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap();
    l.set_momentum_damping(0.5).unwrap();
    let g = gradient(&mut l, &[1., 2.]);
    l.sgd(&g, 0.1).unwrap();
    l.update_snapshot().unwrap().read().unwrap();
    let before = weights(&l);
    let state = states(&l);
    let g = gradient(&mut l, &[1e30, 1.]);
    l.sgd(&g, 1e20).unwrap();
    assert!(l.update_snapshot().unwrap().read().is_err());
    assert_eq!(weights(&l), before);
    assert_eq!(states(&l), state);
    let f = l.forward().unwrap();
    let huge = l.tensor_device().upload(&[1, 2], &[f32::MAX; 2]).unwrap();
    let poisoned = huge.add(&huge).unwrap();
    let bad = l.backward(&f, &poisoned).unwrap();
    l.sgd_weighted(&[(&bad, 0.)], 0.).unwrap();
    assert!(l.update_snapshot().unwrap().read().is_err());
    assert_eq!(weights(&l), before);
    assert_eq!(states(&l), state);
    let good = gradient(&mut l, &[5., 6.]);
    l.sgd(&good, 0.).unwrap();
    assert_eq!(l.update_snapshot().unwrap().read().unwrap(), 4);
    assert_eq!(weights(&l), before);
    assert_eq!(states(&l), state);
    let good = gradient(&mut l, &[5., 6.]);
    l.sgd(&good, 0.1).unwrap();
    assert_eq!(l.update_snapshot().unwrap().read().unwrap(), 5);
    for values in states(&l) {
        close(&values, &[2.75, 3.5]);
    }
}

#[test]
fn momentum_gpu_does_not_validate_an_unused_plain_sgd_candidate() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("learner.momentum.wide").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    for clipping in [false, true] {
        let mut l = gain(runtime.clone());
        l.set_momentum_damping(0.85).unwrap();
        if clipping {
            l.set_grad_clip_max_norm(f32::MAX).unwrap();
        }
        let g = gradient(&mut l, &[2e38, -2e38]);
        l.sgd(&g, 5.).unwrap();
        l.update_snapshot().unwrap().read().unwrap();
        let expected = EmaMomentum::new(0.85)
            .unwrap()
            .transition(2e38, 0.)
            .unwrap();
        close(&states(&l)[0], &[expected, -expected]);
        close(&weights(&l)[0], &[-5. * expected, 5. * expected]);
    }
}
