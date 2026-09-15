use super::*;
use st_kernel_contracts::pointwise::{PointwiseChain, PointwiseExecution, PointwiseStep};

fn runtime() -> Option<WgpuRuntime> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.cotangent.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(runtime)
}

fn chain(steps: &[(&str, Option<usize>)], count: usize) -> PointwiseChain {
    PointwiseChain::new(
        count,
        steps
            .iter()
            .map(|(op, rhs)| PointwiseStep::named(op, *rhs).unwrap())
            .collect(),
    )
    .unwrap()
}

fn definition(rows: usize, dense: bool) -> GraphDefinition {
    let (stages, parameters) = if dense {
        (
            vec![GraphStage::Linear {
                weight: 0,
                bias: 1,
                gelu: false,
            }],
            vec![
                GraphParameter {
                    role: ParameterRole::Weight,
                    shape: vec![2, 2],
                    values: vec![2., 0., 0., 2.],
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![2],
                    values: vec![0.; 2],
                },
            ],
        )
    } else {
        (
            vec![GraphStage::Pointwise {
                chain: chain(&[("multiply", Some(1))], 2),
                parameters: vec![0],
            }],
            vec![GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![2],
                values: vec![2.; 2],
            }],
        )
    };
    GraphDefinition::new(
        NdLayout::contiguous(&[1, rows, 2]).unwrap(),
        stages,
        parameters,
    )
    .unwrap()
}

fn tape(runtime: WgpuRuntime, rows: usize, dense: bool) -> ResidentGraphAutograd {
    let mut g = ResidentGraphAutograd::new(
        runtime,
        definition(rows, dense),
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap();
    g.upload(&[1., 2.].repeat(rows)).unwrap();
    g
}

fn plan(
    d: &TensorDevice,
    inputs: &[&ResidentTensor],
    steps: &[(&str, Option<usize>)],
) -> PointwisePlan {
    PointwisePlan::new(
        d.clone(),
        chain(steps, inputs.len()),
        inputs.iter().map(|t| t.layout().clone()).collect(),
    )
    .unwrap()
}

fn values(g: &GraphGradients) -> Vec<Vec<f32>> {
    std::iter::once(g.input_gradient())
        .chain(g.parameter_gradients())
        .map(|t| t.snapshot().unwrap().read().unwrap())
        .collect()
}

#[test]
fn pointwise_cotangent_matches_materialized_and_analytic_vjps_with_strides() {
    let Some(runtime) = runtime() else { return };
    for dense in [false, true] {
        for rows in [1, 3, 258] {
            let mut g = tape(runtime.clone(), rows, dense);
            let d = g.tensor_device().clone();
            // Nonzero offset and a transposed logical domain; no seed pack.
            let source = d
                .upload(
                    &[2, 1, rows + 1],
                    &[vec![1.; rows + 1], vec![2.; rows + 1]].concat(),
                )
                .unwrap()
                .narrow(2, 1, rows)
                .unwrap()
                .permute(&[1, 2, 0])
                .unwrap();
            let scale = d.upload(&[], &[0.5]).unwrap();
            let inputs = [&source, &scale];
            let p = plan(&d, &inputs, &[("multiply", Some(1)), ("add", Some(0))]);
            let f = g.forward().unwrap();
            let ordinary = g
                .backward(&f, &p.run(&inputs, PointwiseExecution::Fused).unwrap())
                .unwrap();
            let direct = g.backward_pointwise(&f, &p, &inputs).unwrap();
            assert_eq!(values(&ordinary), values(&direct));
            let expected = if dense {
                vec![
                    [3., 6.].repeat(rows),
                    vec![
                        1.5 * rows as f32,
                        3. * rows as f32,
                        3. * rows as f32,
                        6. * rows as f32,
                    ],
                    vec![1.5 * rows as f32, 3. * rows as f32],
                ]
            } else {
                vec![
                    [3., 6.].repeat(rows),
                    vec![1.5 * rows as f32, 6. * rows as f32],
                ]
            };
            assert_eq!(values(&direct), expected);
            let held = direct.input_gradient().reshape(&[2 * rows]).unwrap();
            let pending = held.snapshot().unwrap();
            for _ in 0..8 {
                drop(g.backward_pointwise(&f, &p, &inputs).unwrap());
            }
            drop((g, p, source, scale, ordinary));
            assert_eq!(pending.read().unwrap(), expected[0]);
            assert_eq!(held.snapshot().unwrap().read().unwrap(), expected[0]);
            assert_eq!(values(&direct), expected);
        }
    }
}

#[test]
fn pointwise_cotangent_host_errors_do_not_invalidate_current_tape() {
    let Some(runtime) = runtime() else { return };
    let mut g = tape(runtime.clone(), 3, false);
    let d = g.tensor_device().clone();
    let input = d.upload(&[1, 3, 2], &[1.; 6]).unwrap();
    let p = plan(&d, &[&input], &[("identity", None)]);
    let f = g.forward().unwrap();
    let wrong = input.reshape(&[3, 2]).unwrap();
    let wrong_plan = plan(&d, &[&wrong], &[("identity", None)]);
    assert!(g.backward_pointwise(&f, &wrong_plan, &[&wrong]).is_err());
    assert!(g.backward_pointwise(&f, &p, &[&wrong]).is_err());
    assert!(g.backward_pointwise(&f, &p, &[]).is_err());
    let other = TensorDevice::new(
        pollster::block_on(WgpuRuntime::request_headless("cotangent.foreign")).unwrap(),
    )
    .unwrap();
    let alien = other.upload(&[1, 3, 2], &[1.; 6]).unwrap();
    let alien_plan = plan(&other, &[&alien], &[("identity", None)]);
    assert!(g.backward_pointwise(&f, &p, &[&alien]).is_err());
    assert!(g.backward_pointwise(&f, &alien_plan, &[&alien]).is_err());
    let mut foreign = tape(runtime, 3, false);
    let foreign_f = foreign.forward().unwrap();
    assert!(matches!(
        g.backward_pointwise(&foreign_f, &p, &[&input]),
        Err(TrainingError::StaleForward)
    ));
    assert_eq!(g.submitted_backwards(), 0);
    assert!(g.cotangent_inherited.is_none());
    let good = g.backward_pointwise(&f, &p, &[&input]).unwrap();
    assert_eq!(values(&good)[0], vec![2.; 6]);
    let new_f = g.forward().unwrap();
    assert!(matches!(
        g.backward_pointwise(&f, &p, &[&input]),
        Err(TrainingError::StaleForward)
    ));
    assert_eq!(g.submitted_backwards(), 1);
    g.backward_pointwise(&new_f, &p, &[&input]).unwrap();
    assert_eq!(g.submitted_backwards(), 2);
}

#[test]
fn pointwise_cotangent_masked_failures_guard_all_outputs_and_recover() {
    let Some(runtime) = runtime() else { return };
    for dense in [false, true] {
        let mut g = tape(runtime.clone(), 3, dense);
        let d = g.tensor_device().clone();
        let base = d.upload(&[1, 3, 2], &[1.; 6]).unwrap();
        let huge = d.upload(&[], &[f32::MAX]).unwrap();
        let zero = d.upload(&[], &[0.]).unwrap();
        let neg = d.upload(&[], &[-2.]).unwrap();
        let invalid = huge.mul(&neg).unwrap().relu().unwrap();
        let f = g.forward().unwrap();
        let p = plan(
            &d,
            &[&base, &zero, &invalid],
            &[("add", Some(2)), ("multiply", Some(1))],
        );
        let inherited = g
            .backward_pointwise(&f, &p, &[&base, &zero, &invalid])
            .unwrap();
        // Shrink immediately after the failure, before any other seed clears it.
        let p = plan(&d, &[&base], &[("identity", None)]);
        let fewer = g.backward_pointwise(&f, &p, &[&base]).unwrap();
        assert_eq!(values(&fewer)[0], vec![2.; 6]);
        let p = plan(
            &d,
            &[&base, &huge, &neg],
            &[("multiply", Some(1)), ("multiply", Some(2)), ("relu", None)],
        );
        let masked = g.backward_pointwise(&f, &p, &[&base, &huge, &neg]).unwrap();
        // Producer failures also recover independently of inherited flags.
        let p = plan(&d, &[&base], &[("identity", None)]);
        let good = g.backward_pointwise(&f, &p, &[&base]).unwrap();
        assert_eq!(values(&good)[0], vec![2.; 6]);
        g.upload(&[f32::MAX; 6]).unwrap();
        let bad_f = g.forward().unwrap();
        let zero_seed = d.upload(&[1, 3, 2], &[0.; 6]).unwrap();
        let forward_failed = g.backward_pointwise(&bad_f, &p, &[&zero_seed]).unwrap();
        drop(g);
        for bad in [inherited, masked, forward_failed] {
            for t in std::iter::once(bad.input_gradient()).chain(bad.parameter_gradients()) {
                assert!(matches!(
                    t.snapshot().unwrap().read(),
                    Err(TensorError::NonFinite)
                ));
                assert!(t.mul(&zero).unwrap().snapshot().unwrap().read().is_err());
            }
        }
        assert_eq!(values(&good)[0], vec![2.; 6]);
    }
}

#[test]
fn pointwise_cotangent_rejected_update_preserves_parameters_and_momentum() {
    let Some(runtime) = runtime() else { return };
    for policy in [
        GraphGradientPolicy::Exact,
        GraphGradientPolicy::ModuleCompatible,
    ] {
        let mut l = ResidentGraphLearner::new(
            runtime.clone(),
            definition(3, true),
            policy,
            Default::default(),
            MatmulKernel::Scalar,
            Default::default(),
        )
        .unwrap();
        l.set_momentum_damping(0.5).unwrap();
        l.set_grad_clip_max_norm(0.125).unwrap();
        l.upload(&[1.; 6]).unwrap();
        let d = l.tensor_device().clone();
        let source = d.upload(&[1, 3, 2], &[1.; 6]).unwrap();
        let p = plan(&d, &[&source], &[("identity", None)]);
        let f = l.forward().unwrap();
        let g = l.backward_pointwise(&f, &p, &[&source]).unwrap();
        l.sgd(&g, 0.1).unwrap();
        assert_eq!(l.update_snapshot().unwrap().read().unwrap(), 1);
        let weights = |l: &ResidentGraphLearner| {
            l.parameter_snapshot()
                .unwrap()
                .read()
                .unwrap()
                .parameters()
                .iter()
                .map(|p| p.values.clone())
                .collect::<Vec<_>>()
        };
        let history = |l: &ResidentGraphLearner| {
            l.momentum_tensors()
                .unwrap()
                .iter()
                .map(|t| t.snapshot().unwrap().read().unwrap())
                .collect::<Vec<_>>()
        };
        let before = (weights(&l), history(&l));
        let f = l.forward().unwrap();
        let good = l.backward_pointwise(&f, &p, &[&source]).unwrap();
        let huge = d.upload(&[1, 3, 2], &[f32::MAX; 6]).unwrap();
        let bad = l.backward_pointwise(&f, &p, &[&huge]).unwrap();
        // Zero weight cannot erase an invalid seed/VJP.
        l.sgd_weighted(&[(&good, 1.), (&bad, 0.)], 0.1).unwrap();
        assert!(matches!(
            l.update_snapshot().unwrap().read(),
            Err(TrainingError::Rejected { .. })
        ));
        assert_eq!((weights(&l), history(&l)), before);
        assert!(matches!(
            l.backward_pointwise(&f, &p, &[&source]),
            Err(TrainingError::StaleForward)
        ));
        let f = l.forward().unwrap();
        let good = l.backward_pointwise(&f, &p, &[&source]).unwrap();
        l.sgd(&good, 0.1).unwrap();
        assert_eq!(l.update_snapshot().unwrap().read().unwrap(), 3);
    }
}
