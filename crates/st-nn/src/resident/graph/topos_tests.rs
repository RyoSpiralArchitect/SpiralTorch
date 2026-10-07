use super::*;
use crate::{Module, ToposResonator};
use st_core::dynamics::topos_resonator::ToposResonatorConfig;
use st_tensor::topos::OpenCartesianTopos;

fn layer(gate: &[f32], porosity: f32, max_volume: usize) -> ToposResonator {
    ToposResonator::from_shared_gate(
        "topos",
        Tensor::from_vec(1, gate.len(), gate.to_vec()).unwrap(),
        ToposResonatorConfig::new(0.2, 5).unwrap(),
        OpenCartesianTopos::new(-1., 1e-6, 1., 16, max_volume)
            .unwrap()
            .with_porosity(porosity)
            .unwrap(),
    )
    .unwrap()
}

fn plan(module: &impl Module, shape: &[usize]) -> InferencePlan {
    InferencePlan::from_module(module, NdLayout::contiguous(shape).unwrap()).unwrap()
}

#[test]
fn topos_v5_roundtrip_owns_gate_and_rejects_downgrade_or_invalid_contract() {
    use serde_json::json;
    let mut module = layer(&[0.5, -0.25, 1.], 0.3, 24);
    let plan = plan(&module, &[2, 4, 3]);
    let payload = plan.to_json().unwrap();
    assert!(payload.contains(GRAPH_PLAN_SCHEMA_V5));
    assert_eq!(
        InferencePlan::from_json(&payload)
            .unwrap()
            .to_json()
            .unwrap(),
        payload
    );
    assert_eq!(plan.fuse_pointwise().unwrap().to_json().unwrap(), payload);
    let graph = plan.graph_definition().unwrap();
    assert_eq!(graph.parameters()[0].role, ParameterRole::Gate);
    assert_eq!(graph.parameters()[0].shape, [3]);
    assert!(!graph.module_compatible_row_average(0));
    assert_eq!(plan.source_operation_count(), 1);
    for old in [
        GRAPH_PLAN_SCHEMA,
        GRAPH_PLAN_SCHEMA_V3,
        GRAPH_PLAN_SCHEMA_V4,
    ] {
        assert!(InferencePlan::from_json(&payload.replace(GRAPH_PLAN_SCHEMA_V5, old)).is_err());
    }
    let valid: serde_json::Value = serde_json::from_str(&payload).unwrap();
    for (pointer, value) in [
        ("/stages/0/coupling", json!(1.)),
        ("/stages/0/iterations", json!(4097)),
        ("/stages/0/iterations", json!(0)),
        ("/stages/0/iterations", json!(1.5)),
        ("/stages/0/saturation", json!(0.)),
        ("/stages/0/porosity", json!(1.1)),
        ("/stages/0/max_volume", json!(23)),
        ("/stages/0/max_volume", json!(0)),
        ("/stages/0/gate", json!(1)),
        ("/parameters/0/role", json!("gain")),
        ("/parameters/0/shape", json!([1, 3])),
        ("/parameters/0/values", json!([1., 2.])),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            InferencePlan::from_json(&bad.to_string()).is_err(),
            "{pointer}"
        );
    }
    module
        .visit_parameters_mut(&mut |p| {
            p.value_mut().data_mut().fill(0.75);
            Ok(())
        })
        .unwrap();
    assert_eq!(plan.to_json().unwrap(), payload);
    assert!(InferencePlan::from_module(&module, NdLayout::contiguous(&[9, 3]).unwrap()).is_err());
    let elementwise = ToposResonator::new("elementwise", 2, 3).unwrap();
    assert!(
        InferencePlan::from_module(&elementwise, NdLayout::contiguous(&[2, 3]).unwrap()).is_err()
    );
}

#[test]
fn topos_gate_role_cannot_be_reinterpreted_as_a_scaler_or_tied() {
    let graph = plan(&layer(&[1., 0.5], 0., 8), &[4, 2])
        .graph_definition()
        .unwrap();
    let pointwise = GraphStage::Pointwise {
        chain: PointwiseChain::new(
            2,
            vec![PointwiseStep {
                op: ElementwiseOp::Multiply,
                rhs: Some(1),
            }],
        )
        .unwrap(),
        parameters: vec![0],
    };
    assert!(GraphDefinition::new(
        graph.input_layout().clone(),
        vec![pointwise],
        graph.parameters().to_vec()
    )
    .is_err());
    assert!(GraphDefinition::new(
        graph.input_layout().clone(),
        vec![graph.stages()[0].clone(); 2],
        graph.parameters().to_vec()
    )
    .is_err());
    let positive = ToposResonatorKernel::new(0., 1., 0., 1).unwrap();
    let negative = ToposResonatorKernel::new(-0., 1., 0., 1).unwrap();
    assert_ne!(positive, negative);
}

#[test]
fn topos_handoff_preserves_shape_and_rejects_program_changes_before_mutation() {
    let mut module = layer(&[0.5, -0.25], 0.3, 8);
    let base = plan(&module, &[4, 2]);
    let updated = base.with_graph_values(vec![vec![0.4, 0.6]]).unwrap();
    let graph = updated.graph_definition().unwrap();
    let mut stages = graph.stages().to_vec();
    if let GraphStage::ToposResonator { kernel, .. } = &mut stages[0] {
        *kernel = ToposResonatorKernel::new(0.3, 1., 0.3, 5).unwrap();
    }
    let changed = InferencePlan::from_graph_definition(
        GraphDefinition::new(
            graph.input_layout().clone(),
            stages,
            graph.parameters().to_vec(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(base
        .apply_parameters_to(&mut module, &changed, ModuleOptimizerStatePolicy::Reset)
        .is_err());
    assert_eq!(
        plan(&module, &[4, 2]).to_json().unwrap(),
        base.to_json().unwrap()
    );
    assert_eq!(
        base.apply_parameters_to(&mut module, &updated, ModuleOptimizerStatePolicy::Reject)
            .unwrap(),
        1
    );
    assert_eq!(
        plan(&module, &[4, 2]).to_json().unwrap(),
        updated.to_json().unwrap()
    );
    module
        .visit_parameters(&mut |p| {
            assert_eq!(p.value().shape(), (1, 2));
            Ok(())
        })
        .unwrap();
}

#[test]
fn topos_review_rejected_handoff_preserves_capture_and_backward_replay() {
    let context = crate::execution::RuntimeExecutionContext::from_device_caps_with_config(
        st_core::backend::device_caps::DeviceCaps::cpu(),
        Default::default(),
    );
    let _cpu = crate::execution::push_backend_policy(context.backend_policy());
    for state in 0..3 {
        let mut module = layer(&[0.8, -0.4], 0.3, 8);
        if state == 1 {
            module.attach_hypergrad(-1., 0.01).unwrap();
        }
        if state == 2 {
            module.attach_realgrad(0.01).unwrap();
        }
        let base = plan(&module, &[2, 2]);
        let updated = base.with_graph_values(vec![vec![0.1, 0.2]]).unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.25, 0.5, 0.75, -0.25]).unwrap();
        let seed = Tensor::from_vec(2, 2, vec![0.; 4]).unwrap();
        let expected = module.forward(&input).unwrap();
        if state == 0 {
            module.backward(&input, &seed).unwrap();
        }
        let before = serde_json::to_value(module.parameter().optimizer_checkpoint_state()).unwrap();
        assert!(matches!(
            base.apply_parameters_to(&mut module, &updated, ModuleOptimizerStatePolicy::Reject),
            Err(InferenceError::ModuleUpdate(
                "optimizer state is attached; explicit reset is required"
            ))
        ));
        assert_eq!(
            module.latest_output().as_ref(),
            Some(&expected),
            "state={state}"
        );
        assert_eq!(
            serde_json::to_value(module.parameter().optimizer_checkpoint_state()).unwrap(),
            before
        );
        assert_eq!(
            plan(&module, &[2, 2]).to_json().unwrap(),
            base.to_json().unwrap()
        );
        assert!(module.backward(&input, &seed).is_ok(), "state={state}");
        assert_eq!(
            base.apply_parameters_to(&mut module, &updated, ModuleOptimizerStatePolicy::Reset)
                .unwrap(),
            1
        );
        assert!(module.latest_output().is_none());
    }
}

#[test]
fn topos_review_lowering_checks_the_same_optimizer_alignment_as_host_forward() {
    let mut module = layer(&[0.8, -0.4], 0.3, 8);
    let other = OpenCartesianTopos::new(-1., 1e-6, 0.25, 16, 8)
        .unwrap()
        .with_porosity(0.3)
        .unwrap();
    module
        .parameter_mut()
        .attach_hypergrad_with_topos(-1., 0.01, other)
        .unwrap();
    let input = Tensor::from_vec(2, 2, vec![0.25; 4]).unwrap();
    assert!(module.forward(&input).is_err());
    assert!(InferencePlan::from_module(&module, NdLayout::contiguous(&[2, 2]).unwrap()).is_err());
    assert!(module.resident_parameter_bindings().is_err());
}

#[cfg(feature = "wgpu")]
mod gpu {
    use super::*;
    use crate::{Linear, Sequential};
    use st_backend_wgpu::{
        resident_tensor::TensorDevice,
        runtime::{ensure_default_runtime_blocking, WgpuRuntime},
    };
    use st_core::backend::device_caps::DeviceCaps;

    fn runtime() -> Option<WgpuRuntime> {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return None;
        }
        Some(
            ensure_default_runtime_blocking("nn.topos.graph.tests")
                .unwrap()
                .0,
        )
    }

    fn cpu() -> crate::execution::BackendPolicyGuard {
        let context = crate::execution::RuntimeExecutionContext::from_device_caps_with_config(
            DeviceCaps::cpu(),
            Default::default(),
        );
        crate::execution::push_backend_policy(context.backend_policy())
    }

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 3e-5 * (1. + b.abs()),
                "{i}: {a} != {b}"
            );
        }
    }

    #[test]
    fn topos_graph_updates_match_host_over_many_steps_without_extra_mean() {
        let Some(runtime) = runtime() else { return };
        let _cpu = cpu();
        for porosity in [0., 0.3] {
            for shape in [vec![3], vec![4, 3], vec![2, 2, 3]] {
                let rows = shape.iter().product::<usize>() / 3;
                for policy in [
                    GraphGradientPolicy::Exact,
                    GraphGradientPolicy::ModuleCompatible,
                ] {
                    let mut host = layer(&[0.8, -0.4, 1.1], porosity, 24);
                    let base = plan(&host, &shape);
                    let mut gpu = base
                        .compile_graph_training_wgpu(runtime.clone(), policy)
                        .unwrap();
                    for step in 0..25 {
                        let input: Vec<_> = (0..rows * 3)
                            .map(|i| ((i * 17 + step * 7) % 31) as f32 / 7. - 2.)
                            .collect();
                        let target: Vec<_> = input
                            .iter()
                            .enumerate()
                            .map(|(i, x)| x * 0.13 + i as f32 * 0.01)
                            .collect();
                        let x = Tensor::from_vec(rows, 3, input.clone()).unwrap();
                        host.zero_accumulators().unwrap();
                        let y = host.forward(&x).unwrap();
                        let seed: Vec<_> = y
                            .data()
                            .iter()
                            .zip(&target)
                            .map(|(y, t)| 2. * (y - t) / (rows * 3) as f32)
                            .collect();
                        let dx = host
                            .backward(&x, &Tensor::from_vec(rows, 3, seed).unwrap())
                            .unwrap();
                        let mut dg = Vec::new();
                        host.visit_parameters(&mut |p| {
                            dg = p.gradient().unwrap().data().to_vec();
                            Ok(())
                        })
                        .unwrap();
                        gpu.upload_batch(&input, &target).unwrap();
                        assert_eq!(gpu.step(0.03).unwrap(), (step + 1) as u64);
                        let state = gpu.state_snapshot().unwrap().read().unwrap();
                        close(&state.prediction, y.data());
                        close(&state.input_gradient, dx.data());
                        close(&state.raw_gradients[0], &dg);
                        close(&state.effective_gradients[0], &dg);
                        host.visit_parameters_mut(&mut |p| {
                            for (v, g) in p.value_mut().data_mut().iter_mut().zip(&dg) {
                                *v -= 0.03 * g;
                            }
                            close(&state.graph.parameters()[0].values, p.value().data());
                            Ok(())
                        })
                        .unwrap();
                    }
                }
            }
        }
        eprintln!("Topos graph executed 300 updates across 2 policies, 2 porosities and 3 ranks");
    }

    #[test]
    fn topos_sequential_nd_vjp_matches_host_and_retains_owned_captures() {
        let Some(runtime) = runtime() else { return };
        let _cpu = cpu();
        let device = TensorDevice::new(runtime.clone()).unwrap();
        let mut host = Sequential::new();
        host.push(Linear::new("up", 2, 3).unwrap());
        host.push(layer(&[0.8, -0.4, 1.1], 0.3, 24));
        host.push(Linear::new("down", 3, 2).unwrap());
        host.visit_parameters_mut(&mut |p| {
            if !p.name().starts_with("topos") {
                for (i, v) in p.value_mut().data_mut().iter_mut().enumerate() {
                    *v = (i as f32 - 2.) * 0.2;
                }
            }
            Ok(())
        })
        .unwrap();
        let base = plan(&host, &[2, 2, 2]);
        let mut graph = base.compile_graph_autograd_wgpu(runtime).unwrap();
        let x = vec![-2., 0.3, 1., -0.5, 0.8, 2., 0.1, -1.];
        let seed = vec![0.3, -0.4, 0.5, 0.1, -0.2, 0.8, -1., 0.5];
        let input = Tensor::from_vec(4, 2, x.clone()).unwrap();
        let y = host.forward(&input).unwrap();
        let dx = host
            .backward(&input, &Tensor::from_vec(4, 2, seed.clone()).unwrap())
            .unwrap();
        graph.upload(&x).unwrap();
        let forward = graph.forward().unwrap();
        let gradients = graph
            .backward(&forward, &device.upload(&[2, 2, 2], &seed).unwrap())
            .unwrap();
        let zero = device.upload(&[2, 2, 2], &[0.; 8]).unwrap();
        graph.backward(&forward, &zero).unwrap();
        graph.set_input_tensor(&zero).unwrap();
        graph.forward().unwrap();
        drop(graph);
        close(
            &forward.prediction().snapshot().unwrap().read().unwrap(),
            y.data(),
        );
        close(
            &gradients
                .input_gradient()
                .snapshot()
                .unwrap()
                .read()
                .unwrap(),
            dx.data(),
        );
        let mut id = 0;
        host.visit_parameters(&mut |p| {
            close(
                &gradients.parameter_gradients()[id]
                    .snapshot()
                    .unwrap()
                    .read()
                    .unwrap(),
                p.gradient().unwrap().data(),
            );
            id += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(id, 5);
    }

    #[test]
    fn topos_resident_first_stage_accepts_views_and_invalidates_changed_gate_or_config() {
        let Some(runtime) = runtime() else { return };
        let _cpu = cpu();
        let device = TensorDevice::new(runtime).unwrap();
        let mut module = layer(&[0.8, -0.4, 1.1], 0.3, 24);
        let x = [-2., 0.3, 1., -0.5, 0.8, 2.];
        let view = device
            .upload(&[3, 2], &[-2., -0.5, 0.3, 0.8, 1., 2.])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        let expected = module
            .forward(&Tensor::from_vec(2, 3, x.to_vec()).unwrap())
            .unwrap();
        let held = module.forward_resident(&view).unwrap();
        close(
            &module
                .forward_resident(&view)
                .unwrap()
                .snapshot()
                .unwrap()
                .read()
                .unwrap(),
            expected.data(),
        );
        assert_eq!(module.resident_forward_stats().unwrap().compilations, 1);
        assert_eq!(module.resident_forward_stats().unwrap().cache_hits, 1);
        module
            .visit_parameters_mut(&mut |p| {
                p.value_mut().data_mut()[0] = 0.2;
                Ok(())
            })
            .unwrap();
        module.forward_resident(&view).unwrap();
        assert_eq!(module.resident_forward_stats().unwrap().compilations, 2);
        module = module.with_coupling(0.3).unwrap();
        let new_expected = module
            .forward(&Tensor::from_vec(2, 3, x.to_vec()).unwrap())
            .unwrap();
        close(
            &module
                .forward_resident(&view)
                .unwrap()
                .snapshot()
                .unwrap()
                .read()
                .unwrap(),
            new_expected.data(),
        );
        assert_eq!(module.resident_forward_stats().unwrap().compilations, 3);
        module.clear_resident_forward_cache();
        drop(module);
        close(&held.snapshot().unwrap().read().unwrap(), expected.data());
    }

    #[test]
    fn topos_review_cached_resident_route_revalidates_optimizer_alignment() {
        let Some(runtime) = runtime() else { return };
        let device = TensorDevice::new(runtime).unwrap();
        let mut module = layer(&[0.8, -0.4], 0.3, 8);
        let input = device.upload(&[2, 2], &[0.25; 4]).unwrap();
        let held = module.forward_resident(&input).unwrap();
        let expected = held.snapshot().unwrap().read().unwrap();
        let before = module.resident_forward_stats().unwrap();
        let other = OpenCartesianTopos::new(-1., 1e-6, 0.25, 16, 8)
            .unwrap()
            .with_porosity(0.3)
            .unwrap();
        module
            .parameter_mut()
            .attach_hypergrad_with_topos(-1., 0.01, other)
            .unwrap();
        assert!(module.forward_resident(&input).is_err());
        assert_eq!(module.resident_forward_stats().unwrap(), before);
        close(&held.snapshot().unwrap().read().unwrap(), &expected);
    }

    #[test]
    fn topos_residual_drive_overflow_is_atomic_and_can_retry() {
        let Some(runtime) = runtime() else { return };
        let _cpu = cpu();
        let module = ToposResonator::from_shared_gate(
            "guard",
            Tensor::from_vec(1, 1, vec![1.]).unwrap(),
            ToposResonatorConfig::new(0.5, 1).unwrap(),
            OpenCartesianTopos::new(-1., 1e-6, f32::MAX, 4, 1).unwrap(),
        )
        .unwrap();
        let input = [f32::MAX * 0.8];
        assert!(module
            .forward(&Tensor::from_vec(1, 1, input.to_vec()).unwrap())
            .is_err());
        let base = plan(&module, &[1, 1]);
        let mut inference = base.compile_graph_wgpu(runtime.clone()).unwrap();
        inference.upload(&input).unwrap();
        inference.dispatch().unwrap();
        assert!(inference.snapshot().unwrap().read().is_err());
        let mut training = base
            .compile_graph_training_wgpu(runtime, GraphGradientPolicy::Exact)
            .unwrap();
        training.upload_batch(&input, &input).unwrap();
        training.step(0.1).unwrap();
        assert!(training.state_snapshot().unwrap().read().is_err());
        assert_eq!(
            training
                .parameter_snapshot()
                .unwrap()
                .read()
                .unwrap()
                .parameters()[0]
                .values,
            [1.]
        );
        training.upload_batch(&[0.25], &[0.]).unwrap();
        training.step(0.1).unwrap();
        let state = training.state_snapshot().unwrap().read().unwrap();
        close(&state.prediction, &[0.25]);
        close(&state.graph.parameters()[0].values, &[0.9875]);
    }
}
