use super::*;
use crate::execution::{push_backend_policy, BackendPolicy};
use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn tape(layer: &ToposResonator) -> Arc<ToposResonatorLearningBatch> {
    match &layer.last_step.borrow().as_ref().unwrap().saved {
        ToposResonatorSaved::Captured(batch) => batch.clone(),
        #[cfg(feature = "wgpu")]
        ToposResonatorSaved::Recomputed { .. } => panic!("expected a captured CPU tape"),
    }
}

#[test]
fn cpu_shared_capture_reuses_feature_gate_and_matches_expanded_audits() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    for rows in [0, 1, 7, 257] {
        for features in [1, 3, 17] {
            let topos = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 32, rows.max(1) * features)
                .unwrap()
                .with_porosity(0.3)
                .unwrap();
            let mut layer = ToposResonator::with_shared_gate(
                "shared",
                features,
                ToposResonatorConfig::new(0.9, 16).unwrap(),
                topos,
            )
            .unwrap();
            let gate = Tensor::from_fn(1, features, |_, c| (c % 5) as f32 - 1.5).unwrap();
            *layer.parameter_mut().value_mut() = gate.to_layout(Layout::ColMajor).unwrap();
            let pattern = [0.0, -0.0, f32::from_bits(1), -0.3, 0.5, 1.3, -2.0];
            let input =
                Tensor::from_fn(rows, features, |r, c| pattern[(r * features + c) % 7]).unwrap();
            let op = ToposResonatorOperator::new(layer.config(), layer.topos().clone()).unwrap();
            let expanded = op
                .capture(input.data(), &gate.data().repeat(rows), rows, features)
                .unwrap();
            let output = layer
                .step_resonance(&input.to_layout(Layout::ColMajor).unwrap())
                .unwrap();
            assert_eq!(bits(output.output.data()), bits(expanded.output()));
            assert_eq!(output.audit, expanded.step().audit);
            let saved = tape(&layer);
            assert_eq!(saved.gate_layout(), ToposGateLayout::SharedRows);
            assert_eq!(saved.gate().len(), features);
            assert_eq!(bits(saved.gate()), bits(gate.data()));
            assert_eq!(Arc::strong_count(&saved), 2);
            let mut expected_accumulated = vec![0.0_f32; features];
            for scale in [0.3, -0.2, 0.0] {
                let dy =
                    Tensor::from_fn(rows, features, |r, c| scale * ((r + c) % 11) as f32).unwrap();
                let (expected, audit) = expanded.vjp_audited(dy.data()).unwrap();
                let reduced = Tensor::from_vec(rows, features, expected.grad_gate)
                    .unwrap()
                    .try_sum_axis0_with_backend(TensorUtilBackend::Cpu)
                    .unwrap();
                let actual = layer
                    .backward_resonance(&input, &dy.to_layout(Layout::ColMajor).unwrap())
                    .unwrap();
                assert_eq!(bits(actual.grad_input.data()), bits(&expected.grad_input));
                assert_eq!(actual.audit, audit);
                for (sum, gradient) in expected_accumulated.iter_mut().zip(reduced) {
                    *sum += gradient;
                }
                assert_eq!(
                    bits(layer.parameter().gradient().unwrap().data()),
                    bits(&expected_accumulated)
                );
                assert!(Arc::ptr_eq(&saved, &tape(&layer)));
                assert_eq!(Arc::strong_count(&saved), 2);
            }
        }
    }
}

#[test]
fn compact_sum_failures_preserve_prior_gradient_audit_and_tape() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut layer = ToposResonator::with_shared_gate(
        "shared",
        1,
        ToposResonatorConfig::new(0.0, 1).unwrap(),
        OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 4, 5).unwrap(),
    )
    .unwrap();
    *layer.parameter_mut().value_mut() = Tensor::zeros(1, 1).unwrap();
    let input =
        Tensor::from_vec(5, 1, vec![f32::MAX, f32::MAX, -f32::MAX, -f32::MAX, 1.0]).unwrap();
    layer.forward(&input).unwrap();
    let saved = tape(&layer);
    let valid = Tensor::from_vec(5, 1, vec![1.0; 5]).unwrap();
    let first = layer.backward_resonance(&input, &valid).unwrap();
    assert_eq!(layer.parameter().gradient().unwrap().data(), [1.0]);
    assert!(first.audit.grad_gate_rms.is_finite() && first.audit.grad_gate_rms > 1e38);
    for values in [
        vec![1.0, 1.0, 0.0, 0.0, 0.0],
        vec![2.0, -2.0, 0.0, 0.0, 0.0],
        vec![f32::NAN; 5],
    ] {
        assert!(layer
            .backward(&input, &Tensor::from_vec(5, 1, values).unwrap())
            .is_err());
        assert_eq!(layer.parameter().gradient().unwrap().data(), [1.0]);
        assert_eq!(layer.latest_backward_audit(), Some(first.audit));
        assert!(Arc::ptr_eq(&saved, &tape(&layer)));
    }
    layer
        .backward(&input, &Tensor::from_vec(5, 1, vec![-1.0; 5]).unwrap())
        .unwrap();
    assert_eq!(layer.parameter().gradient().unwrap().data(), [0.0]);
}

#[test]
fn empty_shared_input_checks_feature_gate_before_dispatch() {
    for backend in [DeviceCaps::cpu(), DeviceCaps::wgpu(32, true, 256)] {
        let _policy = push_backend_policy(BackendPolicy::from_device_caps_with_config(
            backend,
            ExecutionConfig::new(AcceleratorFallback::Forbid, 0),
        ));
        for invalid in [f32::NAN, f32::INFINITY] {
            let mut layer = ToposResonator::with_shared_gate(
                "shared",
                1,
                ToposResonatorConfig::default(),
                OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 8, 1).unwrap(),
            )
            .unwrap();
            *layer.parameter_mut().value_mut() = Tensor::from_vec(1, 1, vec![invalid]).unwrap();
            assert!(layer.forward(&Tensor::zeros(0, 1).unwrap()).is_err());
            assert!(layer.latest_audit().is_none());
            assert!(layer.parameter().gradient().is_none());
        }
    }
}

#[test]
fn forward_admission_keeps_core_errors_and_rejects_before_route_events() {
    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = events.clone();
    let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
        captured.lock().unwrap().push(event.op_name);
    })));
    let policies = [
        None,
        Some(BackendPolicy::from_device_caps(DeviceCaps::cpu())),
        Some(BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Allow, 1024),
        )),
        Some(BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 0),
        )),
    ];
    for policy in policies {
        let _policy = policy.map(push_backend_policy);
        for shared in [false, true] {
            for layout in [Layout::RowMajor, Layout::ColMajor] {
                for (input_value, gate_value) in [
                    (f32::NAN, 1.0),
                    (f32::INFINITY, 1.0),
                    (1.0, f32::NAN),
                    (1.0, f32::NEG_INFINITY),
                    (f32::MAX, 2.0),
                    (f32::NAN, f32::INFINITY),
                ] {
                    let config = ToposResonatorConfig::new(0.25, 1).unwrap();
                    let topos = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 4, 6).unwrap();
                    let mut layer = if shared {
                        ToposResonator::with_shared_gate("shared", 3, config, topos).unwrap()
                    } else {
                        ToposResonator::with_config_and_topos("elementwise", 2, 3, config, topos)
                            .unwrap()
                    };
                    let gate =
                        Tensor::from_fn(if shared { 1 } else { 2 }, 3, |_, _| gate_value).unwrap();
                    *layer.parameter_mut().value_mut() = gate.to_layout(layout).unwrap();
                    let input = Tensor::from_fn(2, 3, |_, _| input_value).unwrap();
                    let expected = validate_topos_resonator_state_with_layout(
                        layer.core_request(&input, gate.data()),
                        layer.gate_layout(),
                    )
                    .unwrap_err();
                    events.lock().unwrap().clear();
                    let error = layer
                        .forward(&input.to_layout(layout).unwrap())
                        .unwrap_err();
                    assert_eq!(
                        format!("{error:?}"),
                        format!("{:?}", topos_resonator_error(expected))
                    );
                    assert!(layer.latest_audit().is_none());
                    assert!(layer.parameter().gradient().is_none());
                    assert!(!events.lock().unwrap().iter().any(|name| {
                        matches!(*name, "tensor_util_route" | "topos_resonator_forward")
                    }));
                }
            }
        }
    }
    st_tensor::set_thread_meta_observer(previous);
}

#[test]
fn failed_cpu_forward_preserves_valid_tape_and_prior_gradient() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut layer = ToposResonator::from_shared_gate(
        "shared",
        Tensor::from_vec(1, 2, vec![2.0, 2.0]).unwrap(),
        ToposResonatorConfig::new(0.25, 5).unwrap(),
        OpenCartesianTopos::new(-1.0, 1e-6, 4.0, 8, 4).unwrap(),
    )
    .unwrap();
    let valid = Tensor::from_vec(2, 2, vec![0.1, 0.2, -0.3, 0.4]).unwrap();
    let upstream = Tensor::from_vec(2, 2, vec![0.5; 4]).unwrap();
    layer.forward(&valid).unwrap();
    let expected = layer.backward(&valid, &upstream).unwrap();
    let saved = tape(&layer);
    let audit = layer.latest_backward_audit();
    let gradient = bits(layer.parameter().gradient().unwrap().data());
    for invalid in [f32::NAN, f32::INFINITY, f32::MAX] {
        assert!(layer
            .forward(&Tensor::from_vec(2, 2, vec![invalid; 4]).unwrap())
            .is_err());
        assert!(Arc::ptr_eq(&saved, &tape(&layer)));
        assert_eq!(layer.latest_backward_audit(), audit);
        assert_eq!(bits(layer.parameter().gradient().unwrap().data()), gradient);
    }
    assert_eq!(
        bits(layer.backward(&valid, &upstream).unwrap().data()),
        bits(expected.data())
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn gpu_then_cpu_replays_keep_the_same_compact_tape() {
    if !wgpu_dense::is_available() {
        eprintln!("Topos compact tape replay skipped: WGPU is unavailable");
        return;
    }
    let rows = 257;
    let features = 3;
    let mut layer = ToposResonator::with_shared_gate(
        "shared",
        features,
        ToposResonatorConfig::new(0.25, 5).unwrap(),
        OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 16, rows * features)
            .unwrap()
            .with_porosity(0.3)
            .unwrap(),
    )
    .unwrap();
    let input = Tensor::from_fn(rows, features, |r, c| ((r + c) % 17) as f32 * 0.3 - 2.0).unwrap();
    let dy = Tensor::from_fn(rows, features, |r, c| ((r + c) % 7) as f32 * 0.1 - 0.3).unwrap();
    {
        let _policy = push_backend_policy(crate::test_backend_policy(DeviceCaps::cpu(), 1));
        layer.forward(&input).unwrap();
    }
    let saved = tape(&layer);
    let (expected, audit) = saved.vjp_audited_elementwise(dy.data()).unwrap();
    for (index, caps) in [DeviceCaps::wgpu(32, true, 256), DeviceCaps::cpu()]
        .into_iter()
        .enumerate()
    {
        let _policy = push_backend_policy(BackendPolicy::from_device_caps_with_config(
            caps,
            ExecutionConfig::new(AcceleratorFallback::Forbid, 1),
        ));
        let result = layer.backward_resonance(&input, &dy).unwrap();
        for (actual, expected) in result.grad_input.data().iter().zip(&expected.grad_input) {
            assert!((actual - expected).abs() <= 1e-5);
        }
        assert!((result.audit.grad_gate_rms - audit.grad_gate_rms).abs() <= 1e-5);
        for (actual, expected) in layer
            .parameter()
            .gradient()
            .unwrap()
            .data()
            .iter()
            .zip(&expected.grad_gate)
        {
            assert!((actual - expected * (index + 1) as f32).abs() <= 5e-5);
        }
        assert!(Arc::ptr_eq(&saved, &tape(&layer)));
        assert_eq!(saved.gate().len(), features);
        assert_eq!(Arc::strong_count(&saved), 2);
    }
    eprintln!("Topos compact tape replay executed WGPU then CPU backward on the same CPU capture");
}
