use super::*;
use crate::{Gelu, Linear, ModuleTrainer, Relu, Scaler, Sequential};
use st_backend_wgpu::{resident_tensor::TensorDevice, runtime};
use st_core::backend::device_caps::DeviceCaps;

thread_local! {
    pub(super) static EXACT_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(not(target_arch = "wasm32"))]
fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("nn.module.forward.test").unwrap();
    assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
    Some(TensorDevice::new(runtime).unwrap())
}

fn model() -> Sequential {
    let mut m = Sequential::new();
    m.push(Linear::new("up", 3, 4).unwrap());
    m.push(Gelu::new());
    m.push(Scaler::new("hidden", 4).unwrap());
    m.push(Relu::new());
    m.push(Linear::new("down", 4, 3).unwrap());
    m.push(Scaler::new("out", 3).unwrap());
    m
}

fn cpu() -> crate::execution::BackendPolicyGuard {
    crate::execution::push_backend_policy(
        crate::execution::BackendPolicy::from_device_caps_with_config(
            DeviceCaps::cpu(),
            Default::default(),
        ),
    )
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && (a - b).abs() < 2e-5, "{a} != {b}");
    }
}

#[test]
fn operation_cache_checks_bits_layout_program_and_foreign_values() {
    let same_operations = |a: &[InferenceOp], b: &[InferenceOp]| {
        let mut stamps: Vec<_> = a.iter().map(OperationStamp::capture).collect();
        super::same_operations(a, b, &mut stamps)
    };
    let value = Tensor::from_vec(1, 2, vec![0., 1.]).unwrap();
    let ops = vec![
        InferenceOp::Scale {
            gain: value.clone(),
        },
        InferenceOp::Relu,
    ];
    let saved: Vec<_> = ops.iter().map(InferenceOp::snapshot).collect();
    assert!(same_operations(&saved, &ops));
    assert!(!same_operations(
        &saved,
        &[
            InferenceOp::Scale {
                gain: Tensor::from_vec(1, 2, vec![-0., 1.]).unwrap()
            },
            InferenceOp::Relu
        ]
    ));
    assert!(!same_operations(
        &saved,
        &[
            InferenceOp::Scale {
                gain: value.to_layout(Layout::ColMajor).unwrap()
            },
            InferenceOp::Relu
        ]
    ));
    assert!(!same_operations(
        &saved,
        &[InferenceOp::Scale { gain: value }, InferenceOp::Gelu]
    ));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn terminal_forward_uses_the_original_model_and_shared_cache() {
    let Some(device) = device() else { return };
    let _cpu = cpu();
    let values: Vec<_> = (0..18).map(|i| (i as f32 - 8.) / 16.).collect();
    let input = device
        .upload(&[3, 2, 3], &values)
        .unwrap()
        .permute(&[1, 0, 2])
        .unwrap();
    let host = Tensor::from_vec(6, 3, input.snapshot().unwrap().read().unwrap()).unwrap();
    let modules: Vec<Box<dyn Module>> = vec![
        Box::new(model()),
        Box::new(Linear::new("l", 3, 5).unwrap()),
        Box::new(Scaler::new("g", 3).unwrap()),
        Box::new(Gelu::new()),
        Box::new(Relu::new()),
        Box::new(Sequential::new()),
    ];
    for module in modules {
        let expected = module.forward(&host).unwrap();
        let capture = module.forward_resident_snapshot(&input).unwrap();
        let original = module.forward_resident(&input).unwrap();
        for _ in 0..12 {
            drop(module.forward_resident_snapshot(&input).unwrap());
        }
        if let Some(stats) = module.resident_forward_stats() {
            assert!(stats.compilations <= 1);
            assert!(
                stats.submitted_forwards == 0
                    || stats
                        == ResidentForwardStats {
                            compilations: 1,
                            cache_hits: 13,
                            submitted_forwards: 14
                        }
            );
        }
        module.clear_resident_forward_cache();
        drop(module);
        close(&capture.read().unwrap(), expected.data());
        close(
            &original.snapshot().unwrap().read().unwrap(),
            expected.data(),
        );
    }
    for shape in [vec![], vec![0, 3]] {
        let values = if shape.is_empty() { vec![-0.] } else { vec![] };
        let input = device.upload(&shape, &values).unwrap();
        let capture = Sequential::new().forward_resident_snapshot(&input).unwrap();
        assert_eq!(capture.layout().shape(), shape);
        assert_eq!(
            capture
                .read()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn exact_comparison_preserves_bits_at_odd_lengths_and_nonfinite_payloads() {
    let patterns = [
        0,
        0x8000_0000,
        1,
        0x3f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0x7fc0_0002,
    ];
    for len in [1, 3, 31, 1025] {
        let original = Tensor::from_vec(
            1,
            len,
            (0..len)
                .map(|i| f32::from_bits(patterns[i % patterns.len()]))
                .collect(),
        )
        .unwrap();
        let frozen = original.snapshot();
        assert!(same_tensor(&original, &frozen));
        assert!(same_tensor(&frozen, &frozen.clone()));
        for index in [0, len / 2, len - 1] {
            let mut changed = original.clone();
            changed.data_mut()[index] = f32::from_bits(original.data()[index].to_bits() ^ 1);
            assert!(!same_tensor(&frozen, &changed), "len={len}, index={index}");
            assert!(same_tensor(&frozen, &original));
        }
    }
}

#[test]
fn stamps_skip_scans_refresh_equal_replacements_and_revoke_on_late_export() {
    let mut current = vec![
        InferenceOp::Scale {
            gain: Tensor::from_vec(1, 3, vec![1., 2., 3.]).unwrap(),
        },
        InferenceOp::Gelu,
    ];
    let saved: Vec<_> = current.iter().map(InferenceOp::snapshot).collect();
    let mut stamps: Vec<_> = current.iter().map(OperationStamp::capture).collect();
    EXACT_COMPARISONS.with(|count| count.set(0));
    for _ in 0..20 {
        assert!(same_operations(&saved, &current, &mut stamps));
    }
    EXACT_COMPARISONS.with(|count| assert_eq!(count.get(), 0));
    current[0] = InferenceOp::Scale {
        gain: Tensor::from_vec(1, 3, vec![1., 2., 3.]).unwrap(),
    };
    assert!(same_operations(&saved, &current, &mut stamps));
    EXACT_COMPARISONS.with(|count| assert_eq!(count.get(), 1));
    assert!(same_operations(&saved, &current, &mut stamps));
    EXACT_COMPARISONS.with(|count| assert_eq!(count.get(), 1));
    let InferenceOp::Scale { gain } = &current[0] else {
        panic!()
    };
    let managed = gain.to_dlpack().unwrap();
    let pointer = unsafe { (*managed).dl_tensor.data.cast::<f32>() };
    let _owner = unsafe { Tensor::from_dlpack(managed).unwrap() };
    for _ in 0..2 {
        assert!(same_operations(&saved, &current, &mut stamps));
    }
    EXACT_COMPARISONS.with(|count| assert_eq!(count.get(), 3));
    unsafe {
        *pointer.add(2) = 9.;
    }
    assert!(!same_operations(&saved, &current, &mut stamps));
    assert!(!same_operations(&saved, &current, &mut []));
    unsafe {
        *pointer.add(2) = 3.;
    }
    assert!(same_operations(&saved, &current, &mut stamps));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn native_cached_linear_detects_later_export_and_retains_original_output() {
    let Some(device) = device() else { return };
    let _cpu = cpu();
    let model = Linear::new("late-export", 3, 5).unwrap();
    let input = device.upload(&[2, 3, 3], &[0.25; 18]).unwrap();
    let host = Tensor::from_vec(6, 3, vec![0.25; 18]).unwrap();
    let original_expected = model.forward(&host).unwrap();
    let original = model.forward_resident(&input).unwrap();
    let original_capture = model.forward_resident_snapshot(&input).unwrap();
    model.forward_resident(&input).unwrap();
    let weights = model.weight().value().to_dlpack().unwrap();
    let biases = model.bias().value().to_dlpack().unwrap();
    let wp = unsafe { (*weights).dl_tensor.data.cast::<f32>() };
    let bp = unsafe { (*biases).dl_tensor.data.cast::<f32>() };
    let _weights = unsafe { Tensor::from_dlpack(weights).unwrap() };
    let _biases = unsafe { Tensor::from_dlpack(biases).unwrap() };
    model.forward_resident(&input).unwrap();
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 1);
    unsafe {
        *wp.add(14) += 0.125;
    }
    let changed = model.forward_resident(&input).unwrap();
    let changed_capture = model.forward_resident_snapshot(&input).unwrap();
    let expected = model.forward(&host).unwrap();
    assert_ne!(expected, original_expected);
    close(&changed_capture.read().unwrap(), expected.data());
    close(
        &changed.snapshot().unwrap().read().unwrap(),
        expected.data(),
    );
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 2);
    unsafe {
        *bp.add(4) = f32::NAN;
    }
    let before = model.resident_forward_stats();
    assert!(model.forward_resident(&input).is_err());
    assert!(model.forward_resident_snapshot(&input).is_err());
    assert!(model.forward(&host).is_err());
    assert_eq!(model.resident_forward_stats(), before);
    unsafe {
        *bp.add(4) = 0.;
    }
    model
        .forward_resident(&input)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 2);
    close(
        &original.snapshot().unwrap().read().unwrap(),
        original_expected.data(),
    );
    close(&original_capture.read().unwrap(), original_expected.data());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn ordinary_module_runs_twenty_resident_forwards_with_one_compilation() {
    let Some(device) = device() else { return };
    let _cpu = cpu();
    let model = model();
    let input = device
        .upload(
            &[2, 4, 3],
            &(0..24).map(|i| (i as f32 - 7.) / 16.).collect::<Vec<_>>(),
        )
        .unwrap()
        .permute(&[1, 0, 2])
        .unwrap();
    let logical = input.snapshot().unwrap().read().unwrap();
    let mut reference = Tensor::from_vec(8, 3, logical).unwrap();
    let mut output = input;
    let mut first = None;
    for step in 0..20 {
        reference = model.forward(&reference).unwrap();
        output = model.forward_resident(&output).unwrap();
        if step == 0 {
            first = Some((output.clone(), reference.clone()));
        }
    }
    assert_eq!(output.layout().shape(), &[4, 2, 3]);
    assert_eq!(
        model.resident_forward_stats().unwrap(),
        ResidentForwardStats {
            compilations: 1,
            cache_hits: 19,
            submitted_forwards: 20
        }
    );
    model.clear_resident_forward_cache();
    drop(model);
    close(
        &output.snapshot().unwrap().read().unwrap(),
        reference.data(),
    );
    let (gpu, expected) = first.unwrap();
    close(&gpu.snapshot().unwrap().read().unwrap(), expected.data());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn cache_tracks_trainer_updates_layout_shape_topology_and_explicit_clear() {
    let Some(device) = device() else { return };
    let _cpu = cpu();
    let mut model = model();
    let input = device.upload(&[2, 2, 3], &[0.25; 12]).unwrap();
    let host = Tensor::from_vec(4, 3, vec![0.25; 12]).unwrap();
    let first = model.forward_resident(&input).unwrap();
    let first_expected = model.forward(&host).unwrap();
    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1., 0.01, 0.01);
    trainer.prepare(&mut model).unwrap();
    model
        .backward(&host, &Tensor::from_vec(4, 3, vec![0.5; 12]).unwrap())
        .unwrap();
    trainer.step(&mut model).unwrap();
    let output = model.forward_resident(&input).unwrap();
    let expected = model.forward(&host).unwrap();
    assert_ne!(first_expected, expected);
    close(&output.snapshot().unwrap().read().unwrap(), expected.data());
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 2);
    model
        .visit_parameters_mut(&mut |p| {
            if p.name() == "up::weight" {
                let v = p.value().to_layout(Layout::ColMajor)?;
                p.load_value(&v)?;
            }
            Ok(())
        })
        .unwrap();
    close(
        &model
            .forward_resident(&input)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        expected.data(),
    );
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 3);
    let reshaped = input.reshape(&[4, 3]).unwrap();
    model.forward_resident(&reshaped).unwrap();
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 4);
    model.push(Gelu::new());
    close(
        &model
            .forward_resident(&reshaped)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        model.forward(&host).unwrap().data(),
    );
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 5);
    model.clear_resident_forward_cache();
    model
        .forward_resident(&reshaped)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 6);
    close(
        &first.snapshot().unwrap().read().unwrap(),
        first_expected.data(),
    );
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn external_dlpack_writes_recompile_and_invalid_parameters_never_reuse_old_weights() {
    let Some(device) = device() else { return };
    let gain = Tensor::from_vec(1, 3, vec![1., 2., 3.]).unwrap();
    let export = gain.to_dlpack().unwrap();
    let pointer = unsafe { (*export).dl_tensor.data.cast::<f32>() };
    let owner = unsafe { Tensor::from_dlpack(export).unwrap() };
    let model = Scaler::from_gain("shared", gain).unwrap();
    let input = device.upload(&[2, 3], &[1.; 6]).unwrap();
    let first = model.forward_resident(&input).unwrap();
    unsafe {
        *pointer = 4.;
    }
    close(
        &model
            .forward_resident(&input)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &[4., 2., 3., 4., 2., 3.],
    );
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 2);
    unsafe {
        *pointer = f32::NAN;
    }
    let before = model.resident_forward_stats();
    assert!(model.forward_resident(&input).is_err());
    assert_eq!(model.resident_forward_stats(), before);
    unsafe {
        *pointer = 4.;
    }
    model
        .forward_resident(&input)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 2);
    close(
        &first.snapshot().unwrap().read().unwrap(),
        &[1., 2., 3., 1., 2., 3.],
    );
    drop(owner);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn cache_detects_signed_zero_and_last_word_foreign_linear_updates() {
    let Some(device) = device() else { return };
    let linear = Linear::new("foreign", 33, 35).unwrap();
    let weights = linear.weight().value().to_dlpack().unwrap();
    let biases = linear.bias().value().to_dlpack().unwrap();
    let wp = unsafe { (*weights).dl_tensor.data.cast::<f32>() };
    let bp = unsafe { (*biases).dl_tensor.data.cast::<f32>() };
    let weights = unsafe { Tensor::from_dlpack(weights).unwrap() };
    let biases = unsafe { Tensor::from_dlpack(biases).unwrap() };
    unsafe {
        *wp = 0.;
    }
    let mut model = Sequential::new();
    model.push(linear);
    model.push(Gelu::new());
    let input = device.upload(&[2, 3, 33], &[0.001; 198]).unwrap();
    let reference = || {
        let mut output = Vec::new();
        for _ in 0..6 {
            for col in 0..35 {
                let mut value = 0.;
                for row in 0..33 {
                    value += 0.001 * weights.data()[row * 35 + col];
                }
                value += biases.data()[col];
                output.push(
                    st_kernel_contracts::elementwise::ElementwiseOp::Gelu
                        .apply(value, 0.)
                        .unwrap(),
                );
            }
        }
        output
    };
    let first = model.forward_resident(&input).unwrap();
    let initial = reference();
    close(&first.snapshot().unwrap().read().unwrap(), &initial);
    for change in 0..3 {
        // Producer writes are serialized between forwards, not concurrent with
        // Rust borrows or GPU consumption of a mutable host buffer.
        unsafe {
            match change {
                0 => *wp = -0.,
                1 => *wp.add(1154) += 0.5,
                _ => *bp.add(34) += 0.25,
            }
        }
        let output = model.forward_resident(&input).unwrap();
        close(&output.snapshot().unwrap().read().unwrap(), &reference());
        assert_eq!(
            model.resident_forward_stats().unwrap().compilations,
            change + 2
        );
    }
    let previous = unsafe { *wp.add(1154) };
    let before = model.resident_forward_stats();
    unsafe {
        *wp.add(1154) = f32::NAN;
    }
    assert!(model.forward_resident(&input).is_err());
    assert_eq!(model.resident_forward_stats(), before);
    unsafe {
        *wp.add(1154) = previous;
    }
    close(
        &model
            .forward_resident(&input)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &reference(),
    );
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 4);
    close(&first.snapshot().unwrap().read().unwrap(), &initial);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn invalid_input_flags_survive_zero_gain_and_unary_modules_and_recover() {
    let Some(device) = device() else { return };
    let mut model = Sequential::new();
    model.push(Scaler::from_gain("zero", Tensor::zeros(1, 3).unwrap()).unwrap());
    model.push(Relu::new());
    let large = device.upload(&[2, 3], &[f32::MAX; 6]).unwrap();
    let invalid = large.mul(&large).unwrap();
    let output = model.forward_resident(&invalid).unwrap();
    let output = Gelu::new().forward_resident(&output).unwrap();
    let output = Relu::new().forward_resident(&output).unwrap();
    assert!(output.snapshot().unwrap().read().is_err());
    let good = device.upload(&[2, 3], &[1.; 6]).unwrap();
    close(
        &model
            .forward_resident(&good)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &[0.; 6],
    );
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 1);
    let empty = Sequential::new();
    assert!(empty
        .forward_resident(&invalid)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .is_err());
    assert_eq!(
        empty.resident_forward_stats().unwrap(),
        ResidentForwardStats::default()
    );
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn device_identity_reuses_wrappers_but_rebuilds_for_another_device() {
    let Some(device) = device() else { return };
    let model = model();
    let input = device.upload(&[2, 3], &[0.25; 6]).unwrap();
    let first = model.forward_resident(&input).unwrap();
    let wrapped = runtime::WgpuRuntime::new(
        device.runtime().context().clone(),
        device.runtime().adapter_info().clone(),
    );
    let wrapped = TensorDevice::new(wrapped).unwrap();
    model
        .forward_resident(&wrapped.upload(&[2, 3], &[0.25; 6]).unwrap())
        .unwrap();
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 1);
    let other =
        pollster::block_on(runtime::WgpuRuntime::request_headless("nn.module.other")).unwrap();
    assert!(!other
        .context()
        .shares_handles_with(device.runtime().context()));
    let other = TensorDevice::new(other).unwrap();
    let result = model
        .forward_resident(&other.upload(&[2, 3], &[0.25; 6]).unwrap())
        .unwrap();
    assert_eq!(model.resident_forward_stats().unwrap().compilations, 2);
    close(
        &result.snapshot().unwrap().read().unwrap(),
        &first.snapshot().unwrap().read().unwrap(),
    );
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn committed_tensor_plan_is_not_silently_bypassed() {
    use st_core::backend::{
        device_caps::BackendKind,
        execution_plan::{
            evaluate_runtime_execution_plan, BackendPolicy, RuntimeComponentResolution,
            RuntimeExecutionPlanRequest,
        },
        runtime_probe::{evaluate_runtime_device_probe, RuntimeDeviceProbeRequest},
    };
    let Some(device) = device() else { return };
    let probe = evaluate_runtime_device_probe(RuntimeDeviceProbeRequest {
        requested_backend: BackendKind::Cpu,
        caps: DeviceCaps::cpu(),
        mps_probe: None,
        requested_workgroup: None,
        cols: None,
        tile_hint: None,
        compaction_hint: None,
    })
    .unwrap();
    let plan = evaluate_runtime_execution_plan(RuntimeExecutionPlanRequest {
        runtime_probe: probe,
        execution_config: Default::default(),
        component_resolution: RuntimeComponentResolution::Concrete,
        component_workloads: Vec::new(),
        component_capability_observation: None,
        tensor_util_values: Some(6),
        required_native_components: Vec::new(),
    })
    .unwrap();
    let policy = BackendPolicy::try_from_runtime_plan(&plan).unwrap();
    let net = model();
    let x = device.upload(&[2, 3], &[0.25; 6]).unwrap();
    net.forward_resident(&x).unwrap();
    let before = net.resident_forward_stats();
    let scope = crate::execution::push_backend_policy(policy);
    for module in [&net as &dyn Module, &Gelu::new(), &Sequential::new()] {
        assert!(matches!(
            module.forward_resident_snapshot(&x),
            Err(InferenceError::ResidentForwardPolicy)
        ));
    }
    assert!(matches!(
        net.forward_resident(&x),
        Err(InferenceError::ResidentForwardPolicy)
    ));
    assert!(matches!(
        Gelu::new().forward_resident(&x),
        Err(InferenceError::ResidentForwardPolicy)
    ));
    assert!(matches!(
        Sequential::new().forward_resident(&x),
        Err(InferenceError::ResidentForwardPolicy)
    ));
    assert_eq!(net.resident_forward_stats(), before);
    let binding = st_tensor::execution::current_execution_plan_binding().unwrap();
    let nested = cpu();
    assert!(matches!(
        net.forward_resident(&x),
        Err(InferenceError::ResidentForwardPolicy)
    ));
    drop(nested);
    drop(scope);
    let direct = st_tensor::execution::push_execution_plan_binding(binding);
    assert!(crate::execution::current_backend_policy().is_none());
    assert!(matches!(
        net.forward_resident(&x),
        Err(InferenceError::ResidentForwardPolicy)
    ));
    assert!(matches!(
        Relu::new().forward_resident(&x),
        Err(InferenceError::ResidentForwardPolicy)
    ));
    assert_eq!(net.resident_forward_stats(), before);
    drop(direct);
    net.forward_resident(&x)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(net.resident_forward_stats().unwrap().compilations, 1);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn counter_overflow_rejects_before_replacing_or_dispatching() {
    let Some(device) = device() else { return };
    let x = device.upload(&[2, 3], &[0.25; 6]).unwrap();
    let cache = ResidentForwardCache::default();
    cache.0.borrow_mut().stats.submitted_forwards = u64::MAX;
    assert!(cache.forward(vec![InferenceOp::Relu], &x).is_err());
    assert!(cache.0.borrow().current.is_none());
    cache.0.borrow_mut().stats = ResidentForwardStats::default();
    let first = cache.forward(vec![InferenceOp::Relu], &x).unwrap();
    cache.0.borrow_mut().stats.compilations = u64::MAX;
    let before = cache.stats();
    assert!(cache.forward(vec![InferenceOp::Gelu], &x).is_err());
    assert_eq!(cache.stats(), before);
    close(&first.snapshot().unwrap().read().unwrap(), &[0.25; 6]);
    cache
        .forward(vec![InferenceOp::Relu], &x)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(cache.stats().cache_hits, 1);
}
