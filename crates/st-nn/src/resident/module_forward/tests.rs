use super::*;
use crate::{Gelu, Linear, ModuleTrainer, Relu, Scaler, Sequential};
use st_backend_wgpu::{resident_tensor::TensorDevice, runtime};
use st_core::backend::device_caps::DeviceCaps;

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
