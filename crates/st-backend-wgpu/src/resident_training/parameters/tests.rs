use super::*;

// Exercise the exact same sequence through native Metal and browser WebGPU.
mod checks {
    use crate as st_backend_wgpu;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/support/resident_parameters_checks.rs"
    ));
}

#[test]
fn parameter_update_shader_validates() {
    let module = naga::front::wgsl::parse_str(&source()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn resident_parameter_transactions_and_conv2d_learning() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("parameters.contract").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let result = pollster::block_on(checks::run(&device)).unwrap();
    assert_eq!(result["status"], "passed");
    assert!(matches!(
        ResidentParameters::new(vec![]),
        Err(TrainingError::ParameterLayout)
    ));
    assert!(matches!(
        ResidentParameters::new(vec![device.upload(&[0], &[]).unwrap()]),
        Err(TrainingError::ParameterLayout)
    ));
}

#[test]
fn resident_parameters_reject_foreign_devices_and_preserve_invalid_guards() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("parameters.devices").unwrap();
    let device = TensorDevice::new(runtime).unwrap();
    let other = TensorDevice::new(
        pollster::block_on(WgpuRuntime::request_headless("parameters.other")).unwrap(),
    )
    .unwrap();
    let local = device.upload(&[1], &[1.0]).unwrap();
    let foreign = other.upload(&[1], &[1.0]).unwrap();
    assert!(matches!(
        ResidentParameters::new(vec![local.clone(), foreign.clone()]),
        Err(TrainingError::Tensor(TensorError::DeviceMismatch))
    ));
    let owner = ResidentParameters::new(vec![local]).unwrap();
    assert!(matches!(
        owner.snapshot().bind_gradients(vec![foreign]),
        Err(TrainingError::Tensor(TensorError::DeviceMismatch))
    ));

    let invalid = device
        .upload(&[1], &[f32::MAX])
        .unwrap()
        .mul(&device.upload(&[1], &[2.0]).unwrap())
        .unwrap();
    let mut owner = ResidentParameters::new(vec![invalid]).unwrap();
    let gradient = owner
        .snapshot()
        .bind_gradients(vec![device.upload(&[1], &[0.0]).unwrap()])
        .unwrap();
    let update = owner.sgd(&gradient, 0.0).unwrap();
    assert!(matches!(
        update.snapshot().unwrap().read(),
        Err(TrainingError::Rejected { .. })
    ));
    assert!(matches!(
        owner.snapshot().values()[0].snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
}
