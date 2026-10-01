use super::*;

#[test]
fn scalar_receipt_decoding_preserves_rejection_priority_and_finite_guards() {
    let valid = vec![vec![0; 3], vec![(-2.0f32).to_bits().to_le()], vec![0]];
    assert_eq!(
        ResidentParameterScalarReadback::decode(&valid, 1, 7).unwrap(),
        (7, -2.0)
    );
    for scalar in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut words = valid.clone();
        words[1][0] = scalar.to_bits().to_le();
        assert!(matches!(
            ResidentParameterScalarReadback::decode(&words, 1, 7),
            Err(TrainingError::Tensor(TensorError::NonFinite))
        ));
    }
    let mut rejected = valid.clone();
    rejected[0] = vec![1u32.to_le(), 0, 1u32.to_le()];
    rejected[1][0] = f32::NAN.to_bits().to_le();
    rejected[2][0] = INVALID_TENSOR_FLAG.to_le();
    assert!(matches!(
        ResidentParameterScalarReadback::decode(&rejected, 1, 7),
        Err(TrainingError::Rejected { stage: 0, flags: 1 })
    ));
    rejected[0][2] = 0;
    assert!(matches!(
        ResidentParameterScalarReadback::decode(&rejected, 1, 7),
        Err(TrainingError::InvalidReadback)
    ));
    let mut guarded = valid.clone();
    guarded[2][0] = INVALID_TENSOR_FLAG.to_le();
    assert!(matches!(
        ResidentParameterScalarReadback::decode(&guarded, 1, 7),
        Err(TrainingError::Tensor(TensorError::NonFinite))
    ));
    for words in [vec![], vec![vec![0; 3]], vec![vec![0; 4], vec![0], vec![0]]] {
        assert!(matches!(
            ResidentParameterScalarReadback::decode(&words, 1, 7),
            Err(TrainingError::InvalidReadback)
        ));
    }
    assert!(matches!(
        ResidentParameterScalarReadback::decode(&valid, usize::MAX, 7),
        Err(TrainingError::InvalidReadback)
    ));
}

#[test]
fn restored_parameter_revision_has_a_fresh_owner_identity() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("parameters.restore").unwrap();
    let device = TensorDevice::new(runtime).unwrap();
    let original = ResidentParameters::new(vec![device.upload(&[1], &[2.0]).unwrap()]).unwrap();
    let snapshot = original.snapshot();
    let old = snapshot
        .bind_gradients(vec![device.upload(&[1], &[1.0]).unwrap()])
        .unwrap();
    let mut restored =
        ResidentParameters::from_restored_values(snapshot.values().to_vec(), 0).unwrap();
    assert!(!restored.is_current(&snapshot));
    assert!(matches!(
        restored.sgd(&old, 0.1),
        Err(TrainingError::ParameterVersion)
    ));
    let mut restored =
        ResidentParameters::from_restored_values(snapshot.values().to_vec(), 41).unwrap();
    let gradient = restored
        .snapshot()
        .bind_gradients(vec![device.upload(&[1], &[1.0]).unwrap()])
        .unwrap();
    assert_eq!(
        restored
            .sgd(&gradient, 0.5)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        42
    );
    assert_eq!(
        restored.snapshot().values()[0]
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![1.5]
    );
    assert_eq!(
        snapshot.values()[0].snapshot().unwrap().read().unwrap(),
        vec![2.0]
    );
}

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
    let mut owner = ResidentParameters::new(vec![local]).unwrap();
    assert!(matches!(
        owner.snapshot().bind_gradients(vec![foreign.clone()]),
        Err(TrainingError::Tensor(TensorError::DeviceMismatch))
    ));
    let gradient = owner
        .snapshot()
        .bind_gradients(vec![device.upload(&[1], &[0.0]).unwrap()])
        .unwrap();
    let update = owner.sgd(&gradient, 0.0).unwrap();
    assert!(matches!(
        update.snapshot_with_scalar(&foreign),
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
