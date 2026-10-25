use super::*;

mod checks {
    use crate as st_backend_wgpu;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/support/resident_adamw_checks.rs"
    ));
}

mod numerics_checks {
    use crate as st_backend_wgpu;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/support/resident_adamw_numerics.rs"
    ));
}

#[test]
fn adamw_shaders_validate() {
    for commit in [false, true] {
        let module = naga::front::wgsl::parse_str(&source(commit)).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn retained_adamw_handles_survive_updates_and_owner_drop() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("adamw.retained").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let mut owner = ResidentParameters::new(vec![
        device.upload(&[2], &[1.0, -2.0]).unwrap(),
        device.upload(&[1], &[0.5]).unwrap(),
    ])
    .unwrap();
    assert!(owner.retain_adamw_snapshot().is_none());
    owner
        .enable_adamw(AdamWConfig::new(0.9, 0.999, 1e-5).unwrap())
        .unwrap();
    let mut retained = Vec::new();
    let mut copied = Vec::new();
    for step in 0..=2 {
        retained.push(owner.retain_adamw_snapshot().unwrap());
        copied.push(owner.adamw_snapshot().unwrap().unwrap());
        if step < 2 {
            let gradient = owner
                .snapshot()
                .bind_gradients(vec![
                    device.upload(&[2], &[0.25 + step as f32, -0.5]).unwrap(),
                    device.upload(&[1], &[0.125]).unwrap(),
                ])
                .unwrap();
            owner
                .adamw_with_steps(
                    &gradient,
                    &[
                        AdamWStep::new(0.01, 0.01).unwrap(),
                        AdamWStep::new(0.0, 0.0).unwrap(),
                    ],
                )
                .unwrap();
        }
    }
    drop(owner);
    drop(device);
    // Delayed copies must use the retained shapes, moments and individual clocks,
    // not the final owner state. No readback occurs before both updates finish.
    for (step, (handles, copied)) in retained.into_iter().zip(copied).enumerate() {
        let state = handles.snapshot().unwrap().read().unwrap();
        let expected = copied.read().unwrap();
        assert_eq!(state.config, expected.config);
        for (actual, expected) in state.slots.iter().zip(&expected.slots) {
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(actual.steps, expected.steps);
            assert!(actual
                .first
                .iter()
                .chain(&actual.second)
                .zip(expected.first.iter().chain(&expected.second))
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert_eq!(state.slots.len(), 2);
        assert_eq!(state.slots[0].steps, step as u32);
        assert_eq!(state.slots[1].steps, 0);
    }
}

#[test]
fn adamw_native_matches_torch_and_preserves_transactional_history() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("adamw.contract").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let result = pollster::block_on(checks::run(&TensorDevice::new(runtime).unwrap())).unwrap();
    assert_eq!(result["status"], "passed");
}

#[test]
fn adamw_native_same_gradient_long_replay_matches_torch() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("adamw.numerics").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let result =
        pollster::block_on(numerics_checks::run(&TensorDevice::new(runtime).unwrap())).unwrap();
    assert_eq!(result["status"], "passed", "{result}");
    assert_eq!(result["updates_per_case"], 1024);
    assert_eq!(result["cases"].as_array().unwrap().len(), 3);
    assert_eq!(result["restored_clock_probes"].as_array().unwrap().len(), 8);
}
