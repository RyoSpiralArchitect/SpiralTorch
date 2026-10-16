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
