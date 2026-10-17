use super::*;

#[test]
fn parameter_clip_shader_validates() {
    let module = naga::front::wgsl::parse_str(&source()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

mod checks {
    use crate as st_backend_wgpu;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/support/resident_parameter_clip.rs"
    ));
}

#[test]
fn parameter_clip_global_norm_freeze_guards_and_optimizer_updates() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let runtime = runtime::ensure_default_runtime_blocking("parameters.clip")
        .unwrap()
        .0;
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let report = pollster::block_on(checks::run(&TensorDevice::new(runtime).unwrap())).unwrap();
    assert_eq!(report["passed"], true);
}
