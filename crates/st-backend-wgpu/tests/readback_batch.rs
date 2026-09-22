#![cfg(not(target_arch = "wasm32"))]
#[path = "../examples/support/readback_cases.rs"]
mod cases;

#[test]
fn ordered_owned_readback_batch_when_enabled() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let runtime = pollster::block_on(st_backend_wgpu::runtime::WgpuRuntime::request_headless(
        "batch.test",
    ))
    .unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    pollster::block_on(cases::run(&runtime, |_| {})).unwrap();
}
