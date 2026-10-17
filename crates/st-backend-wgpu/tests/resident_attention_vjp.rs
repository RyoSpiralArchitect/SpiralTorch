#[cfg(not(target_arch = "wasm32"))]
#[path = "support/attention_vjp.rs"]
mod support;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn analytic_attention_range_witnesses() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("attention.range").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = st_backend_wgpu::resident_tensor::TensorDevice::new(runtime).unwrap();
    let report = pollster::block_on(support::attention_range::run(&device)).unwrap();
    println!("{report}");
    assert_eq!(report["passed"], true);
    assert_eq!(report["cases"].as_array().unwrap().len(), 46);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn attention_gradients_and_resident_sgd_match_pytorch() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("attention.vjp.torch").unwrap();
    let report = pollster::block_on(support::run(runtime)).unwrap();
    println!("{report}");
    assert_eq!(report["passed"], true);
    assert_eq!(
        report["range_checks"]["cases"].as_array().unwrap().len(),
        46
    );
    assert_eq!(
        report["shift_checks"]["cases"].as_array().unwrap().len(),
        13
    );
}
