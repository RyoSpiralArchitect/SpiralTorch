#[cfg(not(target_arch = "wasm32"))]
#[path = "support/attention_vjp.rs"]
mod support;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn causal_pair_bias_cpu_reference_matches_frozen_torch() {
    let report = support::incident_cpu_reference().unwrap();
    println!("{report}");
    assert_eq!(report["case"], "mask_2_bias_2");
    assert_eq!(
        (report["torch_query0"].as_f64().unwrap() as f32).to_bits(),
        (-0.036814254_f32).to_bits()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "manual single-case Metal diagnostic; ordinary CI retains the complete suite"]
fn one_causal_pair_bias_pullback_on_metal() {
    assert_eq!(
        std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref(),
        Ok("1")
    );
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("attention.vjp.causal-pair")
            .unwrap();
    let report = pollster::block_on(support::incident_layouts(runtime)).unwrap();
    println!("{report}");
    assert_eq!(report["checks"].as_array().unwrap().len(), 2);
}

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
