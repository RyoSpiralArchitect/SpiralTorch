#[cfg(not(target_arch = "wasm32"))]
#[path = "support/attention_vjp.rs"]
mod support;

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
}
