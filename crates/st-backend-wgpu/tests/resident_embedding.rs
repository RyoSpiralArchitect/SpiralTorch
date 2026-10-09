#[cfg(not(target_arch = "wasm32"))]
#[path = "support/embedding.rs"]
mod support;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn embedding_vjp_and_ce_sgd_match_pytorch() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("embedding.torch").unwrap();
    let report = pollster::block_on(support::run(runtime)).unwrap();
    println!("{report}");
}
