#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "../examples/support/byte_decoder.rs"]
mod support;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn complete_byte_decoder_matches_torch_and_preserves_causality() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte_decoder.torch").unwrap();
    let report = pollster::block_on(support::run(runtime)).unwrap();
    println!("{report}");
}
