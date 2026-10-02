#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

#[path = "../examples/support/attention_chain.rs"]
mod support;

#[test]
fn qkv_geometry_attention_output_matches_pytorch_without_host_intermediates() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let report = pollster::block_on(support::run()).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["checks"].as_array().unwrap().len(), 12);
    eprintln!("{report}");
}
