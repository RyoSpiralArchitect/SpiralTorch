#[path = "support/bias_reduction.rs"]
mod support;

#[test]
fn dense_bias_reduction_preserves_residuals_and_guards() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("bias.reduction").unwrap();
    let report = pollster::block_on(support::run(runtime)).unwrap();
    println!("{report}");
    assert_eq!(report["passed"], true);
    assert_eq!(report["checks"].as_array().unwrap().len(), 8);
    assert_eq!(report["guards"].as_array().unwrap().len(), 5);
    assert!(report["integer_oracle"].as_array().unwrap().len() >= 80);
}
