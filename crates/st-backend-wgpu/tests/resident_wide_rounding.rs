#[path = "support/wide_rounding.rs"]
mod support;

#[test]
fn retained_wide_components_round_like_the_integer_oracle() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("wide.rounding.test").unwrap();
    let report = pollster::block_on(support::run(runtime)).unwrap();
    println!("{report}");
    assert_eq!(report["passed"], true);
    assert_eq!(report["attention"].as_array().unwrap().len(), 12);
}
