#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "../examples/support/residual_attention.rs"]
mod support;

#[test]
fn full_residual_attention_vjp_and_updates_match_torch() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let report = pollster::block_on(support::run()).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["checks"].as_array().unwrap().len(), 60);
    assert_eq!(report["training"].as_array().unwrap().len(), 2);
    for case in report["training"].as_array().unwrap() {
        assert_eq!(case["updates"].as_array().unwrap().len(), 32);
        assert_eq!(case["rejected_update_preserves_all_parameters"], true);
        assert_eq!(case["recovery"], true);
    }
    assert_eq!(report["guards"].as_object().unwrap().len(), 12);
    assert!(report["guards"]
        .as_object()
        .unwrap()
        .values()
        .all(|v| v == true));
    eprintln!("{report}");
}
