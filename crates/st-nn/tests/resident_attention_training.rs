#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "../examples/support/attention_training.rs"]
mod support;

#[test]
fn projected_attention_training_matches_torch_on_the_resident_route() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let report = pollster::block_on(support::run()).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["checks"].as_array().unwrap().len(), 30);
    assert_eq!(report["updates"].as_array().unwrap().len(), 16);
    assert_eq!(
        report["final_parameter_errors"].as_array().unwrap().len(),
        4
    );
    for check in report["checks"].as_array().unwrap() {
        assert_eq!(
            check["max_abs_error"]["parameters"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }
    assert_eq!(report["rejected_update_preserves_all_parameters"], true);
    assert_eq!(report["rejected_update_invalidates_old_gradients"], true);
    eprintln!("{report}");
}
