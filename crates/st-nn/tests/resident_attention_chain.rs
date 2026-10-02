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
    assert!(report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["merged_heads_max_abs_error"].is_number()));
    eprintln!("{report}");
}

#[test]
fn explicit_projection_variants_match_the_same_pytorch_fixture() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/attention_chain_torch.json")).unwrap();
    for projection in ["register8", "register16"] {
        let report = pollster::block_on(support::run_fixture_with_projection(
            fixture.clone(),
            projection,
        ))
        .unwrap();
        assert_eq!(report["passed"], true);
        assert_eq!(report["projection"], projection);
        assert!(report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["merged_heads_max_abs_error"].is_number()));
        assert_eq!(report["checks"].as_array().unwrap().len(), 12);
        eprintln!("{report}");
    }
}
