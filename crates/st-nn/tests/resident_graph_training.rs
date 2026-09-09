#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "../examples/support/resident_graph_training.rs"]
mod fixture;
#[test]
fn owned_gain_graph_training_matches_modules_and_rejects_partial_commits() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("graph.training.test").unwrap();
    futures::executor::block_on(fixture::run(runtime)).unwrap();
}
