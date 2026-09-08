#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

#[path = "../examples/support/resident_training.rs"]
mod fixture;

#[test]
fn resident_vjp_learning_and_transactional_sgd_on_real_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("nn.training.tests").unwrap();
    let result = futures::executor::block_on(fixture::run(runtime)).unwrap();
    assert_eq!(result["status"], "passed");
    assert_eq!(result["vjps"]["cases"].as_array().unwrap().len(), 18);
    assert_eq!(result["learning"]["runs"].as_array().unwrap().len(), 3);
}
