#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "../examples/support/resident_nd_tensor.rs"]
mod fixture;
#[test]
fn nd_tensors_feed_existing_nn_and_preserve_training_guards_on_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("tensor.nd.nn.tests").unwrap();
    let report = futures::executor::block_on(fixture::run(runtime)).unwrap();
    assert_eq!(report["status"], "passed");
    assert_eq!(report["cases"].as_array().unwrap().len(), 3);
    assert_eq!(report["pointwise_cases"].as_array().unwrap().len(), 9);
    assert_eq!(report["pointwise_guards"]["status"], "passed");
    assert_eq!(report["pointwise_vjp"]["status"], "passed");
    assert_eq!(
        report["pointwise_vjp"]["cases"].as_array().unwrap().len(),
        6
    );
    assert_eq!(report["pointwise_vjp"]["nn_bridge"]["status"], "passed");
    assert_eq!(report["pointwise_vjp"]["guards"]["status"], "passed");
    assert_eq!(
        report["pointwise_guards"]["checks"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}
