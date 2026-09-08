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

#[test]
fn zero_rate_probe_preserves_signed_zero_parameters() {
    use st_nn::{module::Module, resident::InferencePlan, Linear};
    use st_tensor::NdLayout;
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let mut linear = Linear::new("probe", 1, 1).unwrap();
    linear
        .visit_parameters_mut(&mut |p| {
            p.value_mut().data_mut().fill(-0.0);
            Ok(())
        })
        .unwrap();
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("nn.probe.identity").unwrap();
    let plan = InferencePlan::from_module(&linear, NdLayout::contiguous(&[1, 1]).unwrap()).unwrap();
    let mut gpu = plan.compile_training_wgpu(runtime).unwrap();
    gpu.upload_batch(&[0.0], &[1.0]).unwrap();
    gpu.step(0.0).unwrap();
    gpu.loss_snapshot().unwrap().read().unwrap();
    let layers = gpu.parameter_snapshot().unwrap().read().unwrap();
    assert_eq!(layers[0].weights[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(layers[0].bias[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(
        plan.with_dense_parameters(layers)
            .unwrap()
            .to_json()
            .unwrap(),
        plan.to_json().unwrap()
    );
}
