#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "../examples/support/resident_graph_forward.rs"]
mod fixture;
#[test]
fn forward_graph_owns_nd_io_and_preserves_deferred_guards() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("graph.forward.test").unwrap();
    futures::executor::block_on(fixture::run(runtime)).unwrap();
}

#[test]
fn single_pass_retains_each_logical_stage_guard_after_reuse() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    use st_backend_wgpu::resident_graph::GraphInferenceError;
    use st_nn::{
        layers::{Relu, Scaler},
        module::Module,
        resident::InferencePlan,
        Linear, Sequential,
    };
    use st_tensor::{NdLayout, Tensor};

    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("graph.stage_guards.test")
            .unwrap();
    for fault in 0..8 {
        let mut model = Sequential::new();
        for index in 0..8 {
            let gain = if index == fault { f32::MAX } else { 1. };
            if index % 2 == 0 {
                let mut linear = Linear::new(format!("linear_{index}"), 1, 1).unwrap();
                linear
                    .visit_parameters_mut(&mut |p| {
                        let value = if p.name().ends_with("::weight") {
                            gain
                        } else {
                            0.
                        };
                        p.value_mut().data_mut()[0] = value;
                        Ok(())
                    })
                    .unwrap();
                model.push(linear);
            } else {
                model.push(
                    Scaler::from_gain(
                        format!("scale_{index}"),
                        Tensor::from_vec(1, 1, vec![gain]).unwrap(),
                    )
                    .unwrap(),
                );
            }
        }
        model.push(Relu::new());
        let mut graph = InferencePlan::from_module(&model, NdLayout::contiguous(&[1]).unwrap())
            .unwrap()
            .compile_graph_wgpu(runtime.clone())
            .unwrap();
        assert_eq!(graph.stage_count(), 9);
        graph.upload(&[-2.]).unwrap();
        graph.dispatch().unwrap();
        let failed = graph.snapshot().unwrap();
        let frozen = graph.output_tensor().unwrap();
        // Deferred flag collection must precede both owning captures and reuse.
        graph.upload(&[0.]).unwrap();
        graph.dispatch().unwrap();
        assert_eq!(graph.snapshot().unwrap().read().unwrap(), vec![0.]);
        drop(graph);
        assert!(
            matches!(failed.read(), Err(GraphInferenceError::NonFinite { stage, .. }) if stage == fault)
        );
        assert!(frozen.relu().unwrap().snapshot().unwrap().read().is_err());
    }
}
