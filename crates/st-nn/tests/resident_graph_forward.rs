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
