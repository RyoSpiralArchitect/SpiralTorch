#![cfg(feature = "wgpu_dense")]

use st_tensor::{AutogradTensor, Tensor, TensorOpMetaEvent, TensorOpMetaObserver};
use std::sync::{Arc, Mutex};

struct RestoreObserver(Option<TensorOpMetaObserver>);

impl Drop for RestoreObserver {
    fn drop(&mut self) {
        st_tensor::set_thread_meta_observer(self.0.take());
    }
}

#[test]
fn prepacked_forward_and_both_reverse_matmuls_execute_on_wgpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    assert_eq!(std::env::var("SPIRALTORCH_STRICT_GPU").as_deref(), Ok("1"));
    assert!(st_tensor::backend::wgpu_dense::is_available());
    let events = Arc::new(Mutex::new(Vec::<TensorOpMetaEvent>::new()));
    let captured = events.clone();
    let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
        captured.lock().unwrap().push(event.clone());
    })));
    let guard = RestoreObserver(previous);
    let x = AutogradTensor::variable(Tensor::from_vec(32, 32, vec![0.25; 1024]).unwrap()).unwrap();
    let mut identity = vec![0.0; 1024];
    for i in 0..32 {
        identity[i * 32 + i] = 2.0;
    }
    let weight = AutogradTensor::variable(Tensor::from_vec(32, 32, identity).unwrap()).unwrap();
    let output = x.matmul_prepacked(&weight.prepack_rhs().unwrap()).unwrap();
    assert_eq!(output.value().data(), &[0.5; 1024]);
    assert_eq!(
        output
            .sum()
            .unwrap()
            .backward()
            .unwrap()
            .leaf_gradient_count,
        2
    );
    assert_eq!(x.grad().unwrap().data(), &[2.0; 1024]);
    assert_eq!(weight.grad().unwrap().data(), &[8.0; 1024]);
    drop(guard);

    let events = events.lock().unwrap();
    let matmuls: Vec<_> = events
        .iter()
        .filter(|e| matches!(e.op_name, "matmul" | "matmul_prepacked"))
        .collect();
    assert_eq!(
        matmuls
            .iter()
            .filter(|e| e.op_name == "matmul_prepacked")
            .count(),
        1
    );
    assert_eq!(matmuls.iter().filter(|e| e.op_name == "matmul").count(), 2);
    for event in matmuls {
        assert_eq!(event.data["executed_backend"], "wgpu", "{event:?}");
        assert_eq!(event.data["event_phase"], "completed");
    }
}
