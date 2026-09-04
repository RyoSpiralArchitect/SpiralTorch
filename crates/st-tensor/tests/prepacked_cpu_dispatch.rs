#![cfg(not(any(feature = "faer", feature = "wgpu_dense")))]

use st_tensor::{PackedB, Tensor, TensorOpMetaEvent, TensorOpMetaObserver, Tile};
use std::sync::{Arc, Mutex};

struct RestoreObserver(Option<TensorOpMetaObserver>);

impl Drop for RestoreObserver {
    fn drop(&mut self) {
        st_tensor::set_thread_meta_observer(self.0.take());
    }
}

#[test]
fn packed_auto_uses_portable_cpu_kernel_without_gpu_or_faer() {
    let events = Arc::new(Mutex::new(Vec::<TensorOpMetaEvent>::new()));
    let captured = events.clone();
    let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
        captured.lock().unwrap().push(event.clone());
    })));
    let guard = RestoreObserver(previous);
    let lhs = Tensor::from_vec(9, 7, vec![1.0; 63]).unwrap();
    let rhs = Tensor::from_vec(7, 13, vec![2.0; 91]).unwrap();
    let packed = PackedB::from_tensor(&rhs, Tile::col_major()).unwrap();
    assert_eq!(lhs.matmul_prepacked(&packed).unwrap().data(), &[14.0; 117]);
    assert_eq!(
        lhs.matmul_prepacked_bias(&packed, &[0.25; 13])
            .unwrap()
            .data(),
        &[14.25; 117]
    );
    drop(guard);
    let events = events.lock().unwrap();
    for name in ["matmul_prepacked", "matmul_prepacked_bias"] {
        let matching: Vec<_> = events.iter().filter(|e| e.op_name == name).collect();
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].data["executed_backend"], "cpu_simd");
        assert_eq!(matching[0].data["event_phase"], "completed");
    }
}
