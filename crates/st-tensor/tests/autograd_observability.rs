// Separate integration process: a lock regression must time out, not stall other tests.
use st_tensor::{AutogradSgd, AutogradTensor, CrossEntropyConfig, Tensor};
use std::sync::{mpsc, Arc};
use std::time::Duration;

#[test]
fn kernel_observers_read_gradients_after_vjp_success_and_failed_commit() {
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let logits = AutogradTensor::variable(Tensor::zeros(1, 2).unwrap()).unwrap();
        let output = logits
            .cross_entropy_with_logits(&[0], CrossEntropyConfig::default())
            .unwrap();
        let captured = logits.clone();
        let observer = Arc::new(move |event: &st_tensor::TensorOpMetaEvent| {
            if event.op_name == "cross_entropy_with_logits_backward" {
                tx.send((
                    captured.grad().map(|gradient| gradient.data().to_vec()),
                    event.data.clone(),
                ))
                .unwrap();
            }
        });
        let previous = st_tensor::set_thread_meta_observer(Some(observer));
        let unit = Tensor::from_vec(1, 1, vec![1.0]).unwrap();
        output.vector_jacobian_product(&logits, &unit).unwrap();
        output.backward().unwrap();
        let large = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        output.backward_with_grad(&large).unwrap();
        assert!(output.backward_with_grad(&large).is_err());
        st_tensor::set_thread_meta_observer(previous);
    });
    let expected = [
        None,
        Some(vec![-0.5, 0.5]),
        Some(vec![-f32::MAX / 2.0, f32::MAX / 2.0]),
        Some(vec![-f32::MAX / 2.0, f32::MAX / 2.0]),
    ];
    for expected in expected {
        let (gradient, metadata) = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("kernel observer blocked on the graph lock");
        assert_eq!(gradient, expected);
        assert_eq!(metadata["backend"], "cpu");
        assert_eq!(metadata["accumulator_dtype"], "f64");
    }
    worker.join().unwrap();
}

#[test]
fn optimizer_kernel_observers_can_read_gradients_after_success_and_failure() {
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let first = AutogradTensor::variable(Tensor::from_vec(1, 1, vec![1.0]).unwrap()).unwrap();
        let last =
            AutogradTensor::variable(Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap()).unwrap();
        first.backward().unwrap();
        last.backward_with_grad(&Tensor::from_vec(1, 1, vec![-f32::MAX]).unwrap())
            .unwrap();
        let captured = first.clone();
        let observer = Arc::new(move |event: &st_tensor::TensorOpMetaEvent| {
            if event.op_name == "add_scaled" || event.op_name == "autograd_sgd_step" {
                tx.send((event.op_name.to_owned(), captured.grad().unwrap().data()[0]))
                    .unwrap();
            }
        });
        let previous = st_tensor::set_thread_meta_observer(Some(observer));
        let mut failed = AutogradSgd::new(vec![first.clone(), last], 1.0).unwrap();
        assert!(failed.step().is_err());
        let mut successful = AutogradSgd::new(vec![first], 0.5).unwrap();
        successful.step().unwrap();
        st_tensor::set_thread_meta_observer(previous);
    });
    for name in ["add_scaled", "add_scaled", "autograd_sgd_step"] {
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5))
                .expect("optimizer observer deadlocked"),
            (name.to_owned(), 1.0),
        );
    }
    worker.join().unwrap();
    assert!(
        rx.try_recv().is_err(),
        "failed step must not emit a commit event"
    );
}
