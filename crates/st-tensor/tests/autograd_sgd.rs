use st_tensor::{AutogradSgd, AutogradTensor, Tensor};

fn variable(values: &[f32]) -> AutogradTensor {
    AutogradTensor::variable(Tensor::from_vec(1, values.len(), values.to_vec()).unwrap()).unwrap()
}

fn seed(parameter: &AutogradTensor, values: &[f32]) {
    parameter
        .backward_with_grad(&Tensor::from_vec(1, values.len(), values.to_vec()).unwrap())
        .unwrap();
}

#[test]
fn updates_all_leaves_without_changing_old_graphs_or_gradients() {
    let x = variable(&[1.0, -2.0]);
    let bias = variable(&[0.0]);
    let old_output = x.hadamard(&x).unwrap().sum().unwrap().add(&bias).unwrap();
    old_output.backward().unwrap();
    let mut optimizer = AutogradSgd::new(vec![x.clone(), bias.clone()], 0.25).unwrap();
    optimizer.step().unwrap();
    assert_eq!(optimizer.parameter(0).unwrap().value().data(), &[0.5, -1.0]);
    assert_eq!(optimizer.parameter(1).unwrap().value().data(), &[-0.25]);
    for parameter in optimizer.parameters() {
        assert!(parameter.grad().is_none());
        assert_eq!(parameter.graph_summary().node_count, 1);
        assert!(parameter.requires_grad());
    }
    assert_ne!(optimizer.parameter(0).unwrap().id(), x.id());
    assert_eq!(x.value().data(), &[1.0, -2.0]);
    assert_eq!(x.grad().unwrap().data(), &[2.0, -4.0]);
    old_output.backward().unwrap();
    assert_eq!(x.grad().unwrap().data(), &[4.0, -8.0]);
    assert_eq!(old_output.item_f32().unwrap(), 5.0);
    assert!(optimizer.parameter(0).unwrap().grad().is_none());
    // A stale graph must not silently update the new leaves a second time.
    assert!(optimizer.step().is_err());
}

#[test]
fn a_late_failure_preserves_the_entire_parameter_set_and_all_gradients() {
    for missing in [true, false] {
        let first = variable(&[1.0]);
        let last = variable(&[f32::MAX]);
        seed(&first, &[2.0]);
        if !missing {
            seed(&last, &[-f32::MAX]);
        }
        let mut optimizer = AutogradSgd::new(vec![first.clone(), last.clone()], 1.0).unwrap();
        assert!(optimizer.step().is_err());
        assert_eq!(optimizer.parameters(), &[first.clone(), last.clone()]);
        assert_eq!(first.value().data(), &[1.0]);
        assert_eq!(first.grad().unwrap().data(), &[2.0]);
        assert_eq!(last.value().data(), &[f32::MAX]);
        assert_eq!(last.grad().is_none(), missing);
        if !missing {
            assert_eq!(last.grad().unwrap().data(), &[-f32::MAX]);
        }
    }
}

#[test]
fn validation_is_atomic_and_rejects_duplicates_constants_and_intermediates() {
    let leaf = variable(&[1.0]);
    let constant = leaf.detach().unwrap();
    let intermediate = leaf.scale(2.0).unwrap();
    for invalid in [constant, intermediate] {
        assert!(AutogradSgd::new(vec![invalid.clone()], 0.1).is_err());
        let mut optimizer = AutogradSgd::new(vec![leaf.clone()], 0.1).unwrap();
        assert!(optimizer.add_parameter(&invalid).is_err());
        assert_eq!(optimizer.parameters(), std::slice::from_ref(&leaf));
    }
    assert!(AutogradSgd::new(vec![leaf.clone(), leaf.clone()], 0.1).is_err());
    let mut optimizer = AutogradSgd::new(vec![leaf.clone()], 0.1).unwrap();
    assert!(optimizer.add_parameter(&leaf).is_err());
    assert!(optimizer.parameter(1).is_err());
    for rate in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(AutogradSgd::new(vec![leaf.clone()], rate).is_err());
        assert!(optimizer.set_learning_rate(rate).is_err());
        assert_eq!(optimizer.learning_rate(), 0.1);
    }
    optimizer.set_learning_rate(0.2).unwrap();
    assert_eq!(optimizer.learning_rate(), 0.2);
}

#[test]
fn empty_registration_is_allowed_but_an_empty_step_is_not() {
    let mut optimizer = AutogradSgd::new(vec![], 0.5).unwrap();
    optimizer.zero_grad();
    assert!(optimizer.step().is_err());
    let leaf = variable(&[2.0]);
    assert_eq!(optimizer.add_parameter(&leaf).unwrap(), 0);
    seed(&leaf, &[1.0]);
    optimizer.step().unwrap();
    assert_eq!(optimizer.parameter(0).unwrap().value().data(), &[1.5]);
}

#[test]
fn zero_grad_and_accumulation_use_the_current_leaves_only() {
    let first = variable(&[1.0]);
    let second = variable(&[2.0]);
    let mut optimizer = AutogradSgd::new(vec![first.clone(), second.clone()], 0.5).unwrap();
    seed(&first, &[1.0]);
    seed(&second, &[2.0]);
    optimizer.zero_grad();
    assert!(first.grad().is_none() && second.grad().is_none());
    seed(&first, &[1.0]);
    seed(&first, &[1.0]);
    seed(&second, &[0.0]);
    optimizer.step().unwrap();
    assert_eq!(optimizer.parameter(0).unwrap().value().data(), &[0.0]);
    assert_eq!(optimizer.parameter(1).unwrap().value().data(), &[2.0]);
    optimizer.zero_grad();
    assert_eq!(first.grad().unwrap().data(), &[2.0]);
    assert_eq!(second.grad().unwrap().data(), &[0.0]);
}

#[test]
fn repeated_training_uses_bounded_fresh_graphs() {
    let mut optimizer = AutogradSgd::new(vec![variable(&[2.0, -3.0])], 0.1).unwrap();
    let target = AutogradTensor::constant(Tensor::zeros(1, 2).unwrap()).unwrap();
    for _ in 0..100 {
        let parameter = optimizer.parameter(0).unwrap();
        let loss = parameter.mean_squared_error(&target).unwrap();
        assert!(loss.graph_summary().node_count <= 5);
        loss.backward().unwrap();
        optimizer.step().unwrap();
    }
    let loss = optimizer
        .parameter(0)
        .unwrap()
        .mean_squared_error(&target)
        .unwrap();
    assert!(loss.item_f32().unwrap() < 1e-7);
}

#[test]
fn concurrent_backward_cannot_split_a_multi_parameter_gradient_snapshot() {
    use std::sync::{Arc, Barrier};
    for _ in 0..20 {
        let first = variable(&[1.0]);
        let last = variable(&[1.0]);
        let loss = first.add(&last).unwrap();
        loss.backward().unwrap();
        let mut optimizer = AutogradSgd::new(vec![first, last], 0.5).unwrap();
        let start = Arc::new(Barrier::new(2));
        let worker_start = Arc::clone(&start);
        let worker = std::thread::spawn(move || {
            worker_start.wait();
            for _ in 0..100 {
                loss.backward().unwrap();
            }
        });
        start.wait();
        optimizer.step().unwrap();
        assert_eq!(
            optimizer.parameter(0).unwrap().value().data(),
            optimizer.parameter(1).unwrap().value().data(),
        );
        worker.join().unwrap();
    }
}
