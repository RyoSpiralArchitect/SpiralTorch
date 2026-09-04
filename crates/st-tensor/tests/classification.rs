use st_tensor::{
    class_indices_from_tensor, AutogradTensor, CrossEntropyConfig, Layout, LossReduction, Tensor,
};

fn scalar(value: f32) -> Tensor {
    Tensor::from_vec(1, 1, vec![value]).unwrap()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= tolerance, "{a} != {b}");
    }
}

#[test]
fn uniform_logits_and_smoothing_have_expected_loss_and_gradient() {
    let logits = Tensor::zeros(2, 3).unwrap();
    for smoothing in [0.0, 0.2, 1.0] {
        let config = CrossEntropyConfig {
            label_smoothing: smoothing,
            ..Default::default()
        };
        close(
            logits
                .cross_entropy_with_logits(&[0, 2], config)
                .unwrap()
                .data(),
            &[3.0_f32.ln()],
            1e-7,
        );
        let gradient = logits
            .cross_entropy_with_logits_backward(&[0, 2], config, &scalar(1.0))
            .unwrap();
        let other = ((1.0 - smoothing) / 6.0) as f32;
        close(
            gradient.data(),
            &[-2.0 * other, other, other, other, other, -2.0 * other],
            1e-7,
        );
    }
}

#[test]
fn ignored_rows_do_not_change_mean_normalizer() {
    let logits = Tensor::from_vec(3, 2, vec![1.0, -1.0, -5.0, 9.0, -1.0, 1.0]).unwrap();
    let labels = [0, -100, 1];
    let mean = logits
        .cross_entropy_with_logits(&labels, CrossEntropyConfig::default())
        .unwrap();
    close(mean.data(), &[(1.0 + (-2.0_f32).exp()).ln()], 1e-7);
    let none = CrossEntropyConfig {
        reduction: LossReduction::None,
        ..Default::default()
    };
    let losses = logits.cross_entropy_with_logits(&labels, none).unwrap();
    assert_eq!(losses.shape(), (3, 1));
    close(losses.data(), &[mean.data()[0], 0.0, mean.data()[0]], 1e-7);
    let seed = Tensor::from_vec(3, 1, vec![2.0, 1e30, -3.0]).unwrap();
    let grad = logits
        .cross_entropy_with_logits_backward(&labels, none, &seed)
        .unwrap();
    assert_eq!(&grad.data()[2..4], &[0.0, 0.0]);
    let sum = CrossEntropyConfig {
        reduction: LossReduction::Sum,
        ..Default::default()
    };
    close(
        logits
            .cross_entropy_with_logits(&labels, sum)
            .unwrap()
            .data(),
        &[2.0 * mean.data()[0]],
        1e-7,
    );
}

#[test]
fn finite_differences_match_smoothed_cross_entropy_vjp() {
    let values = vec![1.0, -0.4, 0.7, 2.1, -1.3, 0.3, -0.2, 0.4, 0.9];
    let labels = [2, -100, 0];
    let epsilon = 0.005;
    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let config = CrossEntropyConfig {
            reduction,
            label_smoothing: 0.17,
            ..Default::default()
        };
        let shape = config.output_shape(3);
        let weights = if reduction == LossReduction::None {
            vec![0.7, 3.0, -1.1]
        } else {
            vec![0.7]
        };
        let seed = Tensor::from_vec(shape.0, shape.1, weights.clone()).unwrap();
        let logits = Tensor::from_vec(3, 3, values.clone()).unwrap();
        let grad = logits
            .cross_entropy_with_logits_backward(&labels, config, &seed)
            .unwrap();
        for index in 0..values.len() {
            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let evaluate = |data| {
                Tensor::from_vec(3, 3, data)
                    .unwrap()
                    .cross_entropy_with_logits(&labels, config)
                    .unwrap()
                    .data()
                    .iter()
                    .zip(&weights)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            };
            let numeric = (evaluate(plus) - evaluate(minus)) / (2.0 * epsilon);
            close(&[grad.data()[index]], &[numeric], 1e-4);
        }
    }
}

#[test]
fn log_softmax_vjp_matches_finite_differences() {
    let values = vec![1.0, -0.4, 0.7, 2.1, -1.3, 0.3];
    let seed = Tensor::from_vec(2, 3, vec![0.2, -0.6, 0.8, 1.0, 0.4, -0.3]).unwrap();
    let logits = Tensor::from_vec(2, 3, values.clone()).unwrap();
    let grad = logits.row_log_softmax_backward(&seed).unwrap();
    for index in 0..values.len() {
        let mut plus = values.clone();
        let mut minus = values.clone();
        plus[index] += 0.001;
        minus[index] -= 0.001;
        let evaluate = |data| {
            Tensor::from_vec(2, 3, data)
                .unwrap()
                .row_log_softmax()
                .unwrap()
                .data()
                .iter()
                .zip(seed.data())
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        close(
            &[grad.data()[index]],
            &[(evaluate(plus) - evaluate(minus)) / 0.002],
            2e-4,
        );
    }
}

#[test]
fn dominant_correct_class_preserves_tiny_loss_and_gradient() {
    let logits = Tensor::from_vec(1, 2, vec![0.0, -80.0]).unwrap();
    let tail = (-80.0_f64).exp() as f32;
    let config = CrossEntropyConfig::default();
    assert_eq!(
        logits
            .cross_entropy_with_logits(&[0], config)
            .unwrap()
            .data(),
        &[tail]
    );
    assert_eq!(
        logits
            .cross_entropy_with_logits_backward(&[0], config, &scalar(1.0))
            .unwrap()
            .data(),
        &[-tail, tail]
    );
    assert_eq!(logits.row_log_softmax().unwrap().data(), &[-tail, -80.0]);
    let seed = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
    assert_eq!(
        logits.row_log_softmax_backward(&seed).unwrap().data(),
        &[tail, -tail]
    );
}

#[test]
fn large_common_offset_does_not_cancel_log_partition() {
    let logits = Tensor::from_vec(2, 3, vec![f32::MAX; 6]).unwrap();
    close(
        logits.row_log_softmax().unwrap().data(),
        &[-3.0_f32.ln(); 6],
        1e-7,
    );
    close(
        logits
            .cross_entropy_with_logits(&[0, 1], CrossEntropyConfig::default())
            .unwrap()
            .data(),
        &[3.0_f32.ln()],
        1e-7,
    );
}

#[test]
fn mean_is_narrowed_only_after_reduction() {
    let logits = Tensor::from_vec(2, 2, vec![f32::MAX, -f32::MAX, 0.0, 0.0]).unwrap();
    let config = CrossEntropyConfig::default();
    assert_eq!(
        logits
            .cross_entropy_with_logits(&[1, 0], config)
            .unwrap()
            .data(),
        &[f32::MAX]
    );
    for reduction in [LossReduction::None, LossReduction::Sum] {
        assert!(logits
            .cross_entropy_with_logits(
                &[1, 0],
                CrossEntropyConfig {
                    reduction,
                    ..config
                }
            )
            .is_err());
    }
    assert!(logits.row_log_softmax().is_err());
}

#[test]
fn extreme_seeds_are_reduced_in_f64() {
    let logits = Tensor::zeros(1, 2).unwrap();
    let seed = Tensor::from_vec(1, 2, vec![f32::MAX; 2]).unwrap();
    assert_eq!(
        logits.row_log_softmax_backward(&seed).unwrap().data(),
        &[0.0, 0.0]
    );
    let grad = logits
        .cross_entropy_with_logits_backward(&[0], CrossEntropyConfig::default(), &scalar(f32::MAX))
        .unwrap();
    assert_eq!(grad.data(), &[-f32::MAX / 2.0, f32::MAX / 2.0]);
}

#[test]
fn log_softmax_retains_small_seeds_between_large_cancelling_terms() {
    let logits = Tensor::zeros(1, 3).unwrap();
    for values in [
        vec![f32::MAX, 1.0, -f32::MAX],
        vec![1.0, f32::MAX, -f32::MAX],
        vec![f32::MAX, -f32::MAX, 1.0],
    ] {
        let seed = Tensor::from_vec(1, 3, values.clone()).unwrap();
        let gradient = logits.row_log_softmax_backward(&seed).unwrap();
        for (&actual, value) in gradient.data().iter().zip(values) {
            let expected = if value == 1.0 { 2.0 / 3.0 } else { value };
            close(&[actual], &[expected], 1e-7);
        }
    }
}

#[test]
fn all_ignored_and_empty_batches_have_explicit_semantics() {
    for rows in [0, 2] {
        let logits = Tensor::zeros(rows, 3).unwrap();
        let labels = vec![-100; rows];
        assert!(logits
            .cross_entropy_with_logits(&labels, CrossEntropyConfig::default())
            .is_err());
        assert!(logits
            .cross_entropy_with_logits_backward(
                &labels,
                CrossEntropyConfig::default(),
                &scalar(1.0)
            )
            .is_err());
        for reduction in [LossReduction::None, LossReduction::Sum] {
            let config = CrossEntropyConfig {
                reduction,
                ..Default::default()
            };
            let loss = logits.cross_entropy_with_logits(&labels, config).unwrap();
            assert!(loss.data().iter().all(|&value| value == 0.0));
            let shape = loss.shape();
            let seed = Tensor::from_vec(shape.0, shape.1, vec![1.0; loss.len()]).unwrap();
            let grad = logits
                .cross_entropy_with_logits_backward(&labels, config, &seed)
                .unwrap();
            assert!(grad.data().iter().all(|&value| value == 0.0));
        }
    }
    assert!(Tensor::zeros(0, 0)
        .unwrap()
        .cross_entropy_with_logits(
            &[],
            CrossEntropyConfig {
                reduction: LossReduction::Sum,
                ..Default::default()
            }
        )
        .is_err());
    for (rows, cols) in [(0, 3), (2, 0), (0, 0)] {
        let input = Tensor::zeros(rows, cols).unwrap();
        assert_eq!(input.row_log_softmax().unwrap().shape(), (rows, cols));
        assert_eq!(
            input.row_log_softmax_backward(&input).unwrap().shape(),
            (rows, cols)
        );
    }
}

#[test]
fn invalid_inputs_are_not_silently_clamped_or_masked() {
    let logits = Tensor::zeros(2, 3).unwrap();
    let config = CrossEntropyConfig::default();
    for labels in [vec![0], vec![-1, 0], vec![3, 0], vec![i64::MAX, 0]] {
        assert!(logits.cross_entropy_with_logits(&labels, config).is_err());
        assert!(logits
            .cross_entropy_with_logits_backward(&labels, config, &scalar(1.0))
            .is_err());
    }
    for smoothing in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let config = CrossEntropyConfig {
            label_smoothing: smoothing,
            ..config
        };
        assert!(logits.cross_entropy_with_logits(&[0, 1], config).is_err());
        assert!(logits
            .cross_entropy_with_logits_backward(&[0, 1], config, &scalar(1.0))
            .is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bad = Tensor::from_vec(1, 1, vec![value]).unwrap();
        assert!(bad.row_log_softmax().is_err());
        assert!(bad.row_log_softmax_backward(&scalar(1.0)).is_err());
        let config = CrossEntropyConfig {
            reduction: LossReduction::Sum,
            ..config
        };
        assert!(bad.cross_entropy_with_logits(&[-100], config).is_err());
        assert!(logits
            .cross_entropy_with_logits_backward(&[0, 1], config, &bad)
            .is_err());
    }
    assert!(logits
        .cross_entropy_with_logits_backward(&[0, 1], config, &Tensor::zeros(2, 1).unwrap())
        .is_err());
    assert!(logits.row_log_softmax_backward(&scalar(1.0)).is_err());
    assert!("avg".parse::<LossReduction>().is_err());
}

#[test]
fn column_major_inputs_and_seeds_are_normalized() {
    let logits = Tensor::from_vec(2, 3, vec![0.0, 1.0, 2.0, -1.0, 0.5, 0.1]).unwrap();
    let seed = Tensor::from_vec(2, 3, vec![1.0, 2.0, -1.0, 0.5, -2.0, 3.0]).unwrap();
    let column = logits.to_layout(Layout::ColMajor).unwrap();
    assert_eq!(
        logits.row_log_softmax().unwrap().data(),
        column.row_log_softmax().unwrap().data()
    );
    assert_eq!(
        logits.row_log_softmax_backward(&seed).unwrap().data(),
        column
            .row_log_softmax_backward(&seed.to_layout(Layout::ColMajor).unwrap())
            .unwrap()
            .data()
    );
    let config = CrossEntropyConfig::default();
    assert_eq!(
        logits
            .cross_entropy_with_logits(&[1, 2], config)
            .unwrap()
            .data(),
        column
            .cross_entropy_with_logits(&[1, 2], config)
            .unwrap()
            .data()
    );
    assert_eq!(
        logits
            .cross_entropy_with_logits_backward(&[1, 2], config, &scalar(1.0))
            .unwrap()
            .data(),
        column
            .cross_entropy_with_logits_backward(&[1, 2], config, &scalar(1.0))
            .unwrap()
            .data()
    );
}

#[test]
fn class_index_tensor_transport_does_not_truncate() {
    assert_eq!(
        class_indices_from_tensor(&Tensor::from_vec(3, 1, vec![0.0, 4.0, -100.0]).unwrap())
            .unwrap(),
        vec![0, 4, -100]
    );
    for value in [0.5, f32::NAN, f32::INFINITY, i64::MAX as f32, -f32::MAX] {
        assert!(class_indices_from_tensor(&scalar(value)).is_err());
    }
    assert!(class_indices_from_tensor(&Tensor::zeros(1, 3).unwrap()).is_err());
    assert_eq!(
        class_indices_from_tensor(&scalar(i64::MIN as f32)).unwrap(),
        vec![i64::MIN]
    );
}

#[test]
fn autograd_owns_labels_and_reuses_core_vjp() {
    let logits = AutogradTensor::variable(
        Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, -0.5, 0.5, 1.0]).unwrap(),
    )
    .unwrap();
    let mut labels = vec![0, 2];
    let config = CrossEntropyConfig {
        label_smoothing: 0.1,
        ..Default::default()
    };
    let loss = logits.cross_entropy_with_logits(&labels, config).unwrap();
    labels.fill(-100);
    assert_eq!(loss.operation_name(), "cross_entropy_with_logits");
    let expected = logits
        .value()
        .cross_entropy_with_logits_backward(&[0, 2], config, &scalar(1.0))
        .unwrap();
    assert_eq!(
        loss.vector_jacobian_product(&logits, &scalar(1.0))
            .unwrap()
            .data(),
        expected.data()
    );
    assert!(logits.grad().is_none());
    loss.backward().unwrap();
    assert_eq!(logits.grad().unwrap().data(), expected.data());
    let log_probs = logits.row_log_softmax().unwrap();
    assert_eq!(log_probs.operation_name(), "row_log_softmax");
    let seed = Tensor::from_vec(2, 3, vec![1.0, -0.2, 0.3, 0.4, -0.5, 0.6]).unwrap();
    let expected = logits.value().row_log_softmax_backward(&seed).unwrap();
    assert_eq!(
        log_probs
            .vector_jacobian_product(&logits, &seed)
            .unwrap()
            .data(),
        expected.data()
    );
}

#[test]
fn overflowing_vjp_never_commits_partial_gradients() {
    let logits =
        AutogradTensor::variable(Tensor::from_vec(1, 3, vec![0.0, -80.0, -80.0]).unwrap()).unwrap();
    let output = logits.row_log_softmax().unwrap();
    let valid_seed = Tensor::from_vec(1, 3, vec![1.0, 0.0, 0.0]).unwrap();
    output.backward_with_grad(&valid_seed).unwrap();
    let before = logits.grad().unwrap().data().to_vec();
    let output_before = output.grad().unwrap().data().to_vec();
    let invalid_seed = Tensor::from_vec(1, 3, vec![-f32::MAX, f32::MAX, f32::MAX]).unwrap();
    assert!(output.backward_with_grad(&invalid_seed).is_err());
    assert_eq!(logits.grad().unwrap().data(), before);
    assert_eq!(output.grad().unwrap().data(), output_before);
}
