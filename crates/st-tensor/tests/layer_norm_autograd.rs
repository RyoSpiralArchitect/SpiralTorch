use st_tensor::{AutogradSgd, AutogradTensor, LayerNormBackend, Tensor, TensorUtilBackend};

fn tensor(rows: usize, cols: usize, values: &[f32]) -> Tensor {
    Tensor::from_vec(rows, cols, values.to_vec()).unwrap()
}

fn close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(actual.shape(), expected.shape());
    for (&a, &b) in actual.data().iter().zip(expected.data()) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * (1.0 + b.abs()),
            "{a} != {b}"
        );
    }
}

fn fixture() -> (Tensor, Tensor, Tensor, Tensor) {
    (
        tensor(2, 3, &[0.45, -0.8, 1.25, -0.35, 0.9, -1.1]),
        tensor(1, 3, &[1.5, -0.75, 0.35]),
        tensor(1, 3, &[0.1, -0.2, 0.05]),
        tensor(2, 3, &[0.2, -0.15, 0.35, -0.25, 0.1, 0.3]),
    )
}

#[test]
fn all_three_vjps_match_finite_difference_and_explicit_affine_scale() {
    let (x, gamma, beta, seed) = fixture();
    let (dx, dg, db) = x.layer_norm_affine_backward(&gamma, &seed, 1e-5).unwrap();
    let loss = |args: &[Tensor; 3]| -> f64 {
        args[0]
            .layer_norm_affine_with_backend(&args[1], &args[2], 1e-5, LayerNormBackend::Cpu)
            .unwrap()
            .data()
            .iter()
            .zip(seed.data())
            .map(|(&v, &g)| f64::from(v) * f64::from(g))
            .sum()
    };
    for (operand, analytic) in [&dx, &dg, &db].into_iter().enumerate() {
        for index in 0..analytic.len() {
            let mut plus = [x.clone(), gamma.clone(), beta.clone()];
            let mut minus = plus.clone();
            plus[operand].data_mut()[index] += 1e-3;
            minus[operand].data_mut()[index] -= 1e-3;
            let numeric = (loss(&plus) - loss(&minus)) / 2e-3;
            assert!(
                (f64::from(analytic.data()[index]) - numeric).abs() < 1e-4,
                "operand={operand} index={index}"
            );
        }
    }
    let (scaled_x, scaled_g, scaled_b) = x
        .layer_norm_affine_backward_with_backend(&gamma, &seed, 1e-5, 0.5, TensorUtilBackend::Cpu)
        .unwrap();
    close(&scaled_x, &dx, 0.0);
    close(&scaled_g, &dg.scale(0.5).unwrap(), 1e-7);
    close(&scaled_b, &db.scale(0.5).unwrap(), 1e-7);
}

#[test]
fn every_trainability_combination_uses_the_same_vjp_without_mutating_queries() {
    let (x, gamma, beta, seed) = fixture();
    let (dx, dg, db) = x.layer_norm_affine_backward(&gamma, &seed, 1e-5).unwrap();
    for mask in 1..8 {
        let parents = [x.clone(), gamma.clone(), beta.clone()]
            .into_iter()
            .enumerate()
            .map(|(i, value)| AutogradTensor::from_tensor(value, mask & (1 << i) != 0).unwrap())
            .collect::<Vec<_>>();
        let y = parents[0]
            .layer_norm_affine(&parents[1], &parents[2], 1e-5)
            .unwrap();
        for (i, expected) in [&dx, &dg, &db].into_iter().enumerate() {
            if parents[i].requires_grad() {
                close(
                    &y.vector_jacobian_product(&parents[i], &seed).unwrap(),
                    expected,
                    1e-6,
                );
            }
            assert!(parents[i].grad().is_none());
        }
        y.backward_with_grad(&seed).unwrap();
        y.backward_with_grad(&seed).unwrap();
        for (i, expected) in [&dx, &dg, &db].into_iter().enumerate() {
            if parents[i].requires_grad() {
                close(
                    &parents[i].grad().unwrap(),
                    &expected.scale(2.0).unwrap(),
                    1e-6,
                );
            } else {
                assert!(parents[i].grad().is_none());
            }
        }
    }
}

#[test]
fn shared_parent_accumulates_input_gamma_and_beta_paths() {
    let value = tensor(1, 3, &[0.3, -0.7, 1.1]);
    let x = AutogradTensor::variable(value.clone()).unwrap();
    let seed = tensor(1, 3, &[0.4, -0.2, 0.8]);
    let y = x.layer_norm_affine(&x, &x, 1e-5).unwrap();
    let (dx, dg, db) = value
        .layer_norm_affine_backward(&value, &seed, 1e-5)
        .unwrap();
    let report = y.backward_with_grad(&seed).unwrap();
    close(
        &x.grad().unwrap(),
        &dx.add(&dg).unwrap().add(&db).unwrap(),
        1e-6,
    );
    assert_eq!(report.leaf_gradient_count, 1);
}

#[test]
fn finite_gradients_survive_large_products_and_cancelled_affine_sums() {
    let input = tensor(3, 1, &[1.0, 2.0, 3.0]);
    let gamma = tensor(1, 1, &[f32::MAX]);
    let seed = tensor(3, 1, &[f32::MAX, f32::MAX, -f32::MAX]);
    let (dx, dg, db) = input
        .layer_norm_affine_backward(&gamma, &seed, 1e-5)
        .unwrap();
    assert_eq!(dx.data(), &[0.0; 3]);
    assert_eq!(dg.data(), &[0.0]);
    assert_eq!(db.data(), &[f32::MAX]);
}

#[test]
fn unrequested_affine_overflow_does_not_poison_input_gradient() {
    let x = AutogradTensor::variable(tensor(2, 1, &[1.0, 2.0])).unwrap();
    let gamma = AutogradTensor::constant(tensor(1, 1, &[f32::MAX])).unwrap();
    let beta = AutogradTensor::constant(tensor(1, 1, &[0.0])).unwrap();
    let y = x.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
    y.backward_with_grad(&tensor(2, 1, &[f32::MAX; 2])).unwrap();
    assert_eq!(x.grad().unwrap().data(), &[0.0; 2]);
    assert!(gamma.grad().is_none() && beta.grad().is_none());
}

#[test]
fn finite_vjp_does_not_require_representable_f32_inverse_std() {
    let input = tensor(1, 2, &[0.0, f32::from_bits(1)]);
    let gamma = tensor(1, 2, &[1.0, 1.0]);
    let seed = tensor(1, 2, &[1.0, 1.0]);
    assert!(input.layer_norm_stats(0.0).is_err());
    let output = input
        .layer_norm_affine_with_backend(
            &gamma,
            &tensor(1, 2, &[0.0, 0.0]),
            0.0,
            LayerNormBackend::Cpu,
        )
        .unwrap();
    assert_eq!(output.data(), &[-1.0, 1.0]);
    let (dx, dg, db) = input
        .layer_norm_affine_backward(&gamma, &seed, 0.0)
        .unwrap();
    assert_eq!(dx.data(), &[0.0, 0.0]);
    assert_eq!(dg.data(), &[-1.0, 1.0]);
    assert_eq!(db.data(), &[1.0, 1.0]);
}

#[test]
fn requested_overflow_preserves_every_existing_gradient() {
    let x = AutogradTensor::variable(tensor(2, 1, &[1.0, 2.0])).unwrap();
    let gamma = AutogradTensor::variable(tensor(1, 1, &[1.0])).unwrap();
    let beta = AutogradTensor::variable(tensor(1, 1, &[0.0])).unwrap();
    let y = x.layer_norm_affine(&gamma, &beta, 1e-5).unwrap();
    y.sum().unwrap().backward().unwrap();
    let before = [
        x.grad().unwrap(),
        gamma.grad().unwrap(),
        beta.grad().unwrap(),
        y.grad().unwrap(),
    ];
    assert!(y.backward_with_grad(&tensor(2, 1, &[f32::MAX; 2])).is_err());
    for (actual, expected) in [x, gamma, beta, y].iter().zip(before) {
        close(&actual.grad().unwrap(), &expected, 0.0);
    }
}

#[test]
fn empty_batches_invalid_epsilon_and_zero_variance_are_consistent() {
    let x = AutogradTensor::variable(Tensor::zeros(0, 3).unwrap()).unwrap();
    let gamma = AutogradTensor::variable(tensor(1, 3, &[1.0; 3])).unwrap();
    let beta = AutogradTensor::variable(Tensor::zeros(1, 3).unwrap()).unwrap();
    x.layer_norm_affine(&gamma, &beta, 1e-5)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(x.grad().unwrap().shape(), (0, 3));
    assert_eq!(gamma.grad().unwrap().data(), &[0.0; 3]);
    assert_eq!(beta.grad().unwrap().data(), &[0.0; 3]);
    for epsilon in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(x.layer_norm_affine(&gamma, &beta, epsilon).is_err());
    }
    let constant = AutogradTensor::variable(tensor(1, 3, &[1.0; 3])).unwrap();
    assert!(constant.layer_norm_affine(&gamma, &beta, 0.0).is_err());
    let zero_width = AutogradTensor::variable(Tensor::zeros(2, 0).unwrap()).unwrap();
    assert!(zero_width.layer_norm_affine(&gamma, &beta, 1e-5).is_err());
}

#[test]
fn centered_gradient_is_translation_invariant_and_supports_logical_layout() {
    let (x, gamma, _, seed) = fixture();
    let (a, b, c) = x.layer_norm_affine_backward(&gamma, &seed, 1e-5).unwrap();
    let column = x.to_layout(st_tensor::Layout::ColMajor).unwrap();
    let (d, e, f) = column
        .layer_norm_affine_backward(
            &gamma,
            &seed.to_layout(st_tensor::Layout::ColMajor).unwrap(),
            1e-5,
        )
        .unwrap();
    close(&a, &d, 0.0);
    close(&b, &e, 0.0);
    close(&c, &f, 0.0);
    let base = tensor(1, 3, &[0.0, 1.0, 2.0]);
    let shifted = tensor(1, 3, &[10000.0, 10001.0, 10002.0]);
    let seed = tensor(1, 3, &[0.2, -0.1, 0.4]);
    let (a, b, _) = base
        .layer_norm_affine_backward(&gamma, &seed, 1e-5)
        .unwrap();
    let (d, e, _) = shifted
        .layer_norm_affine_backward(&gamma, &seed, 1e-5)
        .unwrap();
    close(&a, &d, 0.0);
    close(&b, &e, 0.0);
}

#[test]
fn affine_parameters_learn_through_native_graph_and_sgd() {
    let input = tensor(3, 3, &[0.4, -0.8, 1.2, -0.3, 0.9, -1.1, 0.7, 0.1, -0.2]);
    let target = AutogradTensor::constant(
        input
            .layer_norm_affine_with_backend(
                &tensor(1, 3, &[1.7, 0.5, -0.8]),
                &tensor(1, 3, &[0.2, -0.3, 0.6]),
                1e-5,
                LayerNormBackend::Cpu,
            )
            .unwrap(),
    )
    .unwrap();
    let x = AutogradTensor::constant(input).unwrap();
    let gamma = AutogradTensor::variable(tensor(1, 3, &[1.0; 3])).unwrap();
    let beta = AutogradTensor::variable(tensor(1, 3, &[0.0; 3])).unwrap();
    let mut optimizer = AutogradSgd::new(vec![gamma, beta], 0.1).unwrap();
    let mut losses = Vec::new();
    for _ in 0..400 {
        let parameters = optimizer.parameters();
        let loss = x
            .layer_norm_affine(&parameters[0], &parameters[1], 1e-5)
            .unwrap()
            .mean_squared_error(&target)
            .unwrap();
        losses.push(loss.item_f32().unwrap());
        loss.backward().unwrap();
        optimizer.step().unwrap();
    }
    assert!(
        losses[399] < losses[0] * 1e-4,
        "{} -> {}",
        losses[0],
        losses[399]
    );
}

#[cfg(feature = "wgpu_dense")]
#[test]
fn explicit_hybrid_gpu_vjp_matches_shared_cpu_without_fallback() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let _strict = st_tensor::execution::push_accelerator_fallback(
        st_tensor::execution::AcceleratorFallback::Forbid,
    );
    let (x, gamma, _, seed) = fixture();
    let (a, b, c) = x.layer_norm_affine_backward(&gamma, &seed, 1e-5).unwrap();
    let (d, e, f) = x
        .layer_norm_affine_backward_with_backend(
            &gamma,
            &seed,
            1e-5,
            1.0,
            TensorUtilBackend::GpuWgpu,
        )
        .unwrap();
    close(&a, &d, 2e-5);
    close(&b, &e, 2e-5);
    close(&c, &f, 2e-5);
}
