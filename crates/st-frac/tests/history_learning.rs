use st_frac::learning::{FractionalGlKernel, FractionalLearningError};

fn dot(x: &[f32], y: &[f32]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

#[test]
fn history_decomposes_full_gl_and_has_a_joint_adjoint_on_every_axis() {
    let x: Vec<f32> = (0..24).map(|i| (i as f32 * 0.3).sin()).collect();
    let upstream: Vec<f32> = (0..24).map(|i| (i as f32 * 0.2).cos()).collect();
    let dx: Vec<f32> = x.iter().map(|v| v * 0.2).collect();
    for step in [0.7f32, 1.0, 1.4] {
        let kernel = FractionalGlKernel::new(5, step, 100, 500).unwrap();
        for alpha in [0.45f32, 1.0, 1.5] {
            for axis in 0..3 {
                let full = kernel.forward(&x, &[2, 4, 3], axis, alpha).unwrap();
                let history = kernel.forward_history(&x, &[2, 4, 3], axis, alpha).unwrap();
                let scale = st_frac::gl_step_scale(alpha, step)
                    .unwrap()
                    .inverse_h_alpha();
                for ((&f, &h), &value) in full.output().iter().zip(history.output()).zip(&x) {
                    assert!(
                        (f64::from(f) - f64::from(h) - f64::from(scale) * f64::from(value)).abs()
                            < 5e-7
                    );
                }
                let dy = history.jvp(&dx, 0.3).unwrap();
                let pullback = history.vjp(&upstream).unwrap();
                let left = dot(&dy, &upstream);
                let right = dot(&dx, &pullback.input) + 0.3 * f64::from(pullback.alpha);
                assert!((left - right).abs() < 2e-6 * (1.0 + left.abs()));
                let eps = 1e-3;
                let plus = kernel
                    .forward_history(&x, &[2, 4, 3], axis, alpha + eps)
                    .unwrap();
                let minus = kernel
                    .forward_history(&x, &[2, 4, 3], axis, alpha - eps)
                    .unwrap();
                let finite_difference: Vec<f32> = plus
                    .output()
                    .iter()
                    .zip(minus.output())
                    .map(|(p, m)| (p - m) / (2. * eps))
                    .collect();
                assert!(
                    (dot(&finite_difference, &upstream) - f64::from(pullback.alpha)).abs() < 3e-3
                );
            }
        }
    }
}

#[test]
fn removing_the_zero_lag_tap_avoids_cancellation_and_unused_current_overflow() {
    let kernel = FractionalGlKernel::new(2, 0.7, 2, 4).unwrap();
    let x = [1.0, f32::MAX];
    assert!(kernel.forward(&x, &[2], 0, 0.5).is_err());
    let saved = kernel.forward_history(&x, &[2], 0, 0.5).unwrap();
    let scale = st_frac::gl_step_scale(0.5, 0.7).unwrap().inverse_h_alpha();
    assert_eq!(saved.output().as_slice().unwrap(), &[0.0, -0.5 * scale]);
    let gradient = saved.vjp(&[0.0, 1.0]).unwrap();
    assert_eq!(gradient.input, [-0.5 * scale, 0.0]);
    assert!(gradient.alpha.is_finite() && gradient.alpha != 0.0);
    assert_eq!(saved.vjp(&[1.0, 0.0]).unwrap().alpha, 0.0);
}

#[test]
fn strict_history_has_no_current_future_or_cross_lane_derivative() {
    let kernel = FractionalGlKernel::new(5, 0.8, 24, 120).unwrap();
    let mut x: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let saved = kernel.forward_history(&x, &[2, 4, 3], 1, 0.6).unwrap();
    x[3..].fill(90.0);
    let changed = kernel.forward_history(&x, &[2, 4, 3], 1, 0.6).unwrap();
    assert_eq!(
        &saved.output().as_slice().unwrap()[..6],
        &changed.output().as_slice().unwrap()[..6]
    );
    let mut upstream = vec![0.0; 24];
    upstream[4] = 1.0;
    let gradient = saved.vjp(&upstream).unwrap();
    for (index, &value) in gradient.input.iter().enumerate() {
        if index == 1 {
            assert_ne!(value, 0.0);
        } else {
            assert_eq!(value, 0.0);
        }
    }
    assert_eq!(gradient.alpha, saved.vjp(&upstream).unwrap().alpha);
}

#[test]
fn no_history_is_exactly_zero_including_the_step_scale_derivative() {
    for (kernel_len, shape) in [(1, vec![2, 3]), (8, vec![6, 1])] {
        let kernel = FractionalGlKernel::new(kernel_len, 0.7, 6, 48).unwrap();
        let saved = kernel.forward_history(&[1.0; 6], &shape, 1, 0.6).unwrap();
        assert!(saved.output().iter().all(|&v| v == 0.0));
        let gradient = saved.vjp(&[1.0; 6]).unwrap();
        assert!(gradient.input.iter().all(|&v| v == 0.0));
        assert_eq!(gradient.alpha, 0.0);
        assert!(saved.jvp(&[1.0; 6], 1.0).unwrap().iter().all(|&v| v == 0.0));
    }
}

#[test]
fn empty_history_skips_unrepresentable_discarded_scale_and_coefficients() {
    for (kernel_len, shape) in [(1, vec![2, 3]), (8, vec![6, 1])] {
        for (step, alpha) in [
            (0.1, 38.5),
            (0.1, 100.0),
            (1.0, f32::MAX),
            (f32::MIN_POSITIVE, f32::MAX),
        ] {
            let kernel = FractionalGlKernel::new(kernel_len, step, 6, 48).unwrap();
            let saved = kernel
                .forward_history(&[f32::MAX; 6], &shape, 1, alpha)
                .unwrap();
            assert!(saved.output().iter().all(|&v| v == 0.0));
            let gradient = saved.vjp(&[f32::MAX; 6]).unwrap();
            assert_eq!(gradient.input, [0.0; 6]);
            assert_eq!(gradient.alpha, 0.0);
            assert_eq!(saved.jvp(&[f32::MAX; 6], f32::MAX).unwrap(), [0.0; 6]);
        }
    }
}

#[test]
fn empty_history_still_validates_samples_orders_and_directions() {
    for (kernel_len, shape) in [(1, vec![2, 3]), (8, vec![6, 1])] {
        let kernel = FractionalGlKernel::new(kernel_len, 0.1, 6, 48).unwrap();
        for alpha in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                kernel.forward_history(&[1.0; 6], &shape, 1, alpha),
                Err(FractionalLearningError::Operator(
                    st_frac::FracErr::Alpha { .. }
                ))
            ));
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                kernel.forward_history(&[value; 6], &shape, 1, 38.5),
                Err(FractionalLearningError::Operator(
                    st_frac::FracErr::NonFiniteSample { .. }
                ))
            ));
        }
        let saved = kernel.forward_history(&[1.0; 6], &shape, 1, 38.5).unwrap();
        assert!(saved.vjp(&[]).is_err());
        assert!(saved.vjp(&[f32::NAN; 6]).is_err());
        assert!(saved.jvp(&[], 1.0).is_err());
        assert!(saved.jvp(&[f32::INFINITY; 6], 1.0).is_err());
        assert!(saved.jvp(&[1.0; 6], f32::NAN).is_err());
    }
}

#[test]
fn history_keeps_the_same_shape_budget_and_finite_domain_guards() {
    let kernel = FractionalGlKernel::new(4, 1.0, 10, 20).unwrap();
    assert!(matches!(
        kernel.forward_history(&[1.0; 6], &[6], 0, 0.5),
        Err(FractionalLearningError::Budget)
    ));
    for shape in [vec![], vec![0], vec![usize::MAX, 2], vec![1; 17]] {
        assert!(matches!(
            kernel.forward_history(&[1.0], &shape, 0, 0.5),
            Err(FractionalLearningError::Shape)
        ));
    }
    for alpha in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(kernel.forward_history(&[1.0], &[1], 0, alpha).is_err());
    }
    assert!(kernel.forward_history(&[f32::NAN], &[1], 0, 0.5).is_err());
    let saved = kernel.forward_history(&[1.0], &[1], 0, 0.5).unwrap();
    assert!(saved.vjp(&[]).is_err());
    assert!(saved.vjp(&[f32::NAN]).is_err());
    assert!(saved.jvp(&[1.0], f32::NAN).is_err());
}
