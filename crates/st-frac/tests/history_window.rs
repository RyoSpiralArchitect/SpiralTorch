use st_frac::learning::{FractionalGlKernel, FractionalLearningError};

fn close(a: f32, b: f32) {
    assert!((a - b).abs() <= 3e-6 * (1.0 + a.abs()), "{a} != {b}");
}

#[test]
fn complementary_windows_preserve_forward_and_all_first_order_maps() {
    let x: Vec<_> = (0..54).map(|i| (i as f32 * 0.3).sin()).collect();
    let g: Vec<_> = x.iter().map(|v| v.cos()).collect();
    let dx: Vec<_> = x.iter().map(|v| v * 0.2).collect();
    let kernel = FractionalGlKernel::new(8, 0.7, 54, 432).unwrap();
    for alpha in [0.08, 0.7, 1.0, 2.0, 2.6] {
        for axis in 0..3 {
            let map = |window| {
                kernel
                    .forward_history_log_gain_window(&x, &[2, 9, 3], axis, alpha, 0.3, window)
                    .unwrap()
            };
            let full = kernel
                .forward_history_log_gain(&x, &[2, 9, 3], axis, alpha, 0.3)
                .unwrap();
            let all = map(1..8);
            assert_eq!(full.output(), all.output());
            assert_eq!(full.vjp_input(&g).unwrap(), all.vjp_input(&g).unwrap());
            assert_eq!(
                full.vjp_parameters(&g).unwrap(),
                all.vjp_parameters(&g).unwrap()
            );
            assert_eq!(
                full.jvp(&dx, 0.2, -0.3).unwrap(),
                all.jvp(&dx, 0.2, -0.3).unwrap()
            );
            let (short, tail) = (map(1..3), map(3..8));
            for ((&f, &s), &t) in full.output().iter().zip(short.output()).zip(tail.output()) {
                close(f, s + t);
            }
            let (f, s, t) = (
                full.vjp(&g).unwrap(),
                short.vjp(&g).unwrap(),
                tail.vjp(&g).unwrap(),
            );
            for ((a, b), c) in f.input.iter().zip(&s.input).zip(&t.input) {
                close(*a, b + c);
            }
            close(f.alpha, s.alpha + t.alpha);
            close(f.log_gain, s.log_gain + t.log_gain);
            let (f, s, t) = (
                full.jvp(&dx, 0.2, -0.3).unwrap(),
                short.jvp(&dx, 0.2, -0.3).unwrap(),
                tail.jvp(&dx, 0.2, -0.3).unwrap(),
            );
            for ((a, b), c) in f.iter().zip(&s).zip(&t) {
                close(*a, b + c);
            }
        }
    }
}

#[test]
fn taps_keep_the_full_normalization_not_a_shorter_kernel() {
    let kernel = FractionalGlKernel::new(8, 1.0, 8, 64).unwrap();
    let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let full = kernel
        .forward_history_log_gain(&x, &[8], 0, 0.3, 0.2)
        .unwrap();
    let short = kernel
        .forward_history_log_gain_window(&x, &[8], 0, 0.3, 0.2, 1..3)
        .unwrap();
    let tail = kernel
        .forward_history_log_gain_window(&x, &[8], 0, 0.3, 0.2, 3..8)
        .unwrap();
    for i in 0..8 {
        assert_eq!(
            short.output()[i],
            if (1..3).contains(&i) {
                full.output()[i]
            } else {
                0.0
            }
        );
        assert_eq!(
            tail.output()[i],
            if i >= 3 { full.output()[i] } else { 0.0 }
        );
    }
    let separately_normalized = FractionalGlKernel::new(3, 1.0, 8, 24)
        .unwrap()
        .forward_history_log_gain(&x, &[8], 0, 0.3, 0.2)
        .unwrap();
    assert!((short.output()[1] - separately_normalized.output()[1]).abs() > 0.01);
}

#[test]
fn masked_alpha_and_gain_derivatives_match_joint_finite_differences() {
    let kernel = FractionalGlKernel::new(8, 0.3, 24, 192).unwrap();
    let x: Vec<_> = (0..24).map(|i| (i as f32 * 0.2).cos()).collect();
    let dx: Vec<_> = x.iter().map(|v| v * 0.1 - 0.2).collect();
    let g: Vec<_> = x.iter().map(|v| v * 0.3).collect();
    for lags in [1..3, 2..5, 3..8] {
        for alpha in [0.08, 0.7, 2.0] {
            let eval = |t: f32| {
                let input: Vec<_> = x.iter().zip(&dx).map(|(x, d)| x + t * d).collect();
                kernel
                    .forward_history_log_gain_window(
                        &input,
                        &[2, 12],
                        1,
                        alpha + t * 0.2,
                        0.3 - t * 0.4,
                        lags.clone(),
                    )
                    .unwrap()
            };
            let saved = eval(0.0);
            let (plus, minus) = (eval(0.001), eval(-0.001));
            let jvp = saved.jvp(&dx, 0.2, -0.4).unwrap();
            for ((p, m), d) in plus.output().iter().zip(minus.output()).zip(&jvp) {
                assert!(((p - m) / 0.002 - d).abs() < 0.001);
            }
            let vjp = saved.vjp(&g).unwrap();
            let dot = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>()
            };
            assert!(
                (dot(&jvp, &g) - dot(&dx, &vjp.input) - 0.2 * f64::from(vjp.alpha)
                    + 0.4 * f64::from(vjp.log_gain))
                .abs()
                    < 1e-6
            );
            assert_eq!(saved.vjp_input(&g).unwrap(), vjp.input);
            assert_eq!(saved.vjp_parameters(&g).unwrap(), (vjp.alpha, vjp.log_gain));
        }
    }
}

#[test]
fn zero_integer_order_tail_still_has_an_order_derivative() {
    let kernel = FractionalGlKernel::new(8, 1.0, 8, 64).unwrap();
    let batch = kernel
        .forward_history_log_gain_window(&[1.0; 8], &[8], 0, 2.0, 0.0, 3..8)
        .unwrap();
    assert!(batch.output().iter().all(|&v| v == 0.0));
    assert_eq!(batch.vjp_input(&[1.0; 8]).unwrap(), [0.0; 8]);
    assert_eq!(batch.vjp_log_gain(&[1.0; 8]).unwrap(), 0.0);
    assert_ne!(batch.vjp_alpha(&[1.0; 8]).unwrap(), 0.0);
}

#[test]
fn empty_or_unobservable_windows_are_checked_zero_maps() {
    for (length, lags) in [(1, 1..1), (8, 3..3), (8, 3..8)] {
        let kernel = FractionalGlKernel::new(length, f32::MIN_POSITIVE, 3, 24).unwrap();
        let run = |input: &[f32], alpha, gain| {
            kernel.forward_history_log_gain_window(input, &[3], 0, alpha, gain, lags.clone())
        };
        let batch = run(&[f32::MAX; 3], f32::MAX, 0.0).unwrap();
        assert!(batch.output().iter().all(|&v| v == 0.0));
        let gradient = batch.vjp(&[f32::MAX; 3]).unwrap();
        assert_eq!(gradient.input, [0.0; 3]);
        assert_eq!((gradient.alpha, gradient.log_gain), (0.0, 0.0));
        assert_eq!(
            batch.jvp(&[f32::MAX; 3], f32::MAX, f32::MAX).unwrap(),
            [0.0; 3]
        );
        assert!(run(&[f32::NAN; 3], 1.0, 0.0).is_err());
        assert!(run(&[1.0; 3], 0.0, 0.0).is_err());
        assert!(run(&[1.0; 3], 1.0, 100.0).is_err());
        assert!(batch.vjp(&[f32::NAN; 3]).is_err());
        assert!(batch.jvp(&[1.0; 3], f32::NAN, 0.0).is_err());
    }
}

#[test]
fn window_and_declared_kernel_budgets_are_not_silently_relaxed() {
    let kernel = FractionalGlKernel::new(8, 1.0, 8, 32).unwrap();
    for (start, end) in [(0, 3), (4, 3), (1, 9), (usize::MAX, usize::MAX)] {
        assert!(matches!(
            kernel.validate_history_window(start..end),
            Err(FractionalLearningError::HistoryWindow)
        ));
    }
    assert!(matches!(
        kernel.forward_history_log_gain_window(&[1.0; 8], &[8], 0, 0.7, 0.0, 1..2),
        Err(FractionalLearningError::Budget)
    ));
    assert!(kernel
        .forward_history_log_gain_window(&[1.0], &[1], 1, 0.7, 0.0, 3..8)
        .is_err());
}
