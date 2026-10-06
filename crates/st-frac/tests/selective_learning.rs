use st_frac::learning::FractionalGlKernel;

#[test]
fn selective_pullbacks_match_joint_components_bit_for_bit() {
    let x: Vec<f32> = (0..24).map(|i| (i as f32 * 0.3).sin()).collect();
    let upstream: Vec<f32> = (0..24).map(|i| (i as f32 * 0.2).cos()).collect();
    for step in [0.4, 1.0, 1.4] {
        for alpha in [0.45, 1.0, 1.5] {
            for kernel_len in [1, 5, 8] {
                let kernel = FractionalGlKernel::new(kernel_len, step, 24, 192).unwrap();
                for axis in 0..3 {
                    for history in [false, true] {
                        let saved = if history {
                            kernel.forward_history(&x, &[2, 4, 3], axis, alpha)
                        } else {
                            kernel.forward(&x, &[2, 4, 3], axis, alpha)
                        }
                        .unwrap();
                        let joint = saved.vjp(&upstream).unwrap();
                        let input = saved.vjp_input(&upstream).unwrap();
                        assert_eq!(input.len(), joint.input.len());
                        for (a, b) in input.iter().zip(joint.input) {
                            assert_eq!(a.to_bits(), b.to_bits());
                        }
                        assert_eq!(
                            saved.vjp_alpha(&upstream).unwrap().to_bits(),
                            joint.alpha.to_bits()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn unrequested_input_overflow_does_not_block_a_finite_order_gradient() {
    let kernel = FractionalGlKernel::new(2, 1.0, 2, 4).unwrap();
    for history in [false, true] {
        let saved = if history {
            kernel.forward_history(&[0.0; 2], &[2], 0, 2.0)
        } else {
            kernel.forward(&[0.0; 2], &[2], 0, 2.0)
        }
        .unwrap();
        let upstream = [0.0, f32::MAX];
        assert!(saved.vjp(&upstream).is_err());
        assert!(saved.vjp_input(&upstream).is_err());
        assert_eq!(saved.vjp_alpha(&upstream).unwrap(), 0.0);
    }
}

#[test]
fn unrequested_order_overflow_does_not_block_a_finite_input_gradient() {
    let kernel = FractionalGlKernel::new(2, 1.0, 2, 4).unwrap();
    for history in [false, true] {
        let saved = if history {
            kernel.forward_history(&[f32::MAX * 0.5, 0.0], &[2], 0, 1.0)
        } else {
            kernel.forward(&[f32::MAX * 0.5, 0.0], &[2], 0, 1.0)
        }
        .unwrap();
        let upstream = [0.0, f32::MAX];
        assert!(saved.vjp(&upstream).is_err());
        assert!(saved.vjp_alpha(&upstream).is_err());
        assert_eq!(
            saved.vjp_input(&upstream).unwrap(),
            [-f32::MAX, if history { 0.0 } else { f32::MAX }]
        );
    }
}

#[test]
fn selected_directions_are_validated_even_for_empty_history() {
    for (k, shape, step, alpha) in [
        (5, vec![2, 3], 0.7, 0.6),
        (1, vec![2, 3], 0.1, 100.0),
        (8, vec![6, 1], 0.1, 100.0),
    ] {
        let kernel = FractionalGlKernel::new(k, step, 6, 48).unwrap();
        let saved = kernel.forward_history(&[1.0; 6], &shape, 1, alpha).unwrap();
        for direction in [
            vec![],
            vec![1.0; 5],
            vec![f32::NAN; 6],
            vec![f32::INFINITY; 6],
        ] {
            assert!(saved.vjp_input(&direction).is_err());
            assert!(saved.vjp_alpha(&direction).is_err());
        }
        if k == 1 || shape[1] == 1 {
            assert_eq!(saved.vjp_input(&[f32::MAX; 6]).unwrap(), [0.0; 6]);
            assert_eq!(saved.vjp_alpha(&[f32::MAX; 6]).unwrap(), 0.0);
        }
    }
}

#[test]
fn integer_order_alpha_pullback_keeps_older_lag_derivatives() {
    let saved = FractionalGlKernel::new(4, 1.0, 4, 16)
        .unwrap()
        .forward_history(&[1.0, 0.0, 0.0, 0.0], &[4], 0, 1.0)
        .unwrap();
    assert_eq!(saved.output().as_slice().unwrap()[2], 0.0);
    assert_eq!(saved.vjp_alpha(&[0.0, 0.0, 1.0, 0.0]).unwrap(), 0.5);
}
