use st_core::theory::microlocal::{EllipticLearningError, EllipticWarp};

#[test]
fn causal_feature_attention_vjp_includes_tied_query_key_and_value_paths() {
    let warp = EllipticWarp::for_learning(1.0, 3, 2).unwrap();
    let x = [1., 0.2, 0.3, 1., -0.4, 0.2, 1., 0.5, -0.6, 1., -0.3, -0.2];
    let seed = (0..36).map(|i| (i as f32 * 0.3).cos()).collect::<Vec<_>>();
    let y = warp.differentiate_causal_batch(&x, 2, 2, 4, 8).unwrap();
    let gradient = y.vjp(&seed).unwrap();
    let loss = |x: &[f32]| {
        warp.differentiate_causal_batch(x, 2, 2, 4, 8)
            .unwrap()
            .features()
            .iter()
            .zip(&seed)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    for i in 0..x.len() {
        let mut plus = x;
        let mut minus = x;
        plus[i] += 0.001;
        minus[i] -= 0.001;
        let numeric = (loss(&plus) - loss(&minus)) / f64::from(plus[i] - minus[i]);
        assert!(
            (numeric - f64::from(gradient[i])).abs() < 0.002,
            "i={i} numeric={numeric} actual={}",
            gradient[i]
        );
    }
    let mut first_seed = [0.; 36];
    first_seed[..9].fill(1.);
    let gradient = y.vjp(&first_seed).unwrap();
    assert!(gradient[..3].iter().any(|&v| v != 0.));
    assert_eq!(&gradient[3..], &[0.; 9]);
    assert!(y.vjp(&[f32::NAN; 36]).is_err());
}

#[test]
fn causal_shapes_and_budget_are_checked_before_geometry() {
    let warp = EllipticWarp::for_learning(1., 2, 1).unwrap();
    assert_eq!(
        warp.differentiate_causal_batch(&[f32::NAN; 12], 1, 4, 4, 15)
            .unwrap_err(),
        EllipticLearningError::PairBudget
    );
    for (batch, sequence, rows, pairs) in [
        (0, 4, 4, 16),
        (2, 0, 4, 16),
        (2, 2, 4, 0),
        (3, 2, 4, 16),
        (2, 2, 3, 16),
        (usize::MAX, 2, 4, 16),
    ] {
        assert!(warp
            .differentiate_causal_batch(&[1.; 12], batch, sequence, rows, pairs)
            .is_err());
    }
}

#[test]
fn near_pole_angles_rotors_and_vjp_do_not_collapse() {
    let warp = EllipticWarp::for_learning(1.0, 4, 2).unwrap();
    let orientation = [1e-4, 2e-4, 1.0];
    let batch = warp.differentiate_batch(&orientation, 1).unwrap();
    let expected_angle = 5e-8f32.sqrt().atan2(1.0);
    assert!((batch.features()[0] - expected_angle).abs() < 1e-9);
    assert!((batch.features()[6] + 2e-4).abs() < 1e-9);
    assert!((batch.features()[7] - 1e-4).abs() < 1e-9);
    for feature in 0..9 {
        let mut seed = [0.0; 9];
        seed[feature] = 1.0;
        let gradient = batch.vjp(&seed).unwrap();
        for coordinate in 0..3 {
            let h = if coordinate == 2 { 1e-3 } else { 2e-7 };
            let mut plus = orientation;
            let mut minus = orientation;
            plus[coordinate] += h;
            minus[coordinate] -= h;
            let yp = warp.differentiate_batch(&plus, 1).unwrap().features()[feature];
            let ym = warp.differentiate_batch(&minus, 1).unwrap().features()[feature];
            let numeric = (yp - ym) / (plus[coordinate] - minus[coordinate]);
            // The f32 normal-bias feature rounds to one at this pole distance.
            let atol = if feature == 4 { 3e-4 } else { 2e-4 };
            assert!(
                (gradient[coordinate] - numeric).abs() <= atol + 2e-3 * numeric.abs(),
                "feature={feature} coordinate={coordinate}: analytic={} numeric={numeric}",
                gradient[coordinate]
            );
        }
    }
}

#[test]
fn learning_snapshot_matches_single_rows_and_preserves_large_finite_directions() {
    let warp = EllipticWarp::for_learning(1.5, 3, 2).unwrap();
    let rows = [0.3, 0.4, 0.8, 1e20, 2e20, 3e20];
    let batch = warp.differentiate_batch(&rows, 2).unwrap();
    let (_, single) = warp.map_orientation_with_differential(&rows[..3]).unwrap();
    assert_eq!(&batch.features()[..9], single.feature_slice());
    let scaled = warp.differentiate_batch(&[1.0, 2.0, 3.0], 1).unwrap();
    for (&large, &small) in batch.features()[9..].iter().zip(scaled.features()) {
        assert!((large - small).abs() < 1e-6);
    }
    let gradient = batch.vjp(&[0.5; 18]).unwrap();
    assert!(gradient.iter().all(|v| v.is_finite()));
    assert!(gradient[3..].iter().any(|&v| v != 0.0));
    assert_eq!(batch.telemetry().len(), 2);
}

#[test]
fn chart_validation_and_empty_batches_are_explicit() {
    let warp = EllipticWarp::for_learning(1.0, 2, 1).unwrap();
    for bad in [
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.3],
        [f32::NAN, 1.0, 2.0],
        [1e-30, 0.0, 1e20],
    ] {
        assert!(matches!(
            warp.differentiate_batch(&bad, 1),
            Err(EllipticLearningError::InvalidRow { row: 0 })
        ));
    }
    assert!(warp.differentiate_batch(&[1.0, 2.0], 1).is_err());
    assert!(warp.differentiate_batch(&[1.0; 6], 1).is_err());
    let empty = warp.differentiate_batch(&[], 1).unwrap();
    assert!(empty.features().is_empty());
    assert!(empty.vjp(&[]).unwrap().is_empty());
    assert!(empty.vjp(&[1.0]).is_err());
    let valid = warp.differentiate_batch(&[1.0, 0.3, 0.2], 1).unwrap();
    assert!(valid.vjp(&[f32::NAN; 9]).is_err());
    for radius in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        assert!(EllipticWarp::for_learning(radius, 2, 1).is_err());
    }
    assert!(EllipticWarp::for_learning(1.0, 0, 1).is_err());
    assert!(EllipticWarp::for_learning(1.0, 2, 0).is_err());
    let near_pole = warp.differentiate_batch(&[1e-4, 2e-4, 1.0], 1).unwrap();
    let mut upstream = [0.0; 9];
    upstream[2] = f32::MAX;
    assert_eq!(
        near_pole.vjp(&upstream),
        Err(EllipticLearningError::NonFiniteGradient)
    );
}
