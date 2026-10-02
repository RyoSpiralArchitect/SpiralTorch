use st_core::theory::microlocal::{EllipticLearningError, EllipticWarp};

const INPUT: [f32; 12] = [1., 0.2, 0.3, 1., -0.4, 0.2, 1., 0.5, -0.6, 1., -0.3, -0.2];

#[test]
fn zero_is_exactly_local_with_a_live_gate_derivative() {
    let warp = EllipticWarp::for_learning(1.0, 3, 2).unwrap();
    let local = warp.differentiate_batch(&INPUT, 4).unwrap();
    let seed = (0..36).map(|i| (i as f32 * 0.3).cos()).collect::<Vec<_>>();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for raw in [0.0, -0.0] {
        let anchored = warp.differentiate_anchored_batch(&INPUT, raw, 4).unwrap();
        assert_eq!(bits(anchored.features()), bits(local.features()));
        let gradient = anchored.vjp(&seed).unwrap();
        assert_eq!(
            bits(&gradient.orientations),
            bits(&local.vjp(&seed).unwrap())
        );
        assert!(gradient.raw_mix.is_finite() && gradient.raw_mix.abs() > 0.01);
    }
}

#[test]
fn both_vjps_match_finite_differences_for_signed_gates_and_warps() {
    let seed = (0..36).map(|i| (i as f32 * 0.2).sin()).collect::<Vec<_>>();
    for (radius, sheets, harmonics) in [(1., 2, 1), (1.7, 3, 2)] {
        let warp = EllipticWarp::for_learning(radius, sheets, harmonics).unwrap();
        let loss = |x: &[f32], raw| {
            warp.differentiate_anchored_batch(x, raw, 4)
                .unwrap()
                .features()
                .iter()
                .zip(&seed)
                .map(|(&x, &u)| f64::from(x) * f64::from(u))
                .sum::<f64>()
        };
        for raw in [-0.7, 0.0, 0.8] {
            let gradient = warp
                .differentiate_anchored_batch(&INPUT, raw, 4)
                .unwrap()
                .vjp(&seed)
                .unwrap();
            for i in 0..INPUT.len() {
                let mut plus = INPUT;
                let mut minus = INPUT;
                plus[i] += 0.001;
                minus[i] -= 0.001;
                let numeric =
                    (loss(&plus, raw) - loss(&minus, raw)) / f64::from(plus[i] - minus[i]);
                assert!(
                    (numeric - f64::from(gradient.orientations[i])).abs() < 0.004,
                    "raw={raw}, i={i}"
                );
            }
            let numeric = (loss(&INPUT, raw + 0.001) - loss(&INPUT, raw - 0.001))
                / f64::from((raw + 0.001) - (raw - 0.001));
            assert!((numeric - f64::from(gradient.raw_mix)).abs() < 0.004);
        }
    }
}

#[test]
fn independent_rows_sum_reduction_anchor_and_snapshot_lifetime() {
    let warp = EllipticWarp::for_learning(1.2, 3, 2).unwrap();
    let full = warp.differentiate_anchored_batch(&INPUT, -0.5, 4).unwrap();
    let mut expected_gate = 0.;
    for row in 0..4 {
        let one = warp
            .differentiate_anchored_batch(&INPUT[row * 3..row * 3 + 3], -0.5, 1)
            .unwrap();
        assert_eq!(one.features(), &full.features()[row * 9..row * 9 + 9]);
        let mut upstream = [0.; 36];
        upstream[row * 9..row * 9 + 9].fill(1.);
        let gradient = full.vjp(&upstream).unwrap();
        assert_eq!(&gradient.orientations[..row * 3], vec![0.; row * 3]);
        assert_eq!(&gradient.orientations[row * 3 + 3..], vec![0.; 9 - row * 3]);
        let local = one.vjp(&[1.; 9]).unwrap();
        assert_eq!(
            &gradient.orientations[row * 3..row * 3 + 3],
            local.orientations
        );
        expected_gate += local.raw_mix;
    }
    assert!((full.vjp(&[1.; 36]).unwrap().raw_mix - expected_gate).abs() < 1e-5);
    let anchor = warp.differentiate_batch(&[1., 0., 0.], 1).unwrap();
    for raw in [-0.5, 0.5, 100.] {
        let at_anchor = warp
            .differentiate_anchored_batch(&[1., 0., 0.], raw, 1)
            .unwrap();
        assert_eq!(at_anchor.features(), anchor.features());
        assert_eq!(at_anchor.vjp(&[1.; 9]).unwrap().raw_mix, 0.);
    }
    let saturated = warp.differentiate_anchored_batch(&INPUT, 100., 4).unwrap();
    assert_eq!(saturated.features(), anchor.features().repeat(4));
    assert_eq!(saturated.vjp(&[1.; 36]).unwrap().orientations, vec![0.; 12]);
    assert_eq!(saturated.vjp(&[1.; 36]).unwrap().raw_mix, 0.);
    let before = full.vjp(&[1.; 36]).unwrap();
    let retained = {
        let temporary = EllipticWarp::for_learning(1.2, 3, 2).unwrap();
        temporary
            .differentiate_anchored_batch(&INPUT, -0.5, 4)
            .unwrap()
    };
    assert_eq!(before, retained.vjp(&[1.; 36]).unwrap());
}

#[test]
fn empty_rows_and_invalid_inputs_are_checked_even_at_saturation() {
    let warp = EllipticWarp::for_learning(1., 2, 1).unwrap();
    let empty = warp.differentiate_anchored_batch(&[], -0.5, 1).unwrap();
    assert!(empty.features().is_empty());
    assert_eq!(empty.vjp(&[]).unwrap().raw_mix, 0.);
    for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            warp.differentiate_anchored_batch(&INPUT, raw, 4)
                .unwrap_err(),
            EllipticLearningError::Configuration
        );
    }
    assert!(warp.differentiate_anchored_batch(&INPUT, 0., 0).is_err());
    assert!(warp.differentiate_anchored_batch(&INPUT, 0., 3).is_err());
    assert!(warp
        .differentiate_anchored_batch(&INPUT[..2], 0., 4)
        .is_err());
    assert!(warp
        .differentiate_anchored_batch(&[0., 0., 1.], 100., 4)
        .is_err());
    let full = warp.differentiate_anchored_batch(&INPUT, 0.5, 4).unwrap();
    assert!(full.vjp(&[]).is_err());
    assert!(full.vjp(&[f32::NAN; 36]).is_err());
}
