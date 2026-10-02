use st_core::theory::microlocal::{EllipticLearningError, EllipticWarp};

const INPUT: [f32; 12] = [1., 0.2, 0.3, 1., -0.4, 0.2, 1., 0.5, -0.6, 1., -0.3, -0.2];

#[test]
fn zero_mix_preserves_local_features_and_vjp_but_gate_can_learn() {
    let warp = EllipticWarp::for_learning(1.0, 3, 2).unwrap();
    let local = warp.differentiate_batch(&INPUT, 4).unwrap();
    let seed = (0..36).map(|i| (i as f32 * 0.3).cos()).collect::<Vec<_>>();
    for raw in [0.0, -0.0] {
        let mixed = warp
            .differentiate_gated_causal_batch(&INPUT, [2, 2], raw, 4, 8)
            .unwrap();
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(local.features()), bits(mixed.features()));
        let gradients = mixed.vjp(&seed).unwrap();
        assert_eq!(
            bits(&local.vjp(&seed).unwrap()),
            bits(&gradients.orientations)
        );
        assert!(gradients.raw_mix.is_finite() && gradients.raw_mix.abs() > 0.01);
    }
}

#[test]
fn orientation_and_shared_gate_vjp_match_finite_differences() {
    let warp = EllipticWarp::for_learning(1.0, 3, 2).unwrap();
    let seed = (0..36).map(|i| (i as f32 * 0.2).sin()).collect::<Vec<_>>();
    let loss = |x: &[f32], raw| {
        warp.differentiate_gated_causal_batch(x, [2, 2], raw, 4, 8)
            .unwrap()
            .features()
            .iter()
            .zip(&seed)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    for raw in [-0.7, 0.0, 0.8] {
        let batch = warp
            .differentiate_gated_causal_batch(&INPUT, [2, 2], raw, 4, 8)
            .unwrap();
        let g = batch.vjp(&seed).unwrap();
        for i in 0..INPUT.len() {
            let mut plus = INPUT;
            let mut minus = INPUT;
            plus[i] += 0.001;
            minus[i] -= 0.001;
            let numeric = (loss(&plus, raw) - loss(&minus, raw)) / f64::from(plus[i] - minus[i]);
            assert!(
                (numeric - f64::from(g.orientations[i])).abs() < 0.003,
                "raw={raw} i={i}"
            );
        }
        let numeric = (loss(&INPUT, raw + 0.001) - loss(&INPUT, raw - 0.001))
            / f64::from((raw + 0.001) - (raw - 0.001));
        assert!((numeric - f64::from(g.raw_mix)).abs() < 0.003);
    }
}

#[test]
fn batch_sum_causal_boundaries_saturation_and_validation() {
    let warp = EllipticWarp::for_learning(1., 2, 1).unwrap();
    let full = warp
        .differentiate_gated_causal_batch(&INPUT, [2, 2], 0.5, 4, 8)
        .unwrap();
    let mut seed = [0.; 36];
    seed[..9].fill(1.);
    let g = full.vjp(&seed).unwrap();
    assert_eq!(&g.orientations[3..], &[0.; 9]);
    assert_eq!(g.raw_mix, 0.); // The first token has no other context.
    seed[..18].fill(1.);
    let prefix = warp
        .differentiate_gated_causal_batch(&INPUT[..6], [1, 2], 0.5, 2, 4)
        .unwrap();
    assert_eq!(prefix.features(), &full.features()[..18]);
    let a = full.vjp(&seed).unwrap();
    let b = prefix.vjp(&seed[..18]).unwrap();
    assert_eq!(&a.orientations[..6], b.orientations);
    assert_eq!(a.raw_mix, b.raw_mix);
    let duplicated = [&INPUT[..6], &INPUT[..6]].concat();
    let twice = warp
        .differentiate_gated_causal_batch(&duplicated, [2, 2], 0.5, 4, 8)
        .unwrap()
        .vjp(&[1.; 36])
        .unwrap();
    assert!((twice.raw_mix - 2. * b.raw_mix).abs() < 1e-6);
    let saturated = warp
        .differentiate_gated_causal_batch(&INPUT, [2, 2], 100., 4, 8)
        .unwrap();
    let causal = warp.differentiate_causal_batch(&INPUT, 2, 2, 4, 8).unwrap();
    assert_eq!(saturated.features(), causal.features());
    assert_eq!(saturated.vjp(&[1.; 36]).unwrap().raw_mix, 0.);
    for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            warp.differentiate_gated_causal_batch(&INPUT, [2, 2], raw, 4, 8)
                .unwrap_err(),
            EllipticLearningError::Configuration
        );
    }
    assert_eq!(
        warp.differentiate_gated_causal_batch(&[f32::NAN; 12], [2, 2], 0., 4, 7)
            .unwrap_err(),
        EllipticLearningError::PairBudget
    );
    assert!(full.vjp(&[]).is_err());
    assert!(full.vjp(&[f32::NAN; 36]).is_err());
}
