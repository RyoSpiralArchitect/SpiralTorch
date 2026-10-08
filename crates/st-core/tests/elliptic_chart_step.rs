use st_core::theory::microlocal::{EllipticLearningError, EllipticWarp};

#[test]
fn mean_metric_matches_basis_jvps_and_step_preserves_budget() {
    let warp = EllipticWarp::for_learning(1.3, 3, 2).unwrap();
    let snapshot = warp
        .differentiate_batch(&[1., 16., 5., 1., -12., 3.], 2)
        .unwrap();
    let proposal = [0.01, -0.03, 0.02, 0.04, -0.02, 0.01];
    let result = snapshot.chart_step(&proposal, 0.1).unwrap();
    let x = snapshot.jvp(&[0., 1., 0., 0., 1., 0.]).unwrap();
    let y = snapshot.jvp(&[0., 0., 1., 0., 0., 1.]).unwrap();
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
            / 2.
    };
    for (actual, expected) in
        result
            .metric
            .into_iter()
            .zip([dot(&x, &x), dot(&x, &y), dot(&y, &x), dot(&y, &y)])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!((result.step_l2 / result.proposal_l2 - 1.).abs() < 1e-7);
    assert!(result.cosine.unwrap() > 0. && result.cosine.unwrap() < 1.);
    assert!((1.0..=21.000001).contains(&result.damped_condition));
    let [a, b, _, c] = result.metric;
    let damping = (a + c) * 0.5 * f64::from(0.1f32);
    let mut recovered = Vec::new();
    for row in 0..2 {
        for col in 0..3 {
            let p = f64::from(result.values[col]);
            let q = f64::from(result.values[3 + col]);
            recovered.push(if row == 0 {
                (a + damping) * p + b * q
            } else {
                b * p + (c + damping) * q
            });
        }
    }
    let ratio = recovered[0] / f64::from(proposal[0]);
    for (actual, p) in recovered.into_iter().zip(proposal) {
        assert!((actual - ratio * f64::from(p)).abs() < 1e-9);
    }
}

#[test]
fn metric_mean_is_duplication_invariant_and_radial_scale_does_not_change_direction() {
    let warp = EllipticWarp::for_learning(1., 3, 1).unwrap();
    let x = [1., 0.2, 0.3];
    let proposal = [0.3, -0.1, 0.4, 0.2];
    let one = warp
        .differentiate_batch(&x, 1)
        .unwrap()
        .chart_step(&proposal, 0.1)
        .unwrap();
    let two = warp
        .differentiate_batch(&x.repeat(2), 2)
        .unwrap()
        .chart_step(&proposal, 0.1)
        .unwrap();
    let scaled = warp
        .differentiate_batch(&x.map(|v| v * 1e6), 1)
        .unwrap()
        .chart_step(&proposal, 0.1)
        .unwrap();
    for i in 0..4 {
        assert!((one.metric[i] - two.metric[i]).abs() < 1e-12);
        assert!((one.values[i] - two.values[i]).abs() < 1e-7);
        assert!((one.values[i] - scaled.values[i]).abs() < 1e-6);
    }
}

#[test]
fn zero_empty_nonfinite_and_step_budgets_are_explicit() {
    let warp = EllipticWarp::for_learning(1., 2, 1).unwrap();
    let batch = warp.differentiate_batch(&[1., 0.3, 0.4], 1).unwrap();
    let zero = batch.chart_step(&[0.; 4], 0.1).unwrap();
    assert_eq!(zero.values, [0.; 4]);
    assert_eq!(zero.proposal_l2, 0.);
    assert_eq!(zero.step_l2, 0.);
    assert_eq!(zero.cosine, None);
    for bad in [
        vec![],
        vec![1.],
        vec![f32::NAN; 2],
        vec![f32::INFINITY; 2],
        vec![0.; 131_074],
    ] {
        assert!(matches!(
            batch.chart_step(&bad, 0.1),
            Err(EllipticLearningError::InvalidProposal)
        ));
    }
    for bad in [0., 1e-7, 1.1, f32::INFINITY, f32::NAN] {
        assert!(matches!(
            batch.chart_step(&[1.; 2], bad),
            Err(EllipticLearningError::Configuration)
        ));
    }
    assert!(matches!(
        warp.differentiate_batch(&[], 1)
            .unwrap()
            .chart_step(&[1.; 2], 0.1),
        Err(EllipticLearningError::InvalidMetric)
    ));
}
