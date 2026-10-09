use super::*;

#[test]
fn exact_f32_open_ball_margin_survives_norm_cancellation() {
    let u = 2f32.powi(-24);
    let x = [
        1. - u,
        2f32.powi(-12) * (1. - u),
        2f32.powi(-12),
        u * (1. - u),
    ];
    let points: Vec<_> = x.into_iter().chain(x.map(|x| -x)).collect();
    let spec = PoincareBiasSpec::new([1, 2, 4], 1, -1.).unwrap();
    let f = PoincareBiasForward::new(spec, &points, &[0.]).unwrap();
    let a = f64::from(u).powi(3) * (1. - f64::from(u));
    let expected = -4. * gain(0.).0 * (4. * (1. - a) / (a * a)).sqrt().asinh().powi(2);
    assert!((f64::from(f.scores()[2]) / expected - 1.).abs() < 1e-6);
    let g = f.backward(&[0., 0., 1., 0.]).unwrap();
    assert!(g.coordinates.iter().all(|x| x.is_finite() && *x != 0.));
}

#[test]
fn metric_matches_radial_geodesic_and_both_endpoint_differences() {
    let s = PoincareBiasSpec::new([1, 2, 2], 2, -0.75).unwrap();
    let x = [0.1, -0.2, 0.3, 0.05];
    let gains = [-0.6, 0.4];
    let seed = [0.2, 7., -0.4, 0.3, -0.1, 8., 0.7, -0.2];
    let f = PoincareBiasForward::new(s, &x, &gains).unwrap();
    let g = f.backward(&seed).unwrap();
    let objective = |x: &[f32], gains: &[f32]| -> f64 {
        PoincareBiasForward::new(s, x, gains)
            .unwrap()
            .scores()
            .iter()
            .zip(seed)
            .map(|(&v, g)| f64::from(v) * f64::from(g))
            .sum()
    };
    for i in 0..6 {
        let (mut xp, mut xm, mut gp, mut gm) = (x, x, gains, gains);
        if i < 4 {
            xp[i] += 1e-3;
            xm[i] -= 1e-3;
        } else {
            gp[i - 4] += 1e-3;
            gm[i - 4] -= 1e-3;
        }
        let numeric = (objective(&xp, &gp) - objective(&xm, &gm)) / 0.002;
        let analytic = if i < 4 {
            g.coordinates[i]
        } else {
            g.raw_gain[i - 4]
        };
        assert!(
            (numeric - f64::from(analytic)).abs() < 2e-4,
            "{i}: {numeric} != {analytic}"
        );
    }
    let f = PoincareBiasForward::new(s, &[0., 0., 0.25, 0.], &gains).unwrap();
    let expected =
        -gain(gains[0]).0 * (2. * (0.75f64.sqrt() * 0.25).atanh() / 0.75f64.sqrt()).powi(2);
    assert!((f64::from(f.scores()[2]) - expected).abs() < 1e-7);
}

#[test]
fn coincidence_future_and_document_boundaries_have_exact_zero_vjps() {
    let s = PoincareBiasSpec::new([2, 3, 2], 2, -1.).unwrap();
    let x = [0.1, 0.2, 0.1, 0.2, -0.3, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3, 0.2];
    let f = PoincareBiasForward::new(s, &x, &[0., 1.]).unwrap();
    let mut seed = vec![0.; s.scores_len()];
    seed[1] = f32::MAX; // masked future
    seed[3] = 1.; // identical distinct positions
    let g = f.backward(&seed).unwrap();
    assert!(g.coordinates.iter().chain(&g.raw_gain).all(|&v| v == 0.));
    seed[3] = 0.;
    seed[6] = 1.;
    let g = f.backward(&seed).unwrap();
    assert!(g.coordinates[..6].iter().any(|v| *v != 0.));
    assert!(g.coordinates[6..].iter().all(|&v| v == 0.));
    assert_eq!(g.raw_gain[1], 0.);
}

#[test]
fn tiny_separation_and_gain_keep_recoverable_gradients() {
    let s = PoincareBiasSpec::new([1, 2, 1], 1, -1.).unwrap();
    let f = PoincareBiasForward::new(s, &[0., 1e-25], &[0.]).unwrap();
    let g = f.backward(&[0., 0., 1e30, 0.]).unwrap();
    assert!(g.coordinates[0] > 1e5 && g.coordinates[1] < -1e5);
    assert!(g.raw_gain[0] < 0.);
    let s = PoincareBiasSpec::new([1, 2, 1], 1, -1e-30).unwrap();
    let f = PoincareBiasForward::new(s, &[0., 2e14], &[-110.]).unwrap();
    let g = f.backward(&[0., 0., 1., 0.]).unwrap();
    assert!(f.scores()[2] < 0. && g.raw_gain[0] < 0.);
    assert!(g.coordinates.iter().all(|v| v.is_finite() && *v != 0.));
}

#[test]
fn rejects_bad_shape_domain_finiteness_and_return_overflow() {
    assert_eq!(
        PoincareBiasSpec::new([1, 0, 2], 1, -1.),
        Err(PoincareError::Shape)
    );
    assert_eq!(
        PoincareBiasSpec::new([1, 2, 2], 0, -1.),
        Err(PoincareError::Shape)
    );
    assert_eq!(
        PoincareBiasSpec::new([1, 2, 2], 1, 0.),
        Err(PoincareError::Curvature)
    );
    assert_eq!(
        PoincareBiasSpec::new([1, usize::MAX, 2], 1, -1.),
        Err(PoincareError::Overflow)
    );
    let s = PoincareBiasSpec::new([1, 2, 1], 1, -1.).unwrap();
    for x in [[0., 1.], [0., 1.01]] {
        assert!(matches!(
            PoincareBiasForward::new(s, &x, &[0.]),
            Err(PoincareError::Domain)
        ));
    }
    assert!(matches!(
        PoincareBiasForward::new(s, &[0., f32::NAN], &[0.]),
        Err(PoincareError::NonFinite)
    ));
    assert!(matches!(
        PoincareBiasForward::new(s, &[0., 0.8], &[f32::MAX]),
        Err(PoincareError::NonFinite)
    ));
    let f = PoincareBiasForward::new(s, &[0., 0.8], &[0.]).unwrap();
    assert!(matches!(
        f.backward(&[0., f32::NAN, 0., 0.]),
        Err(PoincareError::NonFinite)
    ));
    assert!(matches!(
        f.backward(&[0., 0., f32::MAX, 0.]),
        Err(PoincareError::NonFinite)
    ));
}
