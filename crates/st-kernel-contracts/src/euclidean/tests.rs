use super::*;

#[test]
fn scaled_chord_has_both_endpoint_and_gain_derivatives() {
    let s = EuclideanBiasSpec::new([1, 2, 1], 1, 4.).unwrap();
    let f = EuclideanBiasForward::new(s, &[0., 1.], &[0.]).unwrap();
    assert_eq!(f.scores()[1], 0.);
    assert!((f64::from(f.scores()[2]) + 4. * 2f64.ln()).abs() < 1e-6);
    let g = f.backward(&[0., 0., 1., 0.]).unwrap();
    assert!((f64::from(g.coordinates[0]) - 8. * 2f64.ln()).abs() < 1e-6);
    assert_eq!(g.coordinates[0], -g.coordinates[1]);
    assert_eq!(g.raw_gain, [-2.]);
}

#[test]
fn validation_is_not_a_ball_domain_check() {
    for shape in [[0, 2, 1], [1, 0, 1], [1, 2, 0], [usize::MAX, 2, 1]] {
        assert!(EuclideanBiasSpec::new(shape, 1, 1.).is_err());
    }
    assert!(EuclideanBiasSpec::new([1, 2, 1], 0, 1.).is_err());
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(EuclideanBiasSpec::new([1, 2, 1], 1, scale).is_err());
    }
    let s = EuclideanBiasSpec::new([1, 2, 1], 1, 1.).unwrap();
    assert!(EuclideanBiasForward::new(s, &[2., 3.], &[0.]).is_ok());
    assert!(EuclideanBiasForward::new(s, &[0.], &[0.]).is_err());
    assert!(EuclideanBiasForward::new(s, &[0., f32::NAN], &[0.]).is_err());
    let f = EuclideanBiasForward::new(s, &[0., 1.], &[0.]).unwrap();
    assert!(f.backward(&[0.; 3]).is_err());
    assert!(f.backward(&[0., f32::NAN, 0., 0.]).is_err());
}

#[test]
fn retained_wide_products_have_nonzero_recoverable_adjoints() {
    let s = EuclideanBiasSpec::new([1, 2, 1], 1, 4.).unwrap();
    for (x, raw, seed) in [(1e-25, 0., 1e30), (2e14, -110., 1.), (1e-20, 1e20, 1.)] {
        let f = EuclideanBiasForward::new(s, &[0., x], &[raw]).unwrap();
        let g = f.backward(&[0., 0., seed, 0.]).unwrap();
        assert!(g
            .coordinates
            .iter()
            .chain(&g.raw_gain)
            .all(|x| x.is_finite() && *x != 0.));
    }
}

#[test]
fn independent_finite_differences_and_translation_invariance() {
    let s = EuclideanBiasSpec::new([2, 3, 2], 2, 4.).unwrap();
    let x: Vec<_> = (0..12).map(|i| i as f32 * 0.11 - 0.5).collect();
    let raw = [-0.7, 0.4];
    let seed: Vec<_> = (0..36).map(|i| (i % 7) as f32 * 0.03 - 0.1).collect();
    let f = EuclideanBiasForward::new(s, &x, &raw).unwrap();
    let g = f.backward(&seed).unwrap();
    let loss = |x: &[f32], gain: &[f32]| {
        EuclideanBiasForward::new(s, x, gain)
            .unwrap()
            .scores()
            .iter()
            .zip(&seed)
            .map(|(&x, &g)| f64::from(x) * f64::from(g))
            .sum::<f64>()
    };
    for i in 0..x.len() + raw.len() {
        let (mut a, mut b) = (x.clone(), x.clone());
        let (mut c, mut d) = (raw, raw);
        let expected = if i < x.len() {
            a[i] += 1e-3;
            b[i] -= 1e-3;
            g.coordinates[i]
        } else {
            c[i - x.len()] += 1e-3;
            d[i - x.len()] -= 1e-3;
            g.raw_gain[i - x.len()]
        };
        let difference = (loss(&a, &c) - loss(&b, &d)) / 2e-3;
        assert!(
            (difference - f64::from(expected)).abs() < 2e-4,
            "{i}: {difference} != {expected}"
        );
    }
    let shifted: Vec<_> = x.iter().map(|v| v + 2.).collect();
    let other = EuclideanBiasForward::new(s, &shifted, &raw).unwrap();
    for (a, b) in f.scores().iter().zip(other.scores()) {
        assert!((a - b).abs() < 3e-6);
    }
}
