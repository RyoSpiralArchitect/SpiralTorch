use super::*;

#[test]
fn distance_and_slope_cover_identity_neighbours_and_disjoint_support() {
    assert_eq!(squared_distance_from_root_chord(0.).unwrap(), (0., 4.));
    let (d, slope) = squared_distance_from_root_chord(2.).unwrap();
    assert!((d - std::f64::consts::PI.powi(2)).abs() < 1e-14);
    assert!((slope - 2. * std::f64::consts::PI).abs() < 1e-14);
    for t in [1e-200, 1e-30, 1e-10, 0.0003999, 0.0004001, 0.1, 1., 1.99] {
        let (d, slope) = squared_distance_from_root_chord(t).unwrap();
        assert!(d > 0. && slope.is_finite());
        let want = 16. * (0.5 * t.sqrt()).asin().powi(2);
        assert!((d / want - 1.).abs() < 2e-15);
        if t > 1e-8 {
            let eps = t * 1e-5;
            let numerical = (squared_distance_from_root_chord(t + eps).unwrap().0
                - squared_distance_from_root_chord(t - eps).unwrap().0)
                / (2. * eps);
            assert!((slope / numerical - 1.).abs() < 2e-9);
        }
    }
    for t in [-1., 2.01, f64::NAN, f64::INFINITY] {
        assert_eq!(
            squared_distance_from_root_chord(t),
            Err(FisherRaoError::RootChord)
        );
    }
}

#[test]
fn full_logit_and_gain_pullbacks_match_finite_differences() {
    let s = FisherRaoBiasSpec::new([2, 3, 4], 2).unwrap();
    let x: Vec<_> = (0..s.coordinates_len())
        .map(|i| ((i * 7 % 23) as f32 - 11.) * 0.17)
        .collect();
    let gains = [-0.7, 1.2];
    let seed: Vec<_> = (0..s.scores_len())
        .map(|i| ((i * 5 % 17) as f32 - 8.) * 0.13)
        .collect();
    let f = FisherRaoBiasForward::new(s, &x, &gains).unwrap();
    let g = f.backward(&seed).unwrap();
    let objective = |x: &[f32], raw: &[f32]| {
        FisherRaoBiasForward::new(s, x, raw)
            .unwrap()
            .scores()
            .iter()
            .zip(&seed)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    for i in 0..x.len() + gains.len() {
        let mut hi = x.clone();
        let mut lo = x.clone();
        let mut gh = gains;
        let mut gl = gains;
        let eps = 0.002;
        let (want, step) = if i < x.len() {
            hi[i] += eps;
            lo[i] -= eps;
            (g.coordinates[i], f64::from(hi[i]) - f64::from(lo[i]))
        } else {
            let k = i - x.len();
            gh[k] += eps;
            gl[k] -= eps;
            (g.raw_gain[k], f64::from(gh[k]) - f64::from(gl[k]))
        };
        let numerical = (objective(&hi, &gh) - objective(&lo, &gl)) / step;
        assert!(
            (numerical - f64::from(want)).abs() < 3e-4,
            "{i}: {numerical} != {want}"
        );
    }
    for row in g.coordinates.chunks_exact(4) {
        assert!(row.iter().map(|&v| f64::from(v)).sum::<f64>().abs() < 2e-7);
    }
}

#[test]
fn causal_mask_shift_invariance_and_identity_are_not_detaches() {
    let s = FisherRaoBiasSpec::new([1, 3, 3], 2).unwrap();
    let x = [0.25, -0.5, 1., 1., -0.25, 0.5, -0.75, 1.5, 0.25];
    let f = FisherRaoBiasForward::new(s, &x, &[0.1, -0.2]).unwrap();
    let mut shifted = x;
    for (row, offset) in shifted.chunks_exact_mut(3).zip([8., -4., 16.]) {
        for v in row {
            *v += offset;
        }
    }
    assert_eq!(
        f.scores(),
        FisherRaoBiasForward::new(s, &shifted, &[0.1, -0.2])
            .unwrap()
            .scores()
    );
    let mut seed = vec![0.; s.scores_len()];
    seed[3] = 1.; // q=1,k=0: both endpoints, but never the suffix.
    let g = f.backward(&seed).unwrap();
    assert!(g.coordinates[..3].iter().any(|v| *v != 0.));
    assert!(g.coordinates[3..6].iter().any(|v| *v != 0.));
    assert_eq!(&g.coordinates[6..], &[0.; 3]);
    for h in 0..2 {
        for q in 0..3 {
            for k in q..3 {
                assert_eq!(f.scores()[(h * 3 + q) * 3 + k], 0.);
            }
        }
    }
    let identical = FisherRaoBiasForward::new(s, &[0.25; 9], &[0.1, -0.2]).unwrap();
    let g = identical.backward(&[1.; 18]).unwrap();
    assert!(identical.scores().iter().all(|v| *v == 0.));
    assert!(g.coordinates.iter().chain(&g.raw_gain).all(|v| *v == 0.));
}

#[test]
fn checked_shapes_nonfinite_and_extreme_finite_logits() {
    for shape in [[0, 2, 3], [1, 0, 3], [1, 2, 0]] {
        assert_eq!(FisherRaoBiasSpec::new(shape, 2), Err(FisherRaoError::Shape));
    }
    assert_eq!(
        FisherRaoBiasSpec::new([1, 2, 3], 0),
        Err(FisherRaoError::Shape)
    );
    assert!(FisherRaoBiasSpec::new([usize::MAX, 2, 2], 1).is_err());
    assert!(FisherRaoBiasSpec::new([1, u32::MAX as usize, 1], 1).is_err());
    let s = FisherRaoBiasSpec::new([1, 2, 2], 1).unwrap();
    let f =
        FisherRaoBiasForward::new(s, &[f32::MAX, -f32::MAX, -f32::MAX, f32::MAX], &[0.]).unwrap();
    assert!(f.scores().iter().all(|v| v.is_finite()));
    assert!(f
        .backward(&[1.; 4])
        .unwrap()
        .coordinates
        .iter()
        .all(|v| *v == 0.));
    assert!(FisherRaoBiasForward::new(s, &[0.; 3], &[0.]).is_err());
    assert!(FisherRaoBiasForward::new(s, &[0.; 4], &[]).is_err());
    for bad in [f32::NAN, f32::INFINITY] {
        assert!(FisherRaoBiasForward::new(s, &[0., 0., 0., bad], &[0.]).is_err());
        assert!(FisherRaoBiasForward::new(s, &[0.; 4], &[bad]).is_err());
        assert!(f.backward(&[0., bad, 0., 0.]).is_err());
    }
    assert!(f.backward(&[0.; 3]).is_err());
}
