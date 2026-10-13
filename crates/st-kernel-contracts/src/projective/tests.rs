use super::*;

#[test]
fn dynamic_range_wedge_retains_primitive_product_residual() {
    let l = 2f32.powi(30);
    let spec = ProjectiveBiasSpec::new([1, 2, 4], 1).unwrap();
    let f =
        ProjectiveBiasForward::new(spec, &[l, 1., l, 2., l, 2., l, 3.], &[2f32.powi(120)]).unwrap();
    // Anchor differences contribute 2; the complex wedge contributes 1.
    assert_eq!(f.scores()[2], -0.75);
    assert_eq!(
        f.backward(&[0., 0., 1., 0.]).unwrap().raw_gain[0],
        -0.75 * 2f32.powi(-120)
    );
}

#[test]
fn base_difference_retains_low_endpoint_before_wedge_cancellation() {
    let e = 2f32.powi(-121);
    let l = 2f32.powi(122);
    let spec = ProjectiveBiasSpec::new([1, 2, 4], 1).unwrap();
    let dx = ProjectiveBiasForward::new(spec, &[e, 0., 1., 0., 1., 0., l, 0.], &[2f32.powi(60)])
        .unwrap()
        .backward(&[0., 0., 2f32.powi(63), 0.])
        .unwrap()
        .coordinates;
    assert_eq!(dx, [-e, 0., l, 0., e, 0., -e, 0.]);
}

#[test]
fn anchor_cancellation_and_mixed_complex_products_preserve_low_terms() {
    let gain = 2f32.powi(120);
    let e = 2f32.powi(-100);
    let l = 2f32.powi(70);
    let d = 1025. * 2f32.powi(29);
    let im = 1025. * 2f32.powi(-36);
    for (x, want) in [
        (
            [1., 0., e, 0., -1., 0., e, 0.],
            [
                -2f32.powi(39),
                0.,
                2f32.powi(-61),
                0.,
                2f32.powi(39),
                0.,
                2f32.powi(-61),
                0.,
            ],
        ),
        (
            [l, 32., l, 0., l, 32., -l, 0.],
            [d, im, -d, 0., d, im, d, 0.],
        ),
    ] {
        let s = ProjectiveBiasSpec::new([1, 2, 4], 1).unwrap();
        let dx = ProjectiveBiasForward::new(s, &x, &[gain])
            .unwrap()
            .backward(&[0., 0., gain, 0.])
            .unwrap()
            .coordinates;
        assert_eq!(dx, want);
    }
}

#[test]
fn cancelling_large_overlap_keeps_homogeneous_anchor_gradient() {
    let l = 2f32.powi(70);
    let gain = 2f32.powi(120);
    let s = ProjectiveBiasSpec::new([1, 2, 4], 1).unwrap();
    let x = [l, 0., l, 0., l, 0., -l, 0.];
    let dx = ProjectiveBiasForward::new(s, &x, &[gain])
        .unwrap()
        .backward(&[0., 0., gain, 0.])
        .unwrap()
        .coordinates;
    let d = 2f32.powi(29);
    assert_eq!(dx, [d, 0., -d, 0., d, 0., d, 0.]);
    // The small middle product must also survive, not only the fixed anchor.
    let s = ProjectiveBiasSpec::new([1, 2, 6], 1).unwrap();
    let x = [l, 0., 32., 0., l, 0., l, 0., 32., 0., -l, 0.];
    let dx = ProjectiveBiasForward::new(s, &x, &[gain])
        .unwrap()
        .backward(&[0., 0., gain, 0.])
        .unwrap()
        .coordinates;
    let d = 1025. * 2f32.powi(29);
    let middle = 1025. * 2f32.powi(-36);
    assert_eq!(dx, [d, 0., middle, 0., -d, 0., d, 0., middle, 0., d, 0.]);
}

#[test]
fn homogeneous_phase_scaling_and_known_chords() {
    let x = [1., 0., 0., 0.];
    assert_eq!(complex_projective_chord_squared(&x, &x).unwrap(), 0.);
    assert_eq!(
        complex_projective_chord_squared(&x, &[0., 0., 0., 1.]).unwrap(),
        1.
    );
    assert_eq!(
        complex_projective_chord_squared(&x, &[1., 0., 0., 1.]).unwrap(),
        0.5
    );
    let y = [0.2, -0.7, 0.9, 0.1];
    let rotated = [0.7, 0.2, -0.1, 0.9];
    assert!(complex_projective_chord_squared(&y, &rotated).unwrap() < 1e-30);
    let large: Vec<_> = y.iter().map(|v| v * 1e308).collect();
    let tiny: Vec<_> = rotated.iter().map(|v| v * 1e-300).collect();
    assert!(complex_projective_chord_squared(&large, &tiny).unwrap() < 1e-30);
    assert_eq!(
        complex_projective_chord_squared(&x, &y).unwrap(),
        complex_projective_chord_squared(&y, &x).unwrap()
    );
}

#[test]
fn chart_agrees_with_homogeneous_definition_and_bloch_chord() {
    let x = [0.3, -0.2];
    let y = [-0.5, 0.8];
    let d = chart_pair(&x, &y).0;
    let homogeneous =
        complex_projective_chord_squared(&[1., 0., x[0], x[1]], &[1., 0., y[0], y[1]]).unwrap();
    assert!((d - homogeneous).abs() < 1e-15);
    let bloch = |z: [f64; 2]| {
        let n = norm(z);
        [
            2. * z[0] / (1. + n),
            2. * z[1] / (1. + n),
            (1. - n) / (1. + n),
        ]
    };
    let chord: f64 = bloch(x)
        .iter()
        .zip(bloch(y))
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!((d - chord / 4.).abs() < 1e-15);
}

#[test]
fn full_vjp_matches_finite_differences_at_both_endpoints_and_gains() {
    let s = ProjectiveBiasSpec::new([2, 3, 4], 2).unwrap();
    let x: Vec<f32> = (0..s.coordinates_len())
        .map(|i| (i as f32 * 0.71).sin() * 0.8)
        .collect();
    let g = [-1.1, 0.7];
    let seed: Vec<f32> = (0..s.scores_len())
        .map(|i| (i as f32 * 0.23).cos())
        .collect();
    let vjp = ProjectiveBiasForward::new(s, &x, &g)
        .unwrap()
        .backward(&seed)
        .unwrap();
    let objective = |x: &[f32], g: &[f32]| {
        ProjectiveBiasForward::new(s, x, g)
            .unwrap()
            .scores()
            .iter()
            .zip(&seed)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    let eps = 0.001;
    for i in 0..x.len() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[i] += eps;
        minus[i] -= eps;
        let fd = (objective(&plus, &g) - objective(&minus, &g)) / f64::from(plus[i] - minus[i]);
        assert!(
            (fd - f64::from(vjp.coordinates[i])).abs() < 9e-5,
            "coordinate {i}: {fd} vs {}",
            vjp.coordinates[i]
        );
    }
    for i in 0..g.len() {
        let mut plus = g;
        let mut minus = g;
        plus[i] += eps;
        minus[i] -= eps;
        let fd = (objective(&x, &plus) - objective(&x, &minus)) / f64::from(plus[i] - minus[i]);
        assert!((fd - f64::from(vjp.raw_gain[i])).abs() < 9e-5);
    }
}

#[test]
fn zero_chart_causal_mask_and_no_distance_deadzone() {
    let s = ProjectiveBiasSpec::new([1, 2, 2], 1).unwrap();
    let zero = ProjectiveBiasForward::new(s, &[0.; 4], &[0.]).unwrap();
    assert_eq!(zero.scores(), &[0.; 4]);
    assert_eq!(zero.backward(&[1.; 4]).unwrap().coordinates, vec![0.; 4]);
    let tape = ProjectiveBiasForward::new(s, &[0., 0., 1e-30, 0.], &[1e30]).unwrap();
    assert_eq!(tape.scores()[1], 0.);
    assert!(tape.scores()[2] < -0.9e-30);
    let vjp = tape.backward(&[0., 0., 1., 0.]).unwrap();
    assert!((vjp.coordinates[0] - 2.).abs() < 1e-6);
    assert!((vjp.coordinates[2] + 2.).abs() < 1e-6);
    let masked = tape.backward(&[0., 7., 0., 0.]).unwrap();
    assert_eq!(masked.coordinates, vec![0.; 4]);
    assert_eq!(masked.raw_gain, vec![0.]);
}

#[test]
fn huge_coordinates_retain_finite_scores_and_vjp() {
    let s = ProjectiveBiasSpec::new([1, 2, 4], 1).unwrap();
    let x = [f32::MAX, 0., 0., 0., 0., 0., f32::MAX, 0.];
    let tape = ProjectiveBiasForward::new(s, &x, &[0.]).unwrap();
    assert!((tape.scores()[2] + std::f32::consts::LN_2).abs() < 1e-7);
    assert!(tape
        .backward(&[1.; 4])
        .unwrap()
        .coordinates
        .iter()
        .all(|v| v.is_finite()));
    let s = ProjectiveBiasSpec::new([1, 2, 2], 1).unwrap();
    let tape = ProjectiveBiasForward::new(s, &[0., 0., 1e20, 0.], &[0.]).unwrap();
    let dx = tape.backward(&[0., 0., 1e30, 0.]).unwrap().coordinates;
    assert!(dx[2] < -1e-30 && dx[2] > -2e-30, "{}", dx[2]);
}

#[test]
fn rejects_invalid_shapes_nonfinite_and_masked_nonfinite_cotangents() {
    for shape in [[0, 1, 2], [1, 0, 2], [1, 1, 0], [1, 1, 3]] {
        assert_eq!(
            ProjectiveBiasSpec::new(shape, 1),
            Err(ProjectiveError::Shape)
        );
    }
    assert_eq!(
        ProjectiveBiasSpec::new([1, 1, 2], 0),
        Err(ProjectiveError::Shape)
    );
    assert_eq!(
        ProjectiveBiasSpec::new([usize::MAX, 2, 2], 1),
        Err(ProjectiveError::Overflow)
    );
    assert_eq!(
        complex_projective_chord_squared(&[0., 0.], &[1., 0.]),
        Err(ProjectiveError::ZeroState)
    );
    assert_eq!(
        complex_projective_chord_squared(&[f64::NAN, 0.], &[1., 0.]),
        Err(ProjectiveError::NonFinite)
    );
    let s = ProjectiveBiasSpec::new([1, 2, 2], 1).unwrap();
    assert!(matches!(
        ProjectiveBiasForward::new(s, &[0.; 3], &[0.]),
        Err(ProjectiveError::Length)
    ));
    assert!(matches!(
        ProjectiveBiasForward::new(s, &[0.; 4], &[f32::INFINITY]),
        Err(ProjectiveError::NonFinite)
    ));
    assert!(matches!(
        ProjectiveBiasForward::new(s, &[0., 0., f32::NAN, 0.], &[0.]),
        Err(ProjectiveError::NonFinite)
    ));
    let tape = ProjectiveBiasForward::new(s, &[0.; 4], &[0.]).unwrap();
    assert!(matches!(
        tape.backward(&[0., f32::NAN, 0., 0.]),
        Err(ProjectiveError::NonFinite)
    ));
    assert!(matches!(
        tape.backward(&[0.; 3]),
        Err(ProjectiveError::Length)
    ));
}
