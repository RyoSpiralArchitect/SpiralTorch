use super::*;

fn near(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tolerance, "{x} != {y}");
    }
}

#[test]
fn shape_curvature_and_nonfinite_inputs_fail_closed() {
    for shape in [[0, 1, 2], [1, 0, 2], [1, 1, 3]] {
        assert_eq!(CausalWaveSpec::new(shape, -1.), Err(CausalWaveError::Shape));
    }
    for c in [0., 1., f32::NAN, f32::NEG_INFINITY] {
        assert_eq!(
            CausalWaveSpec::new([1, 1, 2], c),
            Err(CausalWaveError::Curvature)
        );
    }
    assert_eq!(
        CausalWaveSpec::new([usize::MAX, 2, 2], -1.),
        Err(CausalWaveError::Overflow)
    );
    let s = CausalWaveSpec::new([1, 1, 2], -1.).unwrap();
    assert_eq!(s.validate_lengths(2, 1, 1, 1), Err(CausalWaveError::Length));
    assert!(matches!(
        CausalWaveForward::new(s, &[0., f32::NAN], &[0.], &[0.], &[0.; 2]),
        Err(CausalWaveError::NonFinite)
    ));
    assert!(matches!(
        CausalWaveForward::new(s, &[0.; 2], &[f32::INFINITY], &[0.], &[0.; 2]),
        Err(CausalWaveError::NonFinite)
    ));
    let f = CausalWaveForward::new(s, &[0.; 2], &[0.], &[0.], &[0.; 2]).unwrap();
    assert!(matches!(
        f.backward(&[0., f32::NAN], &[0.; 2]),
        Err(CausalWaveError::NonFinite)
    ));
    assert!(matches!(
        f.backward(&[0.; 2], &[f32::INFINITY, 0.]),
        Err(CausalWaveError::NonFinite)
    ));
}

#[test]
fn zero_state_has_the_nonsingular_chart_pullback() {
    let spec = CausalWaveSpec::new([1, 1, 2], -0.25).unwrap();
    let f = CausalWaveForward::new(spec, &[0.; 2], &[0.], &[0.], &[0.; 2]).unwrap();
    assert_eq!(f.features(), &[0.; 2]);
    let g = f.backward(&[1., -2.], &[0.; 2]).unwrap();
    near(&g.drive, &[1.9 * 0.505, -3.8 * 0.505], 1e-6);
    near(&g.initial_state, &[1.9 * 0.495, -3.8 * 0.495], 1e-6);
    assert_eq!(g.raw_decay, vec![0.]);
    assert_eq!(g.raw_phase, vec![0.]);
}

fn select(
    values: &[f32],
    batch: usize,
    steps: usize,
    cols: usize,
    start: usize,
    len: usize,
) -> Vec<f32> {
    (0..batch)
        .flat_map(|b| {
            values[(b * steps + start) * cols..(b * steps + start + len) * cols]
                .iter()
                .copied()
        })
        .collect()
}

#[test]
fn chunked_state_and_bptt_equal_whole_sequence_without_future_dependence() {
    let spec = CausalWaveSpec::new([2, 5, 4], -1.25).unwrap();
    let x: Vec<_> = (0..40)
        .map(|i| ((i * 7 % 23) as f32 - 11.) * 0.03)
        .collect();
    let seed: Vec<_> = (0..40).map(|i| ((i * 3 % 17) as f32 - 8.) * 0.02).collect();
    let init = vec![0.1; 8];
    let end = vec![-0.03; 8];
    let d = [-0.7, 1.2];
    let p = [0.2, -0.4];
    let full = CausalWaveForward::new(spec, &x, &d, &p, &init).unwrap();
    let left = CausalWaveForward::new(
        CausalWaveSpec::new([2, 2, 4], -1.25).unwrap(),
        &select(&x, 2, 5, 4, 0, 2),
        &d,
        &p,
        &init,
    )
    .unwrap();
    let right = CausalWaveForward::new(
        CausalWaveSpec::new([2, 3, 4], -1.25).unwrap(),
        &select(&x, 2, 5, 4, 2, 3),
        &d,
        &p,
        left.final_state(),
    )
    .unwrap();
    assert_eq!(left.features(), select(full.features(), 2, 5, 4, 0, 2));
    assert_eq!(right.features(), select(full.features(), 2, 5, 4, 2, 3));
    assert_eq!(right.final_state(), full.final_state());
    let whole = full.backward(&seed, &end).unwrap();
    let gr = right.backward(&select(&seed, 2, 5, 4, 2, 3), &end).unwrap();
    let gl = left
        .backward(&select(&seed, 2, 5, 4, 0, 2), &gr.initial_state)
        .unwrap();
    near(&gl.drive, &select(&whole.drive, 2, 5, 4, 0, 2), 2e-6);
    near(&gr.drive, &select(&whole.drive, 2, 5, 4, 2, 3), 2e-6);
    near(&gl.initial_state, &whole.initial_state, 2e-6);
    for (a, b, total) in [
        (&gl.raw_decay, &gr.raw_decay, &whole.raw_decay),
        (&gl.raw_phase, &gr.raw_phase, &whole.raw_phase),
    ] {
        near(
            &a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<_>>(),
            total,
            2e-6,
        );
    }
    let mut changed = x.clone();
    for b in 0..2 {
        changed[(b * 5 + 2) * 4..(b + 1) * 5 * 4].fill(0.8);
    }
    let altered = CausalWaveForward::new(spec, &changed, &d, &p, &init).unwrap();
    assert_eq!(
        select(full.features(), 2, 5, 4, 0, 2),
        select(altered.features(), 2, 5, 4, 0, 2)
    );
    assert_ne!(full.final_state(), altered.final_state());
    let mut prefix_seed = seed.clone();
    for b in 0..2 {
        prefix_seed[(b * 5 + 2) * 4..(b + 1) * 5 * 4].fill(0.);
    }
    let prefix = full.backward(&prefix_seed, &[0.; 8]).unwrap();
    assert!(select(&prefix.drive, 2, 5, 4, 2, 3)
        .iter()
        .all(|&v| v == 0.));
}

#[test]
fn every_input_and_parameter_pullback_matches_finite_differences() {
    let spec = CausalWaveSpec::new([1, 3, 4], -0.75).unwrap();
    let mut args = vec![
        vec![
            0.13, -0.2, 0.05, 0.4, -0.1, 0.3, -0.7, 0.2, 0.6, 0.1, -0.2, -0.4,
        ],
        vec![-0.4, 0.8],
        vec![0.3, -0.6],
        vec![0.2, -0.1, 0.05, 0.7],
    ];
    let seed = vec![
        0.1, -0.3, 0.2, 0.4, -0.2, 0.7, 0.15, -0.11, -0.2, 0.1, 0.25, -0.3,
    ];
    let end = vec![0.12, -0.4, 0.3, 0.2];
    let forward = CausalWaveForward::new(spec, &args[0], &args[1], &args[2], &args[3]).unwrap();
    let g = forward.backward(&seed, &end).unwrap();
    let expected = [g.drive, g.raw_decay, g.raw_phase, g.initial_state];
    let objective = |a: &[Vec<f32>]| {
        let f = CausalWaveForward::new(spec, &a[0], &a[1], &a[2], &a[3]).unwrap();
        f.features()
            .iter()
            .zip(&seed)
            .chain(f.final_state().iter().zip(&end))
            .map(|(&x, &s)| f64::from(x) * f64::from(s))
            .sum::<f64>()
    };
    for group in 0..4 {
        for i in 0..args[group].len() {
            let original = args[group][i];
            args[group][i] = original + 0.001;
            let hi = objective(&args);
            args[group][i] = original - 0.001;
            let lo = objective(&args);
            args[group][i] = original;
            let difference = ((hi - lo) / 0.002 - f64::from(expected[group][i])).abs();
            assert!(
                difference < 0.0002,
                "group {group}, index {i}: {difference}"
            );
        }
    }
}

#[test]
fn chart_handles_large_finite_states_without_squaring_them() {
    let spec = CausalWaveSpec::new([1, 1, 2], -1.).unwrap();
    let f = CausalWaveForward::new(spec, &[1e30, -1e30], &[0.], &[0.], &[0.; 2]).unwrap();
    let norm = f
        .features()
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(norm < 0.951 && norm > 0.949);
    assert!(f
        .backward(&[1., 0.], &[0.; 2])
        .unwrap()
        .drive
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn large_states_preserve_radial_chart_gradients() {
    let spec = CausalWaveSpec::new([1, 1, 4], -1.).unwrap();
    for x in [
        [20000., 0., 0., 0.],
        [20000., 20000., 0., 0.],
        [20000., 12000., -3500., 9000.],
    ] {
        let f = CausalWaveForward::new(spec, &x, &[0.; 2], &[0.; 2], &[0.; 4]).unwrap();
        // Power-of-two scaling keeps the f32 cotangent exactly parallel to s.
        let seed: Vec<_> = f.states().iter().map(|&v| v * 134_217_728.).collect();
        let g = f.backward(&seed, &[0.; 4]).unwrap();
        let norm2: f64 = f.states().iter().map(|&v| f64::from(v).powi(2)).sum();
        for (i, &s) in seed.iter().enumerate() {
            let state_vjp = f64::from(spec.radius()) * f64::from(s) / (1. + norm2).powf(1.5);
            for (actual, factor) in [(g.drive[i], 0.505f32), (g.initial_state[i], 0.495f32)] {
                let expected = state_vjp * f64::from(factor);
                assert!(
                    (f64::from(actual) - expected).abs() <= 3e-6 + 8e-5 * expected.abs(),
                    "drive {x:?}, channel {i}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn mixed_seed_exponents_and_oblique_rotation_keep_their_derivatives() {
    let s = [10000., 6000.];
    let f = CausalWaveForward::new(
        CausalWaveSpec::new([1, 1, 2], -1.).unwrap(),
        &s,
        &[0.],
        &[0.],
        &s,
    )
    .unwrap();
    let g = f.backward(&s.map(|v| v * 134_217_728.), &[0.; 2]).unwrap();
    assert!(g.raw_phase[0].abs() <= 3e-6);

    let s = [10000., 0.];
    let spec = CausalWaveSpec::new([1, 1, 2], -1e-20).unwrap();
    let f = CausalWaveForward::new(spec, &s, &[0.], &[0.], &s).unwrap();
    let seed = [1e38, 1e-10];
    let g = f.backward(&seed, &[0.; 2]).unwrap();
    let chart =
        f64::from(spec.radius()) * f64::from(seed[1]) / (1. + f64::from(s[0]).powi(2)).sqrt();
    let rho = 0.99f32 * 0.5;
    let expected_drive = chart * f64::from(1. - rho);
    let expected_phase = chart * f64::from(rho) * f64::from(s[0]) * f64::from(std::f32::consts::PI);
    assert!((f64::from(g.drive[1]) - expected_drive).abs() <= 8e-5 * expected_drive.abs());
    assert!((f64::from(g.raw_phase[0]) - expected_phase).abs() <= 8e-5 * expected_phase.abs());
    assert!(g.drive[1] > 0. && g.raw_phase[0] > 1.);
}
