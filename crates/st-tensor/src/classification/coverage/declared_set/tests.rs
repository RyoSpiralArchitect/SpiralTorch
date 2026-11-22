use super::*;

fn decl(position: usize, target: usize, rivals: &[usize]) -> DeclaredRivalRow {
    DeclaredRivalRow {
        position,
        target,
        rivals: rivals.to_vec(),
    }
}

#[test]
fn conditional_and_mass_vjps_match_independent_probabilities_and_finite_differences() {
    let values = [0.125, -0.75, 1.0, 0.5, 9., 8., 3., 2., -1., 1., 1.5, -0.5];
    let labels = [2, -100, 1];
    let plan = [decl(0, 2, &[0, 3])];
    let independent = |v: &[f32]| {
        let p: Vec<f64> = v.iter().map(|&x| f64::from(x).exp()).collect();
        let inside = p[0] + p[2] + p[3];
        let within = ((inside / p[2]).ln() + (p[8..].iter().sum::<f64>() / p[9]).ln()) / 2.;
        let coverage = (p[..4].iter().sum::<f64>() / inside).ln() / 2.;
        (within, coverage)
    };
    for layout in [Layout::RowMajor, Layout::ColMajor] {
        let x = Tensor::from_vec(3, 4, values.to_vec())
            .unwrap()
            .to_layout(layout)
            .unwrap();
        let (s, mass, choice) = x
            .declared_set_nll_capture_within(&labels, &plan, |g| {
                Ok::<_, TensorError>(g.data().to_vec())
            })
            .unwrap();
        assert_eq!(s, x.declared_set_nll(&labels, &plan).unwrap());
        assert_eq!((s.target_tokens, s.contrasted_tokens), (2, 1));
        let expected = independent(&values);
        assert!(
            (s.within_nll - expected.0).abs() < 1e-14
                && (s.coverage_nll - expected.1).abs() < 1e-14
        );
        assert_eq!(
            s.full_nll as f32,
            x.cross_entropy_with_logits(&labels, CrossEntropyConfig::default())
                .unwrap()
                .data()[0]
        );
        let (_, reference) = x
            .declared_rival_cross_entropy_mean_gradient(&labels, &plan, 1.)
            .unwrap();
        assert_eq!(choice, reference.data());
        for i in 0..values.len() {
            let (mut a, mut b) = (values, values);
            a[i] += 1. / 1024.;
            b[i] -= 1. / 1024.;
            let (hi, lo) = (independent(&a), independent(&b));
            let dx = f64::from(a[i] - b[i]);
            assert!(((hi.0 - lo.0) / dx - f64::from(choice[i])).abs() < 6e-8);
            assert!(((hi.1 - lo.1) / dx - f64::from(mass.data()[i])).abs() < 6e-8);
        }
    }
}

#[test]
fn mass_vjp_survives_rounding_that_hides_it_in_full_minus_choice() {
    let x = Tensor::from_vec(1, 3, vec![0., 0., -40.]).unwrap();
    let p = [decl(0, 0, &[1])];
    let (s, mass, choice) = x
        .declared_set_nll_capture_within(&[0], &p, |g| Ok::<_, TensorError>(g.data().to_vec()))
        .unwrap();
    let (_, full) = x
        .declared_rival_cross_entropy_mean_gradient(&[0], &p, 0.)
        .unwrap();
    assert_eq!(&full.data()[..2], &choice[..2]);
    assert_eq!(s.full_nll - s.within_nll, 0.);
    let outside = (-40f64).exp() / 2.;
    assert!((s.coverage_nll / outside - 1.).abs() < 1e-14);
    assert_eq!(
        mass.data(),
        &[
            (-outside / 2.) as f32,
            (-outside / 2.) as f32,
            outside as f32
        ]
    );
    let unlikely = Tensor::from_vec(1, 3, vec![-1000., -1000., 1000.]).unwrap();
    assert!(
        (unlikely.declared_set_nll(&[0], &p).unwrap().coverage_nll - (2000. - 2f64.ln())).abs()
            < 1e-12
    );
}

#[test]
fn full_vocabulary_and_undeclared_rows_have_exactly_zero_mass_cost_and_gradient() {
    let x = Tensor::from_vec(2, 3, vec![32., 0., -8., 1., 2., 3.]).unwrap();
    for plan in [vec![], vec![decl(0, 0, &[1, 2])]] {
        let (s, mass, choice) = x
            .declared_set_nll_capture_within(&[0, 2], &plan, |g| {
                Ok::<_, TensorError>(g.data().to_vec())
            })
            .unwrap();
        let (_, full) = x
            .declared_rival_cross_entropy_mean_gradient(&[0, 2], &[], 0.)
            .unwrap();
        assert_eq!(choice, full.data());
        assert_eq!(mass.data(), &[0.; 6]);
        assert_eq!(s.coverage_nll, 0.);
        assert_eq!(s.within_nll, s.full_nll);
    }
}

#[test]
fn reject_bad_membership_before_projection_and_propagate_projection_failure() {
    let x = Tensor::from_vec(2, 3, vec![0.; 6]).unwrap();
    for plan in [
        vec![decl(0, 1, &[2])],
        vec![decl(1, 2, &[0])],
        vec![decl(0, 0, &[2, 1])],
    ] {
        assert!(x
            .declared_set_nll_capture_within(&[0, -100], &plan, |_| -> PureResult<()> {
                panic!("projected invalid inputs")
            })
            .is_err());
    }
    assert!(x.declared_set_nll(&[-100, -100], &[]).is_err());
    let failure = x
        .declared_set_nll_capture_within(&[0, 1], &[], |_| {
            Err::<(), _>(TensorError::InvalidValue {
                label: "projection failed",
            })
        })
        .unwrap_err();
    assert!(matches!(
        failure,
        TensorError::InvalidValue {
            label: "projection failed"
        }
    ));
}
