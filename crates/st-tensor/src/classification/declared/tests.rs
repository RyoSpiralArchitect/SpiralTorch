use super::*;
fn decl(p: usize, t: usize, ids: &[usize]) -> DeclaredRivalRow {
    DeclaredRivalRow {
        position: p,
        target: t,
        rivals: ids.to_vec(),
    }
}

#[test]
fn fixed_subset_keeps_unpaired_ce_and_matches_independent_derivatives() {
    let values = vec![0.1, 3., -0.6, -1.2, 2., 1., 0., -2., 0.1, 0.4, -0.5, 1.2];
    let labels = [0, -100, 2];
    let recipe = [decl(0, 0, &[2, 3])];
    fn objective(v: &[f32], mix: f64) -> f64 {
        let cost = |i: usize, target: usize, rivals: &[usize]| {
            rivals
                .iter()
                .map(|&r| (f64::from(v[i + r]) - f64::from(v[i + target])).exp())
                .sum::<f64>()
                .ln_1p()
        };
        ((1. - mix) * cost(0, 0, &[1, 2, 3]) + mix * cost(0, 0, &[2, 3]) + cost(8, 2, &[0, 1, 3]))
            / 2.
    }
    for mix in [0., 0.5, 1.] {
        let x = Tensor::from_vec(3, 4, values.clone())
            .unwrap()
            .to_layout(Layout::ColMajor)
            .unwrap();
        let (r, g) = x
            .declared_rival_cross_entropy_mean_gradient(&labels, &recipe, mix)
            .unwrap();
        assert_eq!(
            (r.contrasted_tokens, r.rival_ids, r.target_tokens),
            (1, 2, 2)
        );
        assert!((f64::from(r.loss) - objective(&values, mix)).abs() < 2e-7);
        if mix == 1. {
            assert_eq!(g.data()[1], 0.);
        }
        for (i, &analytic) in g.data().iter().enumerate() {
            let (mut lo, mut hi) = (values.clone(), values.clone());
            lo[i] -= 0.001;
            hi[i] += 0.001;
            let fd =
                (objective(&hi, mix) - objective(&lo, mix)) / (f64::from(hi[i]) - f64::from(lo[i]));
            assert!((fd - f64::from(analytic)).abs() < 2e-6);
        }
    }
}

#[test]
fn aliases_empty_or_full_sets_to_ce_and_retains_tiny_pair_loss() {
    let x = Tensor::from_vec(1, 3, vec![0., -40., 1e20]).unwrap();
    let labels = [0];
    let (tiny, g) = x
        .declared_rival_cross_entropy_mean_gradient(&labels, &[decl(0, 0, &[1])], 1.)
        .unwrap();
    assert!(
        tiny.loss > 0.
            && tiny.loss < 1e-16
            && g.data()[0] < 0.
            && g.data()[1] == -g.data()[0]
            && g.data()[2] == 0.
    );
    let ce = CrossEntropyConfig::default();
    let loss = x.cross_entropy_with_logits(&labels, ce).unwrap().data()[0];
    let grad = x
        .cross_entropy_with_logits_backward(&labels, ce, &Tensor::from_vec(1, 1, vec![1.]).unwrap())
        .unwrap();
    for (plan, mix) in [
        (vec![], 1.),
        (vec![decl(0, 0, &[1])], 0.),
        (vec![decl(0, 0, &[1, 2])], 0.7),
    ] {
        let (r, g) = x
            .declared_rival_cross_entropy_mean_gradient(&labels, &plan, mix)
            .unwrap();
        assert_eq!((r.loss, g.data()), (loss, grad.data()));
        assert_eq!(r.declared_ce.is_none(), plan.is_empty());
    }
}

#[test]
fn rejects_bad_positions_labels_classes_and_budget_even_at_zero_mix() {
    let x = Tensor::from_vec(2, 4, vec![0.; 8]).unwrap();
    for plan in [
        vec![decl(2, 0, &[1])],
        vec![decl(1, 0, &[1])],
        vec![decl(0, 1, &[2])],
        vec![decl(0, 0, &[])],
        vec![decl(0, 0, &[0])],
        vec![decl(0, 0, &[2, 1])],
        vec![decl(0, 0, &[1, 1])],
        vec![decl(0, 0, &[4])],
        vec![decl(0, 0, &[1]), decl(0, 0, &[2])],
    ] {
        assert!(x
            .declared_rival_cross_entropy_mean_gradient(&[0, -100], &plan, 0.)
            .is_err());
    }
    assert!(x
        .declared_rival_cross_entropy_mean_gradient(&[0, 1], &[], f64::NAN)
        .is_err());
    let plan = (0..2049)
        .map(|i| decl(i, 32, &(0..32).collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    assert!(DeclaredRivalRow::validate_plan(&plan, 2049, 33, 0.5).is_err());
}
