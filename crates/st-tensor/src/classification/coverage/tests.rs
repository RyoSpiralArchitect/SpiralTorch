use super::*;

fn row(v: &[f32]) -> Tensor {
    Tensor::from_vec(1, v.len(), v.to_vec()).unwrap()
}
fn guard() -> RivalCoverageGuard {
    RivalCoverageGuard::new(0.5).unwrap()
}
fn recipe() -> RivalCrossEntropy {
    RivalCrossEntropy {
        rivals: 1,
        mix: 0.5,
    }
}
fn logsum(v: &[f64]) -> f64 {
    let peak = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    peak + v.iter().map(|x| (x - peak).exp()).sum::<f64>().ln()
}

#[test]
fn guarded_gradient_matches_independent_differences_with_changed_top_rival() {
    // Fixed reference set {0,1}; the candidate's own strongest rival is 2.
    let reference = [1., 2., 0., -0.5];
    let candidate = [0.2f32, -1., 1.8, 0.7];
    let ref_data = row(&reference).rival_coverage_reference(&[0], 1).unwrap();
    let objective = |v: &[f64]| {
        let r = reference.map(f64::from);
        let old = logsum(&r) - logsum(&r[..2]);
        let new = logsum(v) - logsum(&v[..2]);
        let best = v[1..].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (logsum(v) - v[0]) + 0.5 * (logsum(&[v[0], best]) - v[0]) + 0.5 * (new - old).max(0.)
    };
    let (r, g) = row(&candidate)
        .guarded_rival_cross_entropy_mean_gradient(&[0], recipe(), &ref_data, guard())
        .unwrap();
    let q = r.coverage_guard.unwrap();
    let v = candidate.map(f64::from);
    assert_eq!(q.active_targets, 1);
    assert!(q.mean_violation > 0.);
    assert!((f64::from(r.loss) - objective(&v)).abs() < 2e-6);
    for (i, &actual) in g.data().iter().enumerate() {
        let mut lo = v;
        let mut hi = v;
        lo[i] -= 1e-5;
        hi[i] += 1e-5;
        assert!((f64::from(actual) - (objective(&hi) - objective(&lo)) / 2e-5).abs() < 1e-6);
    }
    assert!(g.data().iter().map(|&v| f64::from(v)).sum::<f64>().abs() < 1e-6);
}

#[test]
fn inactive_equality_subset_ties_whole_vocabulary_and_disabled_are_exact() {
    let r = row(&[0., 2., 2., -1.]);
    let reference = r.rival_coverage_reference(&[0], 1).unwrap();
    assert_eq!(reference.ids, [0, 1]);
    for candidate in [r.clone(), row(&[2., 3., 1., -1.])] {
        let (base, gb) = candidate
            .rival_cross_entropy_mean_gradient(&[0], recipe())
            .unwrap();
        let (v, g) = candidate
            .guarded_rival_cross_entropy_mean_gradient(&[0], recipe(), &reference, guard())
            .unwrap();
        assert_eq!(v.loss.to_bits(), base.loss.to_bits());
        assert_eq!(g.data(), gb.data());
        assert_eq!(v.coverage_guard.unwrap().active_targets, 0);
    }
    let other = row(&[0., 0., 4., 3.]);
    let (b, bg) = other
        .rival_cross_entropy_mean_gradient(&[0], recipe())
        .unwrap();
    let (v, g) = other
        .guarded_rival_cross_entropy_mean_gradient(
            &[0],
            recipe(),
            &reference,
            RivalCoverageGuard::new(0.).unwrap(),
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(b).unwrap(),
        serde_json::to_value(v).unwrap()
    );
    assert_eq!(g.data(), bg.data());
    let all = r.rival_coverage_reference(&[0], 3).unwrap();
    let (v, _) = other
        .guarded_rival_cross_entropy_mean_gradient(
            &[0],
            RivalCrossEntropy {
                rivals: 3,
                mix: 0.5,
            },
            &all,
            guard(),
        )
        .unwrap();
    assert_eq!(v.coverage_guard.unwrap().mean_candidate_coverage_nll, 0.);
}

#[test]
fn averages_over_all_active_labels_and_rejects_wrong_reference_alignment() {
    let rf = Tensor::from_vec(3, 3, vec![0., 1., -3., 0., 0., 0., 0., 1., -3.]).unwrap();
    let cand = Tensor::from_vec(3, 3, vec![0., 1., 3., 1., 2., 3., 0., 1., -3.]).unwrap();
    let labels = [0, -100, 0];
    let reference = rf.rival_coverage_reference(&labels, 1).unwrap();
    let (single, sg) = row(&[0., 1., 3.])
        .guarded_rival_cross_entropy_mean_gradient(
            &[0],
            recipe(),
            &row(&[0., 1., -3.])
                .rival_coverage_reference(&[0], 1)
                .unwrap(),
            guard(),
        )
        .unwrap();
    for layout in [Layout::RowMajor, Layout::ColMajor] {
        let (r, g) = cand
            .to_layout(layout)
            .unwrap()
            .guarded_rival_cross_entropy_mean_gradient(&labels, recipe(), &reference, guard())
            .unwrap();
        assert_eq!(
            (r.target_tokens, r.coverage_guard.unwrap().active_targets),
            (2, 1)
        );
        assert_eq!(
            r.coverage_guard.unwrap().mean_violation,
            single.coverage_guard.unwrap().mean_violation / 2.
        );
        for (&a, &b) in g.data()[..3].iter().zip(sg.data()) {
            assert!((a - b / 2.).abs() < 1e-7);
        }
        assert_eq!(&g.data()[3..6], &[0.; 3]);
    }
    for bad in [[1, -100, 0], [0, 0, -100], [-100, -100, -100]] {
        assert!(cand
            .guarded_rival_cross_entropy_mean_gradient(&bad, recipe(), &reference, guard())
            .is_err());
    }
    assert!(cand
        .guarded_rival_cross_entropy_mean_gradient(
            &labels,
            RivalCrossEntropy {
                rivals: 2,
                mix: 0.5
            },
            &reference,
            guard()
        )
        .is_err());
    for s in [-1., 1.1, f64::INFINITY, f64::NAN] {
        assert!(RivalCoverageGuard::new(s).is_err());
    }
}

#[test]
fn tiny_outside_mass_is_not_rounded_to_zero_and_common_shifts_cancel() {
    let f = |shift: f32| {
        let r = row(&[shift, shift - 2., shift - 65.]);
        let v = row(&[shift, shift - 2., shift - 60.]);
        let ref_ = r.rival_coverage_reference(&[0], 1).unwrap();
        v.guarded_rival_cross_entropy_mean_gradient(&[0], recipe(), &ref_, guard())
            .unwrap()
    };
    let (a, ga) = f(0.);
    let (b, gb) = f(4096.);
    let g = a.coverage_guard.unwrap();
    assert_eq!(g.active_targets, 1);
    assert!(g.mean_violation > 0. && g.mean_violation < 1e-24);
    assert_eq!(g.mean_violation, b.coverage_guard.unwrap().mean_violation);
    assert_eq!(ga.data(), gb.data());
}
