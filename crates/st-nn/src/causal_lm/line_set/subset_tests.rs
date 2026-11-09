use super::*;

#[test]
fn every_proper_subset_matches_full_distribution_and_independent_finite_differences() {
    let lines = [vec![1], vec![1, 2], vec![1, 3], vec![2, 3]];
    let prefixes = [vec![], vec![1], vec![1, 2], vec![1, 3], vec![2], vec![2, 3]];
    let v: Vec<f32> = (0..24)
        .map(|i| ((i * 17 % 31) as f32 - 15.) / 13.)
        .collect();
    let z = Tensor::from_vec(6, 4, v.clone()).unwrap();
    for mask in 1..15 {
        let accept: Vec<_> = (0..4).filter(|i| mask & (1 << i) != 0).collect();
        let plan = LineSubset::new(LineSet::new(&lines, 0), &lines, &accept);
        let (r, g) = plan
            .loss_gradient(&z.to_layout(st_tensor::Layout::ColMajor).unwrap())
            .unwrap();
        let oracle = |cell: usize, delta: f64| {
            let mut x: Vec<_> = v.iter().map(|&a| f64::from(a)).collect();
            x[cell] += delta;
            let mass: Vec<_> = lines
                .iter()
                .map(|line| {
                    line.iter()
                        .chain([&0])
                        .enumerate()
                        .map(|(t, &k)| {
                            let u = prefixes.iter().position(|p| p == &line[..t]).unwrap();
                            let row = &x[4 * u..4 * u + 4];
                            row[k].exp() / row.iter().map(|w| w.exp()).sum::<f64>()
                        })
                        .product::<f64>()
                })
                .collect();
            let a = accept.iter().map(|&i| mass[i]).sum::<f64>();
            (-(a / mass.iter().sum::<f64>()).ln(), -a.ln())
        };
        let (want, abs) = oracle(0, 0.);
        assert!(
            (r.conditional_nll - want).abs() < 1e-12 && (r.accepted_set_nll - abs).abs() < 1e-12
        );
        assert!((r.conditional_nll + r.menu_set_nll - r.accepted_set_nll).abs() < 1e-13);
        if accept.len() == 1 {
            let (one, one_g) = LineMenu::new(LineSet::new(&lines, 0), &lines, accept[0], 1.)
                .loss_gradient(&z)
                .unwrap();
            assert!((one.menu_nll - want).abs() < 1e-13);
            for (a, b) in g.data().iter().zip(one_g.data()) {
                assert!((a - b).abs() < 8e-8);
            }
        }
        for (k, &grad) in g.data().iter().enumerate() {
            let num = (oracle(k, 1e-5).0 - oracle(k, -1e-5).0) / 2e-5;
            assert!(
                (f64::from(grad) - num).abs() < 8e-8,
                "mask={mask} k={k}: {grad} vs {num}"
            );
        }
    }
}

#[test]
fn two_equal_aliases_do_not_compete_and_a_sub_epsilon_negative_keeps_its_gradient() {
    let lines = [vec![1], vec![2], vec![3]];
    let z = Tensor::from_vec(
        4,
        4,
        [[0., 20., 20., -20.], [0.; 4], [0.; 4], [0.; 4]].concat(),
    )
    .unwrap();
    let (r, g) = LineSubset::new(LineSet::new(&lines, 0), &lines, &[0, 1])
        .loss_gradient(&z)
        .unwrap();
    let tail = (-40f64).exp() / 2.;
    assert!((r.conditional_nll / tail - 1.).abs() < 1e-12);
    assert!((r.rejected_probability_given_menu / tail - 1.).abs() < 1e-12);
    assert_eq!(g.data()[0], 0.);
    assert_eq!(g.data()[1], g.data()[2]);
    assert!((f64::from(g.data()[1]) / (-tail / 2.) - 1.).abs() < 1e-6);
    assert!((f64::from(g.data()[3]) / tail - 1.).abs() < 1e-6);
}

#[test]
fn balanced_cross_group_pairs_keep_a_rare_extra_branch_and_common_prefix_is_zero() {
    let lines = [
        vec![3, 1, 1],
        vec![3, 1, 2],
        vec![3, 2, 1],
        vec![3, 2, 2],
        vec![3, 2, 3],
    ];
    let fork = [-60., 0., 0., -40.];
    let z = Tensor::from_vec(
        9,
        4,
        [
            [0., 0., 0., -1200.],
            fork,
            fork,
            [0.; 4],
            [0.; 4],
            fork,
            [0.; 4],
            [0.; 4],
            [0.; 4],
        ]
        .concat(),
    )
    .unwrap();
    let (r, g) = LineSubset::new(LineSet::new(&lines, 0), &lines, &[0, 2])
        .loss_gradient(&z)
        .unwrap();
    assert!(r.accepted_set_nll > 1200. && (r.conditional_nll - 2f64.ln()).abs() < 1e-12);
    assert!(g.data()[..4].iter().all(|v| *v == 0.));
    let expected = (-40f64).exp() / 8.;
    assert!((f64::from(g.data()[5]) / (-expected) - 1.).abs() < 1e-6);
    assert_eq!(g.data()[5], -g.data()[6]);
}

#[test]
fn underflowed_accepted_mass_has_finite_loss_and_full_contrast() {
    let lines = [vec![1], vec![2], vec![3]];
    let z = Tensor::from_vec(
        4,
        4,
        [[0., -1200., -1200., 0.], [0.; 4], [0.; 4], [0.; 4]].concat(),
    )
    .unwrap();
    let (r, g) = LineSubset::new(LineSet::new(&lines, 0), &lines, &[0, 1])
        .loss_gradient(&z)
        .unwrap();
    assert!((r.conditional_nll - (1200. - 2f64.ln())).abs() < 1e-12);
    assert_eq!(&g.data()[..4], &[0., -0.5, -0.5, 1.]);
}
