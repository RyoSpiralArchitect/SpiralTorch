use super::*;

fn menu(lines: &[Vec<usize>], gold: usize, mix: f64) -> LineMenu {
    LineMenu::new(LineSet::new(lines, 0), lines, gold, mix)
}

#[test]
fn huge_shared_cost_cancels_before_ranking_and_before_backprop() {
    let lines = [vec![1, 2], vec![1, 3]];
    let z = Tensor::from_vec(
        4,
        4,
        [
            [0., -1e20, 0., 0.],
            [0., -1., 1.1, 0.2],
            [2., 0., 0., 0.],
            [0.4, -0.1, 0., 0.],
        ]
        .concat(),
    )
    .unwrap();
    let (r, g) = menu(&lines, 0, 1.).loss_gradient(&z).unwrap();
    assert_eq!(r.line_nlls[0], r.line_nlls[1]);
    let gap = r.suffix_nlls[0] - r.suffix_nlls[1];
    assert!(gap < -1. && (r.menu_nll - gap.exp().ln_1p()).abs() < 1e-14);
    assert!(g.data()[..4].iter().all(|v| *v == 0.));
    assert_eq!(g.data()[4], 0.);
    assert!(g.data()[6] < -0.01 && g.data()[7] > 0.01);
    let (_, mixed) = menu(&lines, 0, 0.5).loss_gradient(&z).unwrap();
    assert!(mixed.data()[..4].iter().any(|v| v.abs() > 0.1));
}

#[test]
fn very_confident_gold_keeps_the_small_rival_and_a_sparse_fork_seed() {
    let z = Tensor::from_vec(3, 3, [[0., 20., -20.], [0.; 3], [0.; 3]].concat()).unwrap();
    let (r, g) = menu(&[vec![1], vec![2]], 0, 1.).loss_gradient(&z).unwrap();
    let tiny = (-40f64).exp();
    assert_eq!(r.menu_probabilities[0], 1.);
    assert!((r.rival_mass / tiny - 1.).abs() < 1e-12 && (r.menu_nll / tiny - 1.).abs() < 1e-12);
    assert_eq!(g.data()[0], 0.);
    assert!((f64::from(g.data()[2]) / tiny - 1.).abs() < 1e-6);
    assert_eq!(g.data()[1], -g.data()[2]);
    assert!(g.data()[3] < 0. && g.data()[6] > 0.);
}

#[test]
fn earlier_rare_fork_survives_beside_two_order_one_opposing_seeds() {
    // At prefix [1], gold and a common rival still compete, but [2] has
    // already left. Its posterior is ~2*exp(-40), not 1 - rounded_reach.
    let lines = [vec![1, 1], vec![1, 2], vec![2]];
    let z = Tensor::from_vec(
        5,
        4,
        [[-80., 0., -40., -80.], [0.; 4], [0.; 4], [0.; 4], [0.; 4]].concat(),
    )
    .unwrap();
    let (r, g) = menu(&lines, 0, 1.).loss_gradient(&z).unwrap();
    let tail = 2. * (-40f64).exp();
    assert!((r.menu_probabilities[2] / tail - 1.).abs() < 1e-12);
    assert!((f64::from(g.data()[7]) / (tail * 0.25) - 1.).abs() < 1e-6);
    assert_eq!(g.data()[4], g.data()[7]);
    assert_eq!(&g.data()[5..7], &[-0.5, 0.5]);
    assert_eq!(g.data()[0], 0.);
    // The previous two expanded f32 CE seeds round to opposite halves.
    let split = Tensor::from_vec(2, 4, vec![0.; 8])
        .unwrap()
        .cross_entropy_with_logits_backward(
            &[1, 2],
            CrossEntropyConfig {
                reduction: st_tensor::classification::LossReduction::None,
                ..Default::default()
            },
            &Tensor::from_vec(
                2,
                1,
                vec![r.rival_mass as f32, -(r.menu_probabilities[1] as f32)],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(split.data()[3] + split.data()[7], 0.);
}

#[test]
fn enormous_unequal_stop_normalizers_do_not_turn_into_a_tie() {
    let z = Tensor::from_vec(
        3,
        3,
        [[-100., 0., 0.], [-1e20, 0., 0.], [-1e20, 0., -100.]].concat(),
    )
    .unwrap();
    let (r, g) = menu(&[vec![1], vec![2]], 0, 1.)
        .loss_gradient(&z.to_layout(st_tensor::Layout::ColMajor).unwrap())
        .unwrap();
    assert_eq!(r.line_nlls[0], r.line_nlls[1]);
    assert_eq!(r.winner_index, 1);
    assert!((r.menu_nll - 3f64.ln()).abs() < 1e-13);
    assert!((f64::from(g.data()[2]) - 2. / 3.).abs() < 1e-6);
    let tie = Tensor::from_vec(3, 3, vec![0.; 9]).unwrap();
    assert_eq!(
        menu(&[vec![1], vec![2]], 1, 1.)
            .score(&tie)
            .unwrap()
            .winner_index,
        0
    );
}

#[test]
fn underflowed_gold_probability_still_has_a_finite_loss_and_suffix_pullback() {
    let z = Tensor::from_vec(3, 3, [[0., -1200., 0.], [0.; 3], [0.; 3]].concat()).unwrap();
    let (r, g) = menu(&[vec![1], vec![2]], 0, 1.).loss_gradient(&z).unwrap();
    assert_eq!(r.menu_probabilities[0], 0.);
    assert!((r.menu_nll - 1200.).abs() < 1e-10);
    assert_eq!(&g.data()[..3], &[0., -1., 1.]);
    assert!((g.data()[3] + 2. / 3.).abs() < 1e-6);
}

#[test]
fn signed_seed_matches_independent_differences_at_all_logits_including_interior_stops() {
    let lines = [vec![1], vec![1, 2], vec![1, 2, 3], vec![1, 3], vec![2, 3]];
    let prefixes = [
        vec![],
        vec![1],
        vec![1, 2],
        vec![1, 2, 3],
        vec![1, 3],
        vec![2],
        vec![2, 3],
    ];
    let values: Vec<f32> = (0..28)
        .map(|i| ((i * 13 % 29) as f32 - 14.) / 16.)
        .collect();
    let logits = Tensor::from_vec(7, 4, values.clone()).unwrap();
    for gold in 0..lines.len() {
        for mix in [0., 0.37, 1.] {
            let (_, grad) = menu(&lines, gold, mix).loss_gradient(&logits).unwrap();
            let oracle = |cell: usize, delta: f64| {
                let mut z: Vec<_> = values.iter().copied().map(f64::from).collect();
                z[cell] += delta;
                let costs: Vec<_> = lines
                    .iter()
                    .map(|line| {
                        let mut nll = 0.;
                        for (depth, &k) in line.iter().chain([&0]).enumerate() {
                            let i = prefixes.iter().position(|p| p == &line[..depth]).unwrap();
                            let row = &z[i * 4..i * 4 + 4];
                            nll += row.iter().map(|v| v.exp()).sum::<f64>().ln() - row[k];
                        }
                        nll
                    })
                    .collect();
                (1. - mix) * costs[gold] / (lines[gold].len() + 1) as f64
                    + mix * (costs[gold] + costs.iter().map(|v| (-v).exp()).sum::<f64>().ln())
            };
            for (i, &v) in grad.data().iter().enumerate() {
                let numeric = (oracle(i, 1e-5) - oracle(i, -1e-5)) / 2e-5;
                assert!(
                    (f64::from(v) - numeric).abs() < 8e-8,
                    "{gold} {mix} {i}: {v} vs {numeric}"
                );
            }
        }
    }
}
