use super::*;

#[test]
fn enormous_sibling_costs_do_not_erase_the_ratio_of_their_tails() {
    // Both stops cost about 1e20, but the first has two equally likely rivals;
    // the second has one. Their posterior reach is 1/3 vs 2/3, not 1/2 each.
    let root = [-100., 0., 0.];
    let z = Tensor::from_vec(3, 3, [root, [-1e20, 0., 0.], [-1e20, 0., -100.]].concat()).unwrap();
    let (r, g) = LineSet::new(&[vec![1], vec![2]], 0)
        .loss_gradient(&z)
        .unwrap();
    assert_eq!(r.set_nll, f64::from(1e20f32));
    assert!((g.data()[1] - 1. / 6.).abs() < 1e-6);
    assert!((g.data()[2] + 1. / 6.).abs() < 1e-6);
    assert!((g.data()[3] + 1. / 3.).abs() < 1e-6);
    assert!((g.data()[6] + 2. / 3.).abs() < 1e-6);
}

#[test]
fn almost_complete_set_retains_rejection_probability_and_pullback() {
    let x = (-40f64).exp();
    let root = [-20., 20., 20.];
    let stop = [20., -20., -20.];
    let z = Tensor::from_vec(3, 3, [root, stop, stop].concat()).unwrap();
    let plan = LineSet::new(&[vec![1], vec![2]], 0);
    let (r, g) = plan
        .loss_gradient(&z.to_layout(st_tensor::Layout::ColMajor).unwrap())
        .unwrap();
    let nll = (x / 2.).ln_1p() + (2. * x).ln_1p();
    assert!((r.set_nll / nll - 1.).abs() < 1e-14);
    assert_eq!(r.prefix_nodes, 3);
    // One canonical row per prefix. The two branches split posterior reach equally.
    let missed = x / (2. + x);
    let p = x / (1. + 2. * x);
    for (&a, b) in g.data().iter().zip([
        missed,
        -missed / 2.,
        -missed / 2.,
        -p,
        p / 2.,
        p / 2.,
        -p,
        p / 2.,
        p / 2.,
    ]) {
        if b == 0. {
            assert_eq!(a, 0.);
        } else {
            assert!((f64::from(a) / b - 1.).abs() < 1e-6);
        }
    }
}

#[test]
fn child_success_difference_survives_near_one_and_large_equal_head_logits() {
    let root = [0., 1e20, 1e20];
    let z = Tensor::from_vec(3, 3, [root, [20., -20., -20.], [22., -22., -22.]].concat()).unwrap();
    let (r, g) = LineSet::new(&[vec![1], vec![2]], 0)
        .loss_gradient(&z)
        .unwrap();
    let (a, b) = ((-40f64).exp(), (-44f64).exp());
    assert!((r.set_nll / (a + b) - 1.).abs() < 1e-13);
    assert!((f64::from(g.data()[1]) / ((a - b) / 2.) - 1.).abs() < 1e-6);
    assert_eq!(g.data()[1], -g.data()[2]);
}

#[test]
fn rare_shared_prefix_is_scored_even_when_all_line_probabilities_underflow() {
    let root = [0., -1200., 0.];
    let branch = [0., 0., 0.];
    let z = Tensor::from_vec(3, 3, [root, branch, branch].concat()).unwrap();
    let (r, g) = LineSet::new(&[vec![1], vec![1, 2]], 0)
        .loss_gradient(&z)
        .unwrap();
    // root -> 1, then either stop or 2 -> stop.
    assert!((r.set_nll - (1200. + 2f64.ln() - (4f64 / 9.).ln())).abs() < 1e-11);
    assert_eq!(r.target_counts, [2, 3]);
    assert_eq!(&g.data()[..3], &[0.5, -1., 0.5]);
    assert_eq!(r.decoder_lanes, 1);
    assert_eq!(g.shape(), (3, 3));
    assert!((g.data()[6] + 1. / 6.).abs() < 1e-6);
}

#[test]
fn compact_rows_follow_the_decoder_and_keep_interior_stops_for_every_subset() {
    use std::collections::BTreeSet;
    let all = [
        vec![1],
        vec![1, 1],
        vec![1, 1, 1],
        vec![1, 1, 2],
        vec![1, 2],
        vec![1, 2, 2],
        vec![2],
        vec![2, 1],
        vec![2, 1, 2],
    ];
    for mask in 1..(1 << all.len()) {
        let lines: Vec<_> = all
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, l)| l.clone())
            .collect();
        let set = LineSet::new(&lines, 0);
        let leaves: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                !lines
                    .iter()
                    .any(|other| other.len() > l.len() && other.starts_with(l))
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(set.leaves, leaves);
        let mut expected = BTreeSet::new();
        for line in &lines {
            for n in 0..=line.len() {
                expected.insert(&line[..n]);
            }
        }
        let key = |node: &Node| &lines[leaves[node.lane]][..node.depth];
        assert_eq!(expected.len(), set.nodes.len());
        assert_eq!(set.nodes.iter().map(key).collect::<BTreeSet<_>>(), expected);
        assert!(set
            .nodes
            .windows(2)
            .all(|w| (w[0].lane, w[0].depth) < (w[1].lane, w[1].depth)));
        for node in &set.nodes {
            let p = key(node);
            let mut actual = BTreeSet::new();
            for (&token, &child) in &node.edges {
                if let Some(i) = child {
                    assert_eq!(key(&set.nodes[i]), [p, &[token]].concat());
                }
                actual.insert(token);
            }
            assert_eq!(
                actual,
                lines
                    .iter()
                    .filter(|s| s.starts_with(p))
                    .map(|s| s.get(p.len()).copied().unwrap_or(0))
                    .collect()
            );
            assert_eq!(
                node.edges.get(&0),
                lines.iter().any(|s| s == p).then_some(&None)
            );
        }
    }
}
