use super::*;

fn objective(data: &[f32], labels: &[i64], k: usize, mix: f64) -> f64 {
    let mut losses = Vec::new();
    for (row, &label) in data.chunks_exact(5).zip(labels) {
        if label == -100 {
            continue;
        }
        let t = label as usize;
        let mut other: Vec<_> = (0..row.len()).filter(|&i| i != t).collect();
        other.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        let loss = |n: usize| {
            other[..n]
                .iter()
                .map(|&j| (f64::from(row[j]) - f64::from(row[t])).exp())
                .sum::<f64>()
                .ln_1p()
        };
        losses.push((1. - mix) * loss(4) + mix * loss(k));
    }
    losses.iter().sum::<f64>() / losses.len() as f64
}

#[test]
fn rivals_match_independent_masked_loss_and_finite_differences() {
    let data = vec![
        1.2, -0.6, 0.2, 0.7, -1.0, 0., 1., 2., 3., 4., -0.5, 0.25, 0.8, 1.3, -1.1,
    ];
    let labels = [2, -100, 3];
    let x = Tensor::from_vec(3, 5, data.clone()).unwrap();
    let (r, grad) = x
        .rival_cross_entropy_mean_gradient(
            &labels,
            RivalCrossEntropy {
                rivals: 2,
                mix: 0.4,
            },
        )
        .unwrap();
    assert_eq!((r.target_tokens, r.top1_correct), (2, 1));
    assert_eq!(r.mean_target_rank, 2.);
    assert!((f64::from(r.loss) - objective(&data, &labels, 2, 0.4)).abs() < 2e-7);
    assert!((r.rival_ce - objective(&data, &labels, 2, 1.)).abs() < 1e-12);
    assert_eq!(&grad.data()[5..10], &[0.; 5]);
    for (i, &v) in grad.data().iter().enumerate() {
        let mut lo = data.clone();
        let mut hi = data.clone();
        lo[i] -= 0.001;
        hi[i] += 0.001;
        let fd = (objective(&hi, &labels, 2, 0.4) - objective(&lo, &labels, 2, 0.4))
            / (f64::from(hi[i]) - f64::from(lo[i]));
        assert!((fd - f64::from(v)).abs() < 2e-6, "{i}: {fd} != {v}");
    }
    for (k, mix) in [(1, 0.), (4, 0.37), (4, 1.)] {
        let (r, g) = x
            .rival_cross_entropy_mean_gradient(&labels, RivalCrossEntropy { rivals: k, mix })
            .unwrap();
        let ce = CrossEntropyConfig::default();
        assert_eq!(
            r.loss,
            x.cross_entropy_with_logits(&labels, ce).unwrap().data()[0]
        );
        assert_eq!(
            g.data(),
            x.cross_entropy_with_logits_backward(
                &labels,
                ce,
                &Tensor::from_vec(1, 1, vec![1.]).unwrap()
            )
            .unwrap()
            .data()
        );
    }
}

#[test]
fn rival_pool_respects_layout_and_common_logit_shifts() {
    let data = [3., -1., 4., 2., -3., 1., 2., 0.];
    let logits = Tensor::from_vec(2, 4, data.to_vec()).unwrap();
    let shifted = Tensor::from_vec(2, 4, data.iter().map(|v| v + 256.).collect()).unwrap();
    let recipe = RivalCrossEntropy {
        rivals: 2,
        mix: 0.5,
    };
    let (expected, grad) = logits
        .rival_cross_entropy_mean_gradient(&[0, 3], recipe)
        .unwrap();
    for x in [logits.to_layout(Layout::ColMajor).unwrap(), shifted] {
        let (r, g) = x
            .rival_cross_entropy_mean_gradient(&[0, 3], recipe)
            .unwrap();
        assert_eq!(
            (r.loss, r.rival_ce, r.mean_target_minus_best_rival),
            (
                expected.loss,
                expected.rival_ce,
                expected.mean_target_minus_best_rival
            )
        );
        assert_eq!(g.data(), grad.data());
    }
    assert!(Tensor::from_vec(1, 3, vec![0., f32::NAN, 1.])
        .unwrap()
        .rival_cross_entropy_mean_gradient(&[0], recipe)
        .is_err());
}

#[test]
fn tied_rivals_are_stable_and_confident_target_keeps_a_small_gradient() {
    let x = Tensor::from_vec(1, 4, vec![-0., 0., 0., -2.]).unwrap();
    let recipe = RivalCrossEntropy { rivals: 1, mix: 1. };
    let (r, g) = x.rival_cross_entropy_mean_gradient(&[2], recipe).unwrap();
    assert_eq!(g.data(), &[0.5, 0., -0.5, 0.]);
    assert_eq!(
        (
            r.mean_target_minus_best_rival,
            r.mean_target_rank,
            r.top1_correct
        ),
        (0., 3., 0)
    );
    let x = Tensor::from_vec(1, 3, vec![0., -25., -35.]).unwrap();
    let (r, g) = x.rival_cross_entropy_mean_gradient(&[0], recipe).unwrap();
    assert!(r.loss > 0. && g.data()[0] < 0. && g.data()[0] == -g.data()[1] && g.data()[2] == 0.);
    for (n, mix) in [(0, 0.), (3, 0.5), (2, f64::NAN), (2, -0.1), (2, 1.01)] {
        assert!(x
            .rival_cross_entropy_mean_gradient(&[0], RivalCrossEntropy { rivals: n, mix })
            .is_err());
    }
    assert!(RivalCrossEntropy {
        rivals: 33,
        mix: 0.5
    }
    .validate(100)
    .is_err());
    assert!(x
        .rival_cross_entropy_mean_gradient(&[-100], recipe)
        .is_err());
    assert!(x.rival_cross_entropy_mean_gradient(&[3], recipe).is_err());
}
