use super::*;
use CausalLmProjectionBasis::{Dense, Diagonal};
use CausalLmProjectionSite::{Key, Output, Query, Value};

#[test]
fn retention_rejects_a_real_flip_and_refuses_stale_protection_state() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompts = vec![vec![4, 8, 2, 1]];
    let a = NativeCausalLmRankOneProjection::new(&m, 1, Output, &[0., 0.6, -0.8, 0.], &[0.; 4])
        .unwrap();
    let z = a.last_logits(&prompts[0]).unwrap();
    let [y, b] = if z[2] > z[7] { [2, 7] } else { [7, 2] };
    let keep = [[y, b]];
    let rival_objective = [[b, y]];
    let direct_margin = |p: &NativeCausalLmRankOneProjection| {
        p.score_answer_pairs(&prompts, &keep).unwrap()[0].gold_minus_alternative
    };
    let mut found = false;
    for length in [0.04, 0.1, 0.3, 1., 3.] {
        let cfg = CausalLmConditionalAnswerConfig {
            length,
            max_trials: 2,
            mass_inward_cosine: None,
        };
        let mut plain = a.clone();
        let mut stale = plain
            .prepare_conditional_margin_retention(&prompts, &keep)
            .unwrap();
        assert_eq!(stale.baseline_margins(), &[direct_margin(&a)]);
        let r = plain
            .step_conditional_answers(&prompts, &rival_objective, cfg)
            .unwrap();
        if !r.committed || direct_margin(&plain) > 0. {
            continue;
        }
        let plain_owner = plain.projection_id();
        assert!(plain
            .step_conditional_answers_retaining(&prompts, &rival_objective, cfg, &mut stale)
            .is_err());
        assert_eq!(plain.projection_id(), plain_owner);
        let mut guarded = a.clone();
        let mut deck = guarded
            .prepare_conditional_margin_retention(&prompts, &keep)
            .unwrap();
        let r = guarded
            .step_conditional_answers_retaining(&prompts, &rival_objective, cfg, &mut deck)
            .unwrap();
        assert_eq!(r.retention.len(), r.step.trials.len());
        assert!(r
            .retention
            .iter()
            .flatten()
            .any(|c| c.checked == 1 && c.nonpositive == 1));
        assert!(direct_margin(&guarded) > 0.);
        assert_ne!(guarded.projection_id(), plain_owner);
        guarded
            .step_conditional_answers_retaining(&prompts, &rival_objective, cfg, &mut deck)
            .unwrap();
        assert!(direct_margin(&guarded) > 0.);
        found = true;
        break;
    }
    assert!(found, "fixture should expose an unprotected distant flip");
}

#[test]
fn conditional_mass_guard_changes_a_conflicting_real_decoder_step() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let a = NativeCausalLmRankOneProjection::new(
        &m,
        1,
        Query,
        &[0., 0.6, -0.8, 0.],
        &[0.01, -0.02, 0.0, 0.03],
    )
    .unwrap();
    let prompts = vec![vec![4, 8, 2, 1]];
    let config = CausalLmConditionalAnswerConfig {
        length: 0.01,
        max_trials: 2,
        mass_inward_cosine: None,
    };
    for gold in 0..m.config().vocabulary {
        for rival in 0..m.config().vocabulary {
            if gold == rival {
                continue;
            }
            let pair = [[gold, rival]];
            let mut direct = a.clone();
            let d = direct
                .step_conditional_answers(&prompts, &pair, config)
                .unwrap();
            if !d.committed || d.after.candidate_mass_nll <= d.before.candidate_mass_nll {
                continue;
            }
            let mut guarded = a.clone();
            let g = guarded
                .step_conditional_answers(
                    &prompts,
                    &pair,
                    CausalLmConditionalAnswerConfig {
                        mass_inward_cosine: Some(0.05),
                        ..config
                    },
                )
                .unwrap();
            assert_eq!(d.before, g.before);
            if g.committed {
                assert!(g.projected || (g.trials.len() > 1 && !g.trials[0].accepted));
                assert!(g.after.conditional_ce < g.before.conditional_ce);
                assert!(g.after.candidate_mass_nll <= g.before.candidate_mass_nll);
                assert_ne!(guarded.projection_id(), direct.projection_id());
                return;
            }
        }
    }
    panic!("fixture did not exhibit a committed two-objective descent");
}

#[test]
fn finite_mass_floor_covers_initially_wrong_rows_and_refuses_a_sign_safe_step() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let a = NativeCausalLmRankOneProjection::new(
        &m,
        1,
        Query,
        &[0., 0.6, -0.8, 0.],
        &[0.01, -0.02, 0.0, 0.03],
    )
    .unwrap();
    let rows = vec![vec![4, 8, 2, 1], vec![8, 4, 1]];
    let z = a.last_logits(&rows[1]).unwrap();
    let wrong = if z[2] > z[7] { [7, 2] } else { [2, 7] };
    let cfg = CausalLmConditionalAnswerConfig {
        length: 0.01,
        max_trials: 1,
        mass_inward_cosine: None,
    };
    for bad in [-0.01, f64::NAN, 1.01] {
        assert!(a
            .prepare_conditional_margin_and_mass_retention(&rows, &[[2, 7], wrong], bad)
            .is_err());
    }
    let z = a.last_logits(&rows[0]).unwrap();
    for gold in 0..m.config().vocabulary {
        for rival in 0..m.config().vocabulary {
            if z[gold] <= z[rival] {
                continue;
            }
            let keep = [[gold, rival], wrong];
            let objective = &keep[..1];
            let measure = |model: &NativeCausalLmRankOneProjection| {
                model
                    .score_answer_pairs(&rows, &keep)
                    .unwrap()
                    .iter()
                    .map(|s| (-s.gold_nll).exp() + (-s.alternative_nll).exp())
                    .sum::<f64>()
                    / 2.
            };
            let mut plain = a.clone();
            let mut sign = plain
                .prepare_conditional_margin_retention(&rows, &keep)
                .unwrap();
            let ordinary = plain
                .step_conditional_answers_retaining(&rows[..1], objective, cfg, &mut sign)
                .unwrap();
            if !ordinary.step.committed || measure(&plain) >= measure(&a) - 1e-10 {
                continue;
            }
            let mut protected = a.clone();
            let mut guard = protected
                .prepare_conditional_margin_and_mass_retention(&rows, &keep, 0.)
                .unwrap();
            assert_eq!(guard.protected_count(), 1);
            let floor = guard.minimum_mean_pair_mass().unwrap();
            assert!((floor - measure(&a)).abs() < 1e-14);
            let held = protected
                .step_conditional_answers_retaining(&rows[..1], objective, cfg, &mut guard)
                .unwrap();
            let check = held.retention[0].as_ref().unwrap();
            let mass = check.mass.as_ref().unwrap();
            assert_eq!((check.checked, check.nonpositive, mass.rows), (1, 0, 2));
            assert!((mass.mean_pair_mass - measure(&plain)).abs() < 1e-14);
            assert_eq!(mass.minimum_mean_pair_mass, floor);
            assert_eq!(
                held.step.trials[0].projection_id,
                ordinary.step.trials[0].projection_id
            );
            assert!(!held.step.committed);
            assert_eq!(protected.projection_id(), a.projection_id());
            let mut allowed = protected
                .prepare_conditional_margin_and_mass_retention(
                    &rows,
                    &keep,
                    0.01 + floor - measure(&plain),
                )
                .unwrap();
            assert!(
                protected
                    .step_conditional_answers_retaining(&rows[..1], objective, cfg, &mut allowed)
                    .unwrap()
                    .step
                    .committed
            );
            assert_eq!(protected.projection_id(), plain.projection_id());
            return;
        }
    }
    panic!("fixture did not isolate a finite mass-only refusal");
}

#[test]
fn conditional_step_replays_exact_suffix_and_commits_only_verified_trials() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompts = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let pairs = [[2, 7], [7, 2], [2, 7], [2, 7]];
    let evaluate = |a: &NativeCausalLmRankOneProjection| {
        let scores = a.score_answer_pairs(&prompts, &pairs).unwrap();
        let mut c = 0.;
        let mut mass = 0.;
        for s in scores {
            let x = s.gold_minus_alternative;
            let ce = (-x).max(0.) + (-x.abs()).exp().ln_1p();
            c += ce;
            mass += s.gold_nll - ce;
        }
        [c / 4., mass / 4.]
    };
    let mut guarded_commits = 0;
    for site in [Output, Query, Key] {
        for guard in [None, Some(0.05)] {
            let mut a = NativeCausalLmRankOneProjection::new(
                &m,
                1,
                site,
                &[0., 0.6, -0.8, 0.],
                &[0.01, -0.02, 0.01, 0.03],
            )
            .unwrap();
            let full_before = evaluate(&a);
            let owner = a.projection_id();
            let cfg = CausalLmConditionalAnswerConfig {
                length: 0.005,
                max_trials: 4,
                mass_inward_cosine: guard,
            };
            let r = a.step_conditional_answers(&prompts, &pairs, cfg).unwrap();
            let full_after = evaluate(&a);
            for (point, full) in [(r.before, full_before), (r.after, full_after)] {
                assert!((point.conditional_ce - full[0]).abs() < 1e-12);
                assert!((point.candidate_mass_nll - full[1]).abs() < 1e-12);
            }
            assert!(r.trials.len() <= cfg.max_trials);
            assert_eq!(r.trial_forwards, r.trials.len() * 4);
            assert_eq!(r.reused_prefix_block_calls, r.trial_forwards);
            assert_eq!(r.projection_pullbacks, if guard.is_some() { 8 } else { 4 });
            assert_eq!(a.projection_id() != owner, r.committed);
            if r.committed {
                assert!(full_after[0] < full_before[0]);
                assert_eq!(a.projection_id(), r.trials.last().unwrap().projection_id);
                if guard.is_some() {
                    assert!(full_after[1] <= full_before[1]);
                    guarded_commits += 1;
                }
            } else {
                assert_eq!(r.before, r.after);
            }
            let prior = a.projection_id();
            assert!(a
                .step_conditional_answers(&prompts, &[[2, 7], [7, 2], [7, 7], [2, 7]], cfg)
                .is_err());
            assert!(
                !a.step_conditional_answers(
                    &prompts,
                    &pairs,
                    CausalLmConditionalAnswerConfig {
                        length: f64::MIN_POSITIVE,
                        ..cfg
                    }
                )
                .unwrap()
                .committed
            );
            assert_eq!(a.projection_id(), prior);
        }
    }
    assert!(guarded_commits > 0);
}

#[test]
fn binding_observation_splits_ce_without_changing_the_training_adjoint() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let rows = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let gold = [2, 7, 7, 2];
    let binding = CausalLmAnswerBinding {
        ids: [2, 7],
        quartets: &[[0, 1, 2, 3]],
        mix: 0.5,
    };
    for site in [Output, Query, Key, Value] {
        let a = NativeCausalLmRankOneProjection::new(
            &m,
            1,
            site,
            &[0., 0.6, -0.8, 0.],
            &[0.01, -0.02, 0.0, 0.03],
        )
        .unwrap();
        let prior = a.projection_id();
        let ordinary = a.answer_gradients(&rows, &gold, Some(binding)).unwrap();
        let obs = a.observe_binding_gradients(&rows, &gold, binding).unwrap();
        assert_eq!(obs.projection_id, prior);
        assert_eq!(a.projection_id(), prior);
        assert_eq!((obs.prompt_forwards, obs.projection_pullbacks), (4, 12));
        assert_eq!(
            serde_json::to_vec(&obs.training).unwrap(),
            serde_json::to_vec(&ordinary).unwrap()
        );
        assert!(
            (obs.candidate_mass.loss + obs.conditional.loss - ordinary.answer.loss).abs() < 1e-6
        );
        for ((&mass, &pair), &ce) in obs
            .candidate_mass
            .gradient
            .iter()
            .zip(&obs.conditional.gradient)
            .zip(&ordinary.answer.gradient)
        {
            assert!((mass + pair - ce).abs() < 1e-6);
        }
        assert!(a
            .observe_binding_gradients(&rows, &[2, 7, 3, 2], binding)
            .is_err());
        assert_eq!(a.projection_id(), prior);
    }
}

#[test]
fn binding_observation_components_match_separate_finite_differences() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompts = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let binding = CausalLmAnswerBinding {
        ids: [2, 7],
        quartets: &[[0, 1, 2, 3]],
        mix: 0.5,
    };
    let start = [0.01, -0.02, 0., 0.03];
    let eps = 0.01;
    for site in [Query, Key, Value, Output] {
        let projection = |right: &[f64]| {
            NativeCausalLmRankOneProjection::new(&m, 1, site, &[0., 0.6, -0.8, 0.], right).unwrap()
        };
        let base = projection(&start);
        for gold in [[2, 7, 7, 2], [7, 2, 2, 7]] {
            let observed = base
                .observe_binding_gradients(&prompts, &gold, binding)
                .unwrap();
            let reference = |right: &[f64]| {
                let p = projection(right);
                let mut losses = [0., 0.];
                for (ids, y) in prompts.iter().zip(gold) {
                    let z = p.last_logits(ids).unwrap();
                    let pair = binding.ids.map(|i| f64::from(z[i]).exp());
                    let full = z.iter().map(|&v| f64::from(v).exp()).sum::<f64>();
                    losses[0] += (full / (pair[0] + pair[1])).ln();
                    losses[1] += (pair[0] + pair[1]).ln() - f64::from(z[y]);
                }
                losses.map(|loss| loss / prompts.len() as f64)
            };
            let components = [&observed.candidate_mass, &observed.conditional];
            for (component, loss) in components.into_iter().zip(reference(&start)) {
                assert!((component.loss - loss).abs() < 1e-12);
            }
            for i in 0..start.len() {
                let mut plus = start;
                let mut minus = start;
                plus[i] += eps;
                minus[i] -= eps;
                let above = reference(&plus);
                let below = reference(&minus);
                for (j, component) in components.into_iter().enumerate() {
                    let finite = (above[j] - below[j]) / (2. * eps);
                    let actual = component.gradient[i];
                    assert!(
                        (finite - actual).abs() < 1e-5 + finite.abs() * 2e-3,
                        "{site:?} component {j} coordinate {i}: {finite} vs {actual}"
                    );
                }
            }
        }
    }
}

#[test]
fn paired_answer_scores_match_serving_logits_at_each_placement() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompts = vec![vec![4, 8, 2, 1], vec![8, 4, 1]];
    let pairs = [[2, 7], [7, 2]];
    for site in [Query, Key, Value, Output] {
        let a = NativeCausalLmRankOneProjection::new(
            &m,
            0,
            site,
            &[0., 0.6, -0.8, 0.],
            &[0.01, -0.02, 0.0, 0.03],
        )
        .unwrap();
        let owner = a.projection_id();
        let scores = a.score_answer_pairs(&prompts, &pairs).unwrap();
        for ((p, [y, w]), score) in prompts.iter().zip(pairs).zip(scores) {
            let z = a.last_logits(p).unwrap();
            let log_z = z.iter().map(|&v| f64::from(v).exp()).sum::<f64>().ln();
            assert!((score.gold_nll - (log_z - f64::from(z[y]))).abs() < 1e-12);
            assert!((score.alternative_nll - (log_z - f64::from(z[w]))).abs() < 1e-12);
            assert_eq!(
                score.gold_minus_alternative,
                f64::from(z[y]) - f64::from(z[w])
            );
            assert!(z.iter().all(|&v| v <= z[score.top_token]));
        }
        assert!(a.score_answer_pairs(&prompts, &[[2, 7]]).is_err());
        assert!(a.score_answer_pairs(&prompts, &[[2, 7], [7, 7]]).is_err());
        assert!(a
            .score_answer_pairs(&prompts, &[[2, 7], [7, usize::MAX]])
            .is_err());
        assert_eq!(a.projection_id(), owner);
    }
}

#[test]
fn placement_gradient_matches_full_loss_with_a_nonzero_frozen_output_lora() {
    let start = load()
        .with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward)
        .with_lora(0, 2, 2.)
        .unwrap();
    let gradient = start.lora_loss_gradients(&[1, 2, 3, 4], [1, 4]).unwrap();
    let frozen = start.step_lora_sgd(&gradient, 0.4).unwrap();
    assert_ne!(start.lora_artifact_id(), frozen.lora_artifact_id());
    let prompts = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let pairs = [[2, 7], [7, 2], [7, 2], [2, 7]];
    let gold = pairs.map(|[k, _]| k);
    let left = [0., 0.6, -0.8, 0.];
    let binding = CausalLmAnswerBinding {
        ids: [2, 7],
        quartets: &[[0, 1, 2, 3]],
        mix: 0.5,
    };
    for site in [Query, Key, Value, Output] {
        let projection = |right: [f64; 4]| {
            NativeCausalLmRankOneProjection::new(&frozen, 0, site, &left, &right).unwrap()
        };
        let objective = |right| {
            let scores = projection(right)
                .score_answer_pairs(&prompts, &pairs)
                .unwrap();
            let ce = scores.iter().map(|s| s.gold_nll).sum::<f64>() * 0.25;
            let interaction = scores.iter().map(|s| s.gold_minus_alternative).sum::<f64>() * 0.5;
            0.5 * (ce + (-interaction).exp().ln_1p())
        };
        let origin = projection([0.; 4]);
        let grad = origin
            .answer_gradients(&prompts, &gold, Some(binding))
            .unwrap()
            .combined;
        assert!(grad.norm > 1e-6);
        let eps = 0.002;
        let delta = std::array::from_fn(|i| eps * grad.gradient[i] / grad.norm);
        let finite_diff = (objective(delta) - objective(delta.map(|x| -x))) / (2. * eps);
        assert!(
            (finite_diff - grad.norm).abs() < 3e-4,
            "{site:?}: finite difference {finite_diff} vs gradient norm {}",
            grad.norm,
        );
    }
}

#[test]
fn answer_guard_falls_back_or_holds_without_publishing_rejected_weights() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let original =
        NativeCausalLmRankOneProjection::new(&m, 0, Output, &[0., 0.6, -0.8, 0.], &[0.; 4])
            .unwrap();
    let q = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let y = [9, 1, 3, 9];
    let binding = CausalLmAnswerBinding {
        ids: [2, 7],
        quartets: &[[0, 1, 2, 3]],
        mix: 0.5,
    };
    let protect = Some(CausalLmAnswerProtection {
        prompts: &q,
        gold: &y,
    });
    let cfg = CausalLmAnswerGuard {
        length: 0.08,
        max_trials: 4,
        retain_answer: 0.75,
        min_answer_drop: 0.,
        min_interaction_gain: 1e6,
        correction: CausalLmAnswerCorrection::AnchorOnly,
    };
    let point = |a: &NativeCausalLmRankOneProjection| {
        let mut ce = 0.;
        let mut gaps = vec![];
        for (p, &gold) in q.iter().zip(&y) {
            let z = a.last_logits(p).unwrap();
            let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            ce += (max - z[gold] as f64
                + z.iter()
                    .map(|&x| (f64::from(x) - max).exp())
                    .sum::<f64>()
                    .ln())
                / 4.;
            gaps.push(f64::from(z[2]) - f64::from(z[7]));
        }
        (
            ce,
            (1. + (-0.5_f64 * (gaps[0] - gaps[1] - gaps[2] + gaps[3])).exp()).ln(),
        )
    };
    let id = original.projection_id();
    let mut anchor = original.clone();
    let a = anchor
        .step_answers_guarded(&q, &y, binding, protect, cfg)
        .unwrap();
    assert_eq!(a.status, CausalLmAnswerGuardStatus::Anchor);
    assert!(a.after.answer < a.before.answer);
    for (v, measured) in [(&original, a.before), (&anchor, a.after)] {
        let (ce, rel) = point(v);
        assert!((ce - measured.answer).abs() < 1e-12 && (rel - measured.interaction).abs() < 1e-12);
        assert_eq!(Some(measured.answer), measured.protected_answer);
    }
    let cfg = CausalLmAnswerGuard {
        correction: CausalLmAnswerCorrection::Interaction,
        ..cfg
    };
    let mut parent = original.clone();
    let r = parent
        .step_answers_guarded(&q, &y, binding, protect, cfg)
        .unwrap();
    assert_eq!(
        r.trials.iter().filter(|t| t.correction).count(),
        cfg.max_trials
    );
    assert_eq!(r.objective_forwards, (1 + r.trials.len()) * 8);
    assert_eq!(r.status, CausalLmAnswerGuardStatus::Anchor);
    assert_eq!(parent.right(), anchor.right());
    assert_eq!(r.after, a.after);
    let mut held = original.clone();
    let no = held
        .step_answers_guarded(
            &q,
            &y,
            binding,
            None,
            CausalLmAnswerGuard {
                min_answer_drop: 1e6,
                ..cfg
            },
        )
        .unwrap();
    assert_eq!(no.status, CausalLmAnswerGuardStatus::Held);
    assert!(no.trials.iter().all(|t| !t.eligible));
    assert_eq!(no.before, no.after);
    assert_eq!(held.projection_id(), id);
    assert!(held
        .step_answers_guarded(
            &q,
            &y,
            binding,
            Some(CausalLmAnswerProtection {
                prompts: &q,
                gold: &[99; 4]
            }),
            cfg
        )
        .is_err());
    assert!(held
        .step_answers_guarded(
            &q,
            &y,
            binding,
            None,
            CausalLmAnswerGuard {
                retain_answer: f64::NAN,
                ..cfg
            }
        )
        .is_err());
    assert_eq!(held.projection_id(), id);
}

#[test]
fn next_answer_projection_matches_full_vocabulary_and_keeps_quartet_labels_separate() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let rows = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let labels = [9, 1, 3, 9]; // none of the answers need be either contrast ID
    let ids = [2, 7];
    let u = [0., 0.6, -0.8, 0.];
    let mut a = NativeCausalLmRankOneProjection::new(&m, 0, Value, &u, &[0.; 4]).unwrap();
    let plain = a.answer_gradients(&rows, &labels, None).unwrap();
    assert_eq!((plain.prompt_forwards, plain.projection_pullbacks), (4, 4));
    let mut reference = [0.; 4];
    let mut nll = 0.;
    for (q, &y) in rows.iter().zip(&labels) {
        let logits = a.last_logits(q).unwrap();
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let exps = logits
            .iter()
            .map(|&z| (f64::from(z) - max).exp())
            .collect::<Vec<_>>();
        let den = exps.iter().sum::<f64>();
        nll += (max - f64::from(logits[y]) + den.ln()) / 4.;
        for (k, e) in exps.into_iter().enumerate().filter(|(k, _)| *k != y) {
            let dense = m
                .projection_contrast_pullback(
                    q,
                    k,
                    y,
                    CausalLmProjectionSpec {
                        layer: 0,
                        site: Value,
                        basis: Dense,
                    },
                )
                .unwrap();
            for (j, v) in reference.iter_mut().enumerate() {
                *v += e / den / 4. * (0..4).map(|i| u[i] * dense.weight[4 * i + j]).sum::<f64>();
            }
        }
    }
    assert!((nll - plain.answer.loss).abs() < 3e-7);
    for (x, y) in reference.iter().zip(&plain.answer.gradient) {
        assert!((x - y).abs() < 1e-6);
    }
    let groups = [[0, 1, 2, 3], [0, 1, 2, 3]];
    let broken = [[0, 1, 2, 3], [1, 0, 3, 2]];
    let bind = CausalLmAnswerBinding {
        ids,
        quartets: &groups,
        mix: 0.5,
    };
    let mixed = a.answer_gradients(&rows, &labels, Some(bind)).unwrap();
    let ctrl = a
        .answer_gradients(
            &rows,
            &labels,
            Some(CausalLmAnswerBinding {
                quartets: &broken,
                ..bind
            }),
        )
        .unwrap();
    assert_eq!(mixed.answer.gradient, plain.answer.gradient);
    assert_eq!(ctrl.answer.gradient, plain.answer.gradient);
    assert_eq!((mixed.prompt_forwards, mixed.projection_pullbacks), (4, 8));
    let rel = mixed.interaction.unwrap();
    let mut original = a.clone();
    let prior = original
        .step_quartets(
            &rows,
            ids,
            &groups,
            CausalLmQuartetObjective::Interaction,
            0.08,
        )
        .unwrap();
    assert_eq!(rel.loss, prior.loss);
    assert_eq!(rel.gradient, prior.gradient);
    for j in 0..4 {
        assert_eq!(
            mixed.combined.gradient[j],
            0.5 * plain.answer.gradient[j] + 0.5 * rel.gradient[j]
        );
    }
    a.step_answers(
        &rows,
        &labels,
        Some(CausalLmAnswerBinding { mix: 1., ..bind }),
        0.08,
    )
    .unwrap();
    assert_eq!(a.right(), original.right());
    let before = a.projection_id();
    assert!(a
        .step_answers(
            &rows,
            &labels,
            Some(CausalLmAnswerBinding {
                mix: f64::NAN,
                ..bind
            }),
            0.1
        )
        .is_err());
    assert!(a
        .step_answers(&rows, &[0, 1, 2, usize::MAX], None, 0.1)
        .is_err());
    assert!(a.step_answers(&rows, &labels, None, -0.1).is_err());
    assert_eq!(a.projection_id(), before);
    let repeat = [rows[0].clone(), rows[0].clone()];
    assert!(a.step_answers(&repeat, &[2, 3], None, 0.01).is_ok());
}

#[test]
fn projection_prefix_branches_are_immutable_and_expose_head_ordered_causal_keys() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let u = [0., 0.6, -0.8, 0.];
    let a = NativeCausalLmRankOneProjection::new(&m, 1, Key, &u, &[0.; 4]).unwrap();
    let x = a.prefill(vec![4, 8, 2], 4096).unwrap();
    let y = a.prefill(vec![4, 6, 2], 4096).unwrap();
    let ids = x.token_ids().to_vec();
    let before = x.logits().to_vec();
    let mut h = m.embed(&ids, [1, 3]).unwrap();
    for layer in 0..2 {
        let block = &m.blocks[layer];
        let fused = block
            .qkv
            .forward(&block.first_norm.forward(&h, m.config.norm_epsilon).unwrap())
            .unwrap();
        for position in 0..3 {
            let offset = position * 12 + 4;
            assert_eq!(
                x.key_row(layer, position).unwrap(),
                &fused.data()[offset..offset + 4]
            );
        }
        h = m
            .run_block(layer, h, [1, 3], decoder::BlockPass::Forward)
            .unwrap()
            .output;
    }
    assert_eq!(x.key_row(0, 2).unwrap(), y.key_row(0, 2).unwrap());
    assert_ne!(x.key_row(1, 2).unwrap(), y.key_row(1, 2).unwrap());
    let z = x.branch(3).unwrap();
    let expected = a.last_logits(&[4, 8, 2, 3]).unwrap();
    assert!(z
        .logits()
        .iter()
        .zip(expected)
        .all(|(a, b)| (a - b).abs() < 5e-5));
    assert_eq!(z.key_row(1, 2).unwrap(), x.key_row(1, 2).unwrap());
    assert_eq!(x.probe_key_from(&y, 0, 2, &[3]).unwrap(), z.logits());
    assert_eq!(x.probe_key_from(&x, 1, 2, &[3]).unwrap(), z.logits());
    assert!(x.key_row(2, 0).is_err() && x.key_row(0, 3).is_err());
    assert!(x.probe_key_from(&z, 0, 0, &[3]).is_err() && x.probe_key_from(&y, 1, 1, &[3]).is_err());
    assert!(x.probe_key_from(&y, 1, 2, &[]).is_err());
    for (layer, site, right) in [(1, Query, [0.; 4]), (0, Key, [0.; 4]), (1, Key, [0.1; 4])] {
        let other = NativeCausalLmRankOneProjection::new(&m, layer, site, &u, &right)
            .unwrap()
            .prefill(ids.clone(), 4096)
            .unwrap();
        assert_ne!(other.projection_id(), x.projection_id());
        assert!(x.probe_key_from(&other, 1, 2, &[3]).is_err());
    }
    assert_eq!(before, x.logits());
    assert_eq!(ids, x.token_ids());
}

#[test]
fn rank_one_quartet_sites_repack_only_selected_weights_and_contract_the_actual_vjp() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let rows = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let u = [0., 0.6, -0.8, 0.];
    let v = [0.1, -0.12, 0.08, -0.2];
    for site in [Query, Key, Value, Output] {
        let spec = CausalLmProjectionSpec {
            layer: 0,
            site,
            basis: Dense,
        };
        let zero = NativeCausalLmRankOneProjection::new(&m, 0, site, &u, &[0.; 4]).unwrap();
        let (c, g) = zero.contrast_gradient(&rows[0], [2, 7]).unwrap();
        let dense = m
            .projection_contrast_pullback(&rows[0], 2, 7, spec)
            .unwrap();
        assert_eq!(c, dense.contrast);
        for (j, &actual) in g.iter().enumerate() {
            let expect: f64 = (0..4).map(|i| u[i] * dense.weight[i * 4 + j]).sum();
            assert!((actual - expect).abs() < 1e-10);
        }
        let mut a = NativeCausalLmRankOneProjection::new(&m, 0, site, &u, &v).unwrap();
        let mat: Vec<f32> = u
            .iter()
            .flat_map(|x| v.iter().map(move |y| (x * y) as f32))
            .collect();
        let (merged, _) = a.contrast_gradient(&rows[0], [2, 7]).unwrap();
        let overlay = m
            .projection_perturbed_contrast(&rows[0], 2, 7, spec, &[0.; 4], &mat)
            .unwrap();
        assert!((merged.target_minus_alternative - overlay.target_minus_alternative).abs() < 2e-6);
        assert_eq!(zero.contrast_gradient(&rows[0], [2, 7]).unwrap().0, c);
        let mut dup = a.clone();
        let group = [0, 1, 2, 3];
        let step = a
            .step_quartets(
                &rows,
                [2, 7],
                &[group],
                CausalLmQuartetObjective::Interaction,
                0.02,
            )
            .unwrap();
        let s2 = dup
            .step_quartets(
                &rows,
                [2, 7],
                &[group, group],
                CausalLmQuartetObjective::Interaction,
                0.02,
            )
            .unwrap();
        assert_eq!(step.prompt_evaluations, 4);
        assert_eq!(step.gradient, s2.gradient);
        assert_eq!(a.right(), dup.right());
        let after = NativeCausalLmRankOneProjection::new(&m, 0, site, &u, a.right()).unwrap();
        assert_eq!(
            a.last_logits(&rows[1]).unwrap(),
            after.last_logits(&rows[1]).unwrap()
        );
        let before = a.right().to_vec();
        assert!(a
            .step_quartets(
                &rows,
                [2, 7],
                &[[0, 1, 2, 2]],
                CausalLmQuartetObjective::Pairwise,
                0.02
            )
            .is_err());
        assert!(a
            .step_quartets(
                &rows,
                [2, 7],
                &[group],
                CausalLmQuartetObjective::Pairwise,
                f64::INFINITY
            )
            .is_err());
        assert_eq!(a.right(), before);
    }
}

#[test]
fn indexed_output_bank_reuses_roster_for_distinct_counterfactual_pairings() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let rows = vec![
        vec![4, 8, 2, 1],
        vec![8, 4, 1],
        vec![7, 1, 2, 3],
        vec![1, 2, 3],
    ];
    let u = [0., 0.6, -0.8, 0.];
    for left in [None, Some(u.as_slice())] {
        let bank = m.output_contrast_bank(0, &rows, 2, 7, left).unwrap();
        for (i, p) in rows.iter().enumerate() {
            assert_eq!(bank.contrasts()[i], m.next_token_contrast(p, 2, 7).unwrap());
            assert_eq!(
                bank.gradient(i).unwrap().len(),
                if left.is_some() { 4 } else { 16 }
            );
        }
        for pairs in [vec![[0, 1], [2, 3], [0, 1]], vec![[0, 3], [2, 1], [0, 3]]] {
            let (p, n): (Vec<_>, Vec<_>) = pairs
                .iter()
                .map(|&[a, b]| (rows[a].clone(), rows[b].clone()))
                .unzip();
            let direct = m.output_margin_bank(0, &p, &n, 2, 7, left).unwrap();
            for rule in [
                CausalLmOutputMarginRule::EqualMean,
                CausalLmOutputMarginRule::CommonCosine,
            ] {
                let a = bank.direction(&pairs, &[0, 1, 0], rule).unwrap();
                let b = direct.direction(&[0, 1, 0], rule).unwrap();
                assert_eq!(a.coordinates(), b.coordinates());
                assert_eq!(a.certificate(), b.certificate());
            }
        }
        let null = bank
            .direction(
                &[[0, 0], [1, 1]],
                &[0, 1],
                CausalLmOutputMarginRule::CommonCosine,
            )
            .unwrap();
        assert_eq!(null.refusal, Some(CommonDirectionRefusal::ZeroMean));
        assert!(bank
            .direction(
                &[[0, 4], [1, 2]],
                &[0, 1],
                CausalLmOutputMarginRule::EqualMean
            )
            .is_err());
        assert!(bank.gradient(4).is_err());
    }
    assert!(m
        .output_contrast_bank(0, &[vec![0], vec![0]], 2, 7, None)
        .is_err());
    assert!(m
        .output_contrast_bank(0, &vec![vec![0]; MAX_OUTPUT_CONTRAST_ROWS + 1], 2, 7, None)
        .is_err());
}

#[test]
fn repeated_prompts_share_scalar_execution_not_objective_weight_or_snapshot() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let s =
        NativeCausalLmScalarOutput::new(&m, 0, &[1., 0., 0., 0.], &[0., 1., 0., 0.], 0.).unwrap();
    let p = vec![vec![1, 2, 3], vec![1, 2, 3], vec![3, 2, 1]];
    let n = vec![vec![3, 2, 1], vec![4, 3, 1], vec![4, 3, 1]];
    let (next, r) = s.step_pairs(&p, &n, 2, 7, 8., 0.02, 1.).unwrap();
    assert_eq!(r.gradient_prompt_evaluations, 3);
    assert_eq!(r.pairs.len(), 3);
    for (j, (a, b)) in p.iter().zip(&n).enumerate() {
        let (x, dx) = s.contrast_slope(a, 2, 7).unwrap();
        let (y, dy) = s.contrast_slope(b, 2, 7).unwrap();
        assert_eq!(
            r.pairs[j],
            [
                x.target_minus_alternative,
                y.target_minus_alternative,
                dx,
                dy
            ]
        );
    }
    let measured = next.pair_objective(&p, &n, 2, 7).unwrap();
    assert_eq!(measured.prompt_evaluations, 3);
    for (j, (a, b)) in p.iter().zip(&n).enumerate() {
        let z = |ids: &[usize]| {
            let logits = next.last_logits(ids).unwrap();
            f64::from(logits[2]) - f64::from(logits[7])
        };
        assert_eq!(measured.pairs[j], [z(a), z(b)]);
    }
}

#[test]
fn output_margin_bank_is_a_signed_difference_and_fixed_left_is_its_contraction() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let p = vec![vec![4, 8, 2, 1], vec![3, 7, 4], vec![4, 8, 2, 1]];
    let n = vec![vec![8, 4, 1, 2], vec![2, 7, 1], vec![8, 4, 1, 2]];
    let u = [0., 0.6, -0.8, 0.];
    let full = m.output_margin_bank(0, &p, &n, 2, 7, None).unwrap();
    let compact = m.output_margin_bank(0, &p, &n, 2, 7, Some(&u)).unwrap();
    assert_eq!(full.pair_contrasts(), compact.pair_contrasts());
    for i in 0..p.len() {
        let spec = CausalLmProjectionSpec {
            layer: 0,
            site: Output,
            basis: Dense,
        };
        let a = m.projection_contrast_pullback(&p[i], 2, 7, spec).unwrap();
        let b = m.projection_contrast_pullback(&n[i], 2, 7, spec).unwrap();
        assert_eq!(
            full.pair_contrasts()[i],
            [
                a.contrast.target_minus_alternative,
                b.contrast.target_minus_alternative
            ]
        );
        let d = &full.pair_gradients()[i];
        assert_eq!(
            *d,
            a.weight
                .iter()
                .zip(b.weight)
                .map(|(x, y)| x - y)
                .collect::<Vec<_>>()
        );
        for j in 0..4 {
            let projected = (0..4).map(|k| d[k * 4 + j] * u[k]).sum::<f64>();
            assert!((projected - compact.pair_gradients()[i][j]).abs() < 1e-12);
        }
    }
    let dir = compact
        .direction(&[0, 1, 0], CausalLmOutputMarginRule::CommonCosine)
        .unwrap();
    assert_eq!(dir.group_sizes, [2, 1]);
    assert_eq!((dir.target_id, dir.alternative_id), (2, 7));
    assert!(dir.refusal.is_none());
    let right = dir.coordinates().unwrap();
    assert_eq!(right.len(), 4);
    assert!(dir.slopes.iter().all(|x| *x > 0.));
    let s = dir.scalar_adapter(&m).unwrap();
    assert!(dir.scalar_adapter(&load()).is_err());
    for i in 0..2 {
        let a = s.contrast_slope(&p[i], 2, 7).unwrap().1 - s.contrast_slope(&n[i], 2, 7).unwrap().1;
        assert!((a - dir.slopes[i]).abs() < 1e-12);
    }
    assert!(full
        .direction(&[0, 0, 1], CausalLmOutputMarginRule::EqualMean)
        .unwrap()
        .scalar_adapter(&m)
        .is_err());
}

#[test]
fn output_margin_bank_refuses_conflicts_separately_from_bad_inputs() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let p = vec![vec![4, 8, 2, 1], vec![8, 4, 1, 2]];
    let n = vec![p[1].clone(), p[0].clone()];
    let bank = m.output_margin_bank(0, &p, &n, 2, 7, None).unwrap();
    let common = |labels: &[usize]| bank.direction(labels, CausalLmOutputMarginRule::CommonCosine);
    let opposed = common(&[0, 1]).unwrap();
    assert_eq!(opposed.refusal, Some(CommonDirectionRefusal::Unresolved));
    assert!(opposed.coordinates().is_none());
    assert!(common(&[0, 2]).is_err() && common(&[0]).is_err() && common(&[0, 0]).is_err());
    let zero = m
        .output_margin_bank(0, &p, &p, 2, 7, None)
        .unwrap()
        .direction(&[0, 1], CausalLmOutputMarginRule::CommonCosine)
        .unwrap();
    assert_eq!(zero.refusal, Some(CommonDirectionRefusal::ZeroMean));
    assert!(m.output_margin_bank(2, &p, &n, 2, 7, None).is_err());
    assert!(m
        .output_margin_bank(0, &p, &n, 2, 7, Some(&[f64::NAN; 4]))
        .is_err());
    assert!(m.output_margin_bank(0, &p, &n, 2, 2, None).is_err());
}

#[test]
fn scalar_output_uses_current_weights_and_factorized_pullback() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let (u, v) = ([0., 0.6, 0., -0.8], [0.8, 0., 0.6, 0.]);
    let ids = [4, 8, 2, 1];
    for layer in 0..2 {
        let state = |a| NativeCausalLmScalarOutput::new(&m, layer, &u, &v, a).unwrap();
        let spec = CausalLmProjectionSpec {
            layer,
            site: Output,
            basis: Dense,
        };
        let grad = m.projection_contrast_pullback(&ids, 2, 7, spec).unwrap();
        let (c, d) = state(0.).contrast_slope(&ids, 2, 7).unwrap();
        let dense = grad
            .weight
            .iter()
            .enumerate()
            .map(|(i, w)| w * u[i / 4] * v[i % 4])
            .sum::<f64>();
        assert_eq!(c, grad.contrast);
        assert!((dense - d).abs() < 1e-12);
        let s = state(0.34);
        let (c, d) = s.contrast_slope(&ids, 2, 7).unwrap();
        let last = s.last_logits(&ids).unwrap();
        assert_eq!(
            c.target_minus_alternative,
            f64::from(last[2]) - f64::from(last[7])
        );
        let gap = |a| {
            let z = state(a).last_logits(&ids).unwrap();
            f64::from(z[2]) - f64::from(z[7])
        };
        assert!(((gap(0.35) - gap(0.33)) / 0.02 - d).abs() < 1e-5 + 0.015 * d.abs());
        let delta = (0..16)
            .map(|i| (0.34 * u[i / 4] * v[i % 4]) as f32)
            .collect::<Vec<_>>();
        let independent = m
            .projection_perturbed_contrast(&ids, 2, 7, spec, &[0.; 4], &delta)
            .unwrap();
        assert!((c.target_minus_alternative - independent.target_minus_alternative).abs() < 2e-6);
    }
}

#[test]
fn scalar_search_measures_the_same_batch_before_accepting() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let (u, v) = ([0., 0., 0., 1.], [1., 0., 0., 0.]);
    let s = NativeCausalLmScalarOutput::new(&m, 0, &u, &v, 0.).unwrap();
    let p = vec![vec![4, 8, 2, 1], vec![4, 8, 2, 1]];
    let n = vec![vec![8, 4, 1, 2], vec![8, 4, 1, 2]];
    let config = CausalLmScalarSearch {
        learning_rate: 10.,
        max_step: 0.02,
        max_abs: 0.5,
        max_trials: 6,
        min_decrease: 0.,
    };
    let (fixed, g) = s.step_pairs(&p, &n, 2, 7, 10., 0.02, 0.5).unwrap();
    let (next, report) = s.step_pairs_backtracking(&p, &n, 2, 7, config).unwrap();
    assert_eq!(report.status, CausalLmScalarSearchStatus::Accepted);
    assert_eq!(report.gradient, g.gradient);
    assert_eq!(report.derivatives, g.pairs);
    assert_eq!(report.trials[0].coefficient, fixed.coefficient());
    assert_eq!(
        report.baseline.pairs,
        s.pair_objective(&p, &n, 2, 7).unwrap().pairs
    );
    let accepted = report.trials.last().unwrap();
    assert!(accepted.objective.loss < report.baseline.loss);
    assert_eq!(next.steps(), 1);
    assert_eq!(s.steps(), 0);
    assert_eq!(
        next.pair_objective(&p, &n, 2, 7).unwrap().pairs,
        accepted.objective.pairs
    );
}

#[test]
fn scalar_search_refusal_halves_the_feasible_step_and_never_advances() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let (u, v) = ([0., 0., 0., 1.], [1., 0., 0., 0.]);
    let s = NativeCausalLmScalarOutput::new(&m, 0, &u, &v, 0.).unwrap();
    let p = [vec![4, 8, 2, 1]];
    let n = [vec![8, 4, 1, 2]];
    let config = CausalLmScalarSearch {
        learning_rate: 1e8,
        max_step: 100.,
        max_abs: 0.125,
        max_trials: 4,
        min_decrease: 10.,
    };
    let (same, r) = s.step_pairs_backtracking(&p, &n, 2, 7, config).unwrap();
    assert_eq!(r.status, CausalLmScalarSearchStatus::TrialsExhausted);
    assert_eq!(
        same.checkpoint_json().unwrap(),
        s.checkpoint_json().unwrap()
    );
    assert_eq!(r.trials.len(), 4);
    for (i, t) in r.trials.iter().enumerate() {
        assert_eq!(t.coefficient.abs(), 0.125 / (1 << i) as f64);
        assert_eq!(t.required_decrease, 10.);
        let fresh = NativeCausalLmScalarOutput::new(&m, 0, &u, &v, t.coefficient).unwrap();
        assert_eq!(
            fresh.pair_objective(&p, &n, 2, 7).unwrap().pairs,
            t.objective.pairs
        );
    }
    let (same, r) = s.step_pairs_backtracking(&p, &p, 2, 7, config).unwrap();
    assert_eq!(r.status, CausalLmScalarSearchStatus::ZeroGradient);
    assert_eq!(r.baseline.loss, 2f64.ln());
    assert!(r.trials.is_empty());
    assert_eq!(same.steps(), 0);
    for bound in [0, MAX_SCALAR_SEARCH_TRIALS + 1] {
        assert!(s
            .step_pairs_backtracking(
                &p,
                &n,
                2,
                7,
                CausalLmScalarSearch {
                    max_trials: bound,
                    ..config
                }
            )
            .is_err());
    }
    assert!(s
        .step_pairs_backtracking(
            &p,
            &n,
            2,
            7,
            CausalLmScalarSearch {
                min_decrease: f64::NAN,
                ..config
            }
        )
        .is_err());
    assert!(s.pair_objective(&p, &[vec![usize::MAX]], 2, 7).is_err());
}

#[test]
fn scalar_output_checkpoints_resume_against_the_same_immutable_stack() {
    use sha2::{Digest, Sha256};
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let before = m.lora_to_json().unwrap();
    let u = [0., 0., 0., 1.];
    let v = [1., 0., 0., 0.];
    let s = NativeCausalLmScalarOutput::new(&m, 0, &u, &v, 0.).unwrap();
    let train = |s: &NativeCausalLmScalarOutput| {
        s.step_pairs(
            &[vec![4, 8, 2, 1]],
            &[vec![8, 4, 1, 2]],
            2,
            7,
            10.,
            0.02,
            0.5,
        )
        .unwrap()
    };
    let (n, r) = train(&s);
    assert_ne!(r.before, r.after);
    assert!((r.after - r.before).abs() <= 0.02 && n.steps() == 1 && s.coefficient() == 0.);
    let bytes = n.checkpoint_json().unwrap();
    let pin = format!("{:x}", Sha256::digest(bytes.as_bytes()));
    let other = NativeCausalLmScalarOutput::restore(&m, &bytes, &pin).unwrap();
    assert_eq!(
        train(&other).0.checkpoint_json().unwrap(),
        train(&n).0.checkpoint_json().unwrap()
    );
    assert!(NativeCausalLmScalarOutput::restore(&load(), &bytes, &pin).is_err());
    assert!(NativeCausalLmScalarOutput::restore(&m, &bytes, "wrong").is_err());
    assert!(NativeCausalLmScalarOutput::new(&m, 2, &u, &v, 0.).is_err());
    assert!(NativeCausalLmScalarOutput::new(&m, 0, &[0.; 4], &v, 0.).is_err());
    assert!(NativeCausalLmScalarOutput::new(&m, 0, &u, &v, f64::MAX).is_err());
    assert!(s.step_pairs(&[], &[], 2, 7, 1., 0.02, 0.5).is_err());
    assert_eq!(before, m.lora_to_json().unwrap());
}

#[test]
fn projection_probe_reads_native_heads_and_signed_contrast_at_both_layers() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompt = [4, 8, 2, 1];
    let factual = m.next_token_contrast(&prompt, 2, 7).unwrap();
    let owners = (
        m.lora_to_json().unwrap(),
        m.geometry_artifact_id().unwrap().to_string(),
    );
    for layer in 0..2 {
        for site in [Query, Key, Value, Output] {
            let spec = CausalLmProjectionSpec {
                layer,
                site,
                basis: Dense,
            };
            let r = m.projection_contrast_pullback(&prompt, 2, 7, spec).unwrap();
            let s = CausalLmProjectionSpec {
                basis: Diagonal,
                ..spec
            };
            let diag = m.projection_contrast_pullback(&prompt, 2, 7, s).unwrap();
            assert_eq!(r.contrast, factual);
            assert_eq!(diag.bias, r.bias);
            assert_eq!(
                diag.weight,
                (0..4).map(|j| r.weight[j * 4 + j]).collect::<Vec<_>>()
            );
            assert_eq!(r.readout_accumulator, m.readout_accumulator());
            let rev = m.projection_contrast_pullback(&prompt, 7, 2, spec).unwrap();
            assert!(r
                .bias
                .iter()
                .chain(&r.weight)
                .zip(rev.bias.iter().chain(&rev.weight))
                .all(|(a, b)| (*a + *b).abs() < 1e-8));
            assert_eq!(
                m.projection_perturbed_contrast(&prompt, 2, 7, spec, &[0.; 4], &[0.; 16])
                    .unwrap(),
                factual
            );
            let l = r
                .bias
                .iter()
                .chain(&r.weight)
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(l > 1e-5);
            let val = |h: f64| {
                let b = r
                    .bias
                    .iter()
                    .map(|x| (h * x / l) as f32)
                    .collect::<Vec<_>>();
                let w = r
                    .weight
                    .iter()
                    .map(|x| (h * x / l) as f32)
                    .collect::<Vec<_>>();
                m.projection_perturbed_contrast(&prompt, 2, 7, spec, &b, &w)
                    .unwrap()
                    .target_minus_alternative
            };
            let h = 0.003;
            let actual = (val(h) - val(-h)) * 0.5;
            assert!(
                (actual - h * l).abs() < 8e-7 + 0.015 * h * l,
                "{layer} {site:?}: {actual} vs {}",
                h * l
            );
            if site == Key {
                assert!(r.bias.iter().all(|x| x.abs() < 1e-6));
                let gauge = m
                    .projection_perturbed_contrast(
                        &prompt,
                        2,
                        7,
                        spec,
                        &[0.2, -0.1, 0.3, -0.4],
                        &[0.; 16],
                    )
                    .unwrap();
                assert!(
                    (gauge.target_minus_alternative - factual.target_minus_alternative).abs()
                        < 1e-6
                );
            }
        }
    }
    assert_eq!(
        owners,
        (
            m.lora_to_json().unwrap(),
            m.geometry_artifact_id().unwrap().to_string()
        )
    );
    assert_eq!(m.next_token_contrast(&prompt, 2, 7).unwrap(), factual);
}

#[test]
fn projection_overlay_matches_separately_reimported_weights_in_gpt2_row_major_order() {
    let base = load();
    let prompt = [1, 6, 2, 3];
    for layer in 0..2 {
        for (site, name, part, stride) in [
            (Query, "c_attn", 0, 12),
            (Key, "c_attn", 1, 12),
            (Value, "c_attn", 2, 12),
            (Output, "c_proj", 0, 4),
        ] {
            let mut weights = parameters();
            let path = format!("h.{layer}.attn.{name}");
            let mut bias = [0.; 4];
            bias[1] = -0.0625;
            let mut dw = [0.; 16];
            dw[4 + 2] = 0.125;
            dw[12] = -0.0625;
            for (i, &b) in bias.iter().enumerate() {
                weights.get_mut(&(path.clone() + ".bias")).unwrap().1[part * 4 + i] += b;
            }
            for i in 0..4 {
                for j in 0..4 {
                    weights.get_mut(&(path.clone() + ".weight")).unwrap().1
                        [i * stride + part * 4 + j] += dw[i * 4 + j];
                }
            }
            let reference = NativeCausalLm::from_gpt2_json(
                &serde_json::to_vec(&config()).unwrap(),
                &archive(&weights),
                Default::default(),
            )
            .unwrap();
            let expected = reference.next_token_contrast(&prompt, 2, 7).unwrap();
            let r = base
                .projection_perturbed_contrast(
                    &prompt,
                    2,
                    7,
                    CausalLmProjectionSpec {
                        layer,
                        site,
                        basis: Dense,
                    },
                    &bias,
                    &dw,
                )
                .unwrap();
            assert_eq!(r.greedy_id, expected.greedy_id);
            assert!(
                (r.target_logit - expected.target_logit).abs() < 2e-6
                    && (r.alternative_logit - expected.alternative_logit).abs() < 2e-6
            );
        }
    }
}

#[test]
fn projection_probe_admission_and_single_token_softmax_null() {
    let mut m = load();
    let s = CausalLmProjectionSpec {
        layer: 1,
        site: Key,
        basis: Diagonal,
    };
    for site in [Query, Key] {
        let r = m
            .projection_contrast_pullback(&[1], 2, 7, CausalLmProjectionSpec { site, ..s })
            .unwrap();
        assert!(r.bias.iter().chain(&r.weight).all(|&x| x == 0.));
    }
    assert!(m.projection_probe_dimensions(2, 2, 2, s, true).is_err());
    assert!(m
        .projection_probe_dimensions(2, 2, 7, CausalLmProjectionSpec { layer: 2, ..s }, true)
        .is_err());
    assert!(m
        .projection_perturbed_contrast(&[0], 2, 7, s, &[0.; 1], &[0.; 4])
        .is_err());
    assert!(m
        .projection_perturbed_contrast(&[0], 2, 7, s, &[f32::NAN; 4], &[0.; 4])
        .is_err());
    assert!(m.projection_contrast_pullback(&[100], 2, 7, s).is_err());
    m.limits.max_training_checkpoint_elements = 8;
    assert!(m.projection_probe_dimensions(2, 2, 7, s, true).is_ok());
    assert!(m
        .projection_probe_dimensions(2, 2, 7, CausalLmProjectionSpec { layer: 0, ..s }, true)
        .is_err());
    assert!(m
        .projection_probe_dimensions(2, 2, 7, CausalLmProjectionSpec { layer: 0, ..s }, false)
        .is_ok());
}
