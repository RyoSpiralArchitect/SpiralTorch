use super::*;

fn oracle(
    m: &NativeCausalLm,
    p: &[usize],
    lines: &[Vec<usize>],
    gold: usize,
    stop: usize,
    mix: f64,
) -> (f64, Vec<f64>) {
    let costs: Vec<_> = lines
        .iter()
        .map(|c| {
            let mut prefix = p.to_vec();
            let mut nll = 0.;
            for &t in c.iter().chain([&stop]) {
                nll += m.next_token_score(&prefix, t).unwrap().nll;
                prefix.push(t);
            }
            nll
        })
        .collect();
    let cr = (costs.iter().map(|v| (costs[gold] - v).exp()).sum::<f64>()).ln();
    (
        (1. - mix) * costs[gold] / (lines[gold].len() + 1) as f64 + mix * cr,
        costs,
    )
}

#[test]
fn context_token_preference_replays_two_unpadded_ces_with_opposite_signs() {
    for precision in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let mut m = model(1, precision);
        m.limits.max_logit_elements = 2 * m.config.vocabulary;
        let a = &[1usize, 3, 2][..];
        let b = &[4usize, 1, 3, 5, 2][..];
        let target = 6;
        let r = m.context_token_preference(a, b, target, 0.4).unwrap();
        let g = m
            .geometry_context_token_preference_gradients(a, b, target, 0.4)
            .unwrap();
        assert_eq!(g.context_token_preference_report(), Some(&r));
        assert!(g.line_subset_report().is_none());
        assert_eq!(
            (g.target_tokens(), g.window_shape(), g.loss()),
            (2, [2, 6], r.loss as f32)
        );
        let separate = |prefix: &[usize]| {
            let mut ids = prefix.to_vec();
            ids.push(target);
            let mut y = vec![-100; prefix.len()];
            y.push(target as i64);
            m.adaptation_loss_gradients_with_labels(
                &ids,
                [1, ids.len()],
                Some(&y),
                Some(CausalLmAdaptationMode::Geometry),
            )
            .unwrap()
        };
        let (pos, neg) = (separate(a), separate(b));
        assert!((r.preferred_nll - m.next_token_score(a, target).unwrap().nll).abs() < 2e-6);
        assert!((r.other_nll - m.next_token_score(b, target).unwrap().nll).abs() < 2e-6);
        let mut nonzero = false;
        for (site, params) in g.parameters() {
            for (name, values) in params {
                for ((&x, &pa), &nb) in values
                    .iter()
                    .zip(&pos.parameters()[site][name])
                    .zip(&neg.parameters()[site][name])
                {
                    let expected = r.gradient_weight * (f64::from(pa) - f64::from(nb));
                    nonzero |= expected.abs() > 1e-7;
                    assert!((f64::from(x) - expected).abs() < 4e-7 + 4e-5 * expected.abs());
                }
            }
        }
        assert!(nonzero);
        assert!(m
            .geometry_context_token_preference_gradients(a, a, target, 0.)
            .is_err());
        assert!(m
            .geometry_context_token_preference_gradients(a, b, target, f64::MAX)
            .is_err());
        m.limits.max_logit_elements = 2 * m.config.vocabulary - 1;
        assert!(m
            .context_token_preference_shape(3, 5, target, 0., true)
            .is_err());
    }
}

#[test]
fn label_free_scores_match_full_vocabulary_terms_at_interior_and_leaf_stops() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompt = [1, 3];
    let lines = [vec![2], vec![2, 4], vec![2, 4, 7], vec![2, 5]];
    let r = m.score_lines(&prompt, &lines, 0).unwrap();
    assert_eq!(
        (r.prefix_nodes, r.decoder_lanes, r.shared_content_tokens),
        (5, 2, 1)
    );
    let subset = m.line_subset_loss(&prompt, &lines, &[0, 2], 0).unwrap();
    assert_eq!(r.line_nlls, subset.line_nlls);
    assert_eq!(r.menu_set_nll, subset.menu_set_nll);
    assert_eq!(r.winner_index, subset.winner_index);
    for (i, line) in lines.iter().enumerate() {
        let mut ids = prompt.to_vec();
        assert_eq!(r.token_nlls[i].len(), line.len() + 1);
        for (k, &cost) in line.iter().copied().chain([0]).zip(&r.token_nlls[i]) {
            assert!((m.next_token_score(&ids, k).unwrap().nll - cost).abs() < 2e-6);
            ids.push(k);
        }
        assert_eq!((-r.conditional_nlls[i]).exp(), subset.menu_probabilities[i]);
    }
}

#[test]
fn line_set_is_the_absolute_component_of_every_gold_loss_and_pullback() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let (p, lines, stop) = (
        [1, 3],
        vec![vec![2], vec![2, 4], vec![2, 4, 7], vec![2, 5]],
        0,
    );
    let g = m.geometry_line_set_gradients(&p, &lines, stop).unwrap();
    let r = g.line_set_report().unwrap();
    assert!(g.line_menu_report().is_none());
    assert_eq!(
        (
            g.target_tokens(),
            g.window_shape(),
            r.prefix_nodes,
            r.decoder_lanes
        ),
        (12, [4, 6], 5, 2)
    );
    let (_, costs) = oracle(&m, &p, &lines, 0, stop, 1.);
    let absolute = -costs.iter().map(|v| (-v).exp()).sum::<f64>().ln();
    assert!((r.set_nll - absolute).abs() < 2e-6);
    assert_eq!(
        r.set_nll,
        m.line_set_loss(&p, &lines, stop).unwrap().set_nll
    );
    for (gold, line) in lines.iter().enumerate() {
        let c = m
            .geometry_line_menu_gradients(&p, &lines, gold, stop, 1.)
            .unwrap();
        let singleton = m
            .geometry_line_set_gradients(&p, std::slice::from_ref(line), stop)
            .unwrap();
        let mut ids = p.to_vec();
        ids.extend(line);
        ids.push(stop);
        let mut labels = vec![-100; p.len()];
        labels.extend(line.iter().chain([&stop]).map(|&i| i as i64));
        let ce = m
            .adaptation_loss_gradients_with_labels(
                &ids,
                [1, ids.len()],
                Some(&labels),
                Some(CausalLmAdaptationMode::Geometry),
            )
            .unwrap();
        assert!((singleton.line_set_report().unwrap().set_nll - costs[gold]).abs() < 2e-6);
        assert!((r.set_nll + c.line_menu_report().unwrap().menu_nll - costs[gold]).abs() < 2e-6);
        for (path, params) in g.parameters() {
            for (k, values) in params {
                for (i, &v) in values.iter().enumerate() {
                    let want = (line.len() + 1) as f64 * f64::from(ce.parameters()[path][k][i]);
                    let tol = 3e-7 + 4e-5 * want.abs();
                    assert!((f64::from(singleton.parameters()[path][k][i]) - want).abs() < tol);
                    assert!(
                        (f64::from(v) + f64::from(c.parameters()[path][k][i]) - want).abs() < tol
                    );
                }
            }
        }
    }
    let next = m
        .step_adaptation_sgd(&g, CausalLmAdaptationMode::Geometry, 0.25)
        .unwrap();
    assert!(next.line_set_loss(&p, &lines, stop).unwrap().set_nll < r.set_nll);
    assert_eq!(next.lora_artifact_id(), m.lora_artifact_id());
    assert!(next
        .step_adaptation_sgd(&g, CausalLmAdaptationMode::Geometry, 0.25)
        .is_err());
}

#[test]
fn packed_set_admits_actual_head_and_checkpoint_work_not_just_optimistic_lengths() {
    let mut m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let p = [1, 3];
    let compact = vec![vec![2], vec![2, 4], vec![2, 4, 7], vec![2, 5]];
    // Identical lengths, but four maximal lines and eight distinct prefixes.
    let other = vec![vec![2], vec![3, 4], vec![3, 5, 6], vec![8, 9]];
    m.limits.max_logit_elements = 5 * 11;
    m.limits.max_training_checkpoint_elements = 2 * 2 * 5 * 4;
    assert_eq!(m.line_set_shape(2, &[1, 2, 3, 2], 0, true).unwrap(), [4, 6]);
    let g = m.geometry_line_set_gradients(&p, &compact, 0).unwrap();
    assert_eq!(g.target_tokens(), 12);
    assert_eq!(g.loss(), m.line_set_loss(&p, &compact, 0).unwrap().loss);
    assert!(m.line_set_loss(&p, &other, 0).is_err());
    let rank = m
        .geometry_line_menu_gradients(&p, &compact, 0, 0, 1.)
        .unwrap();
    let menu = rank.line_menu_report().unwrap();
    assert_eq!(
        (menu.decoder_lanes, menu.prefix_nodes, rank.target_tokens()),
        (2, 5, 12)
    );
    assert!(m.line_menu_loss(&p, &other, 0, 0, 1.).is_err());
    m.limits.max_logit_elements = 8 * 11;
    assert!(m.line_set_loss(&p, &other, 0).is_ok());
    assert!(m.geometry_line_set_gradients(&p, &other, 0).is_err());
    m.limits.max_training_checkpoint_elements = 2 * 4 * 5 * 4;
    assert!(m.geometry_line_set_gradients(&p, &other, 0).is_ok());
    m.limits.max_logit_elements = 5 * 11 - 1;
    assert!(m.line_set_shape(2, &[1, 2, 3, 2], 0, false).is_err());
}

#[test]
fn packed_fractional_geometry_matches_full_causal_lanes_for_loss_and_every_parameter() {
    let (p, lines, stop) = (
        [3, 1],
        vec![vec![2], vec![2, 4], vec![2, 4, 7], vec![3, 4, 7]],
        0,
    );
    for precision in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let base = model(1, precision);
        let path = "transformer.h.1.attn";
        let mut a = serde_json::to_value(base.geometry_artifact().unwrap().record()).unwrap();
        let placement = &mut a["placements"][0];
        placement["recipe"] = json!({"schema":"spiraltorch.fractional_memory_adapter.v1","features":4,"strength":0.5,
            "kernel":{"kernel_len":4,"step":0.8,"max_values":256,"max_products":1024}});
        placement["tensors"] = json!([
            {"name":"gate","shape":[4],"values":[0.2,-0.1,0.1,0.2],"requires_grad":true,"role":"parameter"},
            {"name":"log_alpha","shape":[],"values":[-0.2],"requires_grad":true,"role":"parameter"}]);
        let m = base
            .with_geometry(
                GeometryAdapterArtifact::create_json(&serde_json::to_vec(&a).unwrap()).unwrap(),
            )
            .unwrap();
        let g = m.geometry_line_set_gradients(&p, &lines, stop).unwrap();
        let r = g.line_set_report().unwrap();
        assert_eq!((r.prefix_nodes, r.decoder_lanes), (7, 2));
        let (_, costs) = oracle(&m, &p, &lines, 0, stop, 1.);
        let mass = costs.iter().map(|n| (-n).exp()).sum::<f64>();
        assert!((r.set_nll + mass.ln()).abs() < 2e-6);
        let mut expected = std::collections::BTreeMap::<String, Vec<f64>>::new();
        for (l, cost) in lines.iter().zip(costs) {
            let mut ids = p.to_vec();
            ids.extend(l);
            ids.push(stop);
            let mut labels = vec![-100; p.len()];
            labels.extend(l.iter().chain([&stop]).map(|&i| i as i64));
            let single = m
                .adaptation_loss_gradients_with_labels(
                    &ids,
                    [1, ids.len()],
                    Some(&labels),
                    Some(CausalLmAdaptationMode::Geometry),
                )
                .unwrap();
            let scale = (l.len() + 1) as f64 * (-cost).exp() / mass;
            for (k, v) in &single.parameters()[path] {
                for (dst, &x) in expected
                    .entry(k.clone())
                    .or_insert_with(|| vec![0.; v.len()])
                    .iter_mut()
                    .zip(v)
                {
                    *dst += scale * f64::from(x);
                }
            }
        }
        assert_eq!(
            expected.keys().collect::<Vec<_>>(),
            g.parameters()[path].keys().collect::<Vec<_>>()
        );
        for (k, vs) in expected {
            assert!(
                vs.iter().any(|v| v.abs() > 1e-8),
                "{k} must exercise a real pullback"
            );
            for (&x, want) in g.parameters()[path][&k].iter().zip(vs) {
                assert!(
                    (f64::from(x) - want).abs() < 4e-7 + 4e-5 * want.abs(),
                    "{k}: {x} vs {want}"
                );
            }
        }
    }
}

#[test]
fn terminated_unequal_lines_match_independent_prefixes_and_parameter_derivatives() {
    let prompt = [1, 3];
    let lines = vec![vec![2], vec![2, 4], vec![2, 4, 7], vec![2, 5]];
    let (gold, stop) = (1, 0);
    for precision in [
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let m = model(1, precision);
        let geom = m.geometry_artifact().unwrap();
        let single: Vec<_> = lines
            .iter()
            .map(|line| {
                let mut ids = prompt.to_vec();
                ids.extend(line);
                ids.push(stop);
                let mut labels = vec![-100; prompt.len()];
                labels.extend(line.iter().map(|&v| v as i64));
                labels.push(stop as i64);
                m.adaptation_loss_gradients_with_labels(
                    &ids,
                    [1, ids.len()],
                    Some(&labels),
                    Some(CausalLmAdaptationMode::Geometry),
                )
                .unwrap()
            })
            .collect();
        for mix in [0., 0.5, 1.] {
            let (loss, costs) = oracle(&m, &prompt, &lines, gold, stop, mix);
            let f = m.line_menu_loss(&prompt, &lines, gold, stop, mix).unwrap();
            let g = m
                .geometry_line_menu_gradients(&prompt, &lines, gold, stop, mix)
                .unwrap();
            let r = g.line_menu_report().unwrap();
            assert_eq!(
                serde_json::to_value(r).unwrap(),
                serde_json::to_value(&f).unwrap()
            );
            assert_eq!(
                (g.window_shape(), g.target_tokens(), r.shared_content_tokens),
                ([4, 6], 12, 1)
            );
            assert_eq!(
                (g.readout_accumulator(), g.lora_artifact_id()),
                (precision, m.lora_artifact_id())
            );
            assert_eq!(g.loss(), r.loss);
            assert!((f64::from(g.loss()) - loss).abs() < 2e-6);
            assert!(g.competition().is_none() && g.declared_rival_report().is_none());
            for (j, &v) in costs.iter().enumerate() {
                assert!((v - r.line_nlls[j]).abs() < 2e-6);
            }
            for (name, vs) in &g.parameters()["transformer.h.1.attn"] {
                for (i, &v) in vs.iter().enumerate() {
                    let param = |j: usize| {
                        f64::from(single[j].parameters()["transformer.h.1.attn"][name][i])
                    };
                    let expected = (1. - mix) * param(gold)
                        + mix
                            * (0..lines.len())
                                .map(|j| {
                                    (lines[j].len() + 1) as f64
                                        * param(j)
                                        * (if j == gold {
                                            r.rival_mass
                                        } else {
                                            -r.menu_probabilities[j]
                                        })
                                })
                                .sum::<f64>();
                    assert!(
                        (f64::from(v) - expected).abs() < 2e-7 + 3e-5 * expected.abs(),
                        "{name}:{i} {v} vs {expected}"
                    );
                }
            }
            if mix == 1. {
                // Derivative through the actual decoder, independently of its CE/VJP machinery.
                let mut max_abs = 0f32;
                for t in &geom.record().placements[0].tensors {
                    for i in 0..t.values.len() {
                        let read = |delta: f32| {
                            let mut rec = geom.record().clone();
                            rec.placements[0]
                                .tensors
                                .iter_mut()
                                .find(|x| x.name == t.name)
                                .unwrap()
                                .values[i] += delta;
                            oracle(
                                &m.with_geometry(GeometryAdapterArtifact::create(rec).unwrap())
                                    .unwrap(),
                                &prompt,
                                &lines,
                                gold,
                                stop,
                                mix,
                            )
                            .0
                        };
                        let v = g.parameters()["transformer.h.1.attn"][&t.name][i];
                        max_abs = max_abs.max(v.abs());
                        let num = (read(0.005) - read(-0.005)) / 0.01;
                        assert!(
                            (num - f64::from(v)).abs() < 6e-5,
                            "{}:{} {v} vs {num}",
                            t.name,
                            i
                        );
                    }
                }
                assert!(max_abs > 1e-4);
                let (next, step) = m
                    .step_paired_geometry_sgd(
                        &g,
                        &g,
                        0.2,
                        CausalLmPairWeighting::CommonDescent,
                        None,
                    )
                    .unwrap();
                assert!(step.applied.both_descend);
                assert_eq!(next.lora_to_json().unwrap(), m.lora_to_json().unwrap());
                assert!(oracle(&next, &prompt, &lines, gold, stop, mix).0 < loss);
                assert!(next
                    .step_adaptation_sgd(&g, CausalLmAdaptationMode::Geometry, 0.1)
                    .is_err());
            }
        }
    }
}

#[test]
fn rejects_invalid_support_and_work_before_allocating_padded_windows() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    let good = vec![vec![1], vec![1, 2]];
    for v in [
        vec![],
        vec![vec![1]],
        vec![vec![], vec![2]],
        vec![vec![1], vec![1]],
        vec![vec![2], vec![1]],
        vec![vec![0], vec![4]],
        vec![vec![1], vec![11]],
    ] {
        assert!(m.line_menu_loss(&[3], &v, 0, 0, 1.).is_err());
    }
    for (p, s, g, stop, w) in [
        (0, vec![1, 1], 0, 0, 1.),
        (usize::MAX, vec![1, 1], 0, 0, 1.),
        (1, vec![usize::MAX, 1], 0, 0, 1.),
        (2, vec![4094, 1], 0, 0, 1.),
        (1, vec![1; 33], 0, 0, 1.),
        (8, vec![1, 1], 0, 0, 1.),
        (1, vec![1, 1], 2, 0, 1.),
        (1, vec![1, 1], 0, 11, 1.),
        (1, vec![1, 1], 0, 0, f64::NAN),
        (1, vec![1, 1], 0, 0, -0.1),
    ] {
        assert!(m.line_menu_shape(p, &s, g, stop, w, true).is_err());
    }
    let base = load();
    assert!(base.line_menu_loss(&[3], &good, 0, 0, 1.).is_ok());
    assert!(base
        .geometry_line_menu_gradients(&[3], &good, 0, 0, 1.)
        .is_err());
    assert!(m
        .geometry_line_menu_gradients(&[11], &good, 0, 0, 1.)
        .is_err());
    let mut limited = m.clone();
    limited.limits.max_logit_elements = 4 * 11;
    assert!(limited.line_menu_shape(1, &[1, 2], 0, 0, 1., true).is_ok());
    limited.limits.max_logit_elements = 3 * 11 - 1;
    assert!(limited.line_menu_shape(1, &[1, 2], 0, 0, 1., true).is_err());
}
