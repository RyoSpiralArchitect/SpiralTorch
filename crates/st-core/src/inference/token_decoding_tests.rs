use super::*;

fn candidates(scores: &[f64], config: TokenDecodingConfig) -> TokenCandidates {
    prepare_token_candidates(scores, &[], &config).unwrap()
}

#[test]
fn ordinary_softmax_and_sampling_boundaries_are_normalized() {
    let c = candidates(&[0., 2., 1.], TokenDecodingConfig::default());
    assert_eq!(c.token_ids(), [1, 2, 0]);
    let total = 1.0 + (-1_f64).exp() + (-2_f64).exp();
    assert!((c.probabilities()[0] - 1. / total).abs() < 1e-15);
    assert_eq!(c.sample(0.).unwrap().token_id, 1);
    assert_eq!(c.sample(c.probabilities()[0]).unwrap().token_id, 2);
    assert_eq!(c.sample(1. - f64::EPSILON).unwrap().token_id, 0);
    for u in [-0.1, 1., f64::NAN, f64::INFINITY] {
        assert!(c.sample(u).is_err());
    }
}

#[test]
fn ties_use_token_id_but_rounded_probability_ties_do_not_change_greedy() {
    assert_eq!(
        candidates(&[-0., 0., 0.], Default::default()).token_ids(),
        [0, 1, 2]
    );
    let c = candidates(
        &[0., 1.],
        TokenDecodingConfig {
            distribution: TokenDistribution::Softmax {
                temperature: f64::MAX,
            },
            ..Default::default()
        },
    );
    assert_eq!(c.probabilities(), [0.5, 0.5]);
    assert_eq!(c.greedy().token_id, 1);
}

#[test]
fn nucleus_keeps_first_crossing_and_never_resurrects_zero_mass() {
    let c = candidates(
        &[0., 0., 0., -10000.],
        TokenDecodingConfig {
            top_p: 0.5,
            ..Default::default()
        },
    );
    assert_eq!(c.token_ids(), [0, 1]);
    assert_eq!(c.probabilities(), [0.5, 0.5]);
    assert_eq!(
        candidates(
            &[0., 0.],
            TokenDecodingConfig {
                top_p: f64::MIN_POSITIVE,
                ..Default::default()
            }
        )
        .token_ids(),
        [0]
    );
    let c = candidates(&[0., -10000.], Default::default());
    assert_eq!(c.token_ids(), [0]);
    assert_eq!(c.sample(1. - f64::EPSILON).unwrap().token_id, 0);
}

#[test]
fn extreme_finite_plain_scores_do_not_overflow_normalization() {
    let c = candidates(
        &[f64::MAX, -f64::MAX],
        TokenDecodingConfig {
            distribution: TokenDistribution::Softmax {
                temperature: f64::MAX,
            },
            ..Default::default()
        },
    );
    assert!((c.probabilities()[0] - 1. / (1. + (-2_f64).exp())).abs() < 1e-15);
    assert_eq!(
        candidates(&[f64::MAX, -f64::MAX], Default::default()).probabilities(),
        [1.]
    );
}

#[test]
fn zspace_uses_existing_probabilities_and_raw_top_k_before_repression() {
    let control = ZSpaceGenerationControlConfig {
        curvature: -1.,
        repression_strength: 10.,
        last_token_repression: 0.,
        ..Default::default()
    };
    let config = TokenDecodingConfig {
        distribution: TokenDistribution::ZSpace {
            config: control.clone(),
        },
        top_k: Some(2),
        ..Default::default()
    };
    let c = prepare_token_candidates(&[3., 2., 1.], &[0], &config).unwrap();
    let reference = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
        logits: vec![3., 2.],
        token_ids: vec![0, 1],
        recent_tokens: vec![0],
        config: control,
    })
    .unwrap();
    assert_eq!(c.token_ids(), [1, 0]);
    assert!((c.probabilities()[0] - reference.probabilities[1]).abs() < 1e-15);
    assert_eq!(
        c.greedy().zspace_contract,
        Some(ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION)
    );
    assert_eq!(c.greedy().raw_top_token, 0);
    assert_eq!(c.greedy().controlled_top_token, 1);
    assert!(!c.token_ids().contains(&2));
}

#[test]
fn invalid_inputs_and_excessive_ngram_work_fail_closed() {
    for scores in [vec![], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(prepare_token_candidates(&scores, &[], &Default::default()).is_err());
    }
    assert!(prepare_token_candidates(&[0.], &[1], &Default::default()).is_err());
    for k in [0, 3] {
        assert!(TokenDecodingConfig {
            top_k: Some(k),
            ..Default::default()
        }
        .validate(2, 0)
        .is_err());
    }
    for p in [0., -1., 1.01, f64::NAN] {
        assert!(TokenDecodingConfig {
            top_p: p,
            ..Default::default()
        }
        .validate(2, 0)
        .is_err());
    }
    for temperature in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(TokenDecodingConfig {
            distribution: TokenDistribution::Softmax { temperature },
            ..Default::default()
        }
        .validate(2, 0)
        .is_err());
    }
    let config = TokenDecodingConfig {
        distribution: TokenDistribution::ZSpace {
            config: ZSpaceGenerationControlConfig {
                ngram_size: 2,
                ngram_window: 1000,
                ngram_repression_strength: 1.,
                ..Default::default()
            },
        },
        ..Default::default()
    };
    assert!(config.validate(MAX_VOCABULARY, 1000).is_err());
    assert!(config.validate(10, 1000).is_ok());
    assert!(config.validate(10, MAX_HISTORY + 1).is_err());
}

// Independent full-sort oracle with the original summation chronology. No partial selector,
// in-place compaction, or optimized ID path is used to produce expected values.
fn full_sort_reference(
    logits: &[f64],
    history: &[usize],
    cfg: &TokenDecodingConfig,
) -> TokenCandidates {
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    ids.truncate(cfg.top_k.unwrap_or(ids.len()));
    let raw_top_token = ids[0];
    let scores: Vec<_> = ids.iter().map(|&i| logits[i]).collect();
    let (adjusted, p, temperature, contract) = match &cfg.distribution {
        TokenDistribution::Softmax { temperature } => {
            let top = scores[0];
            let mut w: Vec<_> = scores
                .iter()
                .map(|&v| {
                    let d = v - top;
                    (if d.is_infinite() && *temperature >= 1. {
                        v / temperature - top / temperature
                    } else {
                        d / temperature
                    })
                    .exp()
                })
                .collect();
            let sum = w.iter().sum::<f64>();
            for v in &mut w {
                *v /= sum;
            }
            (scores, w, *temperature, None)
        }
        TokenDistribution::ZSpace { config } => {
            let c = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
                logits: scores,
                token_ids: ids.iter().map(|&id| id as u64).collect(),
                recent_tokens: history.iter().map(|&id| id as u64).collect(),
                config: config.clone(),
            })
            .unwrap();
            (
                c.adjusted_logits,
                c.probabilities,
                c.effective_temperature,
                Some(ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION),
            )
        }
    };
    let control_entropy = zspace_entropy(&p).unwrap();
    let mut ranked: Vec<_> = (0..ids.len()).collect();
    ranked.sort_by(|&a, &b| {
        adjusted[b]
            .partial_cmp(&adjusted[a])
            .unwrap()
            .then_with(|| ids[a].cmp(&ids[b]))
    });
    let controlled_top_token = ids[ranked[0]];
    let mut kept = Vec::new();
    let mut probs = Vec::new();
    let mut mass = 0.;
    for i in ranked {
        if p[i] == 0. {
            continue;
        }
        kept.push(ids[i]);
        probs.push(p[i]);
        mass += p[i];
        if cfg.top_p < 1. && mass >= cfg.top_p {
            break;
        }
    }
    for x in &mut probs {
        *x /= mass;
    }
    let report = TokenDecision {
        token_id: kept[0],
        probability: probs[0],
        entropy: zspace_entropy(&probs).unwrap(),
        control_entropy,
        effective_temperature: temperature,
        raw_top_token,
        controlled_top_token,
        pre_nucleus_candidates: ids.len(),
        candidates: kept.len(),
        zspace_contract: contract,
    };
    TokenCandidates {
        ids: kept.into_boxed_slice(),
        probabilities: probs.into_boxed_slice(),
        report,
    }
}

fn same_reports(a: &TokenCandidates, b: &TokenCandidates) {
    assert_eq!(a.token_ids(), b.token_ids());
    assert_eq!(
        a.probabilities()
            .iter()
            .map(|p| p.to_bits())
            .collect::<Vec<_>>(),
        b.probabilities()
            .iter()
            .map(|p| p.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        serde_json::to_vec(&a.greedy()).unwrap(),
        serde_json::to_vec(&b.greedy()).unwrap()
    );
    for u in [
        0.,
        0.37,
        b.probabilities()[0].min(1. - f64::EPSILON),
        1. - f64::EPSILON,
    ] {
        assert_eq!(a.sample(u).unwrap(), b.sample(u).unwrap());
    }
}

#[test]
fn f32_and_f64_candidates_are_bit_identical_to_full_sort_at_cutoffs_and_underflow() {
    let x: Vec<f32> = (0..257)
        .map(|i| match i % 11 {
            0 => -0.,
            1 => 0.,
            2 => -15000.,
            3 => 3.,
            _ => ((i * 131 % 47) as f32 - 23.) / 3.,
        })
        .collect();
    let y: Vec<_> = x.iter().map(|&v| f64::from(v)).collect();
    let history = [7, 3, 1, 7, 3, 1];
    for distribution in [
        TokenDistribution::Softmax {
            temperature: f64::MIN_POSITIVE,
        },
        TokenDistribution::Softmax { temperature: 0.7 },
        TokenDistribution::Softmax {
            temperature: f64::MAX,
        },
        TokenDistribution::ZSpace {
            config: ZSpaceGenerationControlConfig {
                repression_strength: 2.,
                ngram_size: 2,
                ngram_window: 8,
                ngram_repression_strength: 4.,
                ..Default::default()
            },
        },
    ] {
        for k in [None, Some(1), Some(2), Some(23), Some(256), Some(257)] {
            for p in [f64::MIN_POSITIVE, 0.35, 0.99, 1.] {
                let cfg = TokenDecodingConfig {
                    distribution: distribution.clone(),
                    top_k: k,
                    top_p: p,
                    ..Default::default()
                };
                let expected = full_sort_reference(&y, &history, &cfg);
                same_reports(
                    &prepare_token_candidates(&y, &history, &cfg).unwrap(),
                    &expected,
                );
                same_reports(
                    &prepare_token_candidates_f32(&x, &history, &cfg).unwrap(),
                    &expected,
                );
                assert_eq!(
                    greedy_token_id_f32(&x, &history, &cfg).unwrap(),
                    expected.greedy().token_id
                );
            }
        }
    }
    for t in [0.5, 1., f64::MAX] {
        let extreme = [-f64::MAX, 0., f64::MAX, -0., f64::MIN_POSITIVE, 1., 1.];
        let cfg = TokenDecodingConfig {
            distribution: TokenDistribution::Softmax { temperature: t },
            ..Default::default()
        };
        same_reports(
            &prepare_token_candidates(&extreme, &[], &cfg).unwrap(),
            &full_sort_reference(&extreme, &[], &cfg),
        );
    }
}

#[test]
fn id_only_greedy_validates_and_respects_real_control_and_lowest_id_ties() {
    let cfg = TokenDecodingConfig {
        top_k: Some(1),
        top_p: f64::MIN_POSITIVE,
        ..Default::default()
    };
    assert_eq!(greedy_token_id_f32(&[-0., 0., -1.], &[], &cfg).unwrap(), 0);
    assert!(greedy_token_id_f32(&[1., f32::NAN], &[], &cfg).is_err());
    assert!(greedy_token_id_f32(&[0., 1.], &[2], &cfg).is_err());
    assert!(greedy_token_id_f32(&[], &[], &cfg).is_err());
    assert!(greedy_token_id_f32(
        &[0.],
        &[],
        &TokenDecodingConfig {
            selection: TokenSelection::Sample,
            ..cfg
        }
    )
    .is_err());
    let c = TokenDecodingConfig {
        distribution: TokenDistribution::ZSpace {
            config: ZSpaceGenerationControlConfig {
                repression_strength: 10.,
                last_token_repression: 0.,
                ..Default::default()
            },
        },
        top_k: Some(2),
        ..Default::default()
    };
    assert_eq!(greedy_token_id_f32(&[3., 2., 1.], &[0], &c).unwrap(), 1);
}
