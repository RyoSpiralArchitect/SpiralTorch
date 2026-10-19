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
