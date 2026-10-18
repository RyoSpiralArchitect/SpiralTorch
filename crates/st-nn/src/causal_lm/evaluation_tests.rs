use super::*;

fn docs() -> Vec<Vec<usize>> {
    vec![
        vec![],
        vec![1],
        vec![2, 3],
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![4, 3, 2, 1],
    ]
}

fn policy(batch: usize) -> CausalLmEvaluationConfig {
    CausalLmEvaluationConfig {
        context_length: 4,
        microbatch_windows: batch,
    }
}

#[test]
fn corpus_nll_matches_target_weighted_manual_windows_and_preserves_geometry() {
    let base = load();
    for model in [
        base.clone(),
        base.with_geometry(artifact(&base, "transformer.h.0.attn", 0.4))
            .unwrap(),
    ] {
        let documents = docs();
        let windows = [
            &documents[2][..],
            &documents[3][..4],
            &documents[3][3..7],
            &documents[3][6..],
            &documents[4][..],
        ];
        let expected = windows
            .iter()
            .map(|w| {
                f64::from(model.next_token_loss(w, [1, w.len()]).unwrap()) * (w.len() - 1) as f64
            })
            .sum::<f64>()
            / 11.;
        let before = model.geometry_artifact().map(|v| v.to_json().to_owned());
        let one = model
            .evaluate_documents(&documents, None, policy(1))
            .unwrap();
        assert_eq!(one.nll, expected);
        assert_eq!(
            (one.window_count, one.target_tokens, one.ignored_targets),
            (5, 11, 0)
        );
        assert_eq!(one.perplexity, Some(one.nll.exp()));
        assert_eq!(
            before,
            model.geometry_artifact().map(|v| v.to_json().to_owned())
        );
        for batch in [2, 7] {
            let other = model
                .evaluate_documents(&documents, None, policy(batch))
                .unwrap();
            assert!((one.nll - other.nll).abs() < 3e-6);
            assert_eq!(one.data_id, other.data_id);
            assert_eq!(one.plan_id, other.plan_id);
        }
    }
}

#[test]
fn masked_targets_and_short_documents_are_validated_and_not_misweighted() {
    let model = load();
    let documents = docs();
    let mut labels: Vec<Vec<i64>> = documents.iter().map(|d| vec![-100; d.len()]).collect();
    labels[3][7] = 2;
    labels[4][2] = 7;
    let score = model
        .evaluate_documents(&documents, Some(&labels), policy(2))
        .unwrap();
    let a = model
        .next_token_loss_with_objective(
            &documents[3][6..],
            [1, 2],
            Some(&labels[3][6..]),
            CausalLmObjective::default(),
        )
        .unwrap();
    let b = model
        .next_token_loss_with_objective(
            &documents[4],
            [1, 4],
            Some(&labels[4]),
            CausalLmObjective::default(),
        )
        .unwrap();
    assert_eq!(score.nll, (f64::from(a) + f64::from(b)) / 2.);
    assert_eq!(
        (
            score.target_tokens,
            score.ignored_targets,
            score.skipped_windows
        ),
        (2, 9, 3)
    );
    labels[1][0] = 99; // Even an unscored single-token document is validated.
    assert!(model
        .evaluate_documents(&documents, Some(&labels), policy(2))
        .is_err());
    labels[1][0] = 1;
    assert_ne!(
        score.data_id,
        model
            .evaluate_documents(&documents, Some(&labels), policy(2))
            .unwrap()
            .data_id
    );
    labels[3][7] = -100;
    labels[4][2] = -100;
    assert!(model
        .evaluate_documents(&documents, Some(&labels), policy(2))
        .is_err());
}

#[test]
fn evaluation_admission_checks_lengths_readouts_and_all_ids() {
    let model = load();
    assert!(model
        .evaluation_plan(&[MAX_EVALUATION_TOKENS + 1], policy(1), false)
        .is_err());
    assert!(model.evaluation_plan(&[1, 1], policy(1), false).is_err());
    assert!(model.evaluation_plan(&[4], policy(0), false).is_err());
    assert!(model
        .evaluation_plan(
            &[4],
            CausalLmEvaluationConfig {
                context_length: 10,
                microbatch_windows: 1,
            },
            false
        )
        .is_err());
    let mut documents = docs();
    documents[1][0] = 99;
    assert!(model
        .evaluate_documents(&documents, None, policy(1))
        .is_err());
    let limited = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&parameters()),
        CausalLmLimits {
            max_logit_elements: 11,
            ..Default::default()
        },
    )
    .unwrap();
    let mut labels = vec![vec![-100; 4]];
    labels[0][3] = 1;
    assert!(limited
        .evaluate_documents(&[vec![1, 2, 3, 4]], Some(&labels), policy(1))
        .is_ok());
    labels[0][2] = 2;
    assert!(limited
        .evaluate_documents(&[vec![1, 2, 3, 4]], Some(&labels), policy(1))
        .is_err());
}

#[test]
fn perplexity_overflow_preserves_finite_nll_and_valid_json() {
    let mut model = load();
    // Frozen untied head creates a finite, very unlikely target, not NaN logits.
    let hidden = model.decode_hidden(&[1], [1, 1]).unwrap();
    let normalized = model
        .final_norm
        .forward(&hidden, model.config.norm_epsilon)
        .unwrap();
    let sign = if normalized.data()[0] > 0. { 1. } else { -1. };
    let mut weights = vec![0.; model.config.width * model.config.vocabulary];
    weights[0] = sign * 1e6;
    model.head = PackedB::from_tensor(
        &Tensor::from_vec(model.config.width, model.config.vocabulary, weights).unwrap(),
        st_tensor::Tile::col_major(),
    )
    .unwrap();
    let score = model
        .evaluate_documents(&[vec![1, 2]], None, policy(1))
        .unwrap();
    assert!(score.nll.is_finite() && score.nll > 710.);
    assert_eq!(score.perplexity, None);
    assert!(serde_json::to_string(&score)
        .unwrap()
        .contains("\"perplexity\":null"));
}
