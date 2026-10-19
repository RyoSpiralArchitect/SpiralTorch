use super::*;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use st_core::inference::generation_control::ZSpaceGenerationControlConfig;
use st_core::inference::token_decoding::{
    prepare_token_candidates, TokenDecodingConfig, TokenDistribution, TokenSelection,
};

fn policy() -> CausalLmGenerationConfig {
    CausalLmGenerationConfig {
        max_new_tokens: 5,
        eos_token_ids: vec![],
        seed: 81,
        decoding: Default::default(),
        execution: Default::default(),
    }
}
fn pin(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[test]
fn generation_checkpoint_binds_readout_policy_and_keeps_default_bytes() {
    for execution in [
        CausalLmGenerationExecution::FullPrefix,
        CausalLmGenerationExecution::KvCache { max_values: 1024 },
    ] {
        for accumulator in [
            CausalLmReadoutAccumulator::F32,
            CausalLmReadoutAccumulator::F64,
        ] {
            let model = load().with_readout_accumulator(accumulator);
            let mut generator = NativeCausalLmGenerator::new(
                model.clone(),
                vec![1, 2],
                CausalLmGenerationConfig {
                    execution: execution.clone(),
                    ..policy()
                },
            )
            .unwrap();
            generator.step().unwrap();
            let payload = generator.to_json().unwrap();
            assert_eq!(
                payload.contains("readout_accumulator"),
                accumulator == CausalLmReadoutAccumulator::F64
            );
            let other =
                model.with_readout_accumulator(if accumulator == CausalLmReadoutAccumulator::F32 {
                    CausalLmReadoutAccumulator::F64
                } else {
                    CausalLmReadoutAccumulator::F32
                });
            assert!(NativeCausalLmGenerator::from_json(
                other,
                payload.as_bytes(),
                &pin(payload.as_bytes())
            )
            .is_err());
            let mut restored = NativeCausalLmGenerator::from_json(
                model,
                payload.as_bytes(),
                &pin(payload.as_bytes()),
            )
            .unwrap();
            assert_eq!(restored.to_json().unwrap(), payload);
            while !generator.finished() {
                assert_eq!(generator.step().unwrap(), restored.step().unwrap());
                assert_eq!(generator.to_json().unwrap(), restored.to_json().unwrap());
            }
        }
    }
}

#[test]
fn greedy_matches_manual_decoder_and_finished_session_is_immutable() {
    let model = load();
    let mut generator =
        NativeCausalLmGenerator::new(model.clone(), vec![1, 2, 3, 4], policy()).unwrap();
    for n in 1..=5 {
        let expected = model
            .next_token_ids(generator.token_ids(), [1, generator.token_ids().len()])
            .unwrap()[0];
        let step = generator.step().unwrap().unwrap();
        assert_eq!((step.step, step.decision.token_id), (n, expected));
    }
    assert_eq!(generator.token_ids().len(), 9); // Last token is not evaluated beyond position capacity.
    assert_eq!(generator.finish_reason(), Some("length"));
    let before = generator.to_json().unwrap();
    assert!(generator.step().unwrap().is_none());
    assert_eq!(before, generator.to_json().unwrap());
}

#[test]
fn sampled_control_continuation_matches_sequential_rng_and_exact_checkpoint() {
    let base = load();
    let model = base
        .with_geometry(artifact(&base, "transformer.h.0.mlp", 0.2))
        .unwrap();
    let config = CausalLmGenerationConfig {
        decoding: TokenDecodingConfig {
            selection: TokenSelection::Sample,
            top_k: Some(6),
            top_p: 0.8,
            distribution: TokenDistribution::ZSpace {
                config: ZSpaceGenerationControlConfig::default(),
            },
        },
        ..policy()
    };
    let mut generator =
        NativeCausalLmGenerator::new(model.clone(), vec![1, 2], config.clone()).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);
    for i in 0..5 {
        let logits: Vec<_> = model
            .logits(
                generator.token_ids(),
                [1, generator.token_ids().len()],
                true,
            )
            .unwrap()
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let expected = prepare_token_candidates(&logits, generator.token_ids(), &config.decoding)
            .unwrap()
            .sample((rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.)
            .unwrap();
        let bytes = generator.to_json().unwrap();
        let mut restored = NativeCausalLmGenerator::from_json(
            model.clone(),
            bytes.as_bytes(),
            &generator.checkpoint_id().unwrap(),
        )
        .unwrap();
        let step = generator.step().unwrap().unwrap();
        assert_eq!(step.decision, expected);
        assert_eq!(step.step, i + 1);
        assert_eq!(restored.step().unwrap().unwrap(), step);
        assert_eq!(restored.to_json().unwrap(), generator.to_json().unwrap());
    }
}

#[test]
fn eos_only_stops_new_tokens_and_takes_precedence_over_length() {
    let model = load();
    let prompt = vec![1, 2];
    let eos = model.next_token_ids(&prompt, [1, 2]).unwrap()[0];
    let mut generator = NativeCausalLmGenerator::new(
        model.clone(),
        prompt,
        CausalLmGenerationConfig {
            max_new_tokens: 1,
            eos_token_ids: vec![eos],
            ..policy()
        },
    )
    .unwrap();
    assert!(!generator.finished());
    assert_eq!(
        generator.step().unwrap().unwrap().finish_reason,
        Some("eos")
    );
    let mut generator = NativeCausalLmGenerator::new(
        model,
        vec![eos],
        CausalLmGenerationConfig {
            eos_token_ids: vec![eos],
            ..policy()
        },
    )
    .unwrap();
    assert!(!generator.finished());
    assert!(generator.step().unwrap().is_some());
}

#[test]
fn config_budgets_owners_and_resealed_invalid_history_are_rejected() {
    let model = load();
    assert!(policy().validate(&model, 5).is_err());
    assert!(policy().validate(&model, usize::MAX).is_err());
    assert!(policy().validate(&model, 0).is_err());
    for eos in [vec![11], vec![1, 1]] {
        assert!(CausalLmGenerationConfig {
            eos_token_ids: eos,
            ..policy()
        }
        .validate(&model, 2)
        .is_err());
    }
    for max_new_tokens in [0, 1025] {
        assert!(CausalLmGenerationConfig {
            max_new_tokens,
            ..policy()
        }
        .validate(&model, 2)
        .is_err());
    }
    let g = NativeCausalLmGenerator::new(model.clone(), vec![1, 2], policy()).unwrap();
    let json = g.to_json().unwrap();
    assert!(NativeCausalLmGenerator::from_json(model.clone(), json.as_bytes(), "wrong").is_err());
    let with_geometry = model
        .with_geometry(artifact(&model, "transformer.h.0.mlp", 0.2))
        .unwrap();
    assert!(NativeCausalLmGenerator::from_json(
        with_geometry,
        json.as_bytes(),
        &pin(json.as_bytes())
    )
    .is_err());
    for (key, value) in [
        ("schema", json!("future")),
        ("rng", json!("foreign")),
        ("base_model_identity", json!("wrong")),
        ("prompt_length", json!(0)),
        ("prompt_length", json!(3)),
        ("tokens", json!([1, 11])),
        ("tokens", json!([1, 2, 3, 4, 5, 6, 7, 8])),
    ] {
        let mut saved: serde_json::Value = serde_json::from_str(&json).unwrap();
        saved[key] = value;
        let bytes = serde_json::to_vec(&saved).unwrap();
        assert!(
            NativeCausalLmGenerator::from_json(model.clone(), &bytes, &pin(&bytes)).is_err(),
            "{key}"
        );
    }
    let mut saved: serde_json::Value = serde_json::from_str(&json).unwrap();
    saved["config"]["eos_token_ids"] = json!([3]);
    saved["tokens"] = json!([1, 2, 3, 4]);
    let bytes = serde_json::to_vec(&saved).unwrap();
    assert!(NativeCausalLmGenerator::from_json(model, &bytes, &pin(&bytes)).is_err());
}

#[test]
fn numerical_control_failure_preserves_token_and_random_position() {
    let mut generator = NativeCausalLmGenerator::new(
        load(),
        vec![1, 1],
        CausalLmGenerationConfig {
            decoding: TokenDecodingConfig {
                selection: TokenSelection::Sample,
                distribution: TokenDistribution::ZSpace {
                    config: ZSpaceGenerationControlConfig {
                        repression_strength: f64::MAX,
                        ..Default::default()
                    },
                },
                ..Default::default()
            },
            ..policy()
        },
    )
    .unwrap();
    let before = generator.to_json().unwrap();
    for _ in 0..2 {
        assert!(generator.step().is_err());
        assert_eq!(generator.to_json().unwrap(), before);
    }
}
