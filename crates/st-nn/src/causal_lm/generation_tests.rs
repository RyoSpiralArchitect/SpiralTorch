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
            assert!(!payload.contains("lora_artifact_id"));
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
fn id_only_chunk_and_reported_paths_share_exact_rng_cache_and_checkpoint() {
    for execution in [
        CausalLmGenerationExecution::FullPrefix,
        CausalLmGenerationExecution::KvCache { max_values: 1024 },
    ] {
        for selection in [TokenSelection::Greedy, TokenSelection::Sample] {
            for distribution in [
                TokenDistribution::Softmax { temperature: 0.7 },
                TokenDistribution::ZSpace {
                    config: ZSpaceGenerationControlConfig::default(),
                },
            ] {
                for readout in [
                    CausalLmReadoutAccumulator::F32,
                    CausalLmReadoutAccumulator::F64,
                ] {
                    let base = load().with_readout_accumulator(readout);
                    let model = base
                        .with_lora(1, 2, 3.)
                        .unwrap()
                        .with_geometry(artifact(&base, "transformer.h.1.attn", 0.2))
                        .unwrap();
                    let cfg = CausalLmGenerationConfig {
                        decoding: TokenDecodingConfig {
                            selection,
                            distribution: distribution.clone(),
                            top_k: Some(6),
                            top_p: 0.8,
                        },
                        execution: execution.clone(),
                        ..policy()
                    };
                    let mut full =
                        NativeCausalLmGenerator::new(model.clone(), vec![1, 2], cfg).unwrap();
                    let mut other = full.clone();
                    assert_eq!(
                        other.step_token_id().unwrap(),
                        Some(full.step().unwrap().unwrap().decision.token_id)
                    );
                    assert_eq!(other.to_json().unwrap(), full.to_json().unwrap());
                    let bytes = other.to_json().unwrap().into_bytes();
                    other =
                        NativeCausalLmGenerator::from_json(model, &bytes, &pin(&bytes)).unwrap();
                    let ids = other.advance_ids(2).unwrap();
                    assert_eq!(
                        ids,
                        (0..2)
                            .map(|_| full.step().unwrap().unwrap().decision.token_id)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(other.to_json().unwrap(), full.to_json().unwrap());
                    assert_eq!(other.step().unwrap(), full.step().unwrap());
                    let last = other.advance_ids(1024).unwrap();
                    assert_eq!(last, vec![full.step().unwrap().unwrap().decision.token_id]);
                    assert_eq!(other.to_json().unwrap(), full.to_json().unwrap());
                    assert!(other.advance_ids(1).unwrap().is_empty());
                    assert!(other.step_token_id().unwrap().is_none());
                    assert!(other.advance_ids(0).is_err());
                    assert!(other.advance_ids(1025).is_err());
                }
            }
        }
    }
}

#[test]
fn late_control_error_rolls_back_the_entire_id_chunk_including_kv_and_rng() {
    let mut cfg = config();
    cfg["tie_word_embeddings"] = json!(false);
    let mut w = parameters();
    w.insert("lm_head.weight".into(), (vec![11, 4], vec![0.; 44]));
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&cfg).unwrap(),
        &archive(&w),
        Default::default(),
    )
    .unwrap();
    for execution in [
        CausalLmGenerationExecution::FullPrefix,
        CausalLmGenerationExecution::KvCache { max_values: 1024 },
    ] {
        for selection in [TokenSelection::Greedy, TokenSelection::Sample] {
            let cfg = CausalLmGenerationConfig {
                execution: execution.clone(),
                decoding: TokenDecodingConfig {
                    selection,
                    top_k: Some(1),
                    distribution: TokenDistribution::ZSpace {
                        config: ZSpaceGenerationControlConfig {
                            repression_strength: f64::MAX,
                            last_token_repression: f64::MAX,
                            ..Default::default()
                        },
                    },
                    ..Default::default()
                },
                ..policy()
            };
            let mut g = NativeCausalLmGenerator::new(model.clone(), vec![1], cfg).unwrap();
            let initial = g.to_json().unwrap();
            let mut proof = g.clone();
            assert_eq!(proof.step().unwrap().unwrap().decision.token_id, 0);
            assert!(proof.step().is_err()); // The newly generated zero is now the last, one-occurrence token.
            assert!(g.advance_ids(2).is_err());
            assert_eq!(g.to_json().unwrap(), initial);
            assert_eq!(g.step_token_id().unwrap(), Some(0));
            assert_eq!(g.to_json().unwrap(), proof.to_json().unwrap());
        }
    }
}

#[test]
fn id_chunk_stops_at_generated_eos_and_never_reemits_prior_ids() {
    let model = load();
    let id = model.next_token_ids(&[1, 2], [1, 2]).unwrap()[0];
    let mut g = NativeCausalLmGenerator::new(
        model,
        vec![1, 2],
        CausalLmGenerationConfig {
            eos_token_ids: vec![id],
            ..policy()
        },
    )
    .unwrap();
    assert_eq!(g.advance_ids(5).unwrap(), vec![id]);
    assert_eq!(g.finish_reason(), Some("eos"));
    let saved = g.to_json().unwrap();
    assert!(g.advance_ids(4).unwrap().is_empty());
    assert_eq!(g.to_json().unwrap(), saved);
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
        ("lora_artifact_id", json!("sha256:foreign-adapter")),
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

#[test]
fn learned_lora_generation_checkpoint_binds_adapter_and_resumes_both_routes() {
    let base = load();
    let initial = base.with_lora(0, 2, 4.).unwrap();
    let gradients = initial.lora_loss_gradients(&[1, 2, 3, 4], [1, 4]).unwrap();
    let model = initial.step_lora_sgd(&gradients, 0.3).unwrap();
    let next = model
        .step_lora_sgd(
            &model.lora_loss_gradients(&[4, 3, 2, 1], [1, 4]).unwrap(),
            0.3,
        )
        .unwrap();
    let restored_model = base.with_lora_json(&model.lora_to_json().unwrap()).unwrap();
    for execution in [
        CausalLmGenerationExecution::FullPrefix,
        CausalLmGenerationExecution::KvCache { max_values: 1024 },
    ] {
        let mut generator = NativeCausalLmGenerator::new(
            model.clone(),
            vec![1, 2],
            CausalLmGenerationConfig {
                execution,
                decoding: TokenDecodingConfig {
                    selection: TokenSelection::Sample,
                    top_k: Some(6),
                    ..Default::default()
                },
                ..policy()
            },
        )
        .unwrap();
        generator.step().unwrap();
        let payload = generator.to_json().unwrap();
        let pinned = pin(payload.as_bytes());
        let saved: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(saved["lora_artifact_id"], json!(model.lora_artifact_id()));
        for wrong in [&base, &initial, &next] {
            assert!(
                NativeCausalLmGenerator::from_json(wrong.clone(), payload.as_bytes(), &pinned)
                    .is_err()
            );
        }
        let mut restored =
            NativeCausalLmGenerator::from_json(restored_model.clone(), payload.as_bytes(), &pinned)
                .unwrap();
        assert_eq!(restored.to_json().unwrap(), payload);
        while !generator.finished() {
            assert_eq!(generator.step().unwrap(), restored.step().unwrap());
            assert_eq!(generator.to_json().unwrap(), restored.to_json().unwrap());
        }
        let mut missing_owner = saved;
        missing_owner
            .as_object_mut()
            .unwrap()
            .remove("lora_artifact_id");
        let missing_owner = serde_json::to_vec(&missing_owner).unwrap();
        assert!(NativeCausalLmGenerator::from_json(
            model.clone(),
            &missing_owner,
            &pin(&missing_owner)
        )
        .is_err());
    }
}
