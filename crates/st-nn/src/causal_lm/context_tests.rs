use super::*;

#[test]
fn prefill_and_absolute_position_appends_match_full_decoder() {
    let base = load();
    for geometry in [false, true] {
        let model = if geometry {
            base.with_geometry(artifact(&base, "transformer.h.0.mlp", 0.3))
                .unwrap()
        } else {
            base.clone()
        };
        let mut context = NativeCausalLmContext::new(model.clone(), vec![1, 2], 1024).unwrap();
        let snapshot = context.clone();
        for token in [3, 4, 5, 6, 7, 8] {
            close(
                &context.logits().unwrap().read_values().unwrap(),
                model
                    .logits(context.token_ids(), [1, context.token_ids().len()], true)
                    .unwrap()
                    .data(),
            );
            let last = context.append(token).unwrap();
            assert_eq!(
                last.read_values().unwrap(),
                context.logits().unwrap().read_values().unwrap()
            );
            assert_eq!(
                context.cache_values(),
                2 * 2 * context.token_ids().len() * 4
            );
        }
        close(
            &context.logits().unwrap().read_values().unwrap(),
            model
                .logits(context.token_ids(), [1, 8], true)
                .unwrap()
                .data(),
        );
        assert_eq!(snapshot.token_ids(), [1, 2]);
        assert_eq!(context.geometry_artifact_id(), model.geometry_artifact_id());
        assert!(context.append(0).is_err());
        assert_eq!(context.token_ids().len(), 8);
    }
}

#[test]
fn cache_admission_and_invalid_append_are_atomic() {
    let model = load();
    for (steps, budget) in [
        (0, 64),
        (1, 0),
        (2, 31),
        (9, 1000),
        (usize::MAX, usize::MAX),
    ] {
        assert!(model.context_prefill_count(steps, budget).is_err());
    }
    let mut context = NativeCausalLmContext::new(model, vec![1, 2], 48).unwrap();
    let before = context.logits().unwrap().read_values().unwrap();
    assert!(context.append(11).is_err());
    assert_eq!(context.token_ids(), [1, 2]);
    assert_eq!(context.logits().unwrap().read_values().unwrap(), before);
    context.append(3).unwrap();
    let before = context.logits().unwrap().read_values().unwrap();
    assert!(context.append(4).is_err());
    assert_eq!(context.token_ids(), [1, 2, 3]);
    assert_eq!(context.logits().unwrap().read_values().unwrap(), before);
}

#[test]
fn cached_work_uses_one_query_budget_not_a_hidden_full_prefix_fallback() {
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&parameters()),
        CausalLmLimits {
            max_attention_pairs: 12,
            ..Default::default()
        },
    )
    .unwrap();
    let mut context = NativeCausalLmContext::new(model.clone(), vec![1], 1024).unwrap();
    context.append(2).unwrap();
    context.append(3).unwrap();
    assert!(model.forward(&[1, 2, 3], [1, 3], true).is_err());
    close(
        &context.logits().unwrap().read_values().unwrap(),
        load().logits(&[1, 2, 3], [1, 3], true).unwrap().data(),
    );
    assert!(context.append(4).is_err());
}

#[test]
fn numerical_failure_after_cached_appends_preserves_continuation() {
    let mut cfg = config();
    cfg["tie_word_embeddings"] = json!(false);
    let mut weights = parameters();
    weights.insert("lm_head.weight".into(), weights["wte.weight"].clone());
    weights.get_mut("wte.weight").unwrap().1[40..44].fill(f32::MAX);
    // Only the poisoned token at the next absolute position overflows embedding addition.
    weights.get_mut("wpe.weight").unwrap().1[16..20].fill(f32::MAX);
    let model = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&cfg).unwrap(),
        &archive(&weights),
        Default::default(),
    )
    .unwrap();
    let mut context = NativeCausalLmContext::new(model, vec![1, 2], 1024).unwrap();
    context.append(3).unwrap();
    context.append(4).unwrap();
    let mut unchanged = context.clone();
    for _ in 0..2 {
        assert!(context.append(10).is_err());
        assert_eq!(context.token_ids(), unchanged.token_ids());
        assert_eq!(context.cache_values(), unchanged.cache_values());
        assert_eq!(
            context.logits().unwrap().read_values().unwrap(),
            unchanged.logits().unwrap().read_values().unwrap()
        );
    }
    assert_eq!(
        context.append(5).unwrap().read_values().unwrap(),
        unchanged.append(5).unwrap().read_values().unwrap()
    );
}

#[test]
fn cached_generation_restores_original_prefill_chronology_and_v1_default_stays_unchanged() {
    let model = load();
    let raw = br#"{"max_new_tokens":5,"decoding":{"selection":"sample","top_k":6},"execution":{"kind":"kv_cache","max_values":1024}}"#;
    let config = CausalLmGenerationConfig::from_json(raw).unwrap();
    let mut generator = NativeCausalLmGenerator::new(model.clone(), vec![1, 2], config).unwrap();
    for _ in 0..5 {
        let saved = generator.to_json().unwrap();
        let mut restored = NativeCausalLmGenerator::from_json(
            model.clone(),
            saved.as_bytes(),
            &generator.checkpoint_id().unwrap(),
        )
        .unwrap();
        assert_eq!(generator.step().unwrap(), restored.step().unwrap());
        assert_eq!(generator.to_json().unwrap(), restored.to_json().unwrap());
    }
    let old = CausalLmGenerationConfig::from_json(br#"{"max_new_tokens":1}"#).unwrap();
    assert!(!serde_json::to_string(&old).unwrap().contains("execution"));
    assert!(CausalLmGenerationConfig::from_json(
        br#"{"max_new_tokens":1,"execution":{"kind":"torch"}}"#
    )
    .is_err());
}
