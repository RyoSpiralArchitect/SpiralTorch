use super::*;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};

#[test]
fn decoder_uses_the_selected_block_and_raw_keys_match_materialized_gather() {
    use st_kernel_contracts::attention::attention_reference;
    let base = load();
    let ids = [2, 4, 1, 8, 1, 5, 2, 3];
    let shape = [2, 4];
    for selected in 0..2 {
        let m = base
            .with_key_lag(CausalLmKeyLag {
                layer: selected,
                lag: 1,
            })
            .unwrap();
        let mut input = m.embed(&ids, shape).unwrap();
        for layer in 0..2 {
            let block = &m.blocks[layer];
            let raw = m
                .split_heads::<3>(
                    &block
                        .qkv
                        .forward(
                            &block
                                .first_norm
                                .forward(&input, m.config.norm_epsilon)
                                .unwrap(),
                        )
                        .unwrap(),
                    shape,
                )
                .unwrap();
            let mut gathered = raw[1].clone();
            if layer == selected {
                let dim = m.config.width / m.config.heads;
                for (new, old) in gathered
                    .chunks_exact_mut(4 * dim)
                    .zip(raw[1].chunks_exact(4 * dim))
                {
                    for j in 1..4 {
                        new[j * dim..(j + 1) * dim].copy_from_slice(&old[(j - 1) * dim..j * dim]);
                    }
                }
            }
            let answer = attention_reference(
                m.attention_spec(layer, shape).unwrap(),
                &raw[0],
                &gathered,
                &raw[2],
                None,
                None,
            )
            .unwrap();
            let actual = m
                .run_block(layer, input, shape, decoder::BlockPass::BackwardReplay)
                .unwrap();
            let cache = actual.cache.unwrap();
            assert_eq!(cache.qkv[1], raw[1]);
            assert_eq!(
                cache.attention_input,
                m.merge_heads([answer.as_slice()], shape).unwrap()
            );
            input = actual.output;
        }
    }
}

#[test]
fn lagged_architecture_has_distinct_owners_and_causal_prefill_append() {
    let base = load();
    let policy = CausalLmKeyLag { layer: 1, lag: 2 };
    let lagged = base.with_key_lag(policy).unwrap();
    let import = |cfg: &serde_json::Value| {
        NativeCausalLm::from_gpt2_json(
            &serde_json::to_vec(cfg).unwrap(),
            &archive(&parameters()),
            Default::default(),
        )
    };
    let mut cfg = config();
    cfg["spiraltorch_key_lag"] = json!({"layer":1,"lag":2});
    assert!(import(&cfg).is_err());
    cfg["model_type"] = json!(CausalLmKeyLag::MODEL_TYPE);
    assert_eq!(import(&cfg).unwrap().base_id, lagged.base_id);
    cfg["spiraltorch_key_lag"] = json!({"layer":1,"lag":0});
    assert!(import(&cfg).is_err());
    cfg.as_object_mut().unwrap().remove("spiraltorch_key_lag");
    assert!(import(&cfg).is_err());
    assert_ne!(lagged.base_id, base.base_id);
    assert_ne!(
        base.with_key_lag(CausalLmKeyLag { layer: 0, lag: 2 })
            .unwrap()
            .base_id,
        lagged.base_id
    );
    let lora = base.with_lora(1, 2, 1.).unwrap();
    assert!(
        lora.with_key_lag(policy).is_err()
            && lagged
                .with_lora_json(&lora.lora_to_json().unwrap())
                .is_err()
    );
    assert!(base
        .with_key_lag(CausalLmKeyLag { layer: 2, lag: 1 })
        .is_err());
    assert!(base
        .with_key_lag(CausalLmKeyLag { layer: 1, lag: 8 })
        .is_err());
    let q = [3, 1, 4, 2, 7];
    let changed = [3, 1, 4, 6, 5];
    let full = lagged.logits(&q, [1, 5], false).unwrap();
    let other = lagged.logits(&changed, [1, 5], false).unwrap();
    assert_eq!(&full.data()[..33], &other.data()[..33]);
    assert_ne!(full.data(), base.logits(&q, [1, 5], false).unwrap().data());
    for len in 1..=q.len() {
        let mut cache =
            NativeCausalLmContext::new(lagged.clone(), q[..len].to_vec(), 4096).unwrap();
        assert_eq!(
            cache.cache_values(),
            lagged.context_cache_values(len, 4096).unwrap()
        );
        for t in len..=q.len() {
            if t > len {
                cache.append(q[t - 1]).unwrap();
            }
            close(
                cache.logits_tensor().data(),
                &full.data()[(t - 1) * 11..t * 11],
            );
        }
    }
}

#[test]
fn fused_qkv_answer_gradient_agrees_with_physical_weight_differences() {
    let policy = CausalLmKeyLag { layer: 1, lag: 1 };
    let m = load().with_key_lag(policy).unwrap();
    let ids = [4, 8, 2, 1, 7, 1, 2, 3];
    let gold = [2, 7];
    for layer in 0..2 {
        let trainer = NativeCausalLmQkvAdamW::new(&m, layer, AdamWConfig::default()).unwrap();
        let (loss, g) = trainer.answer_gradients(&ids, [2, 4], &gold).unwrap();
        let nll = |model: &NativeCausalLm| {
            ids.chunks_exact(4)
                .zip(gold)
                .map(|(q, y)| model.next_token_score(q, y).unwrap().nll / 2.)
                .sum::<f64>()
        };
        assert!((f64::from(loss) - nll(&m)).abs() < 4e-7);
        for part in 0..3 {
            let grad = &g.data()[part * 16..(part + 1) * 16];
            let norm = grad
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(norm > 0.);
            let value = |h: f64| {
                let mut params = parameters();
                let weights = &mut params
                    .get_mut(&format!("h.{layer}.attn.c_attn.weight"))
                    .unwrap()
                    .1;
                for i in 0..4 {
                    for j in 0..4 {
                        weights[i * 12 + part * 4 + j] +=
                            (h * f64::from(grad[j * 4 + i]) / norm) as f32;
                    }
                }
                let b = NativeCausalLm::from_gpt2_json(
                    &serde_json::to_vec(&config()).unwrap(),
                    &archive(&params),
                    Default::default(),
                )
                .unwrap()
                .with_key_lag(policy)
                .unwrap();
                nll(&b)
            };
            let measured = (value(0.01) - value(-0.01)) / 0.02;
            assert!(
                (measured - norm).abs() < 2e-5 + 0.015 * norm,
                "layer {layer} part {part}: {norm} vs {measured}"
            );
        }
    }
}

#[test]
fn logical_qkv_batches_weight_the_remainder_and_advance_only_one_clock() {
    let base = load()
        .with_key_lag(CausalLmKeyLag { layer: 0, lag: 1 })
        .unwrap();
    let wide = artifact(&base, "transformer.h.1.attn", 0.2);
    let mut limited: serde_json::Value = serde_json::from_str(wide.to_json()).unwrap();
    limited["placements"][0]["recipe"]["max_values"] = json!(32);
    let narrow =
        GeometryAdapterArtifact::create_json(&serde_json::to_vec(&limited).unwrap()).unwrap();
    let trainer = |geo| {
        NativeCausalLmQkvAdamW::new(&base.with_geometry(geo).unwrap(), 0, AdamWConfig::default())
            .unwrap()
    };
    let dense = trainer(wide);
    let mut split = trainer(narrow);
    let ids = [3, 1, 4, 2, 1, 6, 5, 9, 2, 7, 3, 6, 3, 8, 9, 1, 9, 6, 2, 0];
    let ys = [7, 4, 2, 3, 10];
    let shape = [5, 4];
    let (loss, g) = dense.answer_gradients(&ids, shape, &ys).unwrap();
    let fast = dense
        .answer_gradients_microbatched(&ids, shape, &ys, 5)
        .unwrap();
    assert_eq!((loss, g.data()), (fast.0, fast.1.data()));
    assert!(split.answer_gradients(&ids, shape, &ys).is_err());
    for chunk in [1, 2] {
        let (cost, grad) = split
            .answer_gradients_microbatched(&ids, shape, &ys, chunk)
            .unwrap();
        assert!((cost - loss).abs() < 5e-7);
        assert!(grad
            .data()
            .iter()
            .zip(g.data())
            .all(|(a, b)| (a - b).abs() < 2e-6));
    }
    let origin = split.checkpoint().unwrap();
    let step = AdamWStep::new(0.01, 0.).unwrap();
    for (labels, chunk) in [([7, 4, 2, 3, 11], 2), (ys, 0), (ys, 3)] {
        assert!(split
            .step_answers_microbatched(&ids, shape, &labels, chunk, step)
            .is_err());
        assert_eq!(split.checkpoint().unwrap(), origin);
    }
    split
        .step_answers_microbatched(&ids, shape, &ys, 2, AdamWStep::new(0., 0.).unwrap())
        .unwrap();
    assert_eq!(split.checkpoint().unwrap(), origin);
    let report = split
        .step_answers_microbatched(&ids, shape, &ys, 2, step)
        .unwrap();
    assert_eq!((report.steps, split.steps()), (1, 1));
    assert_ne!(split.checkpoint().unwrap(), origin);
}

#[test]
fn fused_qkv_checkpoint_resume_and_frozen_prefix_are_atomic() {
    let base = load();
    let m = base
        .with_key_lag(CausalLmKeyLag { layer: 0, lag: 1 })
        .unwrap();
    let ids = [4, 8, 2, 1];
    let gold = [2];
    let cfg = AdamWConfig::default();
    let mut a = NativeCausalLmQkvAdamW::new(&m, 0, cfg).unwrap();
    let start = a.checkpoint().unwrap();
    let prefix = a.prefill(ids.to_vec(), 4096).unwrap();
    let step = AdamWStep::new(0.01, 0.).unwrap();
    a.step_answers(&ids, [1, 4], &gold, AdamWStep::new(0., 0.).unwrap())
        .unwrap();
    assert_eq!(a.checkpoint().unwrap(), start);
    a.step_answers(&ids, [1, 4], &gold, step).unwrap();
    let (bytes, pin) = a.checkpoint().unwrap();
    let mut b = NativeCausalLmQkvAdamW::restore(&m, &bytes, &pin).unwrap();
    assert_eq!(a.serving_id(), b.serving_id());
    assert_ne!(prefix.projection_id(), a.serving_id());
    assert_eq!(
        prefix.logits(),
        m.logits(&ids, [1, 4], true).unwrap().data()
    );
    assert!(NativeCausalLmQkvAdamW::restore(&base, &bytes, &pin).is_err());
    assert!(NativeCausalLmQkvAdamW::restore(&m, &bytes, &start.1).is_err());
    assert!(a.step_answers(&ids, [1, 4], &[99], step).is_err());
    assert_eq!(a.checkpoint().unwrap().1, pin);
    for owner in [&mut a, &mut b] {
        owner.step_answers(&ids, [1, 4], &gold, step).unwrap();
    }
    assert_eq!(a.checkpoint().unwrap(), b.checkpoint().unwrap());
}
