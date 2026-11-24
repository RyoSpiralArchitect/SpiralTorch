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
fn cached_suffix_matches_all_splits_and_preserves_forks_and_final_only_readout() {
    let ids = [2, 4, 1, 8, 1, 5, 3];
    let base = load();
    for m in [
        base.clone(),
        base.with_key_lag(CausalLmKeyLag { layer: 1, lag: 2 })
            .unwrap(),
    ] {
        let m = m
            .with_geometry(artifact(&m, "transformer.h.0.attn", 0.3))
            .unwrap();
        for split in 1..ids.len() {
            let root = NativeCausalLmContext::new(m.clone(), ids[..split].to_vec(), 1024).unwrap();
            let start = root.logit_values().to_vec();
            decoder::work_counts::take();
            let child = root.with_suffix(&ids[split..]).unwrap();
            assert_eq!(decoder::work_counts::take(), [0, 0, 1]);
            let mut serial = root.clone();
            for &id in &ids[split..] {
                serial.append(id).unwrap();
            }
            assert_eq!(decoder::work_counts::take(), [0, 0, ids.len() - split]);
            close(child.logit_values(), serial.logit_values());
            close(
                child.logit_values(),
                m.next_token_logits(&ids).unwrap().data(),
            );
            assert_eq!(child.token_ids(), &ids);
            assert_eq!(child.cache_values(), serial.cache_values());
            assert_eq!(child.geometry_artifact_id(), root.geometry_artifact_id());
            for layer in 0..m.config.layers {
                for pos in 0..ids.len() {
                    let key = child.key_row(layer, pos).unwrap();
                    close(&key, &serial.key_row(layer, pos).unwrap());
                    if pos < split {
                        assert_eq!(key, root.key_row(layer, pos).unwrap());
                    }
                }
            }
            for t in [0, 6] {
                let branch = child.with_suffix(&[t]).unwrap();
                let mut prompt = ids.to_vec();
                prompt.push(t);
                close(
                    branch.logit_values(),
                    m.next_token_logits(&prompt).unwrap().data(),
                );
            }
            assert_eq!(root.token_ids(), &ids[..split]);
            assert_eq!(root.logit_values(), start);
        }
    }
}

#[test]
fn cached_suffix_admits_rectangular_work_and_refuses_before_a_head_or_partial_commit() {
    let m = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&parameters()),
        CausalLmLimits {
            max_attention_pairs: 40,
            ..Default::default()
        },
    )
    .unwrap();
    let root = NativeCausalLmContext::new(m.clone(), vec![1, 2, 3], 1024).unwrap();
    let ok = root.with_suffix(&[4, 5]).unwrap(); // 2 layers * 2 heads * 2 new queries * 5 keys
    assert!(m.next_token_logits(&[1, 2, 3, 4, 5]).is_err());
    close(
        ok.logit_values(),
        load().next_token_logits(&[1, 2, 3, 4, 5]).unwrap().data(),
    );
    decoder::work_counts::take();
    for bad in [&[][..], &[4, 5, 6], &[4, 11]] {
        assert!(root.with_suffix(bad).is_err());
    }
    assert!(root.suffix_input_count(usize::MAX).is_err());
    assert_eq!(decoder::work_counts::take(), [0; 3]);
    assert_eq!(root.token_ids(), &[1, 2, 3]);
    assert_eq!(
        root.with_suffix(&[4, 5]).unwrap().logit_values(),
        ok.logit_values()
    );
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
    assert!(context.with_suffix(&[10, 5]).is_err());
    assert_eq!(context.token_ids(), unchanged.token_ids());
    assert_eq!(context.logit_values(), unchanged.logit_values());
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

fn branching_lines() -> Vec<Vec<usize>> {
    vec![
        vec![1],
        vec![1, 2, 3],
        vec![1, 2, 4],
        vec![1, 5],
        vec![7, 8],
    ]
}

#[test]
fn streamed_readout_tiles_do_not_redecode_attention_or_relax_full_vocabulary_work() {
    let mut m = load();
    m.limits.max_logit_elements = 88;
    let root = NativeCausalLmContext::new(m, vec![1, 2], 192).unwrap();
    let lines = branching_lines();
    let baseline = root.score_lines(&lines, 0).unwrap();
    let unchanged = root.logit_values().to_vec();
    for (cap, batches, peak) in [(11, 7, 11), (21, 7, 11), (22, 6, 22), (88, 6, 22)] {
        decoder::work_counts::take();
        decoder::work_counts::take_suffixes();
        let r = root
            .score_lines_with_tile_limit(&lines, 0, Some(cap))
            .unwrap();
        assert_eq!(decoder::work_counts::take(), [0, 0, batches]);
        assert_eq!(decoder::work_counts::take_suffixes(), (6, 7));
        assert_eq!(
            (
                r.decoder_chunks,
                r.readout_batches,
                r.peak_projected_logit_values
            ),
            (6, batches, peak)
        );
        assert_eq!(
            (
                r.scanned_logit_values,
                r.retained_edge_logits,
                r.peak_fork_cache_values
            ),
            (88, 12, 192)
        );
        assert_eq!(
            serde_json::to_vec(&r.scores).unwrap(),
            serde_json::to_vec(&baseline.scores).unwrap()
        );
    }
    for cap in [0, 10, 89, usize::MAX] {
        assert!(root
            .score_lines_with_tile_limit(&lines, 0, Some(cap))
            .is_err());
    }
    let mut too_many = lines.clone();
    too_many.push(vec![9]); // nine prefix rows; one-row tiles must not bypass the eight-row ceiling.
    assert!(root
        .score_lines_with_tile_limit(&too_many, 0, Some(11))
        .is_err());
    assert_eq!(decoder::work_counts::take(), [0; 3]);
    assert_eq!(root.logit_values(), unchanged);
    assert_eq!(decoder::work_counts::take_suffixes(), (0, 0));
    let r = root
        .score_lines_with_tile_limit(&[vec![1, 2, 3, 4, 5]], 0, Some(22))
        .unwrap();
    assert_eq!(
        (r.decoder_chunks, r.readout_batches, r.scanned_logit_values),
        (1, 3, 66)
    );
    assert_eq!(decoder::work_counts::take_suffixes(), (1, 5));
}
#[test]
fn cached_lines_share_prefixes_and_charge_each_stop_at_its_own_position() {
    let m = load();
    let lines = branching_lines();
    let prompt = [1, 2];
    let root = NativeCausalLmContext::new(m.clone(), prompt.to_vec(), 192).unwrap();
    let full = m.score_lines(&prompt, &lines, 0).unwrap();
    decoder::work_counts::take();
    let r = root.score_lines(&lines, 0).unwrap();
    assert_eq!(decoder::work_counts::take(), [0, 0, 6]);
    assert_eq!(
        (
            r.cached_prompt_tokens,
            r.decoded_content_tokens,
            r.decoder_chunks,
            r.peak_fork_cache_values
        ),
        (2, 7, 6, 192)
    );
    let s = r.scores;
    assert_eq!((s.prefix_nodes, s.decoder_lanes, s.stop_id), (8, 4, 0));
    assert_eq!(s.winner_index, full.winner_index);
    assert_eq!(s.shared_content_tokens, full.shared_content_tokens);
    let near = |a: f64, b: f64| assert!((a - b).abs() < 2e-6, "{a} vs {b}");
    near(s.menu_set_nll, full.menu_set_nll);
    for (i, line) in lines.iter().enumerate() {
        assert_eq!(s.target_counts[i], line.len() + 1);
        near(s.line_nlls[i], full.line_nlls[i]);
        near(s.conditional_nlls[i], full.conditional_nlls[i]);
        let mut prefix = prompt.to_vec();
        let mut total = 0.;
        for (j, id) in line.iter().copied().chain([0]).enumerate() {
            let logits = m.next_token_logits(&prefix).unwrap();
            let v = logits.data();
            let max = f64::from(v.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            let cost = v
                .iter()
                .map(|&x| (f64::from(x) - max).exp())
                .sum::<f64>()
                .ln()
                + max
                - f64::from(v[id]);
            near(s.token_nlls[i][j], cost);
            total += cost;
            prefix.push(id);
        }
        near(s.line_nlls[i], total);
    }
    assert_eq!(root.token_ids(), &prompt);
}

#[test]
fn cached_lines_preflight_every_branch_and_bound_persistent_fork_workspace() {
    let lines = branching_lines();
    for (cache, pairs, logits) in [(191, 32, 88), (192, 31, 88), (192, 32, 87), (192, 32, 88)] {
        let mut m = load();
        m.limits.max_attention_pairs = pairs;
        m.limits.max_logit_elements = logits;
        let root = NativeCausalLmContext::new(m.clone(), vec![1, 2], cache).unwrap();
        let before = root.logit_values().to_vec();
        decoder::work_counts::take();
        let r = root.score_lines(&lines, 0);
        if (cache, pairs, logits) == (192, 32, 88) {
            assert!(r.is_ok(), "{r:?}");
            assert_eq!(decoder::work_counts::take(), [0, 0, 6]);
            assert!(m.score_lines(&[1, 2], &lines, 0).is_err());
        } else {
            assert!(r.is_err());
            assert_eq!(decoder::work_counts::take(), [0; 3]);
        }
        assert_eq!(root.logit_values(), before);
        assert_eq!(root.token_ids(), [1, 2]);
    }
}

#[test]
fn cached_lines_validate_support_and_do_not_feed_a_stop_past_position_capacity() {
    let root = NativeCausalLmContext::new(load(), vec![1; 7], 128).unwrap();
    for bad in [
        vec![],
        vec![vec![]],
        vec![vec![0]],
        vec![vec![11]],
        vec![vec![1], vec![1]],
        vec![vec![2], vec![1]],
        vec![vec![1, 2]],
        vec![vec![1]; 33],
    ] {
        assert!(root.score_lines(&bad, 0).is_err());
    }
    for len in [0, 4096, usize::MAX] {
        assert!(root.admit_line_lengths(&[len], 0).is_err());
    }
    assert!(root.score_lines(&[vec![1]], 11).is_err());
    let r = root.score_lines(&[vec![1]], 0).unwrap();
    assert_eq!(r.scores.target_counts, [2]);
    // The scalar path sum and the trie's reverse reduction may round one ULP apart.
    assert!(
        (r.scores.line_nlls[0] - r.scores.menu_set_nll).abs()
            <= 2. * f64::EPSILON * r.scores.menu_set_nll
    );
    assert_eq!(r.scores.conditional_nlls, [0.]);
    assert_eq!(root.token_ids().len(), 7);
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

#[test]
fn learned_lora_prefill_and_appends_match_full_decoder_and_keep_snapshot() {
    let base = load();
    let initial = base.with_lora(0, 2, 4.).unwrap();
    let gradients = initial.lora_loss_gradients(&[1, 2, 3, 4], [1, 4]).unwrap();
    let model = initial.step_lora_sgd(&gradients, 0.3).unwrap();
    assert_ne!(initial.lora_artifact_id(), model.lora_artifact_id());
    let restored = base.with_lora_json(&model.lora_to_json().unwrap()).unwrap();
    assert_eq!(model.lora_artifact_id(), restored.lora_artifact_id());
    let mut context = NativeCausalLmContext::new(restored.clone(), vec![1, 2], 1024).unwrap();
    let snapshot = context.clone();
    assert_eq!(context.lora_artifact_id(), restored.lora_artifact_id());
    assert_ne!(
        restored.logits(&[1, 2], [1, 2], true).unwrap().data(),
        base.logits(&[1, 2], [1, 2], true).unwrap().data()
    );
    for token in [3, 4, 5, 6, 7, 8] {
        close(
            &context.logits().unwrap().read_values().unwrap(),
            restored
                .logits(context.token_ids(), [1, context.token_ids().len()], true)
                .unwrap()
                .data(),
        );
        context.append(token).unwrap();
    }
    close(
        &context.logits().unwrap().read_values().unwrap(),
        restored
            .logits(&[1, 2, 3, 4, 5, 6, 7, 8], [1, 8], true)
            .unwrap()
            .data(),
    );
    let later = restored
        .step_lora_sgd(
            &restored.lora_loss_gradients(&[4, 3, 2, 1], [1, 4]).unwrap(),
            0.3,
        )
        .unwrap();
    assert_ne!(later.lora_artifact_id(), snapshot.lora_artifact_id());
    assert_eq!(snapshot.lora_artifact_id(), model.lora_artifact_id());
    assert_eq!(snapshot.token_ids(), [1, 2]);
    close(
        &snapshot.logits().unwrap().read_values().unwrap(),
        model.logits(&[1, 2], [1, 2], true).unwrap().data(),
    );
}
