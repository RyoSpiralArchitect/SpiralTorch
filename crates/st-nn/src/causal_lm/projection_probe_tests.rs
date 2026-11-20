use super::*;
use sha2::{Digest, Sha256};
use CausalLmProjectionBasis::{Dense, Diagonal};
use CausalLmProjectionSite::{Key, Output, Query, Value};

fn five_token_readout(head: [f32; 20], norm: Option<([f32; 4], [f32; 4])>) -> NativeCausalLm {
    let mut c = config();
    c["vocab_size"] = json!(5);
    c["tie_word_embeddings"] = json!(false);
    let mut p = parameters();
    let (shape, values) = p.get_mut("wte.weight").unwrap();
    shape[0] = 5;
    values.truncate(20);
    if let Some((gain, bias)) = norm {
        p.get_mut("ln_f.weight").unwrap().1 = gain.to_vec();
        p.get_mut("ln_f.bias").unwrap().1 = bias.to_vec();
    }
    p.insert("lm_head.weight".into(), (vec![5, 4], head.to_vec()));
    NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&c).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap()
}

#[test]
fn binding_observation_ignores_common_vocabulary_weight() {
    let model = five_token_readout(
        std::array::from_fn(|i| if i % 4 == 0 { 67_108_864. } else { 0. }),
        None,
    );
    let prompts = vec![vec![4, 2, 1], vec![2, 1], vec![3, 1, 2], vec![1, 2]];
    let binding = CausalLmAnswerBinding {
        ids: [0, 1],
        quartets: &[[0, 1, 2, 3]],
        mix: 0.5,
    };
    for precision in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let m = model.with_readout_accumulator(precision);
        let p = NativeCausalLmRankOneProjection::new(&m, 1, Output, &[0., 1., 0., 0.], &[0.; 4])
            .unwrap();
        let out = p
            .observe_binding_gradients(&prompts, &[0, 1, 1, 0], binding)
            .unwrap();
        assert!((out.candidate_mass.loss - (2.5_f64).ln()).abs() < 1e-14);
        assert_eq!(out.candidate_mass.gradient, [0.; 4], "{precision:?}");
        assert_eq!(out.conditional.gradient, [0.; 4], "{precision:?}");
    }
}

#[test]
fn binding_observation_retains_saturated_mass_after_decoder_pullback() {
    let m = five_token_readout(
        [
            0., 0., 0., 0., 0., 0., 0., 0., -110., 1., 0., 0., -110., 1., 0., 0., -110., 1., 0., 0.,
        ],
        Some(([0., 1., 1., 1.], [1., 0., 0., 0.])),
    )
    .with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward);
    let prompts = vec![vec![4, 2, 1], vec![2, 1], vec![3, 1, 2], vec![1, 2]];
    let binding = CausalLmAnswerBinding {
        ids: [0, 1],
        quartets: &[[0, 1, 2, 3]],
        mix: 0.5,
    };
    let spec = CausalLmProjectionSpec {
        layer: 1,
        site: Output,
        basis: Diagonal,
    };
    let left = [0., 1., 0., 0.];
    let p =
        NativeCausalLmRankOneProjection::new(&m, spec.layer, spec.site, &left, &[0.; 4]).unwrap();
    let o = p
        .observe_binding_gradients(&prompts, &[0, 1, 1, 0], binding)
        .unwrap();
    let (mut loss, mut right) = (sum::Sum::default(), [0.; 4]);
    for prompt in &prompts {
        let z = p.last_logits(prompt).unwrap();
        assert_eq!(z[..2], [0.; 2]);
        assert!(z[2..].iter().all(|&x| x == z[2]));
        let ratio = 1.5 * f64::from(z[2]).exp();
        let share = ratio / (1. + ratio);
        assert!(share > 0. && share as f32 == 0.);
        loss.add(ratio.ln_1p());
        let (_, gradient) = m.fixed_left_contrast(prompt, 2, 0, spec, &left).unwrap();
        for (reference, normalized) in right.iter_mut().zip(gradient) {
            *reference += share * normalized / prompts.len() as f64;
        }
    }
    let loss = loss.value() / prompts.len() as f64;
    assert!((o.candidate_mass.loss / loss - 1.).abs() < 1e-12);
    assert!(right.iter().any(|x| x.abs() > 1e-51));
    for (actual, reference) in o.candidate_mass.gradient.iter().zip(right) {
        assert!((actual - reference).abs() <= reference.abs() * 1e-6);
    }
}

#[test]
fn prepared_conditional_batch_matches_guarded_and_raw_training_across_updates() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompts = vec![vec![4, 8, 2, 1], vec![8, 4, 1], vec![7, 1, 2, 3]];
    let pairs = [[2, 7], [7, 2], [7, 2]];
    let left = [0., 0.6, -0.8, 0.];
    let zero = [0.; 4];
    let new = |site| NativeCausalLmRankOneProjection::new(&m, 1, site, &left, &zero).unwrap();
    let mut commits = 0;
    for site in [Query, Key] {
        let a = new(site);
        assert!(a
            .prepare_conditional_answer_batch(&prompts, &pairs, 43)
            .is_err());
        assert!(a
            .prepare_conditional_answer_batch(&prompts, &pairs[..2], 44)
            .is_err());
        assert!(a
            .prepare_conditional_answer_batch(&prompts, &[[2, 7], [7, 7], [7, 2]], 44)
            .is_err());
        let prepared = a
            .prepare_conditional_answer_batch(&prompts, &pairs, 44)
            .unwrap();
        assert_eq!((prepared.len(), prepared.prefix_values()), (3, 44));
        assert!(!prepared.is_empty());
        let cfg = CausalLmConditionalAnswerConfig {
            length: 0.01,
            max_trials: 2,
            mass_inward_cosine: Some(0.05),
        };
        let mut unrelated = new(site);
        let owner = unrelated.projection_id();
        assert!(unrelated
            .step_prepared_conditional_answers(&prepared, cfg)
            .is_err());
        assert_eq!(owner, unrelated.projection_id());

        let guards: Vec<_> = prompts
            .iter()
            .map(|p| {
                let logits = a.last_logits(p).unwrap();
                if logits[2] > logits[7] {
                    [2, 7]
                } else {
                    [7, 2]
                }
            })
            .collect();
        let favorable = a
            .prepare_conditional_answer_batch(&prompts, &guards, 44)
            .unwrap();
        for (retained, cone) in [(false, false), (false, true), (true, false), (true, true)] {
            let (prepared, labels, cfg) = if cone {
                (&prepared, &pairs[..], cfg)
            } else {
                (
                    &favorable,
                    guards.as_slice(),
                    CausalLmConditionalAnswerConfig {
                        mass_inward_cosine: None,
                        ..cfg
                    },
                )
            };
            let (mut raw, mut cached) = (a.clone(), a.clone());
            let mut raw_guard = a
                .prepare_conditional_margin_and_mass_retention(&prompts, &guards, 0.02)
                .unwrap();
            let mut cache_guard = a
                .prepare_conditional_margin_and_mass_retention(&prompts, &guards, 0.02)
                .unwrap();
            for _ in 0..3 {
                let (mut cache, ordinary) = if retained {
                    (
                        serde_json::to_value(
                            cached
                                .step_prepared_conditional_answers_retaining(
                                    prepared,
                                    cfg,
                                    &mut cache_guard,
                                )
                                .unwrap(),
                        )
                        .unwrap(),
                        serde_json::to_value(
                            raw.step_conditional_answers_retaining(
                                &prompts,
                                labels,
                                cfg,
                                &mut raw_guard,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    )
                } else {
                    (
                        serde_json::to_value(
                            cached
                                .step_prepared_conditional_answers(prepared, cfg)
                                .unwrap(),
                        )
                        .unwrap(),
                        serde_json::to_value(
                            raw.step_conditional_answers(&prompts, labels, cfg).unwrap(),
                        )
                        .unwrap(),
                    )
                };
                let step = if retained {
                    &mut cache["step"]
                } else {
                    &mut cache
                };
                assert_eq!(
                    step.as_object_mut()
                        .unwrap()
                        .remove("reused_prepared_prefix_block_calls"),
                    Some(serde_json::json!(3))
                );
                commits += usize::from(step["committed"].as_bool().unwrap());
                assert_eq!(cache, ordinary);
                assert_eq!(cached.projection_id(), raw.projection_id());
                assert_eq!(cached.right(), raw.right());
            }
        }
    }
    assert!(commits >= 2);
}

#[test]
fn indexed_conditional_batches_keep_multiplicity_and_skip_real_qkv_calls() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn measure<T>(run: impl FnOnce() -> T) -> (T, usize) {
        let hits = Arc::new(AtomicUsize::new(0));
        let captured = hits.clone();
        let old = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            if event.op_name == "matmul_prepacked_bias"
                && event.data["inner"] == 4
                && event.data["cols"] == 12
            {
                captured.fetch_add(1, Ordering::Relaxed);
            }
        })));
        let result = run();
        st_tensor::set_thread_meta_observer(old);
        (result, hits.load(Ordering::Relaxed))
    }

    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let new = || {
        NativeCausalLmRankOneProjection::new(&m, 1, Query, &[0., 0.6, -0.8, 0.], &[0.; 4]).unwrap()
    };
    let base = new();
    let prompts = vec![vec![4, 8, 2, 1], vec![8, 4, 1], vec![7, 1, 2, 3]];
    let (bank, prep_calls) = measure(|| {
        base.prepare_conditional_answer_prefix_bank(&prompts, 44)
            .unwrap()
    });
    assert_eq!((bank.len(), bank.prefix_values(), prep_calls), (3, 44, 3));
    assert!(!bank.is_empty());
    assert!(base
        .prepare_conditional_answer_prefix_bank(&prompts, 43)
        .is_err());
    assert!(base
        .prepare_conditional_answer_prefix_bank(&[prompts[0].clone(), prompts[0].clone()], 44)
        .is_err());
    let config = CausalLmConditionalAnswerConfig {
        length: 0.01,
        max_trials: 2,
        mass_inward_cosine: None,
    };
    assert!(new()
        .step_indexed_conditional_answers(&bank, &[0], &[[2, 7]], config)
        .is_err());
    let (mut plain, mut indexed) = (base.clone(), base.clone());
    let owner = indexed.projection_id();
    for (indices, labels) in [
        (&[3][..], &[[2, 7]][..]),
        (&[0], &[[2, 2]]),
        (&[0, 1], &[[2, 7]]),
    ] {
        assert!(indexed
            .step_indexed_conditional_answers(&bank, indices, labels, config)
            .is_err());
        assert_eq!(indexed.projection_id(), owner);
    }
    let gold: Vec<[usize; 2]> = prompts
        .iter()
        .map(|p| {
            let logits = base.last_logits(p).unwrap();
            if logits[2] > logits[7] {
                [2, 7]
            } else {
                [7, 2]
            }
        })
        .collect();
    let mut a = base
        .prepare_conditional_margin_and_mass_retention(&prompts, &gold, 0.02)
        .unwrap();
    let mut b = base
        .prepare_conditional_margin_and_mass_retention(&prompts, &gold, 0.02)
        .unwrap();
    let mut saw_commit = false;
    for (round, selected) in [[2, 0, 2], [1, 2, 0], [0, 2, 1]].into_iter().enumerate() {
        let raw: Vec<_> = selected.iter().map(|&i| prompts[i].clone()).collect();
        let mut labels = selected.map(|i| gold[i]);
        if round == 0 {
            labels[2].reverse(); // The repeated prompt contributes a different pair.
        }
        let (x, raw_calls) = measure(|| {
            plain
                .step_conditional_answers_retaining(&raw, &labels, config, &mut a)
                .unwrap()
        });
        let (y, bank_calls) = measure(|| {
            indexed
                .step_indexed_conditional_answers_retaining(
                    &bank, &selected, &labels, config, &mut b,
                )
                .unwrap()
        });
        let mut report = serde_json::to_value(&y).unwrap();
        assert_eq!(
            report["step"]
                .as_object_mut()
                .unwrap()
                .remove("reused_prepared_prefix_block_calls"),
            Some(serde_json::json!(3))
        );
        assert_eq!(report, serde_json::to_value(&x).unwrap());
        assert_eq!(raw_calls, bank_calls + selected.len());
        assert_eq!(plain.projection_id(), indexed.projection_id());
        saw_commit |= y.step.committed;
    }
    assert!(saw_commit);
}

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
        if !r.committed || !r.trials[0].accepted || direct_margin(&plain) > 0. {
            continue;
        }
        let plain_owner = plain.projection_id();
        let bank = a
            .prepare_conditional_answer_prefix_bank(&prompts, 16)
            .unwrap();
        let mut indexed = a.clone();
        let mut indexed_guard = a
            .prepare_conditional_margin_retention(&prompts, &keep)
            .unwrap();
        let before = indexed.checkpoint_with_retention(&indexed_guard).unwrap();
        let veto = indexed
            .step_indexed_conditional_answers_retaining(
                &bank,
                &[0],
                &rival_objective,
                CausalLmConditionalAnswerConfig {
                    max_trials: 1,
                    ..cfg
                },
                &mut indexed_guard,
            )
            .unwrap();
        assert_eq!(veto.step.trials.len(), 1);
        assert_eq!(veto.step.trials[0].projection_id, plain_owner);
        assert!(!veto.step.committed);
        assert_eq!(veto.retention[0].as_ref().unwrap().nonpositive, 1);
        assert_eq!(
            indexed.checkpoint_with_retention(&indexed_guard).unwrap(),
            before
        );
        assert!(plain
            .step_conditional_answers_retaining(&prompts, &rival_objective, cfg, &mut stale)
            .is_err());
        assert_eq!(plain.projection_id(), plain_owner);
        let (bytes, _) = a.checkpoint_with_retention(&stale).unwrap();
        let mut record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        record["projection_id"] = serde_json::json!(plain_owner);
        record["right_bits"] = serde_json::json!(plain
            .right()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>());
        let forged = serde_json::to_vec(&record).unwrap();
        let digest = format!("sha256:{:x}", Sha256::digest(&forged));
        let restore =
            NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &forged, &digest)
                .unwrap();
        assert!(matches!(restore.restore(&prompts, &keep), Err(e)
            if e.to_string().contains("violates the saved guard")));
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
fn retained_checkpoint_replays_the_same_next_update() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let prompts = vec![vec![4, 8, 2, 1], vec![8, 4, 1], vec![7, 1, 2, 3]];
    let cfg = CausalLmConditionalAnswerConfig {
        length: 0.01,
        max_trials: 2,
        mass_inward_cosine: None,
    };
    for allowance in [None, Some(0.02)] {
        let mut a = NativeCausalLmRankOneProjection::new(
            &m,
            1,
            Output,
            &[0., 0.6, -0.8, 0.],
            &[0.01, -0.02, -0.0, 0.03],
        )
        .unwrap();
        assert_eq!(a.right()[2].to_bits(), (-0.0_f64).to_bits());
        let pairs: Vec<_> = prompts
            .iter()
            .map(|p| {
                let z = a.last_logits(p).unwrap();
                if z[2] > z[7] {
                    [2, 7]
                } else {
                    [7, 2]
                }
            })
            .collect();
        let mut guard = match allowance {
            Some(d) => a.prepare_conditional_margin_and_mass_retention(&prompts, &pairs, d),
            None => a.prepare_conditional_margin_retention(&prompts, &pairs),
        }
        .unwrap();
        assert!(
            a.step_conditional_answers_retaining(&prompts, &pairs, cfg, &mut guard)
                .unwrap()
                .step
                .committed
        );
        let (bytes, pin) = a.checkpoint_with_retention(&guard).unwrap();
        let admit = || NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &bytes, &pin);
        let changed = m.with_readout_accumulator(CausalLmReadoutAccumulator::F64);
        assert!(
            NativeCausalLmRankOneProjection::admit_retained_checkpoint(&changed, &bytes, &pin)
                .is_err()
        );
        let mut wrong_pin = pin.clone();
        wrong_pin.replace_range(7..8, if pin.as_bytes()[7] == b'0' { "1" } else { "0" });
        assert!(
            NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &bytes, &wrong_pin)
                .is_err()
        );
        for (key, val) in [
            ("schema", "spiraltorch.native_conditional_retention.v2"),
            ("roster_id", "sha256:not-a-roster-digest"),
        ] {
            let mut record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            record[key] = serde_json::json!(val);
            let invalid = serde_json::to_vec(&record).unwrap();
            let id = format!("sha256:{:x}", Sha256::digest(&invalid));
            assert!(
                NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &invalid, &id)
                    .is_err()
            );
        }
        let mut reversed = pairs.clone();
        reversed[0].reverse();
        assert!(admit().unwrap().restore(&prompts, &reversed).is_err());
        let restore = admit().unwrap();
        assert!(restore.roster_input_count(&[4, 9, 4]).is_err());
        let (mut b, mut restored) = restore.restore(&prompts, &pairs).unwrap();
        assert_eq!(
            b.checkpoint_with_retention(&restored).unwrap(),
            (bytes, pin)
        );
        assert_eq!(guard.baseline_margins(), restored.baseline_margins());
        let step = a
            .step_conditional_answers_retaining(&prompts, &pairs, cfg, &mut guard)
            .unwrap();
        let resumed = b
            .step_conditional_answers_retaining(&prompts, &pairs, cfg, &mut restored)
            .unwrap();
        assert!(step.step.committed);
        assert_eq!(
            serde_json::to_vec(&step).unwrap(),
            serde_json::to_vec(&resumed).unwrap()
        );
        assert_eq!(a.projection_id(), b.projection_id());
        assert_eq!(
            a.checkpoint_with_retention(&guard).unwrap(),
            b.checkpoint_with_retention(&restored).unwrap()
        );
        assert_eq!(
            a.last_logits(&prompts[0]).unwrap(),
            b.last_logits(&prompts[0]).unwrap()
        );
    }
}

#[test]
fn retained_checkpoint_pins_a_floor_below_the_probability_range() {
    let mut config = config();
    config["tie_word_embeddings"] = serde_json::json!(false);
    let mut weights = parameters();
    weights.get_mut("ln_f.weight").unwrap().1 = vec![0., 1., 0., 0.];
    weights.get_mut("ln_f.bias").unwrap().1 = vec![1., 0., 0., 0.];
    let mut head = vec![0.; 11 * 4];
    head[4] = -1.;
    head[8] = 800.;
    head[9] = 1.;
    weights.insert("lm_head.weight".into(), (vec![11, 4], head));
    let m = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config).unwrap(),
        &archive(&weights),
        Default::default(),
    )
    .unwrap();
    let projection = |right: &[f64]| {
        NativeCausalLmRankOneProjection::new(&m, 1, Output, &[1., 0., 0., 0.], right).unwrap()
    };
    let start = [0., -0., 0., 0.];
    let state = projection(&start);
    let prompts = vec![vec![1, 2, 3]];
    let pairs = [[0, 1]];
    let z = state.last_logits(&prompts[0]).unwrap();
    assert_eq!(&z[..2], &[0., -1.]);
    assert!(z[2] > 790.);
    let guard = state
        .prepare_conditional_margin_and_mass_retention(&prompts, &pairs, 0.)
        .unwrap();
    let (bytes, pin) = state.checkpoint_with_retention(&guard).unwrap();
    let (restored, restored_guard) =
        NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &bytes, &pin)
            .unwrap()
            .restore(&prompts, &pairs)
            .unwrap();
    assert_eq!(restored.right()[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(restored.projection_id(), state.projection_id());
    assert_ne!(
        restored.projection_id(),
        projection(&[0.; 4]).projection_id()
    );
    assert_eq!(
        restored.checkpoint_with_retention(&restored_guard).unwrap(),
        (bytes.clone(), pin)
    );
    let record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mass = &record["baseline"]["mass"];
    assert_eq!(mass["minimum_mean_pair_mass_bits"], 0);
    let log_floor = f64::from_bits(mass["minimum_log_mean_pair_mass_bits"].as_u64().unwrap());
    assert!(log_floor < -790.);
    let mut weaker = record.clone();
    weaker["baseline"]["mass"]["minimum_log_mean_pair_mass_bits"] =
        serde_json::json!(log_floor.to_bits() + 1);
    let altered = serde_json::to_vec(&weaker).unwrap();
    let id = format!("sha256:{:x}", Sha256::digest(&altered));
    let plan =
        NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &altered, &id).unwrap();
    assert!(matches!(plan.restore(&prompts, &pairs), Err(e)
        if e.to_string().contains("baseline does not reproduce")));

    let regressed = [-0.5, 0.5]
        .into_iter()
        .map(|delta| {
            let mut right = start;
            right[1] = delta;
            projection(&right)
        })
        .find(|p| p.last_logits(&prompts[0]).unwrap()[2] > z[2] + 0.001)
        .expect("fixture must reduce the log pair mass without changing the sign");
    let scores = regressed.score_answer_pairs(&prompts, &pairs).unwrap();
    assert_eq!(scores[0].gold_minus_alternative, 1.);
    assert_eq!(
        (-scores[0].gold_nll).exp() + (-scores[0].alternative_nll).exp(),
        0.
    );
    let mut endpoint = record;
    endpoint["projection_id"] = serde_json::json!(regressed.projection_id());
    endpoint["right_bits"] = serde_json::json!(regressed
        .right()
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>());
    let altered = serde_json::to_vec(&endpoint).unwrap();
    let id = format!("sha256:{:x}", Sha256::digest(&altered));
    let plan =
        NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &altered, &id).unwrap();
    assert!(matches!(plan.restore(&prompts, &pairs), Err(e)
        if e.to_string().contains("violates the saved guard")));
}

#[test]
fn retained_checkpoint_requires_the_subnormal_residual_after_near_cancellation() {
    for outside_logit in [700., 709.] {
        check_retained_checkpoint_near_cancellation(outside_logit);
    }
}

fn check_retained_checkpoint_near_cancellation(outside_logit: f32) {
    let mut config = config();
    config["tie_word_embeddings"] = serde_json::json!(false);
    let mut weights = parameters();
    weights.get_mut("ln_f.weight").unwrap().1 = vec![0.; 4];
    weights.get_mut("ln_f.bias").unwrap().1 = vec![1., 0., 0., 0.];
    let mut head = vec![0.; 11 * 4];
    head[4] = -1.;
    head[8] = outside_logit;
    weights.insert("lm_head.weight".into(), (vec![11, 4], head));
    let m = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config).unwrap(),
        &archive(&weights),
        Default::default(),
    )
    .unwrap();
    let state =
        NativeCausalLmRankOneProjection::new(&m, 1, Output, &[1., 0., 0., 0.], &[0.; 4]).unwrap();
    let prompts = vec![vec![1, 2, 3]];
    let pairs = [[0, 1]];
    let original = state
        .prepare_conditional_margin_and_mass_retention(&prompts, &pairs, 0.)
        .unwrap();
    let baseline = original.baseline_mean_pair_mass().unwrap();
    let drop = f64::from_bits(baseline.to_bits() - 1);
    let guard = state
        .prepare_conditional_margin_and_mass_retention(&prompts, &pairs, drop)
        .unwrap();
    let minimum = guard.minimum_mean_pair_mass().unwrap();
    assert!(minimum.is_subnormal());
    assert_eq!(guard.minimum_log_mean_pair_mass(), Some(minimum.ln()));
    let (bytes, pin) = state.checkpoint_with_retention(&guard).unwrap();
    let (resumed, restored) =
        NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &bytes, &pin)
            .unwrap()
            .restore(&prompts, &pairs)
            .unwrap();
    assert_eq!(
        resumed.checkpoint_with_retention(&restored).unwrap(),
        (bytes.clone(), pin)
    );

    let mut old: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    old["baseline"]["mass"]["minimum_log_mean_pair_mass_bits"] = serde_json::Value::Null;
    let old = serde_json::to_vec(&old).unwrap();
    let old_pin = format!("sha256:{:x}", Sha256::digest(&old));
    let plan =
        NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &old, &old_pin).unwrap();
    assert!(matches!(plan.restore(&prompts, &pairs), Err(e)
        if e.to_string().contains("baseline does not reproduce")));
}

#[test]
fn retention_rechecks_anchor_limits_on_an_identical_receiving_projection() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let make_projection = |m| {
        NativeCausalLmRankOneProjection::new(m, 1, Output, &[0., 0.6, -0.8, 0.], &[0.; 4]).unwrap()
    };
    let wide = make_projection(&m);
    let anchors = vec![vec![4, 8, 2, 1]];
    let scores = wide.score_answer_pairs(&anchors, &[[2, 7]]).unwrap();
    let keep = if scores[0].gold_minus_alternative > 0. {
        [[2, 7]]
    } else {
        [[7, 2]]
    };
    let mut guard = wide
        .prepare_conditional_margin_and_mass_retention(&anchors, &keep, 0.)
        .unwrap();
    let mut limited = m.clone();
    limited.limits.max_attention_pairs = limited.config.layers * limited.config.heads * 4;
    let mut narrow = make_projection(&limited);
    let prompts = vec![vec![4, 8]];
    let cfg = CausalLmConditionalAnswerConfig {
        length: 0.04,
        max_trials: 1,
        mass_inward_cosine: None,
    };
    let id = wide.projection_id();
    assert_eq!(narrow.projection_id(), id);
    assert!(narrow.score_answer_pairs(&anchors, &keep).is_err());
    assert!(narrow.score_answer_pairs(&prompts, &keep).is_ok());
    assert!(narrow
        .step_conditional_answers_retaining(&prompts, &keep, cfg, &mut guard)
        .unwrap_err()
        .to_string()
        .contains("attention pairs"));
    assert_eq!(narrow.projection_id(), id);
    let mut wide_again = wide.clone();
    assert!(wide_again
        .step_conditional_answers_retaining(&prompts, &keep, cfg, &mut guard)
        .is_ok());
}

#[test]
fn sign_only_checkpoint_refuses_when_a_nonanchor_cannot_be_restored() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let make = |m| {
        NativeCausalLmRankOneProjection::new(m, 1, Output, &[0., 1., 0., 0.], &[0.; 4]).unwrap()
    };
    let wide = make(&m);
    let rows = vec![vec![4, 8], vec![4, 8, 2, 1]];
    let scores = wide.score_answer_pairs(&rows, &[[2, 7]; 2]).unwrap();
    let mut pairs = [[2, 7]; 2];
    for (i, s) in scores.iter().enumerate() {
        assert_ne!(s.gold_minus_alternative, 0.);
        if (s.gold_minus_alternative > 0.) != (i == 0) {
            pairs[i].reverse();
        }
    }
    let guard = wide
        .prepare_conditional_margin_retention(&rows, &pairs)
        .unwrap();
    assert_eq!(guard.protected_count(), 1);
    let mut limited = m.clone();
    limited.limits.max_attention_pairs = limited.config.layers * limited.config.heads * 4;
    let narrow = make(&limited);
    assert_eq!(narrow.projection_id(), wide.projection_id());
    assert!(guard.verify_owner(&narrow).is_ok());
    assert!(narrow
        .checkpoint_with_retention(&guard)
        .unwrap_err()
        .to_string()
        .contains("attention pairs"));
    let (bytes, pin) = wide.checkpoint_with_retention(&guard).unwrap();
    let (restored, guard) =
        NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &bytes, &pin)
            .unwrap()
            .restore(&rows, &pairs)
            .unwrap();
    assert_eq!(
        restored.checkpoint_with_retention(&guard).unwrap(),
        (bytes, pin)
    );
}

#[test]
fn guarded_checkpoint_pins_the_underflowed_outside_ceiling() {
    let m = five_token_readout(
        [
            0., 0., 0., 0., -1., 0., 0., 0., -800., 0., 0., 0., -800., 0., 0., 0., -801., 0., 0.,
            0.,
        ],
        Some(([0.; 4], [1., 0., 0., 0.])),
    );
    let p =
        NativeCausalLmRankOneProjection::new(&m, 1, Output, &[0., 1., 0., 0.], &[0.; 4]).unwrap();
    let rows = vec![vec![1, 2]];
    let pairs = [[0, 1]];
    let guard = p
        .prepare_conditional_margin_and_mass_retention(&rows, &pairs, 0.)
        .unwrap();
    let (bytes, pin) = p.checkpoint_with_retention(&guard).unwrap();
    let mut record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let log_bits = record["baseline"]["mass"]["maximum_log_mean_outside_mass_bits"]
        .as_u64()
        .unwrap();
    assert!(f64::from_bits(log_bits) < -790.);
    let (restored, guard) =
        NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &bytes, &pin)
            .unwrap()
            .restore(&rows, &pairs)
            .unwrap();
    assert_eq!(
        restored.checkpoint_with_retention(&guard).unwrap(),
        (bytes, pin)
    );
    record["baseline"]["mass"]
        .as_object_mut()
        .unwrap()
        .remove("maximum_log_mean_outside_mass_bits");
    let legacy = serde_json::to_vec(&record).unwrap();
    let legacy_pin = format!("sha256:{:x}", Sha256::digest(&legacy));
    let plan = NativeCausalLmRankOneProjection::admit_retained_checkpoint(&m, &legacy, &legacy_pin)
        .unwrap();
    assert!(
        matches!(plan.restore(&rows, &pairs), Err(e) if e.to_string().contains("baseline does not reproduce"))
    );
}

#[test]
fn both_mass_guards_reject_an_underflowed_regression_on_the_same_real_candidate() {
    let make = |outside: f32| {
        let m = five_token_readout(
            [
                0., 1., 0., 0., 0., -1., 0., 0., -800., outside, 0., 0., -800., outside, 0., 0.,
                -800., outside, 0., 0.,
            ],
            Some(([0., 1., 1., 1.], [1., 0., 0., 0.])),
        );
        NativeCausalLmRankOneProjection::new(&m, 1, Output, &[0., 1., 0., 0.], &[0.; 4]).unwrap()
    };
    let rows = vec![vec![4, 2, 1]];
    let probe = make(2.);
    let gap = probe.score_answer_pairs(&rows, &[[0, 1]]).unwrap()[0].gold_minus_alternative;
    assert_ne!(gap, 0.);
    // Improving the oriented margin increases the outside log-odds:
    // the outside logit has slope 2 along the gold coordinate, while
    // logsumexp of the two inside logits has slope at most 1.
    let (start, pairs) = if gap > 0. {
        (probe, [[0, 1]])
    } else {
        (make(-2.), [[1, 0]])
    };
    let cfg = CausalLmConditionalAnswerConfig {
        length: 0.01,
        max_trials: 1,
        mass_inward_cosine: None,
    };
    let id = start.projection_id();
    let mut direct = start.clone();
    let accepted = direct.step_conditional_answers(&rows, &pairs, cfg).unwrap();
    assert!(accepted.committed);
    assert!(accepted.after.conditional_ce < accepted.before.conditional_ce);
    assert_eq!(
        (
            accepted.before.candidate_mass_nll,
            accepted.after.candidate_mass_nll
        ),
        (0., 0.)
    );
    assert!(
        accepted.after.log_candidate_mass_nll.unwrap()
            > accepted.before.log_candidate_mass_nll.unwrap()
    );

    let mut batch = start.clone();
    let held = batch
        .step_conditional_answers(
            &rows,
            &pairs,
            CausalLmConditionalAnswerConfig {
                mass_inward_cosine: Some(0.),
                ..cfg
            },
        )
        .unwrap();
    assert!(!held.committed);
    assert_eq!(held.mass_gradient_l2, Some(0.));
    assert_eq!(
        held.trials[0].projection_id,
        accepted.trials[0].projection_id
    );
    assert_eq!(held.trials[0].point, accepted.after);
    assert_eq!(batch.projection_id(), id);

    let mut roster = start;
    let mut guard = roster
        .prepare_conditional_margin_and_mass_retention(&rows, &pairs, 0.)
        .unwrap();
    let checkpoint = roster.checkpoint_with_retention(&guard).unwrap();
    let held = roster
        .step_conditional_answers_retaining(&rows, &pairs, cfg, &mut guard)
        .unwrap();
    assert!(!held.step.committed);
    assert_eq!(
        held.step.trials[0].projection_id,
        accepted.trials[0].projection_id
    );
    let check = held.retention[0].as_ref().unwrap();
    assert_eq!(check.nonpositive, 0);
    let outside = check.mass.as_ref().unwrap().log_outside.as_ref().unwrap();
    assert!(outside.log_mean_outside_mass.unwrap() > outside.maximum_log_mean_outside_mass);
    assert_eq!(
        roster.checkpoint_with_retention(&guard).unwrap(),
        checkpoint
    );
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
