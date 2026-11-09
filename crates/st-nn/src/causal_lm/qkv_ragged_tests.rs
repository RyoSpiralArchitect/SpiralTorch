use super::*;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};

fn prompts() -> (Vec<Vec<usize>>, Vec<usize>) {
    (
        vec![
            vec![3, 1],
            vec![4, 2, 0, 6],
            vec![1, 8, 3, 7],
            vec![6],
            vec![7, 2, 1, 9, 8],
            vec![5, 6, 3, 2, 1],
            vec![4, 7, 2, 6, 3],
            vec![8, 1, 5],
        ],
        vec![2, 7, 4, 1, 5, 2, 8, 6],
    )
}

fn adapted(lag: usize) -> NativeCausalLm {
    let base = load();
    let model = if lag == 0 {
        base
    } else {
        base.with_key_lag(CausalLmKeyLag { layer: 0, lag }).unwrap()
    };
    model
        .with_lora(1, 2, 1.)
        .unwrap()
        .with_geometry(artifact(&model, "transformer.h.1.attn", 0.2))
        .unwrap()
}

#[test]
fn ragged_qkv_matches_individual_unpadded_prefixes_not_token_weighting() {
    let (owned, labels) = prompts();
    let qs = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    for lag in [0, 1] {
        let trainer =
            NativeCausalLmQkvAdamW::new(&adapted(lag), 0, AdamWConfig::default()).unwrap();
        let mut reference = vec![0.; 48];
        let mut loss = 0.;
        let mut token_loss = 0.;
        for (q, y) in qs.iter().zip(&labels) {
            let (v, g) = trainer.answer_gradients(q, [1, q.len()], &[*y]).unwrap();
            loss += f64::from(v) / qs.len() as f64;
            token_loss += f64::from(v) * q.len() as f64;
            for (a, &x) in reference.iter_mut().zip(g.data()) {
                *a += f64::from(x) / qs.len() as f64;
            }
        }
        token_loss /= qs.iter().map(|q| q.len()).sum::<usize>() as f64;
        assert!(
            (token_loss - loss).abs() > 1e-6,
            "fixture must distinguish answer and prefix-token weights"
        );
        for max_rows in [1, 2, 3, 32] {
            let (v, g) = trainer
                .answer_sequence_gradients(&qs, &labels, max_rows)
                .unwrap();
            assert!((f64::from(v) - loss).abs() < 7e-7);
            assert!(g
                .data()
                .iter()
                .zip(&reference)
                .all(|(&a, &b)| (f64::from(a) - b).abs() < 2e-6));
        }
    }
}

#[test]
fn equal_length_sequence_api_keeps_existing_microbatch_bits() {
    let trainer = NativeCausalLmQkvAdamW::new(&adapted(1), 0, AdamWConfig::default()).unwrap();
    let ids = [2, 4, 6, 1, 3, 8, 7, 9, 1, 6, 2, 5, 4, 2, 8];
    let qs = ids.chunks_exact(3).collect::<Vec<_>>();
    let labels = [7, 3, 5, 9, 1];
    for max_rows in [1, 2, 3, 5, 32] {
        let (v, a) = trainer
            .answer_gradients_microbatched(&ids, [5, 3], &labels, max_rows)
            .unwrap();
        let (w, b) = trainer
            .answer_sequence_gradients(&qs, &labels, max_rows)
            .unwrap();
        assert_eq!(v.to_bits(), w.to_bits());
        assert_eq!(a, b);
    }
}

#[test]
fn ragged_qkv_update_has_one_clock_and_exact_continuation() {
    let model = adapted(1);
    let cfg = AdamWConfig::default();
    let mut a = NativeCausalLmQkvAdamW::new(&model, 0, cfg).unwrap();
    let (owned, labels) = prompts();
    let qs = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let origin = a.checkpoint().unwrap();
    a.step_answer_sequences(&qs, &labels, 2, AdamWStep::new(0., 0.).unwrap())
        .unwrap();
    assert_eq!(a.checkpoint().unwrap(), origin);
    let prefix = a.prefill(owned[0].clone(), 4096).unwrap();
    let step = AdamWStep::new(0.01, 0.02).unwrap();
    let (_, g) = a.answer_sequence_gradients(&qs, &labels, 2).unwrap();
    let expected = a
        .weights()
        .iter()
        .zip(g.data())
        .map(|(&w, &x)| step.candidate(cfg, w, x, 0., 0., 0).unwrap().value)
        .collect::<Vec<_>>();
    let report = a.step_answer_sequences(&qs, &labels, 2, step).unwrap();
    assert_eq!((a.steps(), report.steps), (1, 1));
    assert_eq!(a.weights(), expected);
    assert_ne!(prefix.projection_id(), a.serving_id());
    assert_eq!(
        prefix.logits(),
        model
            .logits(&owned[0], [1, owned[0].len()], true)
            .unwrap()
            .data()
    );
    let (checkpoint, pin) = a.checkpoint().unwrap();
    let mut b = NativeCausalLmQkvAdamW::restore(&model, &checkpoint, &pin).unwrap();
    a.step_answer_sequences(&qs, &labels, 2, step).unwrap();
    b.step_answer_sequences(&qs, &labels, 2, step).unwrap();
    assert_eq!(a.steps(), 2);
    assert_eq!(a.checkpoint().unwrap(), b.checkpoint().unwrap());
}

#[test]
fn ragged_qkv_rejects_bad_tails_and_large_rosters_without_mutating_state() {
    let mut a = NativeCausalLmQkvAdamW::new(&adapted(1), 0, AdamWConfig::default()).unwrap();
    let origin = a.checkpoint().unwrap();
    let step = AdamWStep::new(0.01, 0.).unwrap();
    for tail in [vec![], vec![99], vec![1; 9]] {
        let valid = [1, 2];
        assert!(a
            .step_answer_sequences(&[&valid, &tail], &[2, 3], 1, step)
            .is_err());
        assert_eq!(a.checkpoint().unwrap(), origin);
    }
    let valid = [1, 2];
    for (qs, gold, cap) in [
        (vec![], vec![], 1),
        (vec![&valid[..]], vec![], 1),
        (vec![&valid[..]], vec![99], 1),
        (vec![&valid[..]], vec![2], 0),
        (
            vec![&valid[..]; NativeCausalLmQkvAdamW::MAX_ANSWER_SEQUENCES + 1],
            vec![2; NativeCausalLmQkvAdamW::MAX_ANSWER_SEQUENCES + 1],
            4,
        ),
    ] {
        assert!(a.step_answer_sequences(&qs, &gold, cap, step).is_err());
        assert_eq!(a.checkpoint().unwrap(), origin);
    }
    let widths = (0..1025).map(|i| vec![1; 1 + i % 2]).collect::<Vec<_>>();
    let qs = widths.iter().map(Vec::as_slice).collect::<Vec<_>>();
    assert!(a
        .step_answer_sequences(&qs, &vec![2; qs.len()], 32, step)
        .is_err());
    assert_eq!(a.checkpoint().unwrap(), origin);
}

#[test]
fn ragged_qkv_checks_each_real_group_against_geometry_budget() {
    let base = load();
    let mut record = serde_json::from_str::<serde_json::Value>(
        artifact(&base, "transformer.h.1.attn", 0.2).to_json(),
    )
    .unwrap();
    record["placements"][0]["recipe"]["max_values"] = json!(40);
    let model = base
        .with_geometry(
            GeometryAdapterArtifact::create_json(&serde_json::to_vec(&record).unwrap()).unwrap(),
        )
        .unwrap();
    let mut a = NativeCausalLmQkvAdamW::new(&model, 0, AdamWConfig::default()).unwrap();
    let (owned, labels) = prompts();
    let qs = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let origin = a.checkpoint().unwrap();
    let step = AdamWStep::new(0.01, 0.).unwrap();
    let lengths = qs.iter().map(|q| q.len()).collect::<Vec<_>>();
    assert!(a.answer_sequence_input_count(&lengths, 3).is_err());
    assert_eq!(
        a.answer_sequence_input_count(&lengths, 2).unwrap(),
        lengths.iter().sum::<usize>()
    );
    assert!(a.last_logits_input_count(9).is_err());
    assert_eq!(a.last_logits_input_count(3).unwrap(), 3);
    // Three length-five sequences exceed the bound together; two do not.
    assert!(a.step_answer_sequences(&qs, &labels, 3, step).is_err());
    assert_eq!(a.checkpoint().unwrap(), origin);
    a.step_answer_sequences(&qs, &labels, 2, step).unwrap();
    assert_eq!(a.steps(), 1);
}

fn smoothed_reference(m: &NativeCausalLm, qs: &[&[usize]], gold: &[usize], epsilon: f64) -> f64 {
    qs.iter()
        .zip(gold)
        .map(|(q, &label)| {
            let logits = m.logits(q, [1, q.len()], true).unwrap();
            let row = logits.data();
            let high = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let logz = high
                + row
                    .iter()
                    .map(|&v| (f64::from(v) - high).exp())
                    .sum::<f64>()
                    .ln();
            logz - (1. - epsilon) * f64::from(row[label])
                - epsilon * row.iter().map(|&v| f64::from(v)).sum::<f64>() / row.len() as f64
        })
        .sum::<f64>()
        / qs.len() as f64
}

#[test]
fn smoothed_qkv_uses_shared_full_vocabulary_loss_and_answer_weights() {
    let (owned, labels) = prompts();
    let qs = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    for lag in [0, 1] {
        let m = adapted(lag);
        let train = NativeCausalLmQkvAdamW::new(&m, 0, AdamWConfig::default()).unwrap();
        for epsilon in [0., 0.1, 1.] {
            let expected = smoothed_reference(&m, &qs, &labels, epsilon);
            for cap in [1, 2, 32] {
                let (loss, g) = train
                    .answer_sequence_gradients_with_label_smoothing(&qs, &labels, cap, epsilon)
                    .unwrap();
                assert!((f64::from(loss) - expected).abs() < 2e-6);
                let mut mean = vec![0.; g.data().len()];
                for (q, y) in qs.iter().zip(&labels) {
                    let (_, part) = train
                        .answer_sequence_gradients_with_label_smoothing(&[*q], &[*y], 1, epsilon)
                        .unwrap();
                    for (v, &x) in mean.iter_mut().zip(part.data()) {
                        *v += f64::from(x) / qs.len() as f64;
                    }
                }
                assert!(g
                    .data()
                    .iter()
                    .zip(mean)
                    .all(|(&x, y)| (f64::from(x) - y).abs() < 2e-6));
            }
        }
    }
}

#[test]
fn smoothed_qkv_derivative_matches_physical_parameter_edits() {
    let owned = [vec![1, 2], vec![3, 7, 4, 8]];
    let qs = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let gold = [6, 2];
    for lag in [0, 1] {
        let make = |p: &BTreeMap<String, (Vec<usize>, Vec<f32>)>| {
            let m = NativeCausalLm::from_gpt2_json(
                &serde_json::to_vec(&config()).unwrap(),
                &archive(p),
                Default::default(),
            )
            .unwrap();
            if lag == 0 {
                m
            } else {
                m.with_key_lag(CausalLmKeyLag { layer: 0, lag }).unwrap()
            }
        };
        let m = make(&parameters());
        let a = NativeCausalLmQkvAdamW::new(&m, 0, AdamWConfig::default()).unwrap();
        let (_, g) = a
            .answer_sequence_gradients_with_label_smoothing(&qs, &gold, 2, 0.2)
            .unwrap();
        for output in 0..12 {
            for input in 0..4 {
                let loss = |sign: f32| {
                    let mut p = parameters();
                    p.get_mut("h.0.attn.c_attn.weight").unwrap().1[input * 12 + output] +=
                        sign * 0.002;
                    smoothed_reference(&make(&p), &qs, &gold, 0.2)
                };
                let numerical = (loss(1.) - loss(-1.)) / 0.004;
                assert!((numerical - f64::from(g.data()[output * 4 + input])).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn smoothing_is_opt_in_atomic_and_reported_without_mislabeling_nll() {
    let m = adapted(1);
    let mut a = NativeCausalLmQkvAdamW::new(&m, 0, AdamWConfig::default()).unwrap();
    let (owned, labels) = prompts();
    let qs = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let origin = a.checkpoint().unwrap();
    let step = AdamWStep::new(0.01, 0.).unwrap();
    for epsilon in [-0.1, 1.01, f64::NAN, f64::INFINITY] {
        assert!(a
            .step_answer_sequences_with_label_smoothing(&qs, &labels, 2, epsilon, step)
            .is_err());
        assert_eq!(a.checkpoint().unwrap(), origin);
    }
    a.step_answer_sequences_with_label_smoothing(
        &qs,
        &labels,
        2,
        0.1,
        AdamWStep::new(0., 0.).unwrap(),
    )
    .unwrap();
    assert_eq!(a.checkpoint().unwrap(), origin);
    let mut old = a.clone();
    let plain = old.step_answer_sequences(&qs, &labels, 2, step).unwrap();
    let explicit = a
        .step_answer_sequences_with_label_smoothing(&qs, &labels, 2, 0., step)
        .unwrap();
    assert_eq!(plain.gold_nll.to_bits(), explicit.objective_loss.to_bits());
    assert_eq!(a.checkpoint().unwrap(), old.checkpoint().unwrap());
    let report = a
        .step_answer_sequences_with_label_smoothing(&qs, &labels, 2, 0.1, step)
        .unwrap();
    let value = serde_json::to_value(report).unwrap();
    assert_eq!(value["label_smoothing"], json!(0.1));
    assert!(value.get("gold_nll").is_none());
    let (bytes, pin) = a.checkpoint().unwrap();
    let mut restored = NativeCausalLmQkvAdamW::restore(&m, &bytes, &pin).unwrap();
    a.step_answer_sequences_with_label_smoothing(&qs, &labels, 2, 0.1, step)
        .unwrap();
    restored
        .step_answer_sequences_with_label_smoothing(&qs, &labels, 2, 0.1, step)
        .unwrap();
    assert_eq!(a.checkpoint().unwrap(), restored.checkpoint().unwrap());
}
