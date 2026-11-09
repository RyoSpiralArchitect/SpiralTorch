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
    // Three length-five sequences exceed the bound together; two do not.
    assert!(a.step_answer_sequences(&qs, &labels, 3, step).is_err());
    assert_eq!(a.checkpoint().unwrap(), origin);
    a.step_answer_sequences(&qs, &labels, 2, step).unwrap();
    assert_eq!(a.steps(), 1);
}
