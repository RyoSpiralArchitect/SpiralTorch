use super::*;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};
use st_kernel_contracts::gradient_clip::GlobalNormClip;

fn adapted(lag: usize) -> NativeCausalLm {
    let base = load();
    let base = if lag == 0 {
        base
    } else {
        base.with_key_lag(CausalLmKeyLag { layer: 0, lag }).unwrap()
    };
    base.with_lora(1, 2, 1.)
        .unwrap()
        .with_geometry(artifact(&base, "transformer.h.1.attn", 0.2))
        .unwrap()
}

fn data() -> (Vec<Vec<usize>>, Vec<Vec<i64>>) {
    (
        vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![2, 3, 4],
            vec![1, 3, 5, 7, 9],
        ],
        vec![
            vec![-100, -100, -100, -100, 5],
            vec![-100, -100, 8, 9, 10],
            vec![-100, 3, 4],
            vec![-100, -100, -100, 7, 9],
        ],
    )
}

#[test]
fn qkv_documents_weight_selected_tokens_not_groups_or_documents() {
    let (ids, labels) = data();
    let docs = ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let ys = labels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    for lag in [0, 1] {
        let trainer =
            NativeCausalLmQkvAdamW::new(&adapted(lag), 0, AdamWConfig::default()).unwrap();
        let original = trainer.checkpoint().unwrap();
        for epsilon in [0., 0.1] {
            let parts = docs
                .iter()
                .zip(&ys)
                .map(|(d, y)| {
                    trainer
                        .causal_loss_gradients(d, [1, d.len()], Some(y), epsilon)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let count = parts.iter().map(|p| p.2).sum::<usize>();
            assert_eq!(count, 8);
            let expected_loss = parts
                .iter()
                .map(|p| f64::from(p.0) * p.2 as f64)
                .sum::<f64>()
                / count as f64;
            let wrong_loss = parts.iter().map(|p| f64::from(p.0)).sum::<f64>() / parts.len() as f64;
            assert!((expected_loss - wrong_loss).abs() > 1e-6);
            for cap in [1, 2, 8] {
                let (loss, g, n) = trainer
                    .document_loss_gradients(&docs, Some(&ys), cap, epsilon)
                    .unwrap();
                assert_eq!(n, count);
                assert!((f64::from(loss) - expected_loss).abs() < 1e-6);
                for (index, &x) in g.data().iter().enumerate() {
                    let expected = parts
                        .iter()
                        .map(|p| f64::from(p.1.data()[index]) * p.2 as f64)
                        .sum::<f64>()
                        / count as f64;
                    assert!((f64::from(x) - expected).abs() < 2e-6);
                }
            }
        }
        assert_eq!(trainer.checkpoint().unwrap(), original);
    }
}

#[test]
fn qkv_documents_clip_mean_once_and_resume_one_atomic_step() {
    let model = adapted(1);
    let cfg = AdamWConfig::new(0.9, 0.999, 0.1).unwrap();
    let clip = GlobalNormClip::new(0.001).unwrap();
    let mut a = NativeCausalLmQkvAdamW::new_with_gradient_clip(&model, 0, cfg, Some(clip)).unwrap();
    let (ids, labels) = data();
    let docs = ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let ys = labels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (loss, g, n) = a.document_loss_gradients(&docs, Some(&ys), 2, 0.1).unwrap();
    let squared = g.data().iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    let factors = clip.factors(squared).unwrap();
    assert!(!factors.as_slice().is_empty());
    let step = AdamWStep::new(0.03, 0.02).unwrap();
    let expected = a
        .weights()
        .iter()
        .zip(g.data())
        .map(|(&w, &raw)| {
            let mut grad = raw;
            for &f in factors.as_slice() {
                grad *= f;
            }
            step.candidate(cfg, w, grad, 0., 0., 0).unwrap().value
        })
        .collect::<Vec<_>>();
    let prefix = a.prefill(vec![1, 2], 4096).unwrap();
    let before = a.checkpoint().unwrap();
    a.step_documents(&docs, Some(&ys), 2, 0.1, AdamWStep::new(0., 0.).unwrap())
        .unwrap();
    assert_eq!(before, a.checkpoint().unwrap());
    let report = a.step_documents(&docs, Some(&ys), 2, 0.1, step).unwrap();
    assert_eq!(
        (
            report.steps,
            report.documents,
            report.microbatches,
            report.target_tokens
        ),
        (1, 4, 3, n)
    );
    assert_eq!(report.objective_loss.to_bits(), loss.to_bits());
    assert_eq!(report.gradient_clip.unwrap().factors, factors.as_slice());
    assert_eq!(a.weights(), expected);
    assert_ne!(a.serving_id(), prefix.projection_id());
    let (bytes, pin) = a.checkpoint().unwrap();
    let mut b = NativeCausalLmQkvAdamW::restore(&model, &bytes, &pin).unwrap();
    let ra = a.step_documents(&docs, Some(&ys), 2, 0.1, step).unwrap();
    let rb = b.step_documents(&docs, Some(&ys), 2, 0.1, step).unwrap();
    assert_eq!(
        serde_json::to_value(ra).unwrap(),
        serde_json::to_value(rb).unwrap()
    );
    assert_eq!(a.checkpoint().unwrap(), b.checkpoint().unwrap());
}

#[test]
fn qkv_documents_one_group_matches_windows_and_dense_labels() {
    let model = adapted(0);
    let mut a = NativeCausalLmQkvAdamW::new(&model, 1, AdamWConfig::default()).unwrap();
    let mut b = NativeCausalLmQkvAdamW::new(&model, 1, AdamWConfig::default()).unwrap();
    let flat = [1, 2, 3, 4, 5, 6, 7, 8];
    let docs = flat.chunks_exact(4).collect::<Vec<_>>();
    let labels = docs
        .iter()
        .map(|d| d.iter().map(|&x| x as i64).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let ys = labels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let one = a.causal_loss_gradients(&flat, [2, 4], None, 0.1).unwrap();
    let grouped = a.document_loss_gradients(&docs, None, 2, 0.1).unwrap();
    assert_eq!(one, grouped);
    assert_eq!(
        grouped,
        a.document_loss_gradients(&docs, Some(&ys), 2, 0.1).unwrap()
    );
    let step = AdamWStep::new(0.03, 0.).unwrap();
    a.step_windows(&flat, [2, 4], None, 0.1, step).unwrap();
    b.step_documents(&docs, None, 2, 0.1, step).unwrap();
    assert_eq!(a.checkpoint().unwrap(), b.checkpoint().unwrap());
}

#[test]
fn qkv_documents_admit_sparse_groups_and_reject_invalid_tails_without_mutation() {
    let mut model = adapted(1);
    model.limits.max_logit_elements = 22;
    let mut a = NativeCausalLmQkvAdamW::new(&model, 0, AdamWConfig::default()).unwrap();
    let docs: [&[usize]; 2] = [&[1, 2, 3, 4, 5], &[6, 7, 8, 9, 10]];
    let ys: [&[i64]; 2] = [&[-100, -100, -100, -100, 5], &[-100, -100, -100, -100, 10]];
    assert_eq!(a.document_input_count(&[5, 5], 2, true).unwrap(), 10);
    assert!(a.document_input_count(&[5, 5], 2, false).is_err());
    assert_eq!(
        a.document_loss_gradients(&docs, Some(&ys), 2, 0.)
            .unwrap()
            .2,
        2
    );
    let original = a.checkpoint().unwrap();
    let step = AdamWStep::new(0.03, 0.).unwrap();
    for bad in [vec![-100; 5], vec![-100; 4], vec![99; 5], vec![1; 5]] {
        // The last case fits neither this group's logit budget nor a per-doc budget.
        assert!(a
            .step_documents(&docs, Some(&[ys[0], &bad]), 2, 0., step)
            .is_err());
        assert_eq!(a.checkpoint().unwrap(), original);
    }
    let extra = [-100, -100, -100, 9, 10]; // each doc fits alone, combined group does not
    assert!(a
        .step_documents(&docs, Some(&[ys[0], &extra]), 2, 0., step)
        .is_err());
    assert_eq!(a.checkpoint().unwrap(), original);
    for bad in [vec![], vec![1], vec![99, 2], vec![1; 10]] {
        let mut tail_labels = vec![-100; bad.len()];
        if let Some(last) = tail_labels.last_mut() {
            *last = 1;
        }
        assert!(a
            .step_documents(&[docs[0], &bad], Some(&[ys[0], &tail_labels]), 1, 0., step)
            .is_err());
        assert_eq!(a.checkpoint().unwrap(), original);
    }
    assert!(a.document_input_count(&[], 1, true).is_err());
    assert!(a.document_input_count(&[2], 0, true).is_err());
    assert!(a.document_input_count(&vec![2; 4097], 32, true).is_err());
    assert!(a
        .document_input_count(&(0..1025).map(|i| 2 + i % 2).collect::<Vec<_>>(), 32, true)
        .is_err());
    assert!(a
        .document_loss_gradients(&docs, Some(&ys[..1]), 2, 0.)
        .is_err());
    assert!(a
        .document_loss_gradients(&docs, Some(&ys), 2, f64::NAN)
        .is_err());
}
