use super::*;

// One full prefix per target: no selection/gather, grouped CE, or answer collector.
fn oracle(
    model: &NativeCausalLm,
    doc: &[usize],
    row: &[i64],
    start: usize,
    end: usize,
    width: usize,
) -> (f64, u64) {
    let mut loss = 0.;
    let mut correct = 0;
    for t in start..end {
        let left = (t - 1) / (width - 1) * (width - 1);
        let logits = model.logits(&doc[left..t], [1, t - left], true).unwrap();
        let values = logits.data();
        let mut best = 0;
        for i in 1..values.len() {
            if values[i] > values[best] {
                best = i;
            }
        }
        let maximum = f64::from(values[best]);
        loss += maximum - f64::from(values[row[t] as usize])
            + values
                .iter()
                .map(|&v| (f64::from(v) - maximum).exp())
                .sum::<f64>()
                .ln();
        correct += u64::from(best == row[t] as usize);
    }
    (loss / (end - start) as f64, correct)
}

#[test]
fn answer_scores_match_individual_prefixes_and_preserve_aggregate_receipts() {
    let docs = vec![
        vec![],
        vec![2],
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        vec![4, 3, 2, 1],
        vec![9, 8, 7, 6],
        vec![2, 3, 4, 5],
    ];
    let masks = vec![
        vec![],
        vec![1],
        vec![1, -100, 9, 7, 5, 4, -100, -100, 3, 2],
        vec![-100, -100, -100, 1],
        vec![-100; 4],
        vec![0, 2, 3, 1],
    ];
    let spans = [(2, 2, 6), (2, 8, 10), (3, 3, 4), (5, 1, 4)];
    for arith in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let base = load().with_readout_accumulator(arith);
        let lora = base.with_lora(1, 2, 3.).unwrap();
        let g = lora.lora_loss_gradients(&[1, 2, 3, 4], [1, 4]).unwrap();
        let joint = lora
            .step_lora_sgd(&g, 0.2)
            .unwrap()
            .with_geometry(artifact(&base, "transformer.h.1.attn", 0.4))
            .unwrap();
        let models = [&base, &joint, &base];
        for batch in [1, 2, 8] {
            let cfg = policy(batch);
            let detail =
                NativeCausalLm::evaluate_variants_by_answer(&models, &docs, Some(&masks), cfg)
                    .unwrap();
            let scalar =
                NativeCausalLm::evaluate_variants(&models, &docs, Some(&masks), cfg).unwrap();
            assert_eq!(
                serde_json::to_string(detail.comparison()).unwrap(),
                serde_json::to_string(&scalar).unwrap()
            );
            assert_eq!(
                (
                    detail.document_count(),
                    detail.scored_documents(),
                    detail.answers().len()
                ),
                (6, 3, 4)
            );
            for (answer, &(doc, start, end)) in detail.answers().iter().zip(&spans) {
                assert_eq!(
                    (answer.document, answer.start, answer.end),
                    (doc, start, end)
                );
                for (owner, score) in models.iter().zip(&answer.scores) {
                    let (nll, hits) = oracle(owner, &docs[doc], &masks[doc], start, end, 4);
                    assert!(
                        (score.nll - nll).abs() < 2e-6,
                        "{doc} {start}: {} vs {nll}",
                        score.nll
                    );
                    assert_eq!(score.top1_correct, hits);
                }
            }
            let (a, b) = (
                detail.contrast(0, 1).unwrap(),
                detail.contrast(1, 0).unwrap(),
            );
            assert_eq!(
                (a.improved, a.regressed, a.tied),
                (b.regressed, b.improved, b.tied)
            );
            assert_eq!(a.target_tokens, 10);
            assert_eq!(a.token_weighted_delta_nll, -b.token_weighted_delta_nll);
            assert_eq!(a.answer_weighted_delta_nll, -b.answer_weighted_delta_nll);
            assert_eq!(
                a.reference_all_correct_answers,
                b.candidate_all_correct_answers
            );
            assert_eq!(a.reference_top1_correct, b.candidate_top1_correct);
            let null = detail.contrast(0, 2).unwrap();
            assert_eq!(
                (
                    null.tied,
                    null.improved,
                    null.regressed,
                    null.answer_weighted_delta_nll
                ),
                (4, 0, 0, 0.)
            );
            for (r, c) in [(0, 0), (3, 0), (0, usize::MAX)] {
                assert!(detail.contrast(r, c).is_err());
            }
        }
        let plain =
            NativeCausalLm::evaluate_variants_by_answer(&models, &docs, None, policy(2)).unwrap();
        assert_eq!((plain.answers().len(), plain.scored_documents()), (4, 4));
        for row in plain.answers() {
            assert_eq!((row.start, row.end), (1, docs[row.document].len()));
        }
    }
}

#[test]
fn collecting_answers_does_not_run_extra_decoder_blocks_or_row_partitions() {
    fn count(run: impl FnOnce()) -> [usize; 2] {
        use std::sync::Mutex;
        let counts = Arc::new(Mutex::new([0; 2]));
        let seen = Arc::clone(&counts);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |e| {
            if e.op_name == "cross_entropy_with_logits" {
                seen.lock().unwrap()[0] += 1;
            }
            if e.op_name == "matmul_prepacked_bias" && e.data["inner"] == 4 && e.data["cols"] == 12
            {
                seen.lock().unwrap()[1] += 1;
            }
        })));
        run();
        st_tensor::set_thread_meta_observer(previous);
        let result = *counts.lock().unwrap();
        result
    }
    let base = load();
    let late = base
        .with_geometry(artifact(&base, "transformer.h.1.attn", 0.3))
        .unwrap();
    let models = [&base, &late];
    let docs = vec![vec![1, 2, 3, 4]; 4];
    let masks = vec![
        vec![-100, -100, 4, 3],
        vec![-100, 2, -100, 4],
        vec![-100; 4],
        vec![-100; 4],
    ];
    let aggregate = count(|| {
        NativeCausalLm::evaluate_variants(&models, &docs, Some(&masks), policy(2)).unwrap();
    });
    let detailed = count(|| {
        NativeCausalLm::evaluate_variants_by_answer(&models, &docs, Some(&masks), policy(2))
            .unwrap();
    });
    assert_eq!(aggregate, [2, 3]);
    assert_eq!(detailed, aggregate);
    let n = MAX_ANSWER_SCORE_CELLS / MAX_EVALUATION_VARIANTS;
    let long = vec![vec![1; 2 * n + 3]];
    let labels = vec![(0..long[0].len())
        .map(|i| if i % 2 == 1 { 1 } else { -100 })
        .collect()];
    assert_eq!(
        count(|| {
            let e = NativeCausalLm::evaluate_variants_by_answer(
                &[&base; MAX_EVALUATION_VARIANTS],
                &long,
                Some(&labels),
                policy(2),
            )
            .unwrap_err();
            assert!(e.to_string().contains("score budget"));
        }),
        [0, 0]
    );
    assert!(NativeCausalLm::answer_comparison_plan(
        &[&base; MAX_EVALUATION_VARIANTS],
        &vec![2; n],
        policy(2),
        false
    )
    .is_ok());
    assert!(NativeCausalLm::answer_comparison_plan(
        &[&base; MAX_EVALUATION_VARIANTS],
        &vec![2; n + 1],
        policy(2),
        false
    )
    .is_err());
}

#[test]
fn masked_cells_bound_answers_not_the_number_of_input_documents() {
    let base = load();
    let models = [&base; MAX_EVALUATION_VARIANTS];
    let n = MAX_ANSWER_SCORE_CELLS / models.len() + 1;
    let docs = vec![vec![1, 2]; n];
    let mut labels = vec![vec![-100; 2]; n];
    labels[n - 1][1] = 0;
    let r = NativeCausalLm::evaluate_variants_by_answer(&models, &docs, Some(&labels), policy(2))
        .unwrap();
    assert_eq!(
        (r.document_count(), r.scored_documents(), r.answers().len()),
        (n, 1, 1)
    );
    assert_eq!(r.answers()[0].document, n - 1);
    assert_eq!(r.contrast(0, models.len() - 1).unwrap().tied, 1);
}
