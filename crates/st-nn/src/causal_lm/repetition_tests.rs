use super::*;
use crate::geometry_runtime::NativeGeometryAdamW;
use st_core::runtime::zspace_repetition_unlikelihood::{
    ZSpaceRepetitionUnlikelihoodCandidateSource as Source,
    ZSpaceRepetitionUnlikelihoodConfig as Config,
};

fn recipe(source: Source, strength: f64) -> CausalLmRepetition {
    CausalLmRepetition::NativeV1 {
        config: Config {
            strength,
            candidate_source: source,
            context_window: 16,
            max_candidates_per_position: 3,
        },
    }
}

fn sources() -> [Source; 3] {
    [
        Source::PriorContinuation { ngram_order: 2 },
        Source::ModelTopkHistory { proposal_top_k: 11 },
        Source::ModelTopkPeriodic { proposal_top_k: 11 },
    ]
}

const WINDOWS: [usize; 12] = [1, 2, 1, 2, 1, 3, 4, 5, 6, 7, 8, 9];

#[test]
fn repetition_two_nonempty_batches_use_their_own_position_normalizer() {
    let base = load();
    let model = base.with_geometry(trainable(&base)).unwrap();
    let rows = [[1, 2, 1, 2, 1, 3], [4, 5, 4, 6, 4, 7]];
    let labels = [[1, 2, 1, 2, 1, 3], [-100, -100, -100, 6, 4, 7]];
    let objective = CausalLmObjective {
        repetition: Some(recipe(sources()[0], 0.7)),
        ..Default::default()
    };
    let batches: Vec<_> = rows
        .iter()
        .zip(&labels)
        .map(|(row, labels)| {
            model
                .geometry_loss_gradients_with_objective(row, [1, 6], Some(labels), objective)
                .unwrap()
        })
        .collect();
    let ce: Vec<_> = rows
        .iter()
        .zip(&labels)
        .map(|(row, labels)| {
            model
                .geometry_loss_gradients_with_objective(
                    row,
                    [1, 6],
                    Some(labels),
                    Default::default(),
                )
                .unwrap()
        })
        .collect();
    assert_eq!(
        batches
            .iter()
            .map(|b| b.target_tokens())
            .collect::<Vec<_>>(),
        [5, 3]
    );
    assert_eq!(
        batches
            .iter()
            .map(|b| b.repetition_positions())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let mut acc = model
        .geometry_accumulator_with_objective(objective)
        .unwrap();
    for batch in &batches {
        acc.add(batch).unwrap();
    }
    let expected = (5. * f64::from(ce[0].loss()) + 3. * f64::from(ce[1].loss())) / 8.
        + 0.7
            * (f64::from(batches[0].repetition_loss())
                + 2. * f64::from(batches[1].repetition_loss()))
            / 3.;
    assert!((acc.loss().unwrap() - expected).abs() < 1e-8);
    let wrong = (5. * f64::from(batches[0].loss()) + 3. * f64::from(batches[1].loss())) / 8.;
    assert!((acc.loss().unwrap() - wrong).abs() > 1e-6);
    for (path, group) in acc.parameter_gradients().unwrap() {
        for (name, values) in group {
            for (i, actual) in values.iter().enumerate() {
                let a = f64::from(batches[0].parameters()[&path][&name][i]);
                let b = f64::from(batches[1].parameters()[&path][&name][i]);
                let ca = f64::from(ce[0].parameters()[&path][&name][i]);
                let cb = f64::from(ce[1].parameters()[&path][&name][i]);
                let expected = (5. * ca + 3. * cb) / 8. + ((a - ca) + 2. * (b - cb)) / 3.;
                assert!((f64::from(*actual) - expected).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn repetition_later_over_budget_microbatch_is_rejected_before_any_update() {
    let mut c = config();
    c["n_positions"] = json!(512);
    c["vocab_size"] = json!(64);
    let mut p = parameters();
    p.insert("wpe.weight".into(), (vec![512, 4], vec![0.; 2048]));
    p.insert("wte.weight".into(), (vec![64, 4], vec![0.01; 256]));
    let base = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&c).unwrap(),
        &archive(&p),
        Default::default(),
    )
    .unwrap();
    let mut state = serde_json::to_value(trainable(&base).record()).unwrap();
    for site in state["placements"].as_array_mut().unwrap() {
        site["recipe"]["max_values"] = json!(4096);
    }
    let model = base
        .with_geometry(
            GeometryAdapterArtifact::create_json(&serde_json::to_vec(&state).unwrap()).unwrap(),
        )
        .unwrap();
    let optimizer =
        NativeGeometryAdamW::new(model.geometry_artifact().unwrap(), Default::default(), None)
            .unwrap();
    let recipe = CausalLmRepetition::NativeV1 {
        config: Config {
            strength: 1.,
            candidate_source: Source::ModelTopkPeriodic { proposal_top_k: 64 },
            context_window: 512,
            max_candidates_per_position: 3,
        },
    };
    let training = CausalLmTrainingConfig {
        context_length: 512,
        microbatch_windows: 1,
        accumulation_steps: 1,
        epochs: 1,
        learning_rate: 0.01,
        weight_decay: 0.,
        label_smoothing: 0.,
        repetition: Some(recipe),
    };
    let mut sparse = vec![-100; 512];
    sparse[511] = 1;
    NativeCausalLmTrainer::new(
        model.clone(),
        optimizer.clone(),
        vec![vec![1; 512]],
        Some(vec![sparse.clone()]),
        training,
    )
    .unwrap();
    let result = NativeCausalLmTrainer::new(
        model,
        optimizer,
        vec![vec![1; 512]; 2],
        Some(vec![sparse, vec![1; 512]]),
        training,
    );
    assert!(
        result.is_err(),
        "later work over-budget batch must fail at construction"
    );
    assert!(result.err().unwrap().to_string().contains("work"));
}

#[test]
fn repetition_all_sources_match_parameter_differences_and_unequal_masked_reduction() {
    for arithmetic in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let base = load().with_readout_accumulator(arithmetic);
        let state = trainable(&base);
        let model = base.with_geometry(state.clone()).unwrap();
        let mut labels: Vec<i64> = WINDOWS.iter().map(|&v| v as i64).collect();
        labels[7] = -100;
        labels[8] = -100;
        for source in sources() {
            let objective = CausalLmObjective {
                label_smoothing: 0.1,
                repetition: Some(recipe(source, 0.7)),
            };
            let full = model
                .geometry_loss_gradients_with_objective(&WINDOWS, [2, 6], Some(&labels), objective)
                .unwrap();
            assert!(full.repetition_positions() > 0);
            assert!(full.repetition_loss() > 0.);
            assert_eq!(
                full.loss(),
                model
                    .next_token_loss_with_objective(&WINDOWS, [2, 6], Some(&labels), objective)
                    .unwrap()
            );
            let ce = model
                .geometry_loss_gradients_with_objective(
                    &WINDOWS,
                    [2, 6],
                    Some(&labels),
                    CausalLmObjective {
                        repetition: None,
                        ..objective
                    },
                )
                .unwrap();
            assert_eq!(full.causal_loss(), ce.loss());
            assert_ne!(full.parameters(), ce.parameters());
            let mut accumulated = model
                .geometry_accumulator_with_objective(objective)
                .unwrap();
            for start in [0, 6] {
                accumulated
                    .accumulate_loss_with_labels(
                        &model,
                        &WINDOWS[start..start + 6],
                        [1, 6],
                        Some(&labels[start..start + 6]),
                    )
                    .unwrap();
            }
            assert_eq!(accumulated.target_tokens(), 8);
            assert_eq!(
                accumulated.repetition_positions(),
                full.repetition_positions() as u64
            );
            assert!((accumulated.loss().unwrap() - f64::from(full.loss())).abs() < 3e-7);
            let reduced = accumulated.parameter_gradients().unwrap();
            for (path, group) in full.parameters() {
                for (name, values) in group {
                    for (i, &expected) in values.iter().enumerate() {
                        assert!((reduced[path][name][i] - expected).abs() < 2e-7);
                        let evaluate = |sign: f32| {
                            let mut record = state.record().clone();
                            let site = record
                                .placements
                                .iter_mut()
                                .find(|p| p.path == *path)
                                .unwrap();
                            site.tensors
                                .iter_mut()
                                .find(|t| t.name == *name)
                                .unwrap()
                                .values[i] += sign * 0.005;
                            base.with_geometry(GeometryAdapterArtifact::create(record).unwrap())
                                .unwrap()
                                .next_token_loss_with_objective(
                                    &WINDOWS,
                                    [2, 6],
                                    Some(&labels),
                                    objective,
                                )
                                .unwrap()
                        };
                        let numerical = (evaluate(1.) - evaluate(-1.)) / 0.01;
                        assert!(
                            (numerical - expected).abs() < 8e-5,
                            "{source:?} {path}/{name}[{i}]: {numerical} vs {expected}"
                        );
                    }
                }
            }
            let before = format!("{accumulated:?}");
            assert!(accumulated.add(&ce).is_err());
            let mut bad = labels.clone();
            bad[5] = 10;
            assert!(accumulated
                .accumulate_loss_with_labels(&model, &WINDOWS, [2, 6], Some(&bad))
                .is_err());
            assert_eq!(format!("{accumulated:?}"), before);
        }
    }
}

#[test]
fn repetition_zero_and_empty_candidates_are_exact_ce_and_invalid_recipes_fail() {
    let base = load();
    let model = base.with_geometry(trainable(&base)).unwrap();
    let expected = model.geometry_loss_gradients(&WINDOWS, [2, 6]).unwrap();
    for source in sources() {
        let objective = CausalLmObjective {
            repetition: Some(recipe(source, 0.)),
            ..Default::default()
        };
        let actual = model
            .geometry_loss_gradients_with_objective(&WINDOWS, [2, 6], None, objective)
            .unwrap();
        assert_eq!(actual.loss().to_bits(), expected.loss().to_bits());
        assert_eq!(actual.parameters(), expected.parameters());
        assert_eq!(actual.repetition_positions(), 0);
        let mut acc = model
            .geometry_accumulator_with_objective(objective)
            .unwrap();
        acc.add(&actual).unwrap();
        assert_eq!(
            acc.parameter_gradients().unwrap(),
            expected.parameters().clone()
        );
    }
    let unique = [1, 2, 3, 4, 5, 6];
    let objective = CausalLmObjective {
        repetition: Some(recipe(sources()[0], 1.)),
        ..Default::default()
    };
    let actual = model
        .geometry_loss_gradients_with_objective(&unique, [1, 6], None, objective)
        .unwrap();
    let ce = model.geometry_loss_gradients(&unique, [1, 6]).unwrap();
    assert_eq!(actual.loss().to_bits(), ce.loss().to_bits());
    assert_eq!(actual.parameters(), ce.parameters());
    assert_eq!(actual.repetition_positions(), 0);
    for strength in [-1., 10.1, f64::NAN] {
        assert!(model
            .geometry_accumulator_with_objective(CausalLmObjective {
                repetition: Some(recipe(sources()[0], strength)),
                ..Default::default()
            })
            .is_err());
    }
    assert!(model
        .geometry_accumulator_with_objective(CausalLmObjective {
            repetition: Some(recipe(Source::ModelTopkHistory { proposal_top_k: 12 }, 1.)),
            ..Default::default()
        })
        .is_err());
}

#[test]
fn repetition_training_checkpoint_binds_recipe_and_exact_continuation() {
    for source in sources() {
        let base = load();
        let model = base.with_geometry(trainable(&base)).unwrap();
        let optimizer =
            NativeGeometryAdamW::new(model.geometry_artifact().unwrap(), Default::default(), None)
                .unwrap();
        let docs = vec![WINDOWS.to_vec()];
        let config = CausalLmTrainingConfig {
            context_length: 6,
            microbatch_windows: 1,
            accumulation_steps: 2,
            epochs: 3,
            learning_rate: 0.01,
            weight_decay: 0.,
            label_smoothing: 0.1,
            repetition: Some(recipe(source, 0.7)),
        };
        let mut trainer = NativeCausalLmTrainer::new(
            model.clone(),
            optimizer.clone(),
            docs.clone(),
            None,
            config,
        )
        .unwrap();
        let first = trainer.step().unwrap().unwrap();
        assert!(first.repetition_positions.unwrap() > 0);
        assert!(first.repetition_loss.unwrap() > 0.);
        let payload = trainer.to_json().unwrap();
        let pin = trainer.checkpoint_id().unwrap();
        let mut restored =
            NativeCausalLmTrainer::from_json(base, docs.clone(), None, payload.as_bytes(), &pin)
                .unwrap();
        assert_eq!(restored.config(), config);
        assert_eq!(restored.to_json().unwrap(), payload);
        while !trainer.exhausted() {
            assert_eq!(
                serde_json::to_string(&trainer.step().unwrap()).unwrap(),
                serde_json::to_string(&restored.step().unwrap()).unwrap()
            );
            assert_eq!(trainer.to_json().unwrap(), restored.to_json().unwrap());
        }
        let changed = NativeCausalLmTrainer::new(
            model.clone(),
            optimizer.clone(),
            docs.clone(),
            None,
            CausalLmTrainingConfig {
                repetition: None,
                ..config
            },
        )
        .unwrap();
        assert_ne!(changed.checkpoint_id().unwrap(), pin);
        assert!(!changed.to_json().unwrap().contains("repetition"));
        let mut bad: Vec<Vec<i64>> = docs
            .iter()
            .map(|d| d.iter().map(|&v| v as i64).collect())
            .collect();
        bad[0][5] = 10;
        assert!(NativeCausalLmTrainer::new(model, optimizer, docs, Some(bad), config).is_err());
    }
}
