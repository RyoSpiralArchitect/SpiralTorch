use super::*;
use crate::causal_lm::observation::ResidualSums;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use crate::geometry_runtime::ParameterGradients;
use st_core::runtime::zspace_repetition_unlikelihood::{
    ZSpaceRepetitionUnlikelihoodCandidateSource, ZSpaceRepetitionUnlikelihoodConfig,
};

#[test]
fn residuals_use_all_elements_f64_subtraction_and_no_fictitious_zero_denominator() {
    let base = load();
    let path = "transformer.h.1";
    let artifact = artifact(&base, path, 0.2);
    let gradients = ParameterGradients::from([(path.to_owned(), Default::default())]);
    let tensor = |value, n| Tensor::from_vec(n, 4, vec![value; n * 4]).unwrap();
    let mut sums = ResidualSums::default();
    sums.record(path, &tensor(0., 1), &tensor(2., 1)).unwrap();
    sums.record(path, &tensor(0., 2), &tensor(1., 2)).unwrap();
    let result = sums.finish(&artifact, &gradients).unwrap();
    assert_eq!(result[path].elements, 12);
    assert_eq!(result[path].residual_rms, 2f64.sqrt());
    assert_eq!(result[path].input_rms, 0.);
    assert_eq!(result[path].relative_residual_rms, None);

    let mut sums = ResidualSums::default();
    sums.record(path, &tensor(-f32::MAX, 1), &tensor(f32::MAX, 1))
        .unwrap();
    let result = sums.finish(&artifact, &gradients).unwrap();
    assert_eq!(result[path].residual_rms, 2. * f64::from(f32::MAX));
    assert_eq!(result[path].relative_residual_rms, Some(2.));
}

#[test]
fn reported_residual_is_the_effective_post_minus_pre_at_the_real_model_site() {
    let base = load();
    let path = "transformer.h.1";
    let mut record = artifact(&base, path, 0.4).record().clone();
    record.placements[0].tensors[0].requires_grad = true;
    let artifact = GeometryAdapterArtifact::create(record).unwrap();
    let optimizer = NativeGeometryAdamW::new(&artifact, AdamWConfig::default(), None).unwrap();
    let adapted = base.with_geometry(artifact).unwrap();
    let docs = vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5], vec![3, 4, 5, 6]];
    let mut trainer =
        NativeCausalLmTrainer::new(adapted.clone(), optimizer, docs.clone(), None, config())
            .unwrap();
    let (mut x2, mut d2) = (0f64, 0f64);
    for batch in docs.chunks(2) {
        let shape = [batch.len(), 3];
        let ids = batch
            .iter()
            .flat_map(|d| &d[..3])
            .copied()
            .collect::<Vec<_>>();
        let mut h = base.embed(&ids, shape).unwrap();
        for i in 0..base.config().layers {
            h = base
                .run_block(i, h, shape, decoder::BlockPass::Forward)
                .unwrap()
                .output;
        }
        let actual = adapted
            .geometry
            .as_ref()
            .unwrap()
            .forward_sequence(path, h.data(), shape)
            .unwrap();
        for (&x, &y) in h.data().iter().zip(actual.output().data()) {
            x2 += f64::from(x).powi(2);
            d2 += (f64::from(y) - f64::from(x)).powi(2);
        }
    }
    let observed = trainer.step_observed().unwrap().unwrap();
    let site = &observed.sites[path];
    assert_eq!(site.elements, 36);
    assert!(d2 > 0. && x2 > 0.);
    assert_eq!(site.input_rms, (x2 / 36.).sqrt());
    assert_eq!(site.residual_rms, (d2 / 36.).sqrt());
    assert_eq!(site.relative_residual_rms, Some((d2 / x2).sqrt()));
}

#[test]
fn observation_matches_the_reduced_gradient_without_changing_an_update_or_replaying_counts() {
    let base = load();
    let mut record = serde_json::to_value(trainable(&base).record()).unwrap();
    for (i, site) in record["placements"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .take(2)
        .enumerate()
    {
        let learned = i == 1;
        site["recipe"] = json!({"schema":"spiraltorch.wave_gate_adapter.v2","features":4,"strength":0.3,
            "kernel":{"curvature":-1.0,"saturation":1.0,"porosity":0.05,"max_values":1024},
            "learnable_radius":learned});
        site["tensors"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name":"log_radius","shape":[],"values":[0.2],
            "role":if learned {"parameter"} else {"buffer"},"requires_grad":learned}));
    }
    let artifact =
        GeometryAdapterArtifact::create_json(&serde_json::to_vec(&record).unwrap()).unwrap();
    let optimizer = NativeGeometryAdamW::new(
        &artifact,
        AdamWConfig::default(),
        Some(GlobalNormClip::new(1e-8).unwrap()),
    )
    .unwrap();
    let docs = vec![
        vec![1, 2, 1, 2, 1, 3],
        vec![4, 5, 4, 6, 4, 7],
        vec![2, 3, 2, 3, 2, 4],
    ];
    let labels = vec![
        vec![1, 2, 1, 2, 1, 3],
        vec![-100, -100, -100, 6, 4, 7],
        vec![2, 3, 2, 3, 2, 4],
    ];
    let repetition = Some(CausalLmRepetition::NativeV1 {
        config: ZSpaceRepetitionUnlikelihoodConfig {
            strength: 0.7,
            candidate_source: ZSpaceRepetitionUnlikelihoodCandidateSource::PriorContinuation {
                ngram_order: 2,
            },
            context_window: 16,
            max_candidates_per_position: 3,
        },
    });
    let config = CausalLmTrainingConfig {
        context_length: 6,
        microbatch_windows: 1,
        accumulation_steps: 2,
        epochs: 1,
        repetition,
        ..config()
    };
    let mut plain = NativeCausalLmTrainer::new(
        base.with_geometry(artifact).unwrap(),
        optimizer,
        docs.clone(),
        Some(labels.clone()),
        config,
    )
    .unwrap();
    let mut tapped = plain.clone();
    for (start, end) in [(0, 2), (2, 3)] {
        let model = plain.model();
        let mut acc = model
            .geometry_accumulator_with_objective(CausalLmObjective {
                label_smoothing: config.label_smoothing,
                repetition,
            })
            .unwrap();
        for i in start..end {
            acc.accumulate_loss_with_labels(model, &docs[i], [1, 6], Some(&labels[i]))
                .unwrap();
        }
        let means = acc.parameter_gradients().unwrap();
        let observation = tapped.step_observed().unwrap().unwrap();
        assert_eq!(observation.sites.len(), 3);
        assert_eq!(observation.base_model_identity, model.base_model_identity());
        assert_eq!(observation.data_id, plain.data_id());
        assert!(observation.step.repetition_positions.unwrap() > 0);
        for site in &model.geometry_artifact().unwrap().record().placements {
            let measured = &observation.sites[&site.path];
            assert_eq!(measured.elements, (end - start) as u64 * 5 * 4);
            let g = &means[&site.path];
            let norm = site
                .tensors
                .iter()
                .filter(|t| t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad)
                .flat_map(|t| &g[&t.name])
                .map(|&g| f64::from(g).powi(2))
                .sum::<f64>()
                .sqrt();
            assert_eq!(measured.trainable_gradient_l2, norm);
            let radius = site.tensors.iter().find(|t| t.name == "log_radius");
            assert_eq!(measured.pre_log_radius, radius.map(|r| r.values[0]));
            assert_eq!(
                measured.log_radius_gradient,
                radius
                    .filter(|r| r.requires_grad)
                    .map(|_| g["log_radius"][0])
            );
        }
        let expected = plain.step().unwrap().unwrap();
        assert_eq!(
            serde_json::to_string(&expected).unwrap(),
            serde_json::to_string(&observation.step).unwrap()
        );
        assert_eq!(plain.to_json().unwrap(), tapped.to_json().unwrap());
    }
    let before = tapped.to_json().unwrap();
    assert!(tapped.step_observed().unwrap().is_none());
    assert_eq!(before, tapped.to_json().unwrap());
}

#[test]
fn observed_failure_is_atomic() {
    let mut trainer = create(CausalLmTrainingConfig {
        learning_rate: f32::MAX,
        weight_decay: f32::MAX,
        ..config()
    });
    let before = trainer.to_json().unwrap();
    assert!(trainer.step_observed().is_err());
    assert_eq!(trainer.to_json().unwrap(), before);
}
