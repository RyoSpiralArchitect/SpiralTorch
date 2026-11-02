use super::*;

fn geometry(base: &NativeCausalLm, layer: usize, gate: f32) -> GeometryAdapterArtifact {
    let mut record = artifact(base, &format!("transformer.h.{layer}.attn"), gate)
        .record()
        .clone();
    record.training = true;
    record.placements[0].training = true;
    for tensor in &mut record.placements[0].tensors {
        tensor.requires_grad = true;
    }
    GeometryAdapterArtifact::create(record).unwrap()
}

fn model(layer: usize, policy: CausalLmReadoutAccumulator) -> NativeCausalLm {
    let base = load().with_readout_accumulator(policy);
    let mut lora: serde_json::Value = serde_json::from_str(
        &base
            .with_lora(layer, 2, 3.)
            .unwrap()
            .lora_to_json()
            .unwrap(),
    )
    .unwrap();
    lora["a"] = json!([0.2, -0.1, 0.3, 0.15, -0.2, 0.1, 0.05, 0.3]);
    lora["b"] = json!([0.1, -0.2, 0.15, 0.05, -0.1, 0.2, 0.1, -0.15]);
    base.with_geometry(geometry(&base, layer, 0.2))
        .unwrap()
        .with_lora_json(&lora.to_string())
        .unwrap()
}

fn objective(model: &NativeCausalLm, ids: &[usize], shape: [usize; 2]) -> f64 {
    let p = model
        .prepare_objective(ids, shape, None, CausalLmObjective::default())
        .unwrap();
    let logits = model.logits(&p.input, p.shape, false).unwrap();
    logits
        .data()
        .chunks_exact(model.config.vocabulary)
        .zip(&p.labels)
        .map(|(row, &label)| {
            let m = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            m - f64::from(row[label as usize])
                + row
                    .iter()
                    .map(|&v| (f64::from(v) - m).exp())
                    .sum::<f64>()
                    .ln()
        })
        .sum::<f64>()
        / p.target_tokens as f64
}

#[test]
fn composed_gradients_match_finite_differences_in_both_groups() {
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    for policy in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        for layer in 0..2 {
            let model = model(layer, policy);
            let gradients = model.adaptation_loss_gradients(&ids, [2, 4]).unwrap();
            assert_eq!(gradients.parameters().len(), 2);
            assert_eq!(gradients.target_tokens(), 6);
            assert_eq!(
                gradients.loss(),
                model.next_token_loss(&ids, [2, 4]).unwrap()
            );
            for (path, parameters) in gradients.parameters() {
                for (name, values) in parameters {
                    for (index, &expected) in values.iter().enumerate() {
                        let eps = 0.002;
                        let perturb = |sign: f32| {
                            if path.ends_with("c_proj") {
                                let mut record: serde_json::Value =
                                    serde_json::from_str(&model.lora_to_json().unwrap()).unwrap();
                                record[name][index] = json!(
                                    record[name][index].as_f64().unwrap() as f32 + sign * eps
                                );
                                model.with_lora_json(&record.to_string()).unwrap()
                            } else {
                                let mut record =
                                    model.geometry_artifact().unwrap().record().clone();
                                record.placements[0]
                                    .tensors
                                    .iter_mut()
                                    .find(|t| t.name == *name)
                                    .unwrap()
                                    .values[index] += sign * eps;
                                model
                                    .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
                                    .unwrap()
                            }
                        };
                        let actual = (objective(&perturb(1.), &ids, [2, 4])
                            - objective(&perturb(-1.), &ids, [2, 4]))
                            / (2. * eps as f64);
                        assert!(
                            (actual - f64::from(expected)).abs() < 1e-4,
                            "{policy:?} {path}/{name}[{index}]: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn selective_updates_preserve_frozen_groups_and_seal_both_owners() {
    let model = model(0, CausalLmReadoutAccumulator::F32);
    let grads = model
        .adaptation_loss_gradients(&[1, 2, 3, 4], [1, 4])
        .unwrap();
    let lora = model
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Lora, 0.1)
        .unwrap();
    let geometry = model
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Geometry, 0.1)
        .unwrap();
    let both = model
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, 0.1)
        .unwrap();
    assert_eq!(lora.geometry_artifact_id(), model.geometry_artifact_id());
    assert_eq!(geometry.lora_artifact_id(), model.lora_artifact_id());
    assert_ne!(lora.lora_artifact_id(), model.lora_artifact_id());
    assert_ne!(
        geometry.geometry_artifact_id(),
        model.geometry_artifact_id()
    );
    assert_eq!(both.lora_artifact_id(), lora.lora_artifact_id());
    assert_eq!(both.geometry_artifact_id(), geometry.geometry_artifact_id());
    assert_eq!(both.base_model_identity(), model.base_model_identity());
    // Changing even the group frozen in the requested update invalidates the
    // gradient, since that group changed the original forward function.
    assert!(lora
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Geometry, 0.1)
        .is_err());
    assert!(geometry
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Lora, 0.1)
        .is_err());
    assert!(model
        .with_readout_accumulator(CausalLmReadoutAccumulator::F64)
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, 0.1)
        .is_err());
    assert!(model.lora_loss_gradients(&[1, 2, 3], [1, 3]).is_err());
    assert!(model.geometry_loss_gradients(&[1, 2, 3], [1, 3]).is_err());
    for rate in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(model
            .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, rate)
            .is_err());
    }
    assert_eq!(model.lora_artifact_id(), grads.lora_artifact_id());
    assert_eq!(model.geometry_artifact_id(), grads.geometry_artifact_id());
}

#[test]
fn composition_admission_zero_delta_and_single_group_compatibility() {
    let base = load();
    let geo = geometry(&base, 0, 0.);
    let lora = base.with_lora(0, 2, 2.).unwrap();
    let combined = lora.with_geometry(geo.clone()).unwrap();
    let reverse = base
        .with_geometry(geo)
        .unwrap()
        .with_lora(0, 2, 2.)
        .unwrap();
    assert_eq!(
        combined.logits(&[1, 2, 3], [1, 3], false).unwrap(),
        base.logits(&[1, 2, 3], [1, 3], false).unwrap()
    );
    assert_eq!(
        combined.logits(&[1, 2, 3], [1, 3], false).unwrap(),
        reverse.logits(&[1, 2, 3], [1, 3], false).unwrap()
    );
    assert!(lora.with_geometry(geometry(&base, 1, 0.2)).is_err());
    assert!(base
        .with_geometry(artifact(&base, "transformer.h.0.mlp", 0.2))
        .unwrap()
        .with_lora(0, 2, 2.)
        .is_err());
    let grads = lora.adaptation_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    let old = lora.lora_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert_eq!(
        lora.step_adaptation_sgd(&grads, CausalLmAdaptationMode::Lora, 0.1)
            .unwrap()
            .lora_to_json()
            .unwrap(),
        lora.step_lora_sgd(&old, 0.1)
            .unwrap()
            .lora_to_json()
            .unwrap()
    );
    assert!(lora
        .step_adaptation_sgd(&grads, CausalLmAdaptationMode::Joint, 0.1)
        .is_err());
    let geo = base.with_geometry(geometry(&base, 0, 0.2)).unwrap();
    let grads = geo.adaptation_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    let old = geo.geometry_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert_eq!(
        geo.step_adaptation_sgd(&grads, CausalLmAdaptationMode::Geometry, 0.1)
            .unwrap()
            .geometry_artifact_id(),
        geo.step_geometry_sgd(&old, 0.1)
            .unwrap()
            .geometry_artifact_id()
    );
}
