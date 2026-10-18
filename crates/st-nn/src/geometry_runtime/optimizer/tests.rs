use super::*;
use crate::geometry_runtime::tests::artifact;
use std::collections::BTreeMap;

fn fixture() -> NativeGeometryRuntime {
    let mut record = artifact(true, Some(false), 0.5).record().clone();
    for t in &mut record.placements[0].tensors {
        if t.name == "bias" {
            t.requires_grad = false;
        }
    }
    let mut second = record.placements[0].clone();
    second.path = "later".into();
    record.placements.push(second);
    NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap())
}

fn gradients() -> ParameterGradients {
    BTreeMap::from([(
        "site".into(),
        BTreeMap::from([
            ("gate".into(), vec![3., -4., 0.]),
            ("bias".into(), vec![f32::MAX; 3]),
        ]),
    )])
}

#[test]
fn selected_trainable_parameters_share_adamw_and_global_clip() {
    let runtime = fixture();
    let config = AdamWConfig::new(0.5, 0.75, 0.25).unwrap();
    let optimizer = NativeGeometryAdamW::new(
        runtime.artifact(),
        config,
        Some(GlobalNormClip::new(2.5).unwrap()),
    )
    .unwrap();
    let step = AdamWStep::new(0.1, 0.2).unwrap();
    let (next, state) = runtime
        .with_gradients_adamw(
            runtime.artifact().artifact_id(),
            &gradients(),
            &optimizer,
            step,
        )
        .unwrap();
    assert_eq!(state.revision(), 1);
    assert_eq!(state.state.slots.len(), 2); // No frozen/buffer slots.
    assert_eq!(state.state.slots[1], optimizer.state.slots[1]); // Absent placement.
    for (before, after) in runtime
        .artifact()
        .record()
        .placements
        .iter()
        .zip(&next.artifact().record().placements)
    {
        for (p, q) in before.tensors.iter().zip(&after.tensors) {
            if before.path != "site" || p.name != "gate" {
                assert_eq!(p.values, q.values);
                continue;
            }
            for (i, &g) in [1.5, -2., 0.].iter().enumerate() {
                let expected = step.candidate(config, p.values[i], g, 0., 0., 0).unwrap();
                assert_eq!(q.values[i], expected.value);
                assert_eq!(state.state.slots[0].first[i], expected.first);
                assert_eq!(state.state.slots[0].second[i], expected.second);
                assert_eq!(state.state.slots[0].steps, expected.steps);
            }
        }
    }
    assert_eq!(optimizer.revision(), 0);
    assert_eq!(state.geometry_artifact_id(), next.artifact().artifact_id());
    assert!(next
        .with_gradients_adamw(runtime.artifact().artifact_id(), &gradients(), &state, step,)
        .is_err());
    assert!(next
        .with_gradients_adamw(
            next.artifact().artifact_id(),
            &gradients(),
            &optimizer,
            step,
        )
        .is_err());
}

#[test]
fn geometry_and_moment_checkpoint_continuation_is_exact() {
    let runtime = fixture();
    let optimizer = NativeGeometryAdamW::new(runtime.artifact(), Default::default(), None).unwrap();
    let step = AdamWStep::new(0.01, 0.02).unwrap();
    let (runtime, optimizer) = runtime
        .with_gradients_adamw(
            runtime.artifact().artifact_id(),
            &gradients(),
            &optimizer,
            step,
        )
        .unwrap();
    let saved = runtime.artifact();
    let artifact = GeometryAdapterArtifact::from_json(
        saved.to_json().as_bytes(),
        saved.artifact_id(),
        &saved.record().base_model_identity,
    )
    .unwrap();
    let restored = NativeGeometryRuntime::new(artifact);
    let saved_state = optimizer.to_json().unwrap();
    let restored_state = NativeGeometryAdamW::from_json(
        saved_state.as_bytes(),
        &optimizer.checkpoint_id().unwrap(),
        restored.artifact(),
    )
    .unwrap();
    assert_eq!(saved_state, restored_state.to_json().unwrap());
    let a = runtime
        .with_gradients_adamw(
            runtime.artifact().artifact_id(),
            &gradients(),
            &optimizer,
            step,
        )
        .unwrap();
    let b = restored
        .with_gradients_adamw(
            restored.artifact().artifact_id(),
            &gradients(),
            &restored_state,
            step,
        )
        .unwrap();
    assert_eq!(a.0.artifact().to_json(), b.0.artifact().to_json());
    assert_eq!(a.1.to_json().unwrap(), b.1.to_json().unwrap());
}

#[test]
fn rejects_malformed_checkpoint_and_wrong_pair_before_resume() {
    let runtime = fixture();
    let optimizer = NativeGeometryAdamW::new(runtime.artifact(), Default::default(), None).unwrap();
    let json = optimizer.to_json().unwrap();
    let id = optimizer.checkpoint_id().unwrap();
    assert!(
        NativeGeometryAdamW::from_json(format!("{json} ").as_bytes(), &id, runtime.artifact(),)
            .is_err()
    );
    let original: serde_json::Value = serde_json::from_str(&json).unwrap();
    let changes = [
        ("/schema", serde_json::json!("unsupported")),
        ("/geometry_artifact_id", serde_json::json!("sha256:bad")),
        ("/beta1", serde_json::json!(1.)),
        ("/max_gradient_norm", serde_json::json!(0.)),
        ("/slots/0/shape", serde_json::json!([1, 3])),
        ("/slots/0/parameter", serde_json::json!("bias")),
        ("/slots/0/first", serde_json::json!([])),
        ("/slots/0/first/0", serde_json::json!(0.1)), // Nonzero moments at zero clock.
        ("/slots/0/second/0", serde_json::json!(-0.1)),
        ("/slots/0/steps", serde_json::json!(1)), // Ahead of revision.
        ("/slots", serde_json::json!([])),
    ];
    for (pointer, value) in changes {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        let bytes = serde_json::to_vec(&changed).unwrap();
        assert!(
            NativeGeometryAdamW::from_json(&bytes, &identity(&bytes), runtime.artifact()).is_err(),
            "{pointer}"
        );
    }
    let mut changed = original;
    changed["unexpected"] = serde_json::json!(true);
    let bytes = serde_json::to_vec(&changed).unwrap();
    assert!(NativeGeometryAdamW::from_json(&bytes, &identity(&bytes), runtime.artifact()).is_err());
    let bytes = vec![b' '; MAX_BYTES + 1];
    assert!(NativeGeometryAdamW::from_json(&bytes, "", runtime.artifact()).is_err());
}

#[test]
fn restored_clocks_must_match_the_native_update_owner() {
    let runtime = NativeGeometryRuntime::new(artifact(true, Some(true), 0.5));
    let optimizer = NativeGeometryAdamW::new(runtime.artifact(), Default::default(), None).unwrap();
    let original: serde_json::Value = serde_json::from_str(&optimizer.to_json().unwrap()).unwrap();
    let mut different_ages = original.clone();
    different_ages["revision"] = serde_json::json!(1);
    different_ages["slots"][0]["steps"] = serde_json::json!(1);
    let mut no_updates = original;
    no_updates["revision"] = serde_json::json!(1);
    for value in [different_ages, no_updates] {
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(
            NativeGeometryAdamW::from_json(&bytes, &identity(&bytes), runtime.artifact()).is_err()
        );
    }

    let runtime = fixture(); // One independently updatable parameter per placement.
    let optimizer = NativeGeometryAdamW::new(runtime.artifact(), Default::default(), None).unwrap();
    let mut record: serde_json::Value =
        serde_json::from_str(&optimizer.to_json().unwrap()).unwrap();
    for slot in record["slots"].as_array_mut().unwrap() {
        slot["steps"] = serde_json::json!(2);
    }
    for revision in [2, 3, 4] {
        record["revision"] = serde_json::json!(revision);
        let bytes = serde_json::to_vec(&record).unwrap();
        assert!(
            NativeGeometryAdamW::from_json(&bytes, &identity(&bytes), runtime.artifact()).is_ok()
        );
    }
    for revision in [1, 5] {
        record["revision"] = serde_json::json!(revision);
        let bytes = serde_json::to_vec(&record).unwrap();
        assert!(
            NativeGeometryAdamW::from_json(&bytes, &identity(&bytes), runtime.artifact()).is_err()
        );
    }
}

#[test]
fn invalid_candidates_clocks_and_frozen_derivatives_are_atomic() {
    let runtime = fixture();
    let optimizer = NativeGeometryAdamW::new(runtime.artifact(), Default::default(), None).unwrap();
    let id = runtime.artifact().artifact_id();
    let initial = optimizer.to_json().unwrap();
    let step = AdamWStep::new(0.01, 0.).unwrap();
    for (name, value) in [("gate", f32::MAX), ("bias", f32::NAN)] {
        let mut values = gradients();
        values.get_mut("site").unwrap().get_mut(name).unwrap()[2] = value;
        assert!(runtime
            .with_gradients_adamw(id, &values, &optimizer, step)
            .is_err());
        assert_eq!(initial, optimizer.to_json().unwrap());
        assert_eq!(id, runtime.artifact().artifact_id());
    }
    let mut invalid = gradients();
    invalid
        .get_mut("site")
        .unwrap()
        .insert("log_radius".into(), vec![0.]);
    assert!(runtime
        .with_gradients_adamw(id, &invalid, &optimizer, step)
        .is_err());
    let mut exhausted = optimizer.clone();
    exhausted.revision = u64::MAX;
    assert!(runtime
        .with_gradients_adamw(id, &gradients(), &exhausted, step)
        .is_err());
    exhausted.revision = u64::from(u32::MAX);
    exhausted.state.slots[0].steps = u32::MAX;
    assert!(runtime
        .with_gradients_adamw(id, &gradients(), &exhausted, step)
        .is_err());

    let zero = AdamWStep::new(0., f32::MAX).unwrap();
    let pair = runtime
        .with_gradients_adamw(id, &gradients(), &exhausted, zero)
        .unwrap();
    assert_eq!(pair.0.artifact().to_json(), runtime.artifact().to_json());
    assert_eq!(pair.1.to_json().unwrap(), exhausted.to_json().unwrap());
    let mut invalid = gradients();
    invalid.get_mut("site").unwrap().get_mut("bias").unwrap()[0] = f32::INFINITY;
    assert!(runtime
        .with_gradients_adamw(id, &invalid, &exhausted, zero)
        .is_err());
}

#[test]
fn public_batch_step_preserves_domain_and_original_pair() {
    let runtime = NativeGeometryRuntime::new(artifact(true, Some(true), 0.5));
    let optimizer = NativeGeometryAdamW::new(runtime.artifact(), Default::default(), None).unwrap();
    let batch = runtime.forward("site", &[0.2, -0.3, 0.4], 1).unwrap();
    let saved = optimizer.to_json().unwrap();
    let step = AdamWStep::new(0.01, 0.).unwrap();
    let (next, state) = runtime
        .step_adamw(&batch, &[0.1; 3], &optimizer, step)
        .unwrap();
    assert_ne!(
        next.artifact().artifact_id(),
        runtime.artifact().artifact_id()
    );
    assert_eq!(state.revision(), 1);
    assert_eq!(saved, optimizer.to_json().unwrap());
    let mut values =
        ParameterGradients::from([("site".into(), batch.vjp_parameters(&[0.1; 3]).unwrap())]);
    values
        .get_mut("site")
        .unwrap()
        .get_mut("log_radius")
        .unwrap()[0] = -1.;
    assert!(runtime
        .with_gradients_adamw(
            runtime.artifact().artifact_id(),
            &values,
            &optimizer,
            AdamWStep::new(1e6, 0.).unwrap(),
        )
        .is_err()); // Finite AdamW candidate, invalid Wave radius domain.
    assert_eq!(saved, optimizer.to_json().unwrap());
}
