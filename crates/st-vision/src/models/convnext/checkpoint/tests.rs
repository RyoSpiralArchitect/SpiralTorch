use super::*;

fn config() -> ConvNeXtConfig {
    ConvNeXtConfig {
        input_channels: 2,
        input_hw: (8, 8),
        stage_dims: vec![3, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        curvature: -1.0,
        epsilon: 1e-3,
    }
}

fn checkpoint() -> ConvNeXtTrainingCheckpoint {
    let config = config();
    let model = ConvNeXtBackbone::new(config.clone()).unwrap();
    let mut parameters = Vec::new();
    model
        .visit_parameters(&mut |p| {
            parameters.push(StoredParameter {
                name: p.name().into(),
                shape: [p.value().shape().0, p.value().shape().1],
                values: p.value().data().to_vec(),
            });
            Ok(())
        })
        .unwrap();
    ConvNeXtTrainingCheckpoint::new(config, 2, 7, parameters).unwrap()
}

fn state_bits(model: &ConvNeXtBackbone) -> Vec<Vec<u32>> {
    let mut bits = Vec::new();
    model
        .visit_parameters(&mut |p| {
            bits.push(p.value().data().iter().map(|v| v.to_bits()).collect());
            Ok(())
        })
        .unwrap();
    bits
}

#[test]
fn checkpoint_budget_matches_actual_topology() {
    for depths in [vec![0, 0], vec![1, 1], vec![2, 3]] {
        let config = ConvNeXtConfig {
            stage_depths: depths,
            ..config()
        };
        let model = ConvNeXtBackbone::new(config.clone()).unwrap();
        let mut actual = (0, 0);
        model
            .visit_parameters(&mut |p| {
                actual.0 += 1;
                actual.1 += p.value().data().len();
                Ok(())
            })
            .unwrap();
        assert_eq!(config.parameter_budget().unwrap(), actual);
    }
    assert!(ConvNeXtConfig::default().parameter_budget().unwrap().1 < MAX_PARAMETER_VALUES);
    for bad in [
        ConvNeXtConfig {
            stage_dims: vec![usize::MAX, 4],
            ..config()
        },
        ConvNeXtConfig {
            stage_depths: vec![usize::MAX, 1],
            ..config()
        },
        ConvNeXtConfig {
            patch_size: (0, 1),
            ..config()
        },
        ConvNeXtConfig {
            input_hw: (1, 1),
            ..config()
        },
    ] {
        assert!(ConvNeXtBackbone::new(bad).is_err());
    }
}

#[test]
fn checkpoint_json_preserves_f32_bits_and_host_handoff() {
    let mut state = checkpoint();
    for (slot, value) in state.parameters[0].values.iter_mut().zip([
        -0.0,
        f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        0.12345679,
    ]) {
        *slot = value;
    }
    let restored = ConvNeXtTrainingCheckpoint::from_json(&state.to_json().unwrap()).unwrap();
    assert_eq!(restored.attempted_updates(), 7);
    assert_eq!(restored.batch_size(), 2);
    assert_eq!(restored.config(), &config());
    let host = restored.to_host().unwrap();
    for (actual, expected) in state_bits(&host).iter().zip(&state.parameters) {
        assert_eq!(
            *actual,
            expected
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn checkpoint_rejects_malformed_headers_parameters_and_metadata() {
    let json: serde_json::Value = serde_json::from_str(&checkpoint().to_json().unwrap()).unwrap();
    for (key, value) in [
        ("schema", serde_json::json!("future.v2")),
        ("optimizer", serde_json::json!("adam")),
        ("batch", serde_json::json!(0)),
        ("attempted_updates", serde_json::json!(u64::MAX)),
        ("unknown", serde_json::json!(true)),
    ] {
        let mut invalid = json.clone();
        invalid[key] = value;
        assert!(
            ConvNeXtTrainingCheckpoint::from_json(&invalid.to_string()).is_err(),
            "{key}"
        );
    }
    let mut unknown_config = json.clone();
    unknown_config["config"]["future_layer"] = serde_json::json!(true);
    assert!(ConvNeXtTrainingCheckpoint::from_json(&unknown_config.to_string()).is_err());
    let mut duplicate = checkpoint();
    duplicate.parameters[1].name = duplicate.parameters[0].name.clone();
    assert!(duplicate.to_json().is_err());
    let mut truncated = checkpoint();
    truncated.parameters.last_mut().unwrap().values.pop();
    assert!(truncated.to_host().is_err());
    let mut nonfinite = checkpoint();
    nonfinite.parameters.last_mut().unwrap().values[0] = f32::NAN;
    assert!(nonfinite.to_json().is_err());
    let mut dimensions = checkpoint();
    dimensions.config.stage_dims[0] = usize::MAX;
    assert!(dimensions.to_host().is_err());
    let mut excessive = checkpoint();
    excessive.config.stage_depths[0] = MAX_PARAMETER_TENSORS;
    assert!(excessive.to_host().is_err());
    assert!(ConvNeXtTrainingCheckpoint::from_json("{}").is_err());
}

#[test]
fn checkpoint_host_restore_preflights_all_parameters_before_commit() {
    let good = checkpoint();
    let mut target = ConvNeXtBackbone::new(config()).unwrap();
    let before = state_bits(&target);
    let mut bad = good.clone();
    bad.parameters[0].values[0] = 123.0;
    bad.parameters.last_mut().unwrap().name = "wrong.last.parameter".into();
    assert!(bad.restore_host(&mut target).is_err());
    assert_eq!(state_bits(&target), before);
    let mut wrong_shape = good.clone();
    wrong_shape.parameters.last_mut().unwrap().shape.swap(0, 1);
    assert!(wrong_shape.restore_host(&mut target).is_err());
    assert_eq!(state_bits(&target), before);
    let mut different_config = ConvNeXtBackbone::new(ConvNeXtConfig {
        epsilon: 1e-2,
        ..config()
    })
    .unwrap();
    assert!(good.restore_host(&mut different_config).is_err());
    target.attach_realgrad(0.01).unwrap();
    assert!(good.restore_host(&mut target).is_err());
    assert_eq!(state_bits(&target), before);
}
