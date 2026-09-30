use super::*;
use crate::models::convnext::checkpoint::StoredParameter;
use crate::{create_classification_model, FeatureStage, ImageTensor, ModelKind};
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};

fn config() -> ConvNeXtConfig {
    ConvNeXtConfig {
        input_channels: 1,
        input_hw: (8, 8),
        stage_dims: vec![2, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        epsilon: 1e-3,
        ..Default::default()
    }
}
fn state(model: &ConvNeXtClassifier) -> Vec<StoredParameter> {
    let mut state = Vec::new();
    model
        .visit_parameters(&mut |p| {
            state.push(StoredParameter {
                name: p.name().into(),
                shape: [p.value().shape().0, p.value().shape().1],
                values: p.value().data().to_vec(),
            });
            Ok(())
        })
        .unwrap();
    state
}
fn bits(model: &ConvNeXtClassifier) -> Vec<Vec<u32>> {
    state(model)
        .iter()
        .map(|p| p.values.iter().map(|v| v.to_bits()).collect())
        .collect()
}

#[test]
fn classifier_initialization_and_inference_adapter_share_actual_model() {
    let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let model = ConvNeXtClassifier::new(config(), 3, 42).unwrap();
    let other = ConvNeXtClassifier::new(config(), 3, 43).unwrap();
    assert_eq!(
        bits(&model),
        bits(&ConvNeXtClassifier::new(config(), 3, 42).unwrap())
    );
    for (a, b) in state(&model).iter().zip(state(&other)) {
        if a.name.ends_with("::weight") {
            assert_ne!(a.values, b.values, "{}", a.name);
        }
    }
    let images = [ImageTensor::new(1, 8, 8, (0..64).map(|i| i as f32 / 64.).collect()).unwrap()];
    let x = Tensor::from_vec(1, 64, images[0].as_slice().to_vec()).unwrap();
    let expected = model.forward(&x).unwrap();
    let expected_head = model.features(&x).unwrap();
    let count: usize = state(&model).iter().map(|p| p.values.len()).sum();
    let adapter = model.into_vision_model().unwrap();
    assert_eq!(adapter.forward(&images).unwrap(), expected);
    assert_eq!(
        adapter
            .extract_features(FeatureStage::Head, &images[0])
            .unwrap(),
        expected_head
    );
    assert_eq!(
        adapter
            .extract_features(FeatureStage::Stem, &images[0])
            .unwrap()
            .shape(),
        (2, 16)
    );
    assert_eq!(
        adapter
            .extract_features(FeatureStage::Logits, &images[0])
            .unwrap(),
        expected
    );
    assert_eq!(adapter.parameter_count(), Some(count));
    assert!(!adapter.metadata().has_pretrained);
    assert!(adapter.forward(&[]).is_err());
    assert!(adapter
        .forward(&[ImageTensor::zeros(1, 7, 8).unwrap()])
        .is_err());
}

#[test]
fn classifier_input_and_head_gradients_match_finite_difference() {
    let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut model = ConvNeXtClassifier::new(config(), 2, 7).unwrap();
    let input = Tensor::from_fn(2, 64, |r, c| ((c * 7 + r * 13) % 31) as f32 / 31.).unwrap();
    let seed = Tensor::from_vec(2, 2, vec![0.3, -0.2, 0.6, 0.4]).unwrap();
    let dx = model.backward(&input, &seed).unwrap();
    let objective = |model: &ConvNeXtClassifier, x: &Tensor| -> f32 {
        model
            .forward(x)
            .unwrap()
            .data()
            .iter()
            .zip(seed.data())
            .map(|(a, b)| a * b)
            .sum()
    };
    let epsilon = 1e-3;
    for index in [0, 3, 70, 100] {
        let mut a = input.clone();
        a.data_mut()[index] += epsilon;
        let mut b = input.clone();
        b.data_mut()[index] -= epsilon;
        let numerical = (objective(&model, &a) - objective(&model, &b)) / (2. * epsilon);
        assert!((dx.data()[index] - numerical).abs() < 1e-2 * (1. + numerical.abs()));
    }
    let mut gradients = Vec::new();
    model
        .visit_parameters(&mut |p| {
            gradients.push(p.gradient().unwrap().data().to_vec());
            Ok(())
        })
        .unwrap();
    assert_eq!(gradients.len(), 24);
    for index in [0, 2, 10, 12, 20, 22, 23] {
        let old = state(&model)[index].values[0];
        let set = |model: &mut ConvNeXtClassifier, value| {
            let mut i = 0;
            model
                .visit_parameters_mut(&mut |p| {
                    if i == index {
                        p.value_mut().data_mut()[0] = value;
                    }
                    i += 1;
                    Ok(())
                })
                .unwrap();
        };
        set(&mut model, old + epsilon);
        let a = objective(&model, &input);
        set(&mut model, old - epsilon);
        let b = objective(&model, &input);
        set(&mut model, old);
        let numerical = (a - b) / (2. * epsilon);
        assert!(
            (gradients[index][0] - numerical).abs() < 1e-2 * (1. + numerical.abs()),
            "parameter {index}"
        );
    }
}

#[test]
fn classifier_checkpoint_is_whole_model_and_preserves_failed_handoff() {
    let mut model = ConvNeXtClassifier::new(config(), 3, 7).unwrap();
    let saved = ConvNeXtClassifierCheckpoint::new(config(), 2, 9, 3, state(&model)).unwrap();
    assert_eq!(bits(&saved.to_host().unwrap()), bits(&model));
    let json = saved.to_json().unwrap();
    assert!(ConvNeXtTrainingCheckpoint::from_json(&json).is_err());
    assert_eq!(
        ConvNeXtClassifierCheckpoint::from_json(&json)
            .unwrap()
            .attempted_updates(),
        9
    );
    let before = bits(&model);
    let mut bad: serde_json::Value = serde_json::from_str(&json).unwrap();
    bad["backbone"]["parameters"][0]["values"][0] = serde_json::json!(123.);
    bad["backbone"]["parameters"][21]["name"] = serde_json::json!("wrong_final_name");
    let bad = ConvNeXtClassifierCheckpoint::from_json(&bad.to_string()).unwrap();
    assert!(bad.restore_host(&mut model).is_err());
    assert_eq!(bits(&model), before);
    for key in ["name", "shape", "values"] {
        let mut bad: serde_json::Value = serde_json::from_str(&json).unwrap();
        bad["head"][1][key] = serde_json::Value::Null;
        assert!(ConvNeXtClassifierCheckpoint::from_json(&bad.to_string()).is_err());
    }
    model.head.attach_realgrad(0.01).unwrap();
    assert!(saved.restore_host(&mut model).is_err());
    assert_eq!(bits(&model), before);
}

#[test]
fn convnext_factory_constructs_the_full_tiny_parameter_topology() {
    let model = create_classification_model(ModelKind::ConvNeXtTiny, 3, Some(42)).unwrap();
    let config = ConvNeXtConfig::default();
    let expected = config.parameter_budget().unwrap().1 + 768 * 3 + 3;
    assert_eq!(model.parameter_count(), Some(expected));
    assert_eq!(model.metadata().name, "convnext_tiny");
    assert!(!model.metadata().has_pretrained);
}
