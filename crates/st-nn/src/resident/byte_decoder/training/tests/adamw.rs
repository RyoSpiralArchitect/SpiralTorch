use super::*;
use st_kernel_contracts::classification::ClassReduction;

fn capture(model: &ResidentByteDecoder) -> ByteDecoderCheckpoint {
    model.checkpoint_snapshot().unwrap().read().unwrap()
}

fn train(model: &mut ResidentByteDecoder, frozen: bool) -> f32 {
    let host = ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap();
    let batch = model.prepare_batch(&host).unwrap();
    let forward = model.forward(&batch).unwrap();
    let loss = forward
        .next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.0).unwrap())
        .unwrap();
    let ce = read(loss.value()).unwrap()[0];
    let gradient = model
        .backward(&forward, loss.prediction_gradient())
        .unwrap();
    let mut steps = vec![AdamWStep::new(0.005, 0.01).unwrap(); model.parameter_layout().len()];
    for i in model.parameter_layout().geometry().unwrap().all() {
        steps[i] = AdamWStep::new(if frozen { 0.0 } else { 0.005 }, 0.0).unwrap();
    }
    model
        .adamw_with_steps(&gradient, &steps)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert!(model
        .backward(&forward, loss.prediction_gradient())
        .is_err());
    ce
}

#[test]
fn byte_adamw_joint_geometry_learning_freeze_reject_and_exact_resume() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let runtime = st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.adamw")
        .unwrap()
        .0;
    let plan = super::query_pullback::contextual_plan();
    let geometry = plan.parameter_layout().geometry().unwrap().all();
    let mut model = plan.compile_training_wgpu(runtime.clone()).unwrap();
    model.enable_adamw(AdamWConfig::default()).unwrap();
    let start = capture(&model);
    let initial_loss = train(&mut model, false);
    let trained = capture(&model);
    assert!(geometry
        .clone()
        .any(|i| trained.adamw_state().unwrap().slots[i]
            .first
            .iter()
            .any(|v| *v != 0.0)));
    assert!(!trained.geometry_parameters_match(&start).unwrap());
    train(&mut model, true);
    let frozen = capture(&model);
    assert!(frozen.geometry_parameters_match(&trained).unwrap());
    for i in geometry.clone() {
        let before = &trained.adamw_state().unwrap().slots[i];
        let after = &frozen.adamw_state().unwrap().slots[i];
        assert_eq!(before, after);
        assert_eq!(after.steps, 1);
    }
    let json = frozen.to_json().unwrap();
    let saved = ByteDecoderCheckpoint::from_json(&json).unwrap();
    assert_eq!(saved.to_json().unwrap(), json);
    let mut resumed = saved.restore_wgpu(runtime).unwrap();
    for freeze in [false, true, false] {
        let loss = train(&mut model, freeze);
        let other_loss = train(&mut resumed, freeze);
        assert_eq!(loss.to_bits(), other_loss.to_bits());
        assert!(loss < initial_loss);
        assert_eq!(
            capture(&model).to_json().unwrap(),
            capture(&resumed).to_json().unwrap()
        );
    }
    // One poisoned late parameter rejects geometry, head and optimizer together.
    let before = capture(&model);
    let batch = model
        .prepare_batch(&ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap())
        .unwrap();
    let forward = model.forward(&batch).unwrap();
    let seed = model
        .tensor_device()
        .upload(
            model.output_layout().shape(),
            &vec![0.01; model.output_layout().len()],
        )
        .unwrap();
    let mut gradient = model.backward(&forward, &seed).unwrap();
    let snapshot = model.parameter_snapshot();
    let last = gradient.parameters.len() - 1;
    gradient.parameters[last] = model
        .tensor_device()
        .upload(
            snapshot.values()[last].layout().shape(),
            &vec![f32::MAX; snapshot.values()[last].layout().len()],
        )
        .unwrap();
    gradient.bound = snapshot
        .bind_gradients(gradient.parameters.clone())
        .unwrap();
    assert!(model.sgd(&gradient, 0.01).is_err());
    assert!(matches!(
        model
            .adamw(&gradient, AdamWStep::new(0.01, 0.01).unwrap())
            .unwrap()
            .snapshot()
            .unwrap()
            .read(),
        Err(TrainingError::Rejected { .. })
    ));
    let after = capture(&model);
    let mut expected: serde_json::Value = serde_json::from_str(&before.to_json().unwrap()).unwrap();
    expected["attempted_revision"] =
        serde_json::json!((before.attempted_revision() + 1).to_string());
    let actual: serde_json::Value = serde_json::from_str(&after.to_json().unwrap()).unwrap();
    assert_eq!(actual, expected);
}
