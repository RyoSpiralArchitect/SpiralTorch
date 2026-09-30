use super::*;
use rand::RngCore;

const DATA_ID: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn dataset(count: usize) -> Arc<TensorVisionDataset> {
    Arc::new(
        TensorVisionDataset::from_samples(
            dataset_catalog()[0].clone(),
            (0..count)
                .map(|i| {
                    let values = (0..108)
                        .map(|j| ((i * 13 + j * 7) % 101) as f32 / 101.)
                        .collect();
                    DatasetSample::new(ImageTensor::new(3, 6, 6, values).unwrap())
                        .with_label(i.to_string())
                        .with_target(Tensor::from_vec(1, 1, vec![(i % 2) as f32]).unwrap())
                })
                .collect(),
        )
        .unwrap(),
    )
}

fn pipeline(jitter: bool) -> TransformPipeline {
    let mut pipeline = TransformPipeline::with_seed(29);
    pipeline
        .add(TransformOperation::Resize(Resize::new(8, 8).unwrap()))
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5).unwrap(),
        ))
        .add(TransformOperation::CenterCrop(
            CenterCrop::new(4, 4).unwrap(),
        ));
    if jitter {
        pipeline.add(TransformOperation::ColorJitter(
            ColorJitter::new(0.2, 0.1, 0.1, 0.05).unwrap(),
        ));
    }
    pipeline.add(TransformOperation::Normalize(
        Normalize::new(vec![0.5], vec![0.25]).unwrap(),
    ));
    pipeline
}

fn loader(count: usize, batch: usize, jitter: bool) -> DataLoader<TensorVisionDataset> {
    let mut loader = DataLoader::new(dataset(count), batch, Some(17))
        .unwrap()
        .with_pipeline(pipeline(jitter));
    loader.enable_shuffle(true);
    loader
}

fn next(loader: &mut DataLoader<TensorVisionDataset>) -> VisionBatch {
    if let Some(batch) = loader.next_batch().unwrap() {
        return batch;
    }
    loader.reset();
    loader.next_batch().unwrap().unwrap()
}

fn batch_record(batch: VisionBatch) -> (Vec<Option<String>>, Vec<u32>) {
    (
        batch.labels,
        batch
            .images
            .iter()
            .flat_map(|image| image.as_slice().iter().map(|v| v.to_bits()))
            .collect(),
    )
}

#[test]
fn input_checkpoint_resumes_order_augmentation_and_epoch_shuffle() {
    let mut source = loader(11, 3, true);
    for _ in 0..37 {
        next(&mut source);
    }
    let json = source.checkpoint(DATA_ID).unwrap().to_json().unwrap();
    let state = DataLoaderCheckpoint::from_json(&json).unwrap();
    assert_eq!(state.dataset_len(), 11);
    let expected: Vec<_> = (37..100).map(|_| batch_record(next(&mut source))).collect();
    let end = source.checkpoint(DATA_ID).unwrap();
    drop(source);
    let mut restored = loader(11, 3, true);
    restored.restore_checkpoint(DATA_ID, &state).unwrap();
    for batch in expected {
        assert_eq!(batch_record(next(&mut restored)), batch);
    }
    assert_eq!(restored.checkpoint(DATA_ID).unwrap(), end);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn pinned_shuffle_preserves_previous_native_order() {
    let mut loader = loader(37, 3, false);
    let mut old_rng = StdRng::seed_from_u64(17);
    for _ in 0..5 {
        let mut old_order: Vec<_> = (0..37).collect();
        for i in (1..old_order.len()).rev() {
            let j = old_rng.gen_range(0..=i);
            old_order.swap(i, j);
        }
        assert_eq!(loader.order, old_order);
        assert_eq!(
            loader.shuffle_rng.clone().next_u64(),
            old_rng.clone().next_u64()
        );
        loader.reset();
    }
}

#[test]
fn rng_roundtrip_preserves_partial_blocks_and_large_json_counters() {
    let mut rng = ChaCha12Rng::seed_from_u64(31);
    rng.set_stream(u64::MAX);
    rng.set_word_pos((1_u128 << 63) + 19);
    rng.next_u32();
    let state = RngCheckpoint::capture(&rng);
    let json = serde_json::to_string(&state).unwrap();
    assert!(json.contains("\"18446744073709551615\""));
    let mut restored = serde_json::from_str::<RngCheckpoint>(&json)
        .unwrap()
        .restore()
        .unwrap();
    for _ in 0..1000 {
        assert_eq!(rng.next_u32(), restored.next_u32());
        assert_eq!(rng.next_u64(), restored.next_u64());
    }
    for position in ["-1".into(), "01".into(), (1_u128 << 68).to_string()] {
        let mut bad = state.clone();
        bad.word_position = position;
        assert!(bad.restore().is_err());
    }
}

#[test]
fn invalid_input_restore_is_atomic_and_configuration_bound() {
    let mut target = loader(11, 3, true);
    next(&mut target);
    let before = target.checkpoint(DATA_ID).unwrap();
    let mut cases = Vec::new();
    let mut changed = before.clone();
    changed.order[1] = changed.order[0];
    cases.push(changed);
    let mut changed = before.clone();
    changed.position = 1;
    cases.push(changed);
    let mut changed = before.clone();
    changed.batch_size = 1;
    cases.push(changed);
    let mut changed = before.clone();
    changed.dataset_sha256 = "b".repeat(64);
    cases.push(changed);
    let mut changed = before.clone();
    changed.pipeline = None;
    cases.push(changed);
    let mut changed = before.clone();
    changed.pipeline.as_mut().unwrap().operations.pop();
    cases.push(changed);
    let mut changed = before.clone();
    changed.shuffle_rng.algorithm = "unknown".into();
    cases.push(changed);
    let mut changed = before.clone();
    changed.consumption = "retry_on_rejection".into();
    cases.push(changed);
    for bad in cases {
        assert!(target.restore_checkpoint(DATA_ID, &bad).is_err());
        assert_eq!(target.checkpoint(DATA_ID).unwrap(), before);
    }
    assert!(loader(12, 3, true)
        .restore_checkpoint(DATA_ID, &before)
        .is_err());
    assert!(DataLoaderCheckpoint::from_json(
        &before.to_json().unwrap().replace(LOADER_SCHEMA, "unknown")
    )
    .is_err());
    assert!(target.checkpoint("not-a-content-hash").is_err());
}

#[test]
fn transform_checkpoint_preserves_rng_and_validates_float_bits() {
    let mut pipeline = pipeline(true);
    let image = dataset(1).get(0).unwrap().image;
    pipeline.apply(&mut image.clone()).unwrap();
    let state = pipeline.checkpoint().unwrap();
    let mut expected = image.clone();
    pipeline.apply(&mut expected).unwrap();
    pipeline
        .restore_checkpoint(
            &TransformPipelineCheckpoint::from_json(&state.to_json().unwrap()).unwrap(),
        )
        .unwrap();
    let mut actual = image;
    pipeline.apply(&mut actual).unwrap();
    assert_eq!(actual, expected);
    let before = pipeline.checkpoint().unwrap();
    let mut bad = state;
    bad.operations[0] = TransformConfig::RandomHorizontalFlip {
        probability: f32::NAN.to_bits(),
    };
    assert!(pipeline.restore_checkpoint(&bad).is_err());
    assert_eq!(pipeline.checkpoint().unwrap(), before);
}

#[cfg(all(feature = "wgpu", feature = "nn", not(target_arch = "wasm32")))]
#[test]
fn resident_input_and_classifier_resume_after_rejected_update() {
    use crate::models::{
        ConvNeXtClassifier, ConvNeXtClassifierCheckpoint, ConvNeXtConfig,
        ResidentConvNeXtClassifier,
    };
    use st_backend_wgpu::{resident_tensor::TensorDevice, resident_training::TrainingError};
    use st_nn::loss::{CrossEntropyWithLogits, Loss};
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("vision.input.resume").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let make_loader = || {
        let mut loader = loader(20, 2, false);
        loader
            .pipeline
            .as_mut()
            .unwrap()
            .set_gpu_dispatcher(TransformDispatcher::from_runtime(device.runtime()).unwrap());
        loader
    };
    let step = |model: &mut ResidentConvNeXtClassifier,
                loader: &mut DataLoader<TensorVisionDataset>,
                index: usize| {
        let batch = match loader.next_resident_batch(&device).unwrap() {
            Some(batch) => batch,
            None => {
                loader.reset();
                loader.next_resident_batch(&device).unwrap().unwrap()
            }
        };
        let values = batch.images.snapshot().unwrap().read().unwrap();
        let forward = model.forward(&batch.images).unwrap();
        let rejection = index == 36 || index == 71;
        let labels = if rejection {
            device.upload(&[2, 1], &[0., 2.]).unwrap()
        } else {
            batch.upload_targets().unwrap()
        };
        let loss = CrossEntropyWithLogits::default()
            .evaluate_resident(forward.prediction(), &labels)
            .unwrap();
        let gradient = model
            .backward(&forward, loss.prediction_gradient())
            .unwrap();
        let receipt = model
            .sgd(&gradient, 0.001)
            .unwrap()
            .snapshot()
            .unwrap()
            .read();
        if rejection {
            assert!(matches!(receipt, Err(TrainingError::Rejected { .. })));
        } else {
            assert_eq!(receipt.unwrap(), index as u64 + 1);
        }
        (
            batch.labels,
            values.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        )
    };
    let config = ConvNeXtConfig {
        input_channels: 3,
        input_hw: (4, 4),
        stage_dims: vec![2, 3],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        curvature: -1.,
        epsilon: 0.001,
    };
    let source = ConvNeXtClassifier::new(config, 2, 43).unwrap();
    let mut model = source.compile_resident_training(device.clone(), 2).unwrap();
    let mut input = make_loader();
    for index in 0..37 {
        step(&mut model, &mut input, index);
    }
    // The last submitted batch was rejected. Its input is consumed by contract.
    let input_json = input.checkpoint(DATA_ID).unwrap().to_json().unwrap();
    let model_json = model
        .checkpoint_snapshot()
        .unwrap()
        .read()
        .unwrap()
        .to_json()
        .unwrap();
    let expected: Vec<_> = (37..100).map(|i| step(&mut model, &mut input, i)).collect();
    let end_input = input.checkpoint(DATA_ID).unwrap();
    let end_model = model
        .checkpoint_snapshot()
        .unwrap()
        .read()
        .unwrap()
        .to_json()
        .unwrap();
    drop(model);
    drop(input);
    let mut resumed = ConvNeXtClassifierCheckpoint::from_json(&model_json)
        .unwrap()
        .restore_resident(device.clone())
        .unwrap();
    let mut resumed_input = make_loader();
    resumed_input
        .restore_checkpoint(
            DATA_ID,
            &DataLoaderCheckpoint::from_json(&input_json).unwrap(),
        )
        .unwrap();
    assert!(resumed_input
        .pipeline
        .as_ref()
        .unwrap()
        .has_gpu_dispatcher());
    for (index, record) in (37..100).zip(expected) {
        assert_eq!(step(&mut resumed, &mut resumed_input, index), record);
    }
    assert_eq!(resumed_input.checkpoint(DATA_ID).unwrap(), end_input);
    assert_eq!(
        resumed
            .checkpoint_snapshot()
            .unwrap()
            .read()
            .unwrap()
            .to_json()
            .unwrap(),
        end_model
    );
}
