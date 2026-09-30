use super::*;

#[cfg(not(target_arch = "wasm32"))]
fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("vision.normalize.tests")
            .unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(TensorDevice::new(runtime).unwrap())
}

fn pipeline() -> TransformPipeline {
    let mut pipeline = TransformPipeline::with_seed(271);
    pipeline
        .add(TransformOperation::Normalize(
            Normalize::new(vec![0.25, 0.5], vec![0.5, 2.]).unwrap(),
        ))
        .add(TransformOperation::Resize(Resize::new(7, 9).unwrap()))
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5).unwrap(),
        ))
        .add(TransformOperation::CenterCrop(
            CenterCrop::new(5, 5).unwrap(),
        ))
        .add(TransformOperation::Normalize(
            Normalize::new(vec![0.125], vec![0.75]).unwrap(),
        ))
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5).unwrap(),
        ));
    pipeline
}

fn images() -> Vec<ImageTensor> {
    (0..3)
        .map(|n| {
            ImageTensor::new(
                2,
                8,
                10,
                (0..160)
                    .map(|i| ((i * 31 + n * 17) % 257) as f32 / 256.)
                    .collect(),
            )
            .unwrap()
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn gpu_pipeline(pipeline: TransformPipeline, device: &TensorDevice) -> TransformPipeline {
    pipeline.with_gpu_dispatcher(TransformDispatcher::from_runtime(device.runtime()).unwrap())
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resident_normalization_batch_matches_cpu_and_reuses_prepared_statistics() {
    let Some(device) = device() else {
        return;
    };
    let mut cpu = pipeline();
    let mut gpu = gpu_pipeline(cpu.clone(), &device);
    let inputs = images();
    let mut cached = None;
    for _ in 0..4 {
        let mut expected = Vec::new();
        for image in &inputs {
            let mut image = image.clone();
            cpu.apply(&mut image).unwrap();
            expected.extend_from_slice(image.as_slice());
        }
        let actual = gpu.apply_resident_batch(&inputs, &device).unwrap();
        assert_eq!(actual.layout().shape(), &[3, 2, 5, 5]);
        for (&a, b) in actual
            .snapshot()
            .unwrap()
            .read()
            .unwrap()
            .iter()
            .zip(expected)
        {
            assert!((a - b).abs() < 2e-6, "{a} != {b}");
        }
        let prepared = gpu.normalizers[0].as_ref().unwrap();
        if let Some(previous) = &cached {
            assert!(Shared::ptr_eq(prepared, previous));
        }
        cached = Some(prepared.clone());
    }
    let before = gpu.rng.clone().gen::<u64>();
    assert!(gpu
        .apply_resident_batch(&[ImageTensor::zeros(3, 8, 10).unwrap()], &device)
        .is_err());
    assert_eq!(gpu.rng.clone().gen::<u64>(), before);
    let (shape, values) = pack_images(&inputs).unwrap();
    let view = device
        .upload(&shape, &values)
        .unwrap()
        .permute(&[0, 1, 3, 2])
        .unwrap();
    let mut expected = Vec::new();
    for image in &inputs {
        let mut image = ImageTensor::new(
            2,
            10,
            8,
            (0..160)
                .map(|i| {
                    let c = i / 80;
                    let hw = i % 80;
                    image.as_slice()[c * 80 + (hw % 8) * 10 + hw / 8]
                })
                .collect(),
        )
        .unwrap();
        cpu.apply(&mut image).unwrap();
        expected.extend_from_slice(image.as_slice());
    }
    let actual = gpu
        .apply_from_resident(&view)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2e-6);
    }
    assert!(!Shared::ptr_eq(
        gpu.normalizers[0].as_ref().unwrap(),
        cached.as_ref().unwrap()
    ));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resident_normalization_cannot_hide_failure_by_cropping_or_mapping_retry() {
    let Some(device) = device() else {
        return;
    };
    let mut pipeline = gpu_pipeline(TransformPipeline::with_seed(11), &device);
    pipeline
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5).unwrap(),
        ))
        .add(TransformOperation::Normalize(
            Normalize::new(vec![0.], vec![0.5]).unwrap(),
        ))
        .add(TransformOperation::CenterCrop(
            CenterCrop::new(1, 1).unwrap(),
        ));
    let mut image =
        ImageTensor::new(1, 3, 3, vec![f32::MAX, 0., 0., 0., 1., 0., 0., 0., 0.]).unwrap();
    let initial = image.clone();
    let rng = pipeline.rng.clone().gen::<u64>();
    assert!(pollster::block_on(pipeline.apply_gpu_async(&mut image, &device)).is_err());
    assert_eq!(image, initial);
    assert_eq!(pipeline.rng.clone().gen::<u64>(), rng);
    let output = pipeline.apply_resident(&image, &device).unwrap();
    assert!(output.relu().unwrap().snapshot().unwrap().read().is_err());
    assert_ne!(pipeline.rng.clone().gen::<u64>(), rng);
    let input = device
        .upload(&[1, 1, 3, 3], &[f32::MAX; 9])
        .unwrap()
        .mul(&device.upload(&[], &[2.]).unwrap())
        .unwrap();
    let mut crop = gpu_pipeline(TransformPipeline::new(), &device);
    crop.add(TransformOperation::CenterCrop(
        CenterCrop::new(1, 1).unwrap(),
    ));
    assert!(crop
        .apply_from_resident(&input)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .is_err());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resident_dataloader_preserves_targets_and_has_explicit_submission_cursor() {
    let Some(device) = device() else {
        return;
    };
    let dataset = TensorVisionDataset::from_samples(
        dataset_catalog()[0].clone(),
        images()
            .into_iter()
            .enumerate()
            .map(|(i, image)| {
                DatasetSample::new(image)
                    .with_label(i.to_string())
                    .with_target(Tensor::from_vec(1, 1, vec![i as f32]).unwrap())
            })
            .collect(),
    )
    .unwrap();
    let mut loader = DataLoader::new(Arc::new(dataset), 2, Some(3))
        .unwrap()
        .with_pipeline(gpu_pipeline(pipeline(), &device));
    let batch = loader.next_resident_batch(&device).unwrap().unwrap();
    assert_eq!(loader.position, 2);
    assert_eq!(batch.images.layout().shape(), &[2, 2, 5, 5]);
    assert_eq!(batch.labels, vec![Some("0".into()), Some("1".into())]);
    assert_eq!(
        batch
            .upload_targets()
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![0., 1.]
    );
    let mut tail = loader.next_resident_batch(&device).unwrap().unwrap();
    assert_eq!(tail.images.layout().shape(), &[1, 2, 5, 5]);
    tail.targets[0] = None;
    assert!(tail.upload_targets().is_err());
    assert!(loader.next_resident_batch(&device).unwrap().is_none());
    loader.reset();
    assert_eq!(loader.position, 0);
    loader
        .pipeline
        .as_mut()
        .unwrap()
        .add(TransformOperation::ColorJitter(
            ColorJitter::new(0.1, 0., 0., 0.).unwrap(),
        ));
    let rng = loader.pipeline.as_ref().unwrap().rng.clone().gen::<u64>();
    assert!(loader.next_resident_batch(&device).is_err());
    assert_eq!(loader.position, 0);
    assert_eq!(
        loader.pipeline.as_ref().unwrap().rng.clone().gen::<u64>(),
        rng
    );
}
