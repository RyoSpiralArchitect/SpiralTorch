use st_backend_wgpu::transform::TransformDispatcher;
use st_vision::{
    dataset_catalog, CenterCrop, DataLoader, DatasetSample, Normalize, RandomHorizontalFlip,
    Resize, TensorVisionDataset, TransformOperation, TransformPipeline,
};
use std::sync::Arc;

/// Synthetic images exercise the actual loader-to-classifier boundary, not
/// real-dataset quality. All observations occur after four submitted updates.
pub(super) async fn run(device: &TensorDevice) -> Result<serde_json::Value> {
    let config = ConvNeXtConfig {
        input_channels: 1,
        input_hw: (8, 8),
        stage_dims: vec![2, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        epsilon: 1e-3,
        ..Default::default()
    };
    let mut cpu = ConvNeXtClassifier::new(config.clone(), 2, 17)?;
    let mut gpu =
        ConvNeXtClassifier::new(config, 2, 17)?.compile_resident_training(device.clone(), 2)?;
    let dataset = TensorVisionDataset::from_samples(
        dataset_catalog()[0].clone(),
        (0..2)
            .map(|n| {
                Ok(DatasetSample::new(ImageTensor::new(
                    1,
                    12,
                    14,
                    (0..168)
                        .map(|i| ((i * 31 + n * 17) % 257) as f32 / 256.)
                        .collect(),
                )?)
                .with_target(Tensor::from_vec(1, 1, vec![n as f32])?))
            })
            .collect::<Result<Vec<_>>>()?,
    )?;
    let dataset = Arc::new(dataset);
    let mut pipeline = TransformPipeline::with_seed(77);
    pipeline
        .add(TransformOperation::Normalize(Normalize::new(
            vec![0.5],
            vec![0.25],
        )?))
        .add(TransformOperation::Resize(Resize::new(10, 10)?))
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5)?,
        ))
        .add(TransformOperation::CenterCrop(CenterCrop::new(8, 8)?));
    let mut host_loader =
        DataLoader::new(dataset.clone(), 2, Some(19))?.with_pipeline(pipeline.clone());
    let mut resident_loader = DataLoader::new(dataset, 2, Some(19))?.with_pipeline(
        pipeline.with_gpu_dispatcher(TransformDispatcher::from_runtime(device.runtime())?),
    );
    let targets = Tensor::from_vec(2, 1, vec![0., 1.])?;
    let mut loss = CrossEntropyWithLogits::default();
    let mut captures = Vec::new();
    for _ in 0..4 {
        host_loader.reset();
        resident_loader.reset();
        let input = host_loader
            .next_batch()?
            .ok_or("missing host batch")?
            .stack()?;
        let batch = resident_loader
            .next_resident_batch(device)?
            .ok_or("missing resident batch")?;
        let y = batch.upload_targets()?;
        let reference = {
            let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
            let logits = cpu.forward(&input)?;
            let value = loss.forward(&logits, &targets)?;
            let seed = loss.backward(&logits, &targets)?;
            cpu.backward(&input, &seed)?;
            let gradients = values(&cpu, true)?;
            cpu.visit_parameters_mut(&mut |p| p.apply_step(RATE))?;
            (input, logits, value, gradients, values(&cpu, false)?)
        };
        let forward = gpu.forward(&batch.images)?;
        let value = loss.evaluate_resident(forward.prediction(), &y)?;
        let gradients = gpu.backward(&forward, value.prediction_gradient())?;
        let update = gpu.sgd(&gradients, RATE)?;
        captures.push((
            batch.images,
            forward,
            value.value().clone(),
            gradients,
            update,
            gpu.parameter_snapshot(),
            reference,
        ));
    }
    let mut error = 0.;
    let mut losses = Vec::new();
    for (step, (input, forward, loss, gradients, update, weights, expected)) in
        captures.iter().enumerate()
    {
        if receipt(update).await? != (step + 1) as u64 {
            return Err("normalized input update rejected".into());
        }
        close(&read(input).await?, expected.0.data(), false, &mut error)?;
        close(
            &read(forward.prediction()).await?,
            expected.1.data(),
            false,
            &mut error,
        )?;
        let value = read(loss).await?;
        close(&value, expected.2.data(), false, &mut error)?;
        losses.push(value[0]);
        for (i, gradient) in gradients.parameter_gradients().iter().enumerate() {
            close(&read(gradient).await?, &expected.3[i], false, &mut error)?;
            close(
                &read(&weights.values()[i]).await?,
                &expected.4[i],
                false,
                &mut error,
            )?;
        }
    }
    let mut invalid_pipeline = TransformPipeline::new()
        .with_gpu_dispatcher(TransformDispatcher::from_runtime(device.runtime())?);
    invalid_pipeline.add(TransformOperation::Normalize(Normalize::new(
        vec![0.],
        vec![0.5],
    )?));
    let invalid =
        invalid_pipeline.apply_packed_resident_batch(&[2, 1, 8, 8], &[f32::MAX; 128], device)?;
    let saved = gpu.parameter_snapshot();
    let forward = gpu.forward(&invalid)?;
    let value = loss.evaluate_resident(
        forward.prediction(),
        &device.upload(&[2, 1], targets.data())?,
    )?;
    let gradients = gpu.backward(&forward, value.prediction_gradient())?;
    let update = gpu.sgd(&gradients, RATE)?;
    if !matches!(receipt(&update).await, Err(TrainingError::Rejected { .. })) {
        return Err("invalid normalization reached a committed update".into());
    }
    for (before, after) in saved.values().iter().zip(gpu.parameter_snapshot().values()) {
        close(&read(before).await?, &read(after).await?, true, &mut error)?;
    }
    Ok(
        serde_json::json!({"steps":4,"parameters":24,"source":"synthetic_12x14_images",
        "pipeline":"Normalize -> Resize -> Flip -> Crop -> ConvNeXtClassifier -> CE -> SGD",
        "input_to_model_readbacks":0,"training_loop_readbacks":0,"max_scaled_error":error,
        "losses":losses,"invalid_normalization_rejects_all_weights":true}),
    )
}
