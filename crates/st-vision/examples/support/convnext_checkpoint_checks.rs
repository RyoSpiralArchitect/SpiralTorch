use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{
        parameters::{ResidentParameterSnapshot, ResidentParameterUpdate},
        TrainingError,
    },
};
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    execution::{push_backend_policy, BackendPolicy},
    module::Module,
    resident::InferenceError,
};
use st_tensor::Tensor;
use st_vision::models::{
    ConvNeXtBackbone, ConvNeXtCheckpointSnapshot, ConvNeXtConfig, ConvNeXtTrainingCheckpoint,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const STEPS: usize = 8;
const CURSOR: usize = 4;
const TOLERANCE: f32 = 2e-4;

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

// The caller, not the model checkpoint, owns this data cursor and rate schedule.
fn sample(cursor: usize) -> (Vec<f32>, Vec<f32>, f32) {
    let input = (0..256)
        .map(|i| ((i * 17 + cursor * 7) % 113) as f32 / 113.0 - 0.5)
        .collect();
    let target = (0..32)
        .map(|i| ((i * 5 + cursor * 3) % 29) as f32 / 29.0 - 0.5)
        .collect();
    (input, target, 0.0001 / (1.0 + cursor as f32 * 0.1))
}

async fn read_checkpoint(
    snapshot: ConvNeXtCheckpointSnapshot,
) -> Result<ConvNeXtTrainingCheckpoint> {
    #[cfg(not(target_arch = "wasm32"))]
    let checkpoint = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let checkpoint = snapshot.read_async().await?;
    Ok(checkpoint)
}

async fn read_tensor(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    Ok(values)
}

async fn read_parameters(snapshot: &ResidentParameterSnapshot) -> Result<Vec<Vec<f32>>> {
    let refs: Vec<_> = snapshot.values().iter().collect();
    let batch = refs[0].device().snapshot_many(&refs)?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = batch.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = batch.read_async().await?;
    Ok(values)
}

async fn read_update(update: &ResidentParameterUpdate) -> std::result::Result<u64, TrainingError> {
    let snapshot = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    return snapshot.read();
    #[cfg(target_arch = "wasm32")]
    return snapshot.read_async().await;
}

fn compare(actual: &[f32], expected: &[f32], exact: bool, error: &mut f32) -> Result<()> {
    if actual.len() != expected.len() {
        return Err("checkpoint comparison length differs".into());
    }
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let scaled = (a - b).abs() / (1.0 + b.abs());
        if !a.is_finite()
            || !b.is_finite()
            || scaled > TOLERANCE
            || (exact && a.to_bits() != b.to_bits())
        {
            return Err(format!("checkpoint comparison [{i}]: {a} != {b}, exact={exact}").into());
        }
        *error = error.max(scaled);
    }
    Ok(())
}

fn host_values(model: &ConvNeXtBackbone) -> Result<Vec<Vec<f32>>> {
    let mut values = Vec::new();
    model.visit_parameters(&mut |p| {
        values.push(p.value().data().to_vec());
        Ok(())
    })?;
    Ok(values)
}

fn compare_parameters(
    actual: &[Vec<f32>],
    expected: &[Vec<f32>],
    exact: bool,
    error: &mut f32,
) -> Result<()> {
    if actual.len() != 22 || expected.len() != 22 {
        return Err("checkpoint parameter count differs".into());
    }
    for (a, b) in actual.iter().zip(expected) {
        compare(a, b, exact, error)?;
    }
    Ok(())
}

/// The same deterministic learning/restart fixture runs natively and in WebGPU.
/// Optional imported JSON exercises a checkpoint actually produced by another runtime.
pub async fn run(device: &TensorDevice, imported_json: Option<&str>) -> Result<serde_json::Value> {
    let mut source = ConvNeXtBackbone::new(config())?;
    let initial = host_values(&source)?;
    let mut uninterrupted = source.compile_resident_training(device.clone(), 2)?;
    let mut captures = Vec::new();
    let mut midpoint = None;
    for cursor in 0..STEPS {
        let (input, target, rate) = sample(cursor);
        let input = device.upload(&[2, 2, 8, 8], &input)?;
        let target = device.upload(&[2, 16], &target)?;
        let forward = uninterrupted.forward(&input)?;
        let loss = forward.prediction().mean_squared_error(&target)?;
        let gradients = uninterrupted.backward(&forward, loss.prediction_gradient())?;
        let receipt = uninterrupted.sgd(&gradients, rate)?;
        captures.push((
            forward,
            gradients,
            receipt,
            uninterrupted.parameter_snapshot(),
        ));
        if cursor + 1 == CURSOR {
            midpoint = Some(uninterrupted.checkpoint_snapshot()?);
        }
    }
    let terminal = uninterrupted.checkpoint_snapshot()?;
    drop(uninterrupted);
    // No GPU mapping before both segments finish and the original owner is gone.
    let checkpoint = read_checkpoint(midpoint.ok_or("missing midpoint")?).await?;
    let terminal = read_checkpoint(terminal).await?;
    if checkpoint.attempted_updates() != CURSOR as u64
        || terminal.attempted_updates() != STEPS as u64
        || checkpoint.config() != &config()
        || checkpoint.batch_size() != 2
    {
        return Err("checkpoint metadata changed after later learning/model drop".into());
    }
    let json = checkpoint.to_json()?;
    let restored = ConvNeXtTrainingCheckpoint::from_json(&json)?;
    let mut error = 0.0;
    compare_parameters(&host_values(&source)?, &initial, true, &mut error)?;
    let midpoint_values = read_parameters(&captures[CURSOR - 1].3).await?;
    let final_values = read_parameters(&captures[STEPS - 1].3).await?;
    compare_parameters(
        &host_values(&restored.to_host()?)?,
        &midpoint_values,
        true,
        &mut error,
    )?;
    compare_parameters(
        &host_values(&terminal.to_host()?)?,
        &final_values,
        true,
        &mut error,
    )?;
    restored.restore_host(&mut source)?;
    compare_parameters(&host_values(&source)?, &midpoint_values, true, &mut error)?;
    let host_prediction = {
        let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        source.forward(&Tensor::from_vec(2, 128, sample(CURSOR).0)?)?
    };
    compare(
        &read_tensor(captures[CURSOR].0.prediction()).await?,
        host_prediction.data(),
        false,
        &mut error,
    )?;

    let mut resume = restored.restore_resident(device.clone())?;
    if resume.parameter_snapshot().revision() != CURSOR as u64
        || !matches!(
            resume.sgd(&captures[CURSOR].1, sample(CURSOR).2),
            Err(InferenceError::Training(TrainingError::ParameterVersion))
        )
    {
        return Err("restored model reused old parameter identity".into());
    }
    let seed = device.upload(&[2, 16], &[0.1; 32])?;
    if !matches!(
        resume.backward(&captures[CURSOR].0, &seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ) {
        return Err("restored model reused old forward tape".into());
    }
    let mut resumed = Vec::new();
    for cursor in CURSOR..STEPS {
        let (input, target, rate) = sample(cursor);
        let forward = resume.forward(&device.upload(&[2, 2, 8, 8], &input)?)?;
        let loss = forward
            .prediction()
            .mean_squared_error(&device.upload(&[2, 16], &target)?)?;
        let gradients = resume.backward(&forward, loss.prediction_gradient())?;
        let receipt = resume.sgd(&gradients, rate)?;
        resumed.push((forward, receipt, resume.parameter_snapshot()));
    }
    for (offset, (forward, receipt, weights)) in resumed.iter().enumerate() {
        let cursor = CURSOR + offset;
        if read_update(receipt).await? != (cursor + 1) as u64 {
            return Err("restored update clock differs".into());
        }
        compare(
            &read_tensor(forward.prediction()).await?,
            &read_tensor(captures[cursor].0.prediction()).await?,
            true,
            &mut error,
        )?;
        compare_parameters(
            &read_parameters(weights).await?,
            &read_parameters(&captures[cursor].3).await?,
            true,
            &mut error,
        )?;
    }
    for (cursor, (_, _, receipt, _)) in captures.iter().enumerate() {
        if read_update(receipt).await? != (cursor + 1) as u64 {
            return Err("uninterrupted update rejected".into());
        }
    }

    // Rejected attempts still advance the clock; checkpoint weights remain valid.
    let bad_seed = device
        .upload(&[2, 16], &[-f32::MAX; 32])?
        .mul(&device.upload(&[2, 16], &[2.0; 32])?)?
        .relu()?;
    let (input, target, rate) = sample(STEPS);
    let input = device.upload(&[2, 2, 8, 8], &input)?;
    let target = device.upload(&[2, 16], &target)?;
    let forward = resume.forward(&input)?;
    let bad_gradient = resume.backward(&forward, &bad_seed)?;
    let rejected = resume.sgd(&bad_gradient, 0.0)?;
    let rejected_checkpoint = read_checkpoint(resume.checkpoint_snapshot()?).await?;
    if !matches!(
        read_update(&rejected).await,
        Err(TrainingError::Rejected { .. })
    ) || rejected_checkpoint.attempted_updates() != 9
    {
        return Err("rejected attempt not preserved by checkpoint".into());
    }
    compare_parameters(
        &host_values(&rejected_checkpoint.to_host()?)?,
        &final_values,
        true,
        &mut error,
    )?;
    let mut retry = ConvNeXtTrainingCheckpoint::from_json(&rejected_checkpoint.to_json()?)?
        .restore_resident(device.clone())?;
    let mut retried = Vec::new();
    for model in [&mut resume, &mut retry] {
        let forward = model.forward(&input)?;
        let loss = forward.prediction().mean_squared_error(&target)?;
        let gradients = model.backward(&forward, loss.prediction_gradient())?;
        let receipt = model.sgd(&gradients, rate)?;
        retried.push((receipt, model.parameter_snapshot()));
    }
    for (receipt, _) in &retried {
        if read_update(receipt).await? != 10 {
            return Err("resumed retry clock differs".into());
        }
    }
    compare_parameters(
        &read_parameters(&retried[0].1).await?,
        &read_parameters(&retried[1].1).await?,
        true,
        &mut error,
    )?;

    let mut imported_steps = 0;
    if let Some(external) = imported_json {
        let external = ConvNeXtTrainingCheckpoint::from_json(external)?;
        if external.config() != &config()
            || external.batch_size() != 2
            || external.attempted_updates() != CURSOR as u64
        {
            return Err("imported checkpoint does not match fixture/cursor".into());
        }
        let mut model = external.restore_resident(device.clone())?;
        compare_parameters(
            &read_parameters(&model.parameter_snapshot()).await?,
            &host_values(&external.to_host()?)?,
            true,
            &mut error,
        )?;
        let mut imported = Vec::new();
        for cursor in CURSOR..STEPS {
            let (input, target, rate) = sample(cursor);
            let forward = model.forward(&device.upload(&[2, 2, 8, 8], &input)?)?;
            let loss = forward
                .prediction()
                .mean_squared_error(&device.upload(&[2, 16], &target)?)?;
            let gradients = model.backward(&forward, loss.prediction_gradient())?;
            let receipt = model.sgd(&gradients, rate)?;
            imported.push((forward, receipt, model.parameter_snapshot()));
        }
        for (offset, (forward, receipt, weights)) in imported.iter().enumerate() {
            let cursor = CURSOR + offset;
            if read_update(receipt).await? != (cursor + 1) as u64 {
                return Err("imported model update rejected".into());
            }
            compare(
                &read_tensor(forward.prediction()).await?,
                &read_tensor(captures[cursor].0.prediction()).await?,
                false,
                &mut error,
            )?;
            compare_parameters(
                &read_parameters(weights).await?,
                &read_parameters(&captures[cursor].3).await?,
                false,
                &mut error,
            )?;
            imported_steps += 1;
        }
    }
    Ok(serde_json::json!({
        "schema": "spiraltorch.convnext_checkpoint.contract.v1", "status": "passed",
        "steps": STEPS, "resume_cursor": CURSOR, "resumed_steps": STEPS - CURSOR,
        "parameters": 22, "training_loop_readbacks": 0,
        "same_runtime_resume": "bitwise", "imported_steps": imported_steps,
        "max_scaled_error": error, "scaled_error_tolerance": TOLERANCE,
        "checkpoint_json": json,
        "checks": ["metadata", "snapshot_after_later_updates", "snapshot_after_owner_drop",
            "all_parameter_bits", "host_source_isolation", "explicit_host_handoff",
            "host_forward_parity", "fresh_owner_identity", "old_forward_rejected",
            "every_resumed_prediction", "every_resumed_weight", "receipts",
            "rejected_attempt_clock", "rejected_weights_unchanged", "resumed_retry"],
        "caller_state": {"data_cursor": CURSOR, "schedule": "0.0001 / (1 + cursor * 0.1)",
            "loss": "mean_squared_error", "fixture": "deterministic_modular_inputs_and_targets"},
    }))
}
