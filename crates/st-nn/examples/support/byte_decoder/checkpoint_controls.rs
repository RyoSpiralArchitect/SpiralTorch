//! Same-run exact resume checks; cross-device bitwise equality is not implied.
use super::*;
use st_nn::resident::{ByteDecoderCheckpoint, ResidentByteDecoder};

async fn checkpoint(model: &ResidentByteDecoder) -> Result<ByteDecoderCheckpoint> {
    let capture = model.checkpoint_snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let checkpoint = capture.read()?;
    #[cfg(target_arch = "wasm32")]
    let checkpoint = capture.read_async().await?;
    Ok(checkpoint)
}

async fn step(model: &mut ResidentByteDecoder, host: &ByteLmBatch, rate: f32) -> Result<f32> {
    let batch = model.prepare_batch(host)?;
    let forward = model.forward(&batch)?;
    let loss = forward.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
    let gradient = model.backward(&forward, loss.prediction_gradient())?;
    let update = model.sgd(&gradient, rate)?;
    let receipt = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let revision = receipt.read()?;
    #[cfg(target_arch = "wasm32")]
    let revision = receipt.read_async().await?;
    if revision != model.parameter_snapshot().revision() {
        return Err("checkpoint update acceptance revision mismatch".into());
    }
    Ok(read(loss.value()).await?[0])
}

pub(super) async fn run(runtime: &WgpuRuntime, case: &Value) -> Result<Value> {
    let plan = plan(case, 4)?;
    let windows: Vec<Vec<u8>> = case["windows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| sizes(row).into_iter().map(|v| v as u8).collect())
        .collect();
    let host = batch(&windows, 4)?;
    let altered: Vec<Vec<_>> = windows
        .iter()
        .map(|row| row.iter().map(|v| v.wrapping_add(17)).collect())
        .collect();
    let other = batch(&altered, 4)?;
    let mut uninterrupted = plan.compile_training_wgpu(runtime.clone())?;
    let first_losses = [
        step(&mut uninterrupted, &host, 0.04).await?,
        step(&mut uninterrupted, &other, 0.03).await?,
    ];
    let capture = uninterrupted.checkpoint_snapshot()?;
    let resident = uninterrupted.prepare_batch(&host)?;
    let before = uninterrupted.forward(&resident)?;
    let loss = before.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
    let gradient = uninterrupted.backward(&before, loss.prediction_gradient())?;
    let update = uninterrupted.sgd(&gradient, 0.02)?;
    let receipt = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    receipt.read()?;
    #[cfg(target_arch = "wasm32")]
    receipt.read_async().await?;
    // Read only after the original owner has advanced: a live-weight alias
    // would incorrectly serialize revision 3 values under the captured clock.
    #[cfg(not(target_arch = "wasm32"))]
    let saved = capture.read()?;
    #[cfg(target_arch = "wasm32")]
    let saved = capture.read_async().await?;
    if saved.attempted_revision() != 2 {
        return Err("checkpoint followed later owner updates".into());
    }
    let payload = saved.to_json()?;
    let imported = ByteDecoderCheckpoint::from_json(&payload)?;
    if imported.to_json()? != payload {
        return Err("checkpoint transport changed model bits or topology".into());
    }
    let mut resumed = imported.restore_wgpu(runtime.clone())?;
    if checkpoint(&resumed).await?.to_json()? != payload {
        return Err("restoration changed parameters or attempted revision".into());
    }
    let resumed_batch = resumed.prepare_batch(&host)?;
    // Match the original tape's submission index as well as its revision;
    // only owner identity should prevent a cross-model backward here.
    for _ in 0..2 {
        resumed.forward(&resumed_batch)?;
    }
    let local = resumed.forward(&resumed_batch)?;
    if !matches!(
        resumed.backward(&before, loss.prediction_gradient()),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ) || !matches!(
        resumed.sgd(&gradient, 0.02),
        Err(InferenceError::Training(TrainingError::ParameterVersion))
    ) {
        return Err("restored owner accepted a foreign tape or gradients".into());
    }
    let local_loss =
        local.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
    let local_gradient = resumed.backward(&local, local_loss.prediction_gradient())?;
    if !same_bits(
        &read(before.prediction()).await?,
        &read(local.prediction()).await?,
    ) || !same_bits(&read(loss.value()).await?, &read(local_loss.value()).await?)
    {
        return Err("resumed logits/loss differ from uninterrupted training".into());
    }
    let a = read_many(
        uninterrupted.tensor_device(),
        gradient.parameter_gradients(),
    )
    .await?;
    let b = read_many(
        resumed.tensor_device(),
        local_gradient.parameter_gradients(),
    )
    .await?;
    if a.len() != b.len() || a.iter().zip(&b).any(|(a, b)| !same_bits(a, b)) {
        return Err("resumed parameter pullbacks differ".into());
    }
    let update = resumed.sgd(&local_gradient, 0.02)?;
    let receipt = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    receipt.read()?;
    #[cfg(target_arch = "wasm32")]
    receipt.read_async().await?;
    let after_three = checkpoint(&resumed).await?;
    if after_three.to_json()? != checkpoint(&uninterrupted).await?.to_json()?
        || after_three.plan().initial_checkpoint().to_json()?
            == saved.plan().initial_checkpoint().to_json()?
    {
        return Err("resume update is mismatched or inert".into());
    }
    let a = step(&mut uninterrupted, &other, 0.01).await?;
    let b = step(&mut resumed, &other, 0.01).await?;
    let after_four = checkpoint(&resumed).await?;
    if a.to_bits() != b.to_bits()
        || after_four.to_json()? != checkpoint(&uninterrupted).await?.to_json()?
    {
        return Err("continued resume trajectory differs".into());
    }
    let tape = resumed.forward(&resumed_batch)?;
    let huge = resumed.tensor_device().upload(
        resumed.output_layout().shape(),
        &vec![f32::MAX; resumed.output_layout().len()],
    )?;
    let invalid = resumed.backward(&tape, &huge.add(&huge)?)?;
    let rejected = resumed.sgd(&invalid, 0.)?;
    let receipt = rejected.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let status = receipt.read();
    #[cfg(target_arch = "wasm32")]
    let status = receipt.read_async().await;
    if !matches!(status, Err(TrainingError::Rejected { .. })) {
        return Err("checkpoint rejection control unexpectedly updated".into());
    }
    let after_rejection = checkpoint(&resumed).await?;
    if after_rejection.attempted_revision() != 5
        || after_rejection.plan().initial_checkpoint().to_json()?
            != after_four.plan().initial_checkpoint().to_json()?
    {
        return Err("checkpoint confused attempted revision with accepted steps".into());
    }
    let mut restored_rejection = after_rejection.restore_wgpu(runtime.clone())?;
    let a = step(&mut resumed, &other, 0.01).await?;
    let b = step(&mut restored_rejection, &other, 0.01).await?;
    if a.to_bits() != b.to_bits()
        || checkpoint(&resumed).await?.to_json()?
            != checkpoint(&restored_rejection).await?.to_json()?
    {
        return Err("resume after rejected update differs".into());
    }
    // Revision is a decimal string, never a JS number. Advance past 2^53.
    let large_payload = payload.replacen(
        "\"attempted_revision\":\"2\"",
        "\"attempted_revision\":\"9007199254740993\"",
        1,
    );
    let mut large =
        ByteDecoderCheckpoint::from_json(&large_payload)?.restore_wgpu(runtime.clone())?;
    step(&mut large, &host, 0.02).await?;
    let large_revision = checkpoint(&large).await?.attempted_revision();
    if large_revision != 9_007_199_254_740_994 {
        return Err("large checkpoint revision lost precision".into());
    }
    Ok(
        json!({"captured_revision":2, "restored_updates":2, "final_revision":6,
        "parameter_count":plan.parameter_layout().len(), "first_losses":first_losses,
        "json_bytes":payload.len(), "large_revision":large_revision.to_string(),
        "exact_logits_loss":true, "exact_parameter_gradients":true, "exact_parameters":true,
        "immutable_capture":true, "foreign_tape":true, "foreign_gradients":true,
        "rejected_update_resume":true, "nonzero_update_control":true}),
    )
}
