use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{parameters::ResidentParameterUpdate, TrainingError},
};
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    execution::{push_backend_policy, BackendPolicy},
    loss::{CrossEntropyWithLogits, Loss},
    module::Module,
};
use st_tensor::Tensor;
use st_vision::{
    models::{
        ConvNeXtClassifier, ConvNeXtClassifierCheckpoint, ConvNeXtClassifierCheckpointSnapshot,
        ConvNeXtConfig,
    },
    ImageTensor,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const STEPS: usize = 8;
const RATE: f32 = 0.01;

mod normalized_input {
    use super::*;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/support/vision_input_classifier_checks.rs"
    ));
}

async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    Ok(values)
}
async fn checkpoint(
    snapshot: ConvNeXtClassifierCheckpointSnapshot,
) -> Result<ConvNeXtClassifierCheckpoint> {
    #[cfg(not(target_arch = "wasm32"))]
    let state = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let state = snapshot.read_async().await?;
    Ok(state)
}
async fn receipt(update: &ResidentParameterUpdate) -> std::result::Result<u64, TrainingError> {
    let snapshot = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    return snapshot.read();
    #[cfg(target_arch = "wasm32")]
    return snapshot.read_async().await;
}
fn values(model: &ConvNeXtClassifier, gradient: bool) -> Result<Vec<Vec<f32>>> {
    let mut values = Vec::new();
    model.visit_parameters(&mut |p| {
        values.push(
            if gradient {
                p.gradient().unwrap()
            } else {
                p.value()
            }
            .data()
            .to_vec(),
        );
        Ok(())
    })?;
    Ok(values)
}
fn close(actual: &[f32], expected: &[f32], exact: bool, error: &mut f32) -> Result<()> {
    if actual.len() != expected.len() {
        return Err("classifier length differs".into());
    }
    for (&a, &b) in actual.iter().zip(expected) {
        let scaled = (a - b).abs() / (1. + b.abs());
        if !a.is_finite()
            || !b.is_finite()
            || scaled > 2e-4
            || (exact && a.to_bits() != b.to_bits())
        {
            return Err(format!("classifier {a} != {b}; scaled={scaled}, exact={exact}").into());
        }
        *error = error.max(scaled);
    }
    Ok(())
}

/// Full supervised classifier and checkpoint handoff, shared by native and browser.
pub async fn run(device: &TensorDevice) -> Result<serde_json::Value> {
    let config = ConvNeXtConfig {
        input_channels: 1,
        input_hw: (8, 8),
        stage_dims: vec![2, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        epsilon: 1e-3,
        ..Default::default()
    };
    let source = ConvNeXtClassifier::new(config.clone(), 2, 7)?;
    let mut cpu = ConvNeXtClassifier::new(config, 2, 7)?;
    let mut gpu = source.compile_resident_training(device.clone(), 2)?;
    let input = Tensor::from_fn(2, 64, |r, c| {
        let shape = if (r == 0 && c % 8 < 4) || (r == 1 && c / 8 < 4) {
            0.8
        } else {
            0.0
        };
        shape + ((c * 7 + r * 11) % 31) as f32 / 155.
    })?;
    let targets = Tensor::from_vec(2, 1, vec![0., 1.])?;
    let x = device.upload(&[2, 1, 8, 8], input.data())?;
    let y = device.upload(&[2, 1], targets.data())?;
    let mut loss = CrossEntropyWithLogits::default();
    let mut captures = Vec::new();
    let mut expected = Vec::new();
    let mut midpoint = None;
    for step in 0..STEPS {
        let reference = {
            let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
            let logits = cpu.forward(&input)?;
            let value = loss.forward(&logits, &targets)?;
            let seed = loss.backward(&logits, &targets)?;
            let dx = cpu.backward(&input, &seed)?;
            let gradients = values(&cpu, true)?;
            cpu.visit_parameters_mut(&mut |p| p.apply_step(RATE))?;
            (logits, value, dx, gradients, values(&cpu, false)?)
        };
        let forward = gpu.forward(&x)?;
        let value = loss.evaluate_resident(forward.prediction(), &y)?;
        let gradients = gpu.backward(&forward, value.prediction_gradient())?;
        let update = gpu.sgd(&gradients, RATE)?;
        captures.push((
            forward,
            value.value().clone(),
            gradients,
            update,
            gpu.parameter_snapshot(),
        ));
        expected.push(reference);
        if step == 3 {
            midpoint = Some(gpu.checkpoint_snapshot()?);
        }
    }
    let next = gpu.forward(&x)?;
    let final_loss = loss.evaluate_resident(next.prediction(), &y)?;
    // Only now observe all eight consecutive update submissions.
    let mut error = 0.;
    let mut losses = Vec::new();
    for (i, ((forward, loss, gradients, update, weights), reference)) in
        captures.iter().zip(&expected).enumerate()
    {
        if receipt(update).await? != (i + 1) as u64 {
            return Err("classifier update rejected".into());
        }
        close(
            &read(forward.prediction()).await?,
            reference.0.data(),
            false,
            &mut error,
        )?;
        let loss = read(loss).await?;
        close(&loss, reference.1.data(), false, &mut error)?;
        losses.push(loss[0]);
        close(
            &read(gradients.input_gradient()).await?,
            reference.2.data(),
            false,
            &mut error,
        )?;
        if gradients.parameter_gradients().len() != 24 {
            return Err("missing classifier parameter".into());
        }
        for (p, tensor) in gradients.parameter_gradients().iter().enumerate() {
            close(&read(tensor).await?, &reference.3[p], false, &mut error)?;
            close(
                &read(&weights.values()[p]).await?,
                &reference.4[p],
                false,
                &mut error,
            )?;
        }
    }
    let final_ce = read(final_loss.value()).await?[0];
    if final_ce >= losses[0] {
        return Err(format!("classifier loss did not decrease: {losses:?} -> {final_ce}").into());
    }
    let initial = values(&source, false)?;
    for range in [0..2, 2..10, 10..12, 12..20, 20..22, 22..24] {
        if !range
            .clone()
            .any(|i| initial[i] != expected.last().unwrap().4[i])
        {
            return Err(format!("classifier parameter group {range:?} unchanged").into());
        }
    }
    let saved = checkpoint(midpoint.ok_or("missing classifier checkpoint")?).await?;
    let mut resumed = ConvNeXtClassifierCheckpoint::from_json(&saved.to_json()?)?
        .restore_resident(device.clone())?;
    if resumed.sgd(&captures[4].2, RATE).is_ok() {
        return Err("classifier old gradient identity restored".into());
    }
    let mut restarts = Vec::new();
    for _ in 4..STEPS {
        let forward = resumed.forward(&x)?;
        let value = loss.evaluate_resident(forward.prediction(), &y)?;
        let gradient = resumed.backward(&forward, value.prediction_gradient())?;
        let update = resumed.sgd(&gradient, RATE)?;
        restarts.push((forward, update, resumed.parameter_snapshot()));
    }
    for (i, (forward, update, weights)) in restarts.iter().enumerate() {
        if receipt(update).await? != (i + 5) as u64 {
            return Err("classifier restart update rejected".into());
        }
        close(
            &read(forward.prediction()).await?,
            &read(captures[i + 4].0.prediction()).await?,
            true,
            &mut error,
        )?;
        for (a, b) in weights.values().iter().zip(captures[i + 4].4.values()) {
            close(&read(a).await?, &read(b).await?, true, &mut error)?;
        }
    }
    let final_state = checkpoint(gpu.checkpoint_snapshot()?).await?;
    let images = input
        .data()
        .chunks_exact(input.shape().1)
        .map(|data| ImageTensor::new(1, 8, 8, data.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let common_model = final_state.to_host()?.into_vision_model()?;
    let public_logits = {
        let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        common_model.forward(&images)?
    };
    close(
        &read(next.prediction()).await?,
        public_logits.data(),
        false,
        &mut error,
    )?;
    let invalid_labels = device.upload(&[2, 1], &[0., 2.])?;
    let forward = gpu.forward(&x)?;
    let bad_loss = loss.evaluate_resident(forward.prediction(), &invalid_labels)?;
    let bad_gradient = gpu.backward(&forward, bad_loss.prediction_gradient())?;
    let rejected = gpu.sgd(&bad_gradient, 0.0)?;
    if !matches!(
        receipt(&rejected).await,
        Err(TrainingError::Rejected { .. })
    ) {
        return Err("invalid classification labels committed".into());
    }
    let after_rejection = checkpoint(gpu.checkpoint_snapshot()?).await?;
    for (a, b) in values(&after_rejection.to_host()?, false)?
        .iter()
        .zip(values(&final_state.to_host()?, false)?)
    {
        close(a, &b, true, &mut error)?;
    }
    let retry = gpu.forward(&x)?;
    close(
        &read(retry.prediction()).await?,
        &read(next.prediction()).await?,
        true,
        &mut error,
    )?;
    let retry_loss = loss.evaluate_resident(retry.prediction(), &y)?;
    let retry_gradient = gpu.backward(&retry, retry_loss.prediction_gradient())?;
    if receipt(&gpu.sgd(&retry_gradient, RATE)?).await? != (STEPS + 2) as u64 {
        return Err("classifier retry revision differs".into());
    }
    let expected_retry = {
        let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let logits = cpu.forward(&input)?;
        let seed = loss.backward(&logits, &targets)?;
        cpu.backward(&input, &seed)?;
        cpu.visit_parameters_mut(&mut |p| p.apply_step(RATE))?;
        values(&cpu, false)?
    };
    for (actual, expected) in gpu.parameter_snapshot().values().iter().zip(expected_retry) {
        close(&read(actual).await?, &expected, false, &mut error)?;
    }

    // A pool VJP must not overflow from derivatives of its non-trainable filter.
    let pool =
        st_nn::layers::GlobalAveragePool2d::new(1, (1, 2))?.compile_resident(device.clone())?;
    let huge = device.upload(&[2, 1, 1, 2], &[f32::MAX; 4])?;
    close(
        &read(&pool.backward(&huge, &device.upload(&[2, 1], &[1e10; 2])?)?).await?,
        &[5e9; 4],
        true,
        &mut error,
    )?;
    let invalid = device
        .upload(&[2, 1, 1, 2], &[-f32::MAX; 4])?
        .mul(&device.upload(&[1], &[2.])?)?
        .relu()?;
    if read(&pool.backward(&invalid, &device.upload(&[2, 1], &[1.; 2])?)?)
        .await
        .is_ok()
    {
        return Err("pool VJP lost inherited guard".into());
    }
    let normalized_input = normalized_input::run(device).await?;
    Ok(
        serde_json::json!({ "schema":"spiraltorch.convnext_classifier.contract.v1", "status":"passed",
        "normalized_input":normalized_input,
        "steps":STEPS,"parameters":24,"learning_rate":RATE,"training_loop_readbacks":0,
        "losses":losses,"final_cross_entropy":final_ce,"max_scaled_error":error,"scaled_error_tolerance":2e-4,
        "resumed_steps":4,"resume_comparison":"bitwise","updated_groups":6,
        "checks":["every_prediction_cpu_parity","every_input_vjp_cpu_parity","all_24_parameter_vjps",
            "all_24_parameter_updates","all_six_groups_update","loss_decrease","all_receipts",
            "whole_classifier_checkpoint","fresh_owner_identity","every_resumed_prediction_and_weight",
            "trained_common_inference_entry","invalid_labels_reject_all_weights","valid_retry_after_rejection",
            "fixed_pool_gradient_no_spurious_overflow","pool_guard_inheritance"] }),
    )
}
