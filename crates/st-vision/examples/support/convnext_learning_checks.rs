use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{parameters::ResidentParameterUpdate, TrainingError},
};
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    execution::{push_backend_policy, BackendPolicy},
    loss::{Loss, MeanSquaredError},
    module::Module,
    resident::InferenceError,
};
use st_tensor::Tensor;
use st_vision::models::{ConvNeXtBackbone, ConvNeXtConfig};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const STEPS: usize = 8;
const RATE: f32 = 0.0001;
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

async fn read_tensor(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    Ok(values)
}

async fn read_update(update: &ResidentParameterUpdate) -> std::result::Result<u64, TrainingError> {
    let snapshot = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    return snapshot.read();
    #[cfg(target_arch = "wasm32")]
    return snapshot.read_async().await;
}

fn close(actual: &[f32], expected: &[f32], label: &str, errors: &mut [f32; 2]) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: length differs").into());
    }
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let absolute = (a - b).abs();
        let scaled = absolute / (1.0 + b.abs());
        if !a.is_finite() || !b.is_finite() || scaled > TOLERANCE {
            return Err(format!("{label}[{i}]: {a} != {b}; scaled error {scaled}").into());
        }
        errors[0] = errors[0].max(absolute);
        errors[1] = errors[1].max(scaled);
    }
    Ok(())
}

fn host_values(model: &ConvNeXtBackbone, gradients: bool) -> Result<Vec<Vec<f32>>> {
    let mut values = Vec::new();
    model.visit_parameters(&mut |p| {
        let value = if gradients {
            p.gradient().unwrap()
        } else {
            p.value()
        };
        values.push(value.data().to_vec());
        Ok(())
    })?;
    Ok(values)
}

fn same_bits(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("retained parameters changed bits".into());
    }
    Ok(())
}

/// One fixture for native Metal/Vulkan and browser WebGPU; no GPU mapping in the learning loop.
pub async fn run(device: &TensorDevice) -> Result<serde_json::Value> {
    let mut source = ConvNeXtBackbone::new(config())?;
    let mut reference = ConvNeXtBackbone::new(config())?;
    let mut model = source.compile_resident_training(device.clone(), 2)?;
    let mut foreign = source.compile_resident_training(device.clone(), 2)?;
    if model.input_shape() != &[2, 2, 8, 8]
        || model.output_shape() != &[2, 16]
        || model.parameter_names().len() != 22
        || source.compile_resident_training(device.clone(), 0).is_ok()
    {
        return Err("compiled model shape/parameter validation".into());
    }
    let initial = model.parameter_snapshot();
    let initial_host = host_values(&source, false)?;
    source.visit_parameters_mut(&mut |p| {
        p.value_mut().data_mut().fill(0.25);
        Ok(())
    })?;
    let detached_host = host_values(&source, false)?;
    let input = Tensor::from_fn(2, 128, |row, col| {
        ((row * 41 + col * 17) % 101) as f32 / 101.0 - 0.5
    })?;
    let target = Tensor::from_fn(2, 16, |row, col| {
        ((row * 13 + col * 7) % 37) as f32 / 37.0 - 0.4
    })?;
    // An offset, non-contiguous view with the same logical NCHW values as the CPU input.
    let mut packed = vec![99.0; 128];
    for n in 0..2 {
        for c in 0..2 {
            for y in 0..8 {
                for x in 0..8 {
                    packed.push(input.data()[(n * 2 + c) * 64 + x * 8 + y]);
                }
            }
        }
    }
    let resident_input = device
        .upload(&[3, 2, 8, 8], &packed)?
        .narrow(0, 1, 2)?
        .permute(&[0, 1, 3, 2])?;
    let resident_target = device.upload(&[2, 16], target.data())?;
    let malformed = device.upload(&[1], &[0.0])?;
    let mut cpu_loss = MeanSquaredError::new();
    let mut captures = Vec::new();
    let mut references = Vec::new();
    for step in 0..STEPS {
        let expected = {
            let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
            let prediction = reference.forward(&input)?;
            let loss = cpu_loss.forward(&prediction, &target)?;
            let seed = cpu_loss.backward(&prediction, &target)?;
            let dx = reference.backward(&input, &seed)?;
            let gradients = host_values(&reference, true)?;
            if step == 0
                && reference
                    .compile_resident_training(device.clone(), 2)
                    .is_ok()
            {
                return Err("pending host derivatives silently discarded".into());
            }
            reference.visit_parameters_mut(&mut |p| p.apply_step(RATE))?;
            (
                prediction.data().to_vec(),
                loss.data().to_vec(),
                dx.data().to_vec(),
                gradients,
                host_values(&reference, false)?,
            )
        };
        let forward = model.forward(&resident_input)?;
        let loss = forward.prediction().mean_squared_error(&resident_target)?;
        if model.forward(&malformed).is_ok() || model.backward(&forward, &malformed).is_ok() {
            return Err("malformed input/cotangent accepted".into());
        }
        let gradients = model.backward(&forward, loss.prediction_gradient())?;
        if step == 0 {
            let other = foreign.forward(&resident_input)?;
            let other_gradients = foreign.backward(&other, loss.prediction_gradient())?;
            if !matches!(
                model.backward(&other, loss.prediction_gradient()),
                Err(InferenceError::Training(TrainingError::StaleForward))
            ) || !matches!(
                model.sgd(&other_gradients, RATE),
                Err(InferenceError::Training(TrainingError::ParameterVersion))
            ) {
                return Err("foreign model token accepted".into());
            }
            for rate in [-1.0, f32::NAN, f32::INFINITY] {
                if !matches!(
                    model.sgd(&gradients, rate),
                    Err(InferenceError::Training(TrainingError::LearningRate))
                ) || model.parameter_snapshot().revision() != 0
                {
                    return Err("invalid rate changed model version".into());
                }
            }
        }
        let receipt = model.sgd(&gradients, RATE)?;
        if !matches!(
            model.sgd(&gradients, RATE),
            Err(InferenceError::Training(TrainingError::ParameterVersion))
        ) || !matches!(
            model.backward(&forward, loss.prediction_gradient()),
            Err(InferenceError::Training(TrainingError::StaleForward))
        ) {
            return Err("stale model derivative/forward accepted".into());
        }
        captures.push((
            forward,
            loss.value().clone(),
            gradients,
            receipt,
            model.parameter_snapshot(),
        ));
        references.push(expected);
    }
    let final_forward = model.forward(&resident_input)?;
    let final_loss = final_forward
        .prediction()
        .mean_squared_error(&resident_target)?;
    let expected_final = {
        let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        reference.forward(&input)?
    };

    // Read only after all consecutive updates and the next forward have been submitted.
    let mut errors = [0.0; 2];
    let mut losses = Vec::new();
    for (step, ((forward, loss, gradients, receipt, weights), expected)) in
        captures.iter().zip(&references).enumerate()
    {
        if read_update(receipt).await? != (step + 1) as u64 {
            return Err("unexpected model update revision".into());
        }
        close(
            &read_tensor(forward.prediction()).await?,
            &expected.0,
            "prediction",
            &mut errors,
        )?;
        let loss = read_tensor(loss).await?;
        close(&loss, &expected.1, "MSE", &mut errors)?;
        losses.push(loss[0]);
        close(
            &read_tensor(gradients.input_gradient()).await?,
            &expected.2,
            "input VJP",
            &mut errors,
        )?;
        if gradients.parameter_gradients().len() != 22 {
            return Err("missing model parameter derivative".into());
        }
        for (i, gradient) in gradients.parameter_gradients().iter().enumerate() {
            close(
                &read_tensor(gradient).await?,
                &expected.3[i],
                &format!("gradient {i}"),
                &mut errors,
            )?;
            close(
                &read_tensor(&weights.values()[i]).await?,
                &expected.4[i],
                &format!("weight {i}"),
                &mut errors,
            )?;
        }
    }
    close(
        &read_tensor(final_forward.prediction()).await?,
        expected_final.data(),
        "next forward",
        &mut errors,
    )?;
    let final_mse = read_tensor(final_loss.value()).await?[0];
    if !final_mse.is_finite() || final_mse >= losses[0] {
        return Err(format!("ConvNeXt learning did not decrease MSE: rate={RATE}, losses={losses:?}, final={final_mse}").into());
    }
    let trained = model.parameter_snapshot();
    let mut trained_values = Vec::new();
    for (tensor, expected) in initial.values().iter().zip(&initial_host) {
        same_bits(&read_tensor(tensor).await?, expected)?;
    }
    for tensor in trained.values() {
        trained_values.push(read_tensor(tensor).await?);
    }
    let groups = [0..2, 2..10, 10..12, 12..20, 20..22];
    for group in &groups {
        if !group.clone().any(|i| initial_host[i] != trained_values[i]) {
            return Err(format!("parameter group {group:?} was not updated").into());
        }
    }
    if host_values(&source, false)? != detached_host {
        return Err("resident learning modified source host model".into());
    }

    // An invalid upstream seed must reject ALL weights, even at zero learning rate.
    let bad_seed = device
        .upload(&[2, 16], &[-f32::MAX; 32])?
        .mul(&device.upload(&[2, 16], &[2.0; 32])?)?
        .relu()?;
    for rate in [RATE, 0.0] {
        let forward = model.forward(&resident_input)?;
        let gradient = model.backward(&forward, &bad_seed)?;
        for tensor in
            std::iter::once(gradient.input_gradient()).chain(gradient.parameter_gradients())
        {
            if read_tensor(tensor).await.is_ok() {
                return Err("inherited seed failure lost in full-model backward".into());
            }
        }
        let receipt = model.sgd(&gradient, rate)?;
        if !matches!(
            read_update(&receipt).await,
            Err(TrainingError::Rejected { .. })
        ) {
            return Err("invalid model update accepted".into());
        }
        for (tensor, expected) in model
            .parameter_snapshot()
            .values()
            .iter()
            .zip(&trained_values)
        {
            same_bits(&read_tensor(tensor).await?, expected)?;
        }
    }
    let retry = model.forward(&resident_input)?;
    close(
        &read_tensor(retry.prediction()).await?,
        expected_final.data(),
        "retry forward",
        &mut errors,
    )?;
    if model
        .backward(&final_forward, final_loss.prediction_gradient())
        .is_ok()
    {
        return Err("reused stale intermediate tape".into());
    }
    let loss = retry.prediction().mean_squared_error(&resident_target)?;
    let gradient = model.backward(&retry, loss.prediction_gradient())?;
    let receipt = model.sgd(&gradient, RATE)?;
    if read_update(&receipt).await? != STEPS as u64 + 3 {
        return Err("valid model retry did not commit".into());
    }
    let expected_retry = {
        let _cpu = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let prediction = reference.forward(&input)?;
        let seed = cpu_loss.backward(&prediction, &target)?;
        reference.backward(&input, &seed)?;
        reference.visit_parameters_mut(&mut |p| p.apply_step(RATE))?;
        reference.forward(&input)?
    };
    let after_retry = model.forward(&resident_input)?;
    close(
        &read_tensor(after_retry.prediction()).await?,
        expected_retry.data(),
        "retry next forward",
        &mut errors,
    )?;
    drop(model);
    for (tensor, expected) in trained.values().iter().zip(&trained_values) {
        same_bits(&read_tensor(tensor).await?, expected)?;
    }
    close(
        &read_tensor(captures[0].2.input_gradient()).await?,
        &references[0].2,
        "retained VJP",
        &mut errors,
    )?;
    if read_update(&captures[0].3).await? != 1 {
        return Err("retained model receipt lost identity".into());
    }
    Ok(serde_json::json!({
        "schema": "spiraltorch.convnext_learning.contract.v1",
        "status": "passed", "steps": STEPS, "parameters": 22,
        "input_shape": [2, 2, 8, 8], "stage_dims": [3, 4], "stage_depths": [1, 1],
        "epsilon": 1e-3, "learning_rate": RATE, "training_loop_readbacks": 0,
        "losses": losses, "final_mse": final_mse, "updated_parameter_groups": groups.len(),
        "max_absolute_error": errors[0], "max_scaled_error": errors[1],
        "scaled_error_tolerance": TOLERANCE,
        "checks": ["strided_offset_input", "every_step_cpu_parity", "next_forward_uses_updates",
            "all_parameter_groups_update", "host_source_isolation", "pending_host_gradient_rejected",
            "shape_validation", "rate_validation", "foreign_model", "stale_gradients",
            "stale_forward", "whole_model_rejection", "invalid_zero_rate", "valid_retry",
            "retained_parameters", "retained_gradients", "retained_receipts"],
    }))
}
