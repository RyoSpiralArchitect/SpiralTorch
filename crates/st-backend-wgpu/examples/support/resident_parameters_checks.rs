use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{
        parameters::{ResidentParameterSnapshot, ResidentParameterUpdate, ResidentParameters},
        TrainingError,
    },
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn read_tensor(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    Ok(values)
}

async fn read_parameters(snapshot: &ResidentParameterSnapshot) -> Result<Vec<Vec<f32>>> {
    let mut values = Vec::new();
    for tensor in snapshot.values() {
        values.push(read_tensor(tensor).await?);
    }
    Ok(values)
}

async fn read_update(update: &ResidentParameterUpdate) -> std::result::Result<u64, TrainingError> {
    let snapshot = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    return snapshot.read();
    #[cfg(target_arch = "wasm32")]
    return snapshot.read_async().await;
}

fn close(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 2e-5 * (1.0 + b.abs()))
    {
        return Err(format!("parameter parity: {actual:?} != {expected:?}").into());
    }
    Ok(())
}

fn same_bits(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> Result<()> {
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(a, b)| {
            a.len() != b.len() || a.iter().zip(b).any(|(a, b)| a.to_bits() != b.to_bits())
        })
    {
        return Err("parameter values were not preserved bit-for-bit".into());
    }
    Ok(())
}

/// Native and browser execute this same Rust contract fixture.
pub async fn run(device: &TensorDevice) -> Result<serde_json::Value> {
    let strided = device
        .upload(&[3, 2], &[99.0, 99.0, 1.0, 2.0, 3.0, 4.0])?
        .narrow(0, 1, 2)?
        .permute(&[1, 0])?;
    let bias = device.upload(&[2], &[-0.0, 2.0])?;
    let mut owner = ResidentParameters::new(vec![strided, bias])?;
    let initial = owner.snapshot();
    let initial_values = read_parameters(&initial).await?;
    let derivatives = initial.bind_gradients(vec![
        device
            .upload(&[2, 2], &[1.0, 2.0, 3.0, 4.0])?
            .permute(&[1, 0])?,
        device.upload(&[2], &[1.0, -2.0])?,
    ])?;
    let zero = owner.sgd(&derivatives, -0.0)?;
    if read_update(&zero).await? != 1 {
        return Err("zero-rate revision".into());
    }
    same_bits(&read_parameters(&owner.snapshot()).await?, &initial_values)?;
    if !matches!(
        owner.sgd(&derivatives, 0.25),
        Err(TrainingError::ParameterVersion)
    ) {
        return Err("old zero-rate gradient was accepted".into());
    }
    let current = owner.snapshot();
    if !matches!(
        current.bind_gradients(vec![]),
        Err(TrainingError::ParameterLayout)
    ) || !matches!(
        current.bind_gradients(vec![
            device.upload(&[4], &[1.0; 4])?,
            device.upload(&[2], &[1.0; 2])?,
        ]),
        Err(TrainingError::ParameterLayout)
    ) {
        return Err("invalid gradient layout accepted".into());
    }
    let derivatives = current.bind_gradients(vec![
        device
            .upload(&[2, 2], &[1.0, 2.0, 3.0, 4.0])?
            .permute(&[1, 0])?,
        device.upload(&[2], &[1.0, -2.0])?,
    ])?;
    for rate in [-1.0, f32::NAN, f32::INFINITY] {
        if !matches!(
            owner.sgd(&derivatives, rate),
            Err(TrainingError::LearningRate)
        ) || owner.snapshot().revision() != 1
        {
            return Err("invalid rate changed ownership".into());
        }
    }
    let mut foreign = ResidentParameters::new(current.values().to_vec())?;
    let foreign_zero = foreign.snapshot().bind_gradients(vec![
        device.upload(&[2, 2], &[0.0; 4])?,
        device.upload(&[2], &[0.0; 2])?,
    ])?;
    foreign.sgd(&foreign_zero, 0.0)?;
    if foreign.snapshot().revision() != current.revision() {
        return Err("foreign-owner test must compare the same revision".into());
    }
    if !matches!(
        foreign.sgd(&derivatives, 0.25),
        Err(TrainingError::ParameterVersion)
    ) {
        return Err("foreign owner accepted gradient identity".into());
    }
    let accepted = owner.sgd(&derivatives, 0.25)?;
    let accepted_values = read_parameters(&owner.snapshot()).await?;
    close(&accepted_values[0], &[0.75, 2.25, 1.5, 3.0])?;
    close(&accepted_values[1], &[-0.25, 2.5])?;
    same_bits(&read_parameters(&initial).await?, &initial_values)?;

    let wide_values: Vec<_> = (0..513).map(|i| i as f32).collect();
    let mut wide = ResidentParameters::new(vec![device.upload(&[513], &wide_values)?])?;
    let wide_gradient = device.upload(&[514], &[2.0; 514])?.narrow(0, 1, 513)?;
    let gradient = wide.snapshot().bind_gradients(vec![wide_gradient])?;
    read_update(&wide.sgd(&gradient, 0.25)?).await?;
    close(
        &read_parameters(&wide.snapshot()).await?[0],
        &wide_values.iter().map(|v| v - 0.5).collect::<Vec<_>>(),
    )?;

    // Only the final parameter is bad: even earlier finite candidates must roll back.
    let snapshot = owner.snapshot();
    let overflow = snapshot.bind_gradients(vec![
        device.upload(&[2, 2], &[1.0; 4])?,
        device.upload(&[2], &[f32::MAX; 2])?,
    ])?;
    let rejected = owner.sgd(&overflow, 2.0)?;
    if !matches!(
        read_update(&rejected).await,
        Err(TrainingError::Rejected { stage: 1, .. })
    ) {
        return Err("finite SGD multiplication overflow was accepted".into());
    }
    same_bits(&read_parameters(&owner.snapshot()).await?, &accepted_values)?;
    if !matches!(
        owner.sgd(&overflow, 0.0),
        Err(TrainingError::ParameterVersion)
    ) {
        return Err("rejected update left old gradients current".into());
    }
    let poisoned = device
        .upload(&[2], &[f32::MAX; 2])?
        .mul(&device.upload(&[2], &[2.0; 2])?)?;
    let poisoned = owner
        .snapshot()
        .bind_gradients(vec![device.upload(&[2, 2], &[1.0; 4])?, poisoned])?;
    let invalid_zero = owner.sgd(&poisoned, 0.0)?;
    if !matches!(
        read_update(&invalid_zero).await,
        Err(TrainingError::Rejected { stage: 1, .. })
    ) {
        return Err("zero-rate update masked an invalid gradient guard".into());
    }
    same_bits(&read_parameters(&owner.snapshot()).await?, &accepted_values)?;
    let valid = owner.snapshot().bind_gradients(vec![
        device.upload(&[2, 2], &[0.0; 4])?,
        device.upload(&[2], &[0.0; 2])?,
    ])?;
    let retry = owner.sgd(&valid, 0.5)?;
    if read_update(&retry).await? != 5 || read_update(&accepted).await? != 2 {
        return Err("retry or retained receipt revision".into());
    }
    let retained = owner.snapshot();
    drop(owner);
    same_bits(&read_parameters(&retained).await?, &accepted_values)?;
    same_bits(&read_parameters(&initial).await?, &initial_values)?;

    // Complete Conv2d forward/loss/VJP/update chains, with no mapping inside the loop.
    let input_values = [1.0, 2.0, 3.0, 4.0];
    let input = device.upload(&[1, 1, 2, 2], &input_values)?;
    let target = device.upload(&[1, 1, 2, 2], &[0.5, 1.0, 1.5, 2.0])?;
    let mut conv = ResidentParameters::new(vec![
        device.upload(&[1, 1, 1, 1], &[1.0])?,
        device.upload(&[1], &[0.0])?,
    ])?;
    let mut reference = [1.0f32, 0.0];
    let mut receipts = Vec::new();
    let mut losses = Vec::new();
    for _ in 0..16 {
        let parameters = conv.snapshot();
        let weights = &parameters.values()[0];
        let bias = &parameters.values()[1];
        let prediction = input.conv2d(weights, bias, (1, 1), (0, 0), (1, 1))?;
        let loss = prediction.mean_squared_error(&target)?;
        let gradient =
            input.conv2d_vjp(weights, loss.prediction_gradient(), (1, 1), (0, 0), (1, 1))?;
        let derivatives = parameters.bind_gradients(gradient[1..].to_vec())?;
        receipts.push(conv.sgd(&derivatives, 0.03)?);
        losses.push(loss.value().clone());
        let mut dw = 0.0;
        let mut db = 0.0;
        for value in input_values {
            let seed = 0.5 * (reference[0] * value + reference[1] - 0.5 * value);
            dw += seed * value;
            db += seed;
        }
        reference[0] -= 0.03 * dw;
        reference[1] -= 0.03 * db;
    }
    let parameters = conv.snapshot();
    let prediction = input.conv2d(
        &parameters.values()[0],
        &parameters.values()[1],
        (1, 1),
        (0, 0),
        (1, 1),
    )?;
    let final_loss = prediction.mean_squared_error(&target)?;
    for (i, receipt) in receipts.iter().enumerate() {
        if read_update(receipt).await? != (i + 1) as u64 {
            return Err("convolution update revision".into());
        }
    }
    let values = read_parameters(&parameters).await?;
    close(&values[0], &reference[..1])?;
    close(&values[1], &reference[1..])?;
    let expected: Vec<_> = input_values
        .iter()
        .map(|x| reference[0] * x + reference[1])
        .collect();
    close(&read_tensor(&prediction).await?, &expected)?;
    let initial_loss = read_tensor(&losses[0]).await?[0];
    let final_loss = read_tensor(final_loss.value()).await?[0];
    if !final_loss.is_finite() || !initial_loss.is_finite() || final_loss >= initial_loss * 0.1 {
        return Err("convolution training did not reduce MSE".into());
    }
    Ok(serde_json::json!({
        "schema": "spiraltorch.resident_parameters.contract.v1",
        "status": "passed",
        "parameter_checks": ["strided_offset_values", "strided_gradients", "signed_zero",
            "stale_gradients", "foreign_owner", "layout_validation", "rate_validation",
            "whole_update_overflow_rejection", "invalid_zero_rate", "valid_retry",
            "retained_receipts", "retained_snapshots", "multi_workgroup"],
        "conv2d_steps": 16,
        "conv2d_training_loop_readbacks": 0,
        "conv2d_initial_mse": initial_loss,
        "conv2d_final_mse": final_loss,
        "conv2d_final_parameters": values,
        "conv2d_reference_parameters": reference,
    }))
}
