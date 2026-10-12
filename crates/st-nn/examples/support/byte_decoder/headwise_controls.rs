//! One failing metric group must reject the complete model update.
use super::*;

pub(super) async fn run(runtime: &WgpuRuntime, case: &Value) -> Result<Value> {
    let mut probe = case.clone();
    let parameters = probe["parameters"]
        .as_array_mut()
        .ok_or("missing parameters")?;
    // Alternating, widely separated coordinates give a deterministic overflow
    // in only the final Poincare group when its finite raw gain becomes MAX.
    for p in parameters.iter_mut() {
        let name = p["name"].as_str().ok_or("missing parameter name")?;
        let values = match name {
            "token_embedding" => Some(
                (0..256 * 4)
                    .map(|i| if (i / 4 + i % 4) % 2 == 0 { 10. } else { -10. })
                    .collect::<Vec<f32>>(),
            ),
            "position_embedding"
            | "geometry.projection.bias"
            | "geometry.raw_phase"
            | "geometry.raw_gain.0" => Some(vec![0.; data(&p["values"]).len()]),
            "geometry.projection.weight" => Some(
                (0..16)
                    .map(|i| if i / 4 == i % 4 { 1. } else { 0. })
                    .collect(),
            ),
            "geometry.raw_decay" => Some(vec![-50.; 2]),
            _ => None,
        };
        if let Some(values) = values {
            p["values"] = json!(values);
        }
    }
    let host = ByteLmBatch::from_windows(&[b"abaca", b"xyxyz"])?;
    let mut finite = plan(&probe, 4)?.compile_training_wgpu(runtime.clone())?;
    let batch = finite.prepare_batch(&host)?;
    let forward = finite.forward(&batch)?;
    read(forward.prediction()).await?;
    let scores = read(forward.geometry_pair_bias(0).ok_or("missing routed bias")?).await?;
    let trigger = (0..2)
        .flat_map(|b| scores[(b * 4 + 3) * 16..(b * 4 + 4) * 16].iter())
        .map(|v| f64::from(v.abs()))
        .fold(0., f64::max);
    if trigger <= 2. {
        return Err("headwise overflow control lost its trigger".into());
    }
    probe["parameters"][6]["values"][3] = json!(f32::MAX);
    let mut model = plan(&probe, 4)?.compile_training_wgpu(runtime.clone())?;
    let before = read_many(model.tensor_device(), model.parameter_snapshot().values()).await?;
    let batch = model.prepare_batch(&host)?;
    let forward = model.forward(&batch)?;
    for output in [
        forward.prediction(),
        forward.geometry_pair_bias(0).ok_or("missing routed bias")?,
    ] {
        if !matches!(read(output).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
        {
            return Err("one invalid head group did not guard concatenated outputs".into());
        }
    }
    let seed = model.tensor_device().upload(
        model.output_layout().shape(),
        &vec![0.001; model.output_layout().len()],
    )?;
    let gradient = model.backward(&forward, &seed)?;
    for tensor in gradient
        .parameter_gradients()
        .iter()
        .chain([gradient.embedding_output_gradient()])
    {
        if !matches!(read(tensor).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
        {
            return Err("one invalid metric group left a readable gradient".into());
        }
    }
    let update = model.sgd(&gradient, 0.)?;
    let receipt = update.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let status = receipt.read_async().await;
    #[cfg(not(target_arch = "wasm32"))]
    let status = receipt.read();
    if !matches!(status, Err(TrainingError::Rejected { stage: 0, flags }) if flags & st_backend_wgpu::resident_tensor::INVALID_TENSOR_FLAG != 0)
    {
        return Err("invalid head group update was accepted or misclassified".into());
    }
    let after = read_many(model.tensor_device(), model.parameter_snapshot().values()).await?;
    if before.len() != after.len() || before.iter().zip(&after).any(|(a, b)| !same_bits(a, b)) {
        return Err("headwise rejection partially mutated the model".into());
    }
    Ok(
        json!({"schema":"spiraltorch.byte_headwise_group_guard.v1", "fault_head":3,
        "finite_control":true, "baseline_max_abs_bias":trigger, "all_gradients_guarded":true,
        "atomic_rejection":true, "parameter_tensors":before.len()}),
    )
}
