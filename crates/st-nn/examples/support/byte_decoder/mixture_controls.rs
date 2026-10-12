//! A zero-mass component is not permission to hide its invalid computation.
use super::*;

pub(super) async fn wide_pullback(runtime: WgpuRuntime) -> Result<Value> {
    let device = TensorDevice::new(runtime)?;
    let mut checks = Vec::new();
    for (name, shape, values, logits, seed) in [
        (
            "finite_final",
            [1, 1, 1, 2],
            vec![1., 2.],
            vec![0.; 2],
            vec![f32::MAX],
        ),
        (
            "position_centering",
            [1, 1, 2, 2],
            vec![f32::MAX, f32::MAX, 1., 2.],
            vec![0.; 2],
            vec![1.; 2],
        ),
        (
            "batch_centering",
            [2, 1, 1, 2],
            vec![f32::MAX, f32::MAX, 1., 2.],
            vec![0.; 2],
            vec![1.; 2],
        ),
        (
            "identical_components",
            [2, 1, 1, 3],
            vec![f32::MAX; 6],
            vec![-0.3, 0.2, 0.7],
            vec![f32::MAX; 2],
        ),
    ] {
        let forward = device
            .upload(&shape, &values)?
            .headwise_mixture(&device.upload(&[shape[1], shape[3]], &logits)?)?;
        let gradient = forward.backward(&device.upload(&shape[..3], &seed)?)?;
        checks.push(json!({"name":name, "output":read(forward.output()).await?,
            "logits":read(gradient.logits()).await?,
            "components":read(gradient.components()).await?}));
    }
    let forward = device
        .upload(&[1, 1, 1, 2], &[1., 10.])?
        .headwise_mixture(&device.upload(&[1, 2], &[0., 0.])?)?;
    read(forward.output()).await?;
    let gradient = forward.backward(&device.upload(&[1, 1, 1], &[f32::MAX])?)?;
    for tensor in [gradient.components(), gradient.logits()] {
        if !matches!(read(tensor).await,Err(e) if matches!(e.downcast_ref::<TensorError>(),Some(TensorError::NonFinite)))
        {
            return Err("nonfinite final mixture derivative was not guarded".into());
        }
    }
    Ok(
        json!({"schema":"spiraltorch.mixture_wide_pullback.v1", "checks":checks,
        "nonfinite_final_rejected":true}),
    )
}

pub(super) async fn run(runtime: &WgpuRuntime, case: &Value) -> Result<Value> {
    let mut probe = case.clone();
    for p in probe["parameters"]
        .as_array_mut()
        .ok_or("missing parameters")?
    {
        let values = match p["name"].as_str().ok_or("missing name")? {
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
            "geometry.mixture_logits.0" => {
                Some(vec![f32::MAX, -f32::MAX, 0., f32::MAX, -f32::MAX, 0.])
            }
            _ => None,
        };
        if let Some(v) = values {
            p["values"] = json!(v);
        }
    }
    probe["config"]["causal_geometry"]["mixture"]["scales"] = json!([[1., 1., 1., 1., 1., 1.]]);
    let host = ByteLmBatch::from_windows(&[b"abaca", b"xyxyz"])?;
    let mut finite = plan(&probe, 4)?.compile_training_wgpu(runtime.clone())?;
    let batch = finite.prepare_batch(&host)?;
    let f = finite.forward(&batch)?;
    read(f.prediction()).await?;
    let scores = read(f.geometry_pair_bias(0).ok_or("missing mixture bias")?).await?;
    let trigger = scores.iter().map(|v| f64::from(v.abs())).fold(0., f64::max);
    if trigger <= 2. {
        return Err("mixture overflow trigger is insensitive".into());
    }
    for p in probe["parameters"].as_array_mut().unwrap() {
        if p["name"] == "geometry.mixture_logits.0" {
            p["values"] = json!([-f32::MAX, f32::MAX, 0., -f32::MAX, f32::MAX, 0.]);
        }
    }
    let mut inactive = plan(&probe, 4)?.compile_training_wgpu(runtime.clone())?;
    let batch = inactive.prepare_batch(&host)?;
    let f = inactive.forward(&batch)?;
    read(f.prediction()).await?;
    if read(
        f.geometry_mixture_weights(0)
            .ok_or("missing mixture mass")?,
    )
    .await?
        != [0., 1., 0., 0., 1., 0.]
    {
        return Err("inactive component control lost zero mass".into());
    }
    probe["config"]["causal_geometry"]["mixture"]["scales"][0][3] = json!(f32::MAX);
    let mut broken = plan(&probe, 4)?.compile_training_wgpu(runtime.clone())?;
    let before = read_many(broken.tensor_device(), broken.parameter_snapshot().values()).await?;
    let batch = broken.prepare_batch(&host)?;
    let f = broken.forward(&batch)?;
    for v in [
        f.prediction(),
        f.geometry_pair_bias(0).unwrap(),
        f.geometry_mixture_weights(0).unwrap(),
    ] {
        if !matches!(read(v).await,Err(e) if matches!(e.downcast_ref::<TensorError>(),Some(TensorError::NonFinite)))
        {
            return Err("inactive invalid component did not guard the complete forward".into());
        }
    }
    let seed = broken.tensor_device().upload(
        broken.output_layout().shape(),
        &vec![0.001; broken.output_layout().len()],
    )?;
    let g = broken.backward(&f, &seed)?;
    for v in g
        .parameter_gradients()
        .iter()
        .chain([g.embedding_output_gradient()])
    {
        if !matches!(read(v).await,Err(e) if matches!(e.downcast_ref::<TensorError>(),Some(TensorError::NonFinite)))
        {
            return Err("inactive invalid component left a readable derivative".into());
        }
    }
    let update = broken.sgd(&g, 0.)?;
    let receipt = update.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let status = receipt.read_async().await;
    #[cfg(not(target_arch = "wasm32"))]
    let status = receipt.read();
    if !matches!(status,Err(TrainingError::Rejected {stage:0,flags}) if flags & st_backend_wgpu::resident_tensor::INVALID_TENSOR_FLAG != 0)
    {
        return Err("invalid mixture accepted or misclassified".into());
    }
    let after = read_many(broken.tensor_device(), broken.parameter_snapshot().values()).await?;
    if before.len() != after.len() || before.iter().zip(&after).any(|(a, b)| !same_bits(a, b)) {
        return Err("mixture rejection partially changed model values".into());
    }
    Ok(
        json!({"schema":"spiraltorch.byte_mixture_guard.v1","finite_control":true,
        "zero_mass_control":true,"baseline_max_abs_bias":trigger,"fault_head":1,
        "fault_component":0,"all_gradients_guarded":true,"atomic_rejection":true,"parameter_tensors":before.len()}),
    )
}
