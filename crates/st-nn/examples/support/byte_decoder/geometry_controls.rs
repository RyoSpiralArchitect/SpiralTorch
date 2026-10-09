use super::*;

// Absolute tolerance alone can accept a severed, tiny geometry derivative.
pub(super) fn relative_gradient(actual: &[f32], expected: &[f32]) -> Result<f64> {
    if actual.len() != expected.len() || actual.iter().chain(expected).any(|v| !v.is_finite()) {
        return Err("invalid geometry gradient family".into());
    }
    let norm = expected
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    let actual_norm = actual
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (f64::from(a) - f64::from(e)).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm <= 1e-8 || actual_norm == 0. || error / norm > 0.002 {
        return Err(format!("geometry derivative is inert or inaccurate: norm={norm}, actual={actual_norm}, relative={}",error/norm).into());
    }
    Ok(error / norm)
}

pub(super) async fn run(
    runtime: &WgpuRuntime,
    case: &Value,
    coupled_output: &[f32],
    coupled_gradient: &[f32],
) -> Result<Value> {
    let mut ordinary = case.clone();
    let full_plan = plan(case, 4)?;
    let width = full_plan.embedding_width();
    let qk_zero = full_plan.parameter_layout().blocks().iter().all(|r| {
        let weights = data(&case["parameters"][r.start + 2]["values"]);
        let bias = data(&case["parameters"][r.start + 3]["values"]);
        weights
            .chunks_exact(3 * width)
            .all(|row| row[..2 * width].iter().all(|&v| v == 0.))
            && bias[..2 * width].iter().all(|&v| v == 0.)
    });
    if case["config"]["causal_geometry"]["metric_only_scores"].as_bool() != Some(qk_zero) {
        return Err("zero-Q/K isolation does not match actual fixture parameters".into());
    }
    let slots = full_plan
        .parameter_layout()
        .geometry()
        .ok_or("geometry missing")?
        .all();
    ordinary["parameters"]
        .as_array_mut()
        .unwrap()
        .drain(slots.clone());
    ordinary["config"]["causal_geometry"] = Value::Null;
    let ordinary_plan = plan(&ordinary, 4)?;
    if ordinary_plan.parameter_layout().geometry().is_some() {
        return Err("off control still has geometry".into());
    }
    let mut model = ordinary_plan.compile_training_wgpu(runtime.clone())?;
    let windows: Vec<Vec<u8>> = case["windows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            sizes(v)
                .into_iter()
                .map(|n| u8::try_from(n).unwrap())
                .collect()
        })
        .collect();
    let batch = model.prepare_batch(&batch(&windows, 4)?)?;
    let external = biases(model.tensor_device(), &ordinary, 4)?;
    let off = model.forward_with_external_biases(&batch, &borrowed(&external))?;
    let off_values = read(off.prediction()).await?;
    let off_error = close(
        &off_values,
        &data(&case["controls"]["off_output"]),
        "geometry off logits",
    )?;
    let separation = off_values
        .iter()
        .zip(coupled_output.iter().copied())
        .map(|(&a, b)| f64::from((a - b).abs()))
        .fold(0., f64::max);
    if separation <= 1e-6 {
        return Err("geometry-off control is insensitive".into());
    }
    let difference = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a - b).collect::<Vec<_>>();
    let output_contrast_relative = relative_gradient(
        &difference(coupled_output, &off_values),
        &difference(
            &data(&case["output"]),
            &data(&case["controls"]["off_output"]),
        ),
    )?;
    ordinary["biases"] = case["controls"]["combined_biases"].clone();
    let frozen_bias = biases(model.tensor_device(), &ordinary, 4)?;
    let detached = model.forward_with_external_biases(&batch, &borrowed(&frozen_bias))?;
    let detached_error = close(
        &read(detached.prediction()).await?,
        &data(&case["output"]),
        "detached geometry same logits",
    )?;
    let seed = model
        .tensor_device()
        .upload(model.output_layout().shape(), &data(&case["cotangent"]))?;
    let gradient = model.backward(&detached, &seed)?;
    let detached_gradient = read(gradient.embedding_output_gradient()).await?;
    close(
        &detached_gradient,
        &data(&case["controls"]["detached_embedding_gradient"]),
        "detached geometry input VJP",
    )?;
    let gradient_delta = detached_gradient
        .iter()
        .zip(coupled_gradient.iter().copied())
        .map(|(&a, b)| f64::from((a - b).abs()))
        .fold(0., f64::max);
    if gradient_delta <= 1e-8 {
        return Err("detached geometry control cannot detect the missing pullback".into());
    }
    let pullback_contrast_relative = relative_gradient(
        &difference(coupled_gradient, &detached_gradient),
        &difference(
            &data(&case["embedding_output_gradient"]),
            &data(&case["controls"]["detached_embedding_gradient"]),
        ),
    )?;
    // Verify that the relative checker rejects zero and wrong gradients.
    for bad in [vec![0., 0.], vec![1., -1.], vec![f32::NAN, 0.]] {
        if relative_gradient(&bad, &[1., 2.]).is_ok() {
            return Err("weak geometry gradient checker".into());
        }
    }
    if relative_gradient(&[1., 2.], &[1., 2.])? != 0. {
        return Err("relative checker positive control failed".into());
    }
    Ok(
        json!({"off_output_max_abs_error":off_error,"off_output_separation":separation,
        "detached_output_max_abs_error":detached_error,"detached_embedding_gradient_separation":gradient_delta,
        "relative_checker_negative_controls":3,"geometry_parameter_slots":[slots.start,slots.end],
        "output_contrast_relative_l2":output_contrast_relative,"pullback_contrast_relative_l2":pullback_contrast_relative,
        "qk_scores_zero":qk_zero}),
    )
}
