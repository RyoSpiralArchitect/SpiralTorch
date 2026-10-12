//! Contextual-routing witnesses, independent of absolute parameter tolerances.
use super::*;

fn difference(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}

fn update_relative(
    actual: &[f32],
    before: &[f32],
    expected: &[f32],
    previous: &[f32],
) -> Result<f64> {
    if actual.len() != before.len()
        || actual.len() != expected.len()
        || actual.len() != previous.len()
        || actual
            .iter()
            .chain(before)
            .chain(expected)
            .chain(previous)
            .any(|v| !v.is_finite())
    {
        return Err("invalid contextual update comparison".into());
    }
    let mut norm = 0f64;
    let mut error = 0f64;
    let mut observed = 0f64;
    for ((&a, &b), (&e, &p)) in actual.iter().zip(before).zip(expected.iter().zip(previous)) {
        let delta = f64::from(a) - f64::from(b);
        let want = f64::from(e) - f64::from(p);
        norm += want * want;
        observed += delta * delta;
        error += (delta - want).powi(2);
    }
    if norm.sqrt() <= 1e-8 || observed == 0. || (error / norm).sqrt() > 0.002 {
        return Err(format!(
            "contextual update inert or inaccurate: ref={}, rel={}",
            norm.sqrt(),
            (error / norm).sqrt()
        )
        .into());
    }
    Ok((error / norm).sqrt())
}

pub(super) fn check(case: &Value, report: &Value) -> Result<Value> {
    let zero = case["query_controls"]["zero_init"]
        .as_bool()
        .ok_or("missing initialization")?;
    let output = data(&report["output"]);
    let shared = data(&case["query_controls"]["shared_output"]);
    let shared_error = output
        .iter()
        .zip(&shared)
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0., f64::max);
    let router_pullback = if zero {
        close(&output, &shared, "zero query recovers shared forward")?;
        None
    } else {
        if shared_error <= 1e-6 {
            return Err("contextual forward control is insensitive".into());
        }
        let detached = data(&case["query_controls"]["router_detached_embedding_gradient"]);
        let reference = difference(&data(&case["embedding_output_gradient"]), &detached);
        let delta = difference(&data(&report["embedding_output_gradient"]), &detached);
        let relative = geometry_controls::relative_gradient(&delta, &reference)?;
        if geometry_controls::relative_gradient(&vec![0.; reference.len()], &reference).is_ok() {
            return Err("detached-router control was not rejected".into());
        }
        Some(relative)
    };
    let descriptors = case["parameters"].as_array().ok_or("missing parameters")?;
    let actual = report["learning"]["trace"]
        .as_array()
        .ok_or("missing actual trajectory")?;
    let expected = case["learning"]["trace"]
        .as_array()
        .ok_or("missing reference trajectory")?;
    if actual.len() != 16 || expected.len() != 16 {
        return Err("incomplete contextual trajectory".into());
    }
    let mut max_relative = 0f64;
    let mut controls = 0;
    for (step, (a, e)) in actual.iter().zip(expected).enumerate() {
        for (slot, p) in descriptors.iter().enumerate() {
            let name = p["name"].as_str().ok_or("missing parameter name")?;
            if !name.starts_with("geometry.query_projection.")
                && !name.starts_with("geometry.mixture_logits.")
            {
                continue;
            }
            let before = data(if step == 0 {
                &p["values"]
            } else {
                &actual[step - 1]["parameters"][slot]
            });
            let previous = data(if step == 0 {
                &p["values"]
            } else {
                &expected[step - 1]["parameters"][slot]
            });
            let a = data(&a["parameters"][slot]);
            let e = data(&e["parameters"][slot]);
            max_relative = max_relative.max(
                update_relative(&a, &before, &e, &previous).map_err(|error| {
                    format!("{} step {} {name}: {error}", case["name"], step + 1)
                })?,
            );
            if update_relative(&before, &before, &e, &previous).is_ok() {
                return Err("frozen contextual update was not rejected".into());
            }
            controls += 1;
        }
    }
    Ok(
        json!({"zero_initialization":zero,"shared_output_max_difference":shared_error,
        "router_coordinate_vjp_relative_l2":router_pullback,"router_update_relative_l2":max_relative,
        "frozen_update_rejections":controls,"detached_router_rejected":!zero}),
    )
}

pub(super) async fn guard(runtime: &WgpuRuntime, case: &Value) -> Result<Value> {
    let mut probe = case.clone();
    let last = case["config"]["blocks"]
        .as_array()
        .ok_or("missing blocks")?
        .len()
        - 1;
    let weight = format!("geometry.query_projection.{last}.weight");
    let bias = format!("geometry.query_projection.{last}.bias");
    let table = format!("geometry.mixture_logits.{last}");
    for p in probe["parameters"]
        .as_array_mut()
        .ok_or("missing parameters")?
    {
        let n = data(&p["values"]).len();
        if p["name"] == weight || p["name"] == table {
            p["values"] = json!(vec![0.; n]);
        }
        if p["name"] == bias {
            p["values"] = json!(vec![f32::MAX; n]);
        }
    }
    let host = ByteLmBatch::from_windows(&[b"abaca", b"xyxyz"])?;
    let mut healthy = plan(&probe, 4)?.compile_training_wgpu(runtime.clone())?;
    let batch = healthy.prepare_batch(&host)?;
    read(healthy.forward(&batch)?.prediction()).await?;
    for p in probe["parameters"].as_array_mut().unwrap() {
        if p["name"] == table {
            p["values"] = json!(vec![f32::MAX; data(&p["values"]).len()]);
        }
    }
    let mut rejected = 0;
    for rate in [0., 0.125] {
        let mut broken = plan(&probe, 4)?.compile_training_wgpu(runtime.clone())?;
        let before =
            read_many(broken.tensor_device(), broken.parameter_snapshot().values()).await?;
        let batch = broken.prepare_batch(&host)?;
        let f = broken.forward(&batch)?;
        for value in [
            f.prediction(),
            f.geometry_mixture_weights(last)
                .ok_or("missing query weights")?,
        ] {
            if !matches!(read(value).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
            {
                return Err("invalid query projection left a readable forward".into());
            }
        }
        let seed = broken.tensor_device().upload(
            broken.output_layout().shape(),
            &vec![0.001; broken.output_layout().len()],
        )?;
        let gradient = broken.backward(&f, &seed)?;
        for value in gradient
            .parameter_gradients()
            .iter()
            .chain([gradient.embedding_output_gradient()])
        {
            if !matches!(read(value).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
            {
                return Err("invalid late query projection left a readable derivative".into());
            }
        }
        let update = broken.sgd(&gradient, rate)?;
        let receipt = update.snapshot()?;
        #[cfg(target_arch = "wasm32")]
        let status = receipt.read_async().await;
        #[cfg(not(target_arch = "wasm32"))]
        let status = receipt.read();
        if !matches!(status, Err(TrainingError::Rejected { stage:0, flags }) if flags & st_backend_wgpu::resident_tensor::INVALID_TENSOR_FLAG != 0)
        {
            return Err("invalid query update was not rejected".into());
        }
        let after = read_many(broken.tensor_device(), broken.parameter_snapshot().values()).await?;
        if before.len() != after.len() || before.iter().zip(&after).any(|(a, b)| !same_bits(a, b)) {
            return Err("query failure partially changed model parameters".into());
        }
        rejected += 1;
    }
    Ok(
        json!({"finite_control":true,"fault_block":last,"atomic_rejections":rejected,
        "all_parameter_gradients_guarded":true,"embedding_gradient_guarded":true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_all_or_some_mixture_snapshots_cannot_skip_validation() {
        assert!(require_mixture_count(0, 0).is_ok());
        assert!(require_mixture_count(2, 2).is_ok());
        for actual in [0, 1, 3] {
            assert!(require_mixture_count(2, actual).is_err());
        }
        assert!(require_mixture_count(0, 1).is_err());
    }
    #[test]
    fn delta_checks_reject_unchanged_tiny_parameters_and_nonfinite_inputs() {
        let before = [1e-4, -1e-4];
        let after = [1e-4 + 2e-7, -1e-4 - 3e-7];
        assert_eq!(
            update_relative(&after, &before, &after, &before).unwrap(),
            0.
        );
        assert!(update_relative(&before, &before, &after, &before).is_err());
        assert!(update_relative(&[f32::NAN; 2], &before, &after, &before).is_err());
        assert!(update_relative(&after, &before, &before, &before).is_err());
    }
}
