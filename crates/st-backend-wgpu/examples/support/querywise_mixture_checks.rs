use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice, TensorError, INVALID_TENSOR_FLAG},
    resident_training::{parameters::ResidentParameters, TrainingError},
    runtime::WgpuRuntime,
};
use st_kernel_contracts::mixture::{QuerywiseMixtureForward, QuerywiseMixtureSpec};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn read(t: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = t.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    Ok(values)
}
fn data(v: &Value, len: usize) -> Result<Vec<f32>> {
    let values = v.as_array().ok_or("missing numeric array")?;
    if values.len() != len {
        return Err("array length mismatch".into());
    }
    values
        .iter()
        .map(|v| {
            let x = v.as_f64().ok_or("invalid number")? as f32;
            if !x.is_finite() {
                return Err("nonfinite float32".into());
            }
            Ok(x)
        })
        .collect()
}
fn close(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.len() != b.len() {
        return Err("comparison length".into());
    }
    let mut maximum = 0f64;
    for (&a, &b) in a.iter().zip(b) {
        let error = (f64::from(a) - f64::from(b)).abs();
        if !a.is_finite() || !b.is_finite() || error > 3e-6 + 5e-5 * f64::from(b).abs() {
            return Err(format!("querywise mismatch: {a} != {b}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn router_close(a: &[f32], b: &[f32]) -> Result<f64> {
    close(a, b)?;
    let norm: f64 = b.iter().map(|&v| f64::from(v).powi(2)).sum();
    let error: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    if norm.sqrt() <= 1e-8
        || (error / norm).sqrt() > 0.002
        || a.iter().zip(b).any(|(&a, &b)| b.abs() > 1e-8 && a == 0.)
    {
        return Err("query router has no qualified gradient".into());
    }
    Ok((error / norm).sqrt())
}
fn update_close(initial: &[f32], actual: &[f32], expected: &[f32]) -> Result<f64> {
    close(actual, expected)?;
    if initial.len() != expected.len() {
        return Err("update length".into());
    }
    let norm: f64 = initial
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(b) - f64::from(a)).powi(2))
        .sum();
    let error: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(b) - f64::from(a)).powi(2))
        .sum();
    if norm.sqrt() <= 1e-8 || (error / norm).sqrt() > 0.002 || same_bits(initial, actual) {
        return Err("unqualified query parameter update".into());
    }
    Ok((error / norm).sqrt())
}
fn strided(d: &TensorDevice, shape: &[usize], values: &[f32]) -> Result<ResidentTensor> {
    let padded = [shape, &[2]].concat();
    Ok(d.upload(
        &padded,
        &values.iter().flat_map(|&v| [99., v]).collect::<Vec<_>>(),
    )?
    .select(shape.len(), 1)?)
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| a.to_bits() == b.to_bits())
}
async fn guards(d: &TensorDevice) -> Result<Value> {
    let x = d.upload(&[1, 1, 2, 1, 2], &[1., 2., 3., 4.])?;
    let logits = d.upload(&[1, 1, 2, 2], &[f32::MAX, -f32::MAX, 0., 0.])?;
    let seed = d.upload(&[1, 1, 2, 1], &[1., 1.])?;
    let good = x.querywise_mixture(&logits)?;
    let expected = read(good.output()).await?;
    let large = d.upload(&[1], &[f32::MAX])?;
    let invalid = large.add(&large)?;
    let healthy = d.upload(&[2], &[7., -3.])?;
    let healthy_gradient = d.upload(&[2], &[0.5, -0.75])?;
    let mut rejected = 0;
    for operand in 0..3 {
        let failed = |v: &ResidentTensor| -> Result<ResidentTensor> {
            Ok(d.guard_together(&[v, &invalid])?[0].clone())
        };
        let f = if operand == 0 { failed(&x)? } else { x.clone() }.querywise_mixture(
            &if operand == 1 {
                failed(&logits)?
            } else {
                logits.clone()
            },
        )?;
        let g = f.backward(&if operand == 2 {
            failed(&seed)?
        } else {
            seed.clone()
        })?;
        for tensor in [g.components(), g.logits()] {
            if !matches!(read(tensor).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
            {
                return Err("query VJP lost its family guard".into());
            }
        }
        for rate in [0., 0.05] {
            let mut owner =
                ResidentParameters::new(vec![x.clone(), logits.clone(), healthy.clone()])?;
            let bound = owner.snapshot().bind_gradients(vec![
                g.components().clone(),
                g.logits().clone(),
                healthy_gradient.clone(),
            ])?;
            let receipt = owner.sgd(&bound, rate)?.snapshot()?;
            #[cfg(target_arch = "wasm32")]
            let status = receipt.read_async().await;
            #[cfg(not(target_arch = "wasm32"))]
            let status = receipt.read();
            if !matches!(status, Err(TrainingError::Rejected {stage:0,flags}) if flags & INVALID_TENSOR_FLAG != 0)
            {
                return Err("query invalid SGD was not atomically rejected".into());
            }
            let snapshot = owner.snapshot();
            if !same_bits(&read(&snapshot.values()[0]).await?, &[1., 2., 3., 4.])
                || !same_bits(
                    &read(&snapshot.values()[1]).await?,
                    &[f32::MAX, -f32::MAX, 0., 0.],
                )
                || !same_bits(&read(&snapshot.values()[2]).await?, &[7., -3.])
            {
                return Err("query rejected update mutated parameters".into());
            }
            rejected += 1;
        }
    }
    let mut positive = ResidentParameters::new(vec![healthy])?;
    let bound = positive.snapshot().bind_gradients(vec![healthy_gradient])?;
    let receipt = positive.sgd(&bound, 0.05)?.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    receipt.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    receipt.read()?;
    let updated = read(&positive.snapshot().values()[0]).await?;
    close(&updated, &[6.975, -2.9625])?;
    if same_bits(&updated, &[7., -3.]) {
        return Err("healthy sibling did not qualify a nonzero update".into());
    }
    if !same_bits(&expected, &read(good.output()).await?) {
        return Err("retained forward mutated".into());
    }
    if router_close(&[0.; 2], &[-1e-4, 1e-4]).is_ok() || same_bits(&[0.], &[-0.]) {
        return Err("invalid checker accepted negative control".into());
    }
    Ok(
        json!({"guarded_operands":3,"atomic_rejections":rejected,"retained":true,"negative_controls":2,
            "healthy_sibling_unchanged":true,"healthy_positive_control":true}),
    )
}
pub async fn run(runtime: WgpuRuntime, input: &str) -> Result<Value> {
    if input.len() > 32 * 1024 * 1024 {
        return Err("fixture exceeds 32 MiB".into());
    }
    let fixture: Value = serde_json::from_str(input)?;
    if fixture["schema"] != "spiraltorch.querywise_mixture.qualification.v1"
        || fixture["rate"].as_f64() != Some(0.05)
    {
        return Err("query fixture identity".into());
    }
    let cases = fixture["cases"].as_array().ok_or("missing cases")?;
    if cases.is_empty() || cases.len() > 64 {
        return Err("bounded query case count".into());
    }
    let adapter = format!("{:?}", runtime.adapter_info());
    let device = TensorDevice::new(runtime)?;
    let mut checks = Vec::new();
    let mut names = std::collections::HashSet::new();
    for case in cases {
        let name = case["name"].as_str().ok_or("missing name")?;
        if !names.insert(name) {
            return Err("duplicate name".into());
        }
        let dims = case["shape"].as_array().ok_or("missing shape")?;
        let shape: Vec<usize> = dims
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("invalid dimension")
            })
            .collect::<std::result::Result<_, _>>()?;
        let shape: [usize; 5] = shape.try_into().map_err(|_| "expected five axes")?;
        let spec = QuerywiseMixtureSpec::new(shape)?;
        if spec.len() > 1_000_000 {
            return Err("bounded query volume".into());
        }
        let [b, h, q, s, k] = shape;
        let x = data(&case["input"], spec.len())?;
        let l = data(&case["logits"], spec.router_len())?;
        let seed = data(&case["seed"], spec.output_len())?;
        let expected_output = data(&case["output"], spec.output_len())?;
        let expected_weights = data(&case["weights"], spec.router_len())?;
        let expected_x = data(&case["input_gradient"], spec.len())?;
        let expected_l = data(&case["logits_gradient"], spec.router_len())?;
        let cpu = QuerywiseMixtureForward::new(spec, &x, &l)?;
        let cpu_g = cpu.backward(&seed)?;
        let cpu_error = close(cpu.output(), &expected_output)?
            .max(close(cpu.weights(), &expected_weights)?)
            .max(close(&cpu_g.input, &expected_x)?)
            .max(close(&cpu_g.logits, &expected_l)?);
        router_close(&cpu_g.logits, &expected_l)?;
        let x_gpu = strided(&device, &shape, &x)?;
        let l_gpu = strided(&device, &[b, h, q, k], &l)?;
        let f = x_gpu.querywise_mixture(&l_gpu)?;
        let g = f.backward(&strided(&device, &[b, h, q, s], &seed)?)?;
        let output = read(f.output()).await?;
        let weights = read(f.weights()).await?;
        let gx = read(g.components()).await?;
        let gl = read(g.logits()).await?;
        let output_error = close(&output, &expected_output)?;
        let weight_error = close(&weights, &expected_weights)?;
        let gx_error = close(&gx, &expected_x)?;
        let gl_error = close(&gl, &expected_l)?;
        let router_error = router_close(&gl, &expected_l)?;
        let again = f.backward(&device.upload(&[b, h, q, s], &seed)?)?;
        if !same_bits(&gx, &read(again.components()).await?)
            || !same_bits(&gl, &read(again.logits()).await?)
        {
            return Err("query repeated backward differs".into());
        }
        let mut owner = ResidentParameters::new(vec![x_gpu, l_gpu])?;
        let bound = owner
            .snapshot()
            .bind_gradients(vec![g.components().clone(), g.logits().clone()])?;
        let receipt = owner.sgd(&bound, 0.05)?.snapshot()?;
        #[cfg(target_arch = "wasm32")]
        receipt.read_async().await?;
        #[cfg(not(target_arch = "wasm32"))]
        receipt.read()?;
        let snapshot = owner.snapshot();
        let after_x = read(&snapshot.values()[0]).await?;
        let after_l = read(&snapshot.values()[1]).await?;
        let expected_after_x = data(&case["updated_input"], spec.len())?;
        let expected_after_l = data(&case["updated_logits"], spec.router_len())?;
        let step_error =
            close(&after_x, &expected_after_x)?.max(close(&after_l, &expected_after_l)?);
        let component_update_error = update_close(&x, &after_x, &expected_after_x)?;
        let router_update_error = update_close(&l, &after_l, &expected_after_l)?;
        if update_close(&x, &x, &expected_after_x).is_ok()
            || update_close(&l, &l, &expected_after_l).is_ok()
        {
            return Err("a skipped query update passed verification".into());
        }
        let updated = snapshot.values()[0].querywise_mixture(&snapshot.values()[1])?;
        let after_error = close(
            &read(updated.output()).await?,
            &data(&case["updated_output"], spec.output_len())?,
        )?;
        if !same_bits(&output, &read(f.output()).await?) {
            return Err("SGD mutated a retained forward".into());
        }
        checks.push(json!({"name":name,"shape":shape,"output_max_abs":output_error,"weights_max_abs":weight_error,
            "components_vjp_max_abs":gx_error,"logits_vjp_max_abs":gl_error,"router_relative_l2":router_error,
            "cpu_max_abs":cpu_error,"sgd_max_abs":step_error,"updated_output_max_abs":after_error,
            "component_update_relative_l2":component_update_error,"router_update_relative_l2":router_update_error,
            "frozen_updates_rejected":true,
            "strided":true,"repeat_bits":true,"retained":true}));
    }
    Ok(
        json!({"schema":"spiraltorch.querywise_mixture.validation.v2","passed":true,"adapter":adapter,
        "checks":checks,"guards":guards(&device).await?,
        "scope":"Query-local primitive and frozen LM cotangents, not an input-conditioned LM or quality/speed advantage"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_skipped_update_cannot_hide_under_absolute_tolerance() {
        let initial = [1., -1.];
        let updated = [1.0000005, -1.0000005];
        assert!(close(&initial, &updated).is_ok());
        assert!(update_close(&initial, &initial, &updated).is_err());
        assert_eq!(update_close(&initial, &updated, &updated).unwrap(), 0.);
        assert!(update_close(&initial, &initial, &initial).is_err());
    }
    #[test]
    fn invalid_comparison_and_frozen_gradient_controls_reject() {
        assert!(close(&[f32::NAN], &[0.]).is_err());
        assert!(router_close(&[0., 0.], &[-1e-4, 1e-4]).is_err());
        assert!(data(&json!([true, 0]), 2).is_err());
        assert!(data(&json!([1e300]), 1).is_err());
        assert!(!same_bits(&[0.], &[-0.]));
        assert!(!same_bits(&[0.], &[]));
    }
}
