use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::TrainingError,
    runtime::WgpuRuntime,
};
use st_kernel_contracts::attention::AttentionMask;
use st_nn::{
    resident::{AttentionInferencePlan, InferenceError},
    Tensor,
};
use st_tensor::NdLayout;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn data(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn shape(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}
fn close(actual: &[f32], expected: &[f32], name: &str) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err(format!("{name}: length mismatch").into());
    }
    let mut maximum = 0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - f64::from(e)).abs();
        if !a.is_finite() || !e.is_finite() || error > 3e-6 + 5e-5 * f64::from(e).abs() {
            return Err(format!("{name}[{i}]: {a} != {e}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    Ok(values)
}
fn plan(case: &Value) -> Result<AttentionInferencePlan> {
    let parameters = case["projections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let dims = shape(&p["weight_shape"]);
            Ok((
                Tensor::from_vec(dims[0], dims[1], data(&p["weight"]))?,
                Tensor::from_vec(1, dims[1], data(&p["bias"]))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(AttentionInferencePlan::from_parameters(
        NdLayout::contiguous(&shape(&case["input_shape"]))?,
        case["heads"].as_u64().unwrap() as usize,
        if case["causal"].as_bool().unwrap() {
            AttentionMask::Causal { query_offset: 0 }
        } else {
            AttentionMask::None
        },
        std::array::from_fn(|i| (&parameters[i].0, &parameters[i].1)),
    )?)
}
fn biases(device: &TensorDevice, case: &Value) -> Result<[Option<ResidentTensor>; 2]> {
    let input = shape(&case["input_shape"]);
    let dims = [
        input[0],
        case["heads"].as_u64().unwrap() as usize,
        input[1],
        input[1],
    ];
    let bias = |name: &str, shape: &[usize]| -> Result<_> {
        if case[name].is_null() {
            Ok(None)
        } else {
            Ok(Some(device.upload(shape, &data(&case[name]))?))
        }
    };
    Ok([bias("z_bias", &dims[..3])?, bias("pair_bias", &dims)?])
}

fn validate_fixture(fixture: &Value) -> Result<()> {
    if fixture["schema"] != "spiraltorch.resident_attention_training_torch.v1"
        || fixture["cases"].as_array().map(Vec::len) != Some(30)
        || fixture["training"]["final_parameters"]
            .as_array()
            .map(Vec::len)
            != Some(4)
        || fixture["training"]["losses"].as_array().map(Vec::len) != Some(16)
    {
        return Err("wrong fixture contract or reference counts".into());
    }
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(std::iter::once(&fixture["training"]))
    {
        for field in ["projections", "parameter_gradients"] {
            if case[field].as_array().map(Vec::len) != Some(4) {
                return Err(format!("wrong {field} reference count").into());
            }
        }
    }
    Ok(())
}

/// Shared native/browser probe. The update loop does not map any activation,
/// gradient, weight or acceptance flag; terminal observations happen afterward.
pub async fn run() -> Result<Value> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/resident_attention_training_torch.json"
    ))?;
    validate_fixture(&fixture)?;
    let runtime = WgpuRuntime::request_headless("attention.training.fixture").await?;
    #[cfg(not(target_arch = "wasm32"))]
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("real GPU required".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let mut checks = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let plan = plan(case)?;
        let mut model = plan.compile_training_wgpu(runtime.clone())?;
        let input = device.upload(plan.input_layout().shape(), &data(&case["input"]))?;
        let seed = device.upload(plan.output_layout().shape(), &data(&case["upstream"]))?;
        let [z, pair] = biases(&device, case)?;
        let forward = model.forward(&input, z.as_ref(), pair.as_ref())?;
        let gradients = model.backward(&forward, &seed)?;
        drop((model, input, seed, z, pair));
        let mut errors = json!({
            "forward": close(&read(forward.prediction()).await?, &data(&case["expected"]), "forward")?,
            "input": close(&read(gradients.input_gradient()).await?, &data(&case["input_gradient"]), "input")?,
        });
        if gradients.parameter_gradients().len() != 4 {
            return Err("parameter gradient count".into());
        }
        let mut parameter_errors = Vec::new();
        for (gradient, expected) in gradients
            .parameter_gradients()
            .iter()
            .zip(case["parameter_gradients"].as_array().unwrap())
        {
            parameter_errors.push(close(&read(gradient).await?, &data(expected), "parameter")?);
        }
        errors["parameters"] = json!(parameter_errors);
        for (gradient, name) in [
            (gradients.z_bias_gradient(), "z_bias_gradient"),
            (gradients.pair_bias_gradient(), "pair_bias_gradient"),
        ] {
            if gradient.is_none() != case[name].is_null() {
                return Err("geometry gradient presence".into());
            }
            if let Some(gradient) = gradient {
                errors[name] = json!(close(&read(gradient).await?, &data(&case[name]), name)?);
            }
        }
        checks.push(json!({"name": case["name"], "max_abs_error": errors}));
    }

    let case = &fixture["training"];
    let plan = plan(case)?;
    let mut model = plan.compile_training_wgpu(runtime.clone())?;
    let input = device.upload(plan.input_layout().shape(), &data(&case["input"]))?;
    let target = device.upload(plan.output_layout().shape(), &data(&case["target"]))?;
    let [z, pair] = biases(&device, case)?;
    let mut observations = Vec::new();
    for _ in 0..16 {
        let forward = model.forward(&input, z.as_ref(), pair.as_ref())?;
        let loss = forward.prediction().mean_squared_error(&target)?;
        let gradients = model.backward(&forward, loss.prediction_gradient())?;
        let update = model.sgd(&gradients, case["rate"].as_f64().unwrap() as f32)?;
        observations.push((loss.value().clone(), update));
    }
    let mut updates = Vec::new();
    for (i, (loss, update)) in observations.iter().enumerate() {
        let snapshot = update.snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        let revision = snapshot.read()?;
        #[cfg(target_arch = "wasm32")]
        let revision = snapshot.read_async().await?;
        if revision != (i + 1) as u64 {
            return Err("attempted update revision".into());
        }
        let loss = read(loss).await?;
        let expected = case["losses"][i].as_f64().unwrap() as f32;
        let error = close(&loss, &[expected], "loss")?;
        updates.push(json!({"revision": revision, "loss": loss[0], "reference_loss": expected, "max_abs_error": error}));
    }
    let before_rejection = model.parameter_snapshot();
    if before_rejection.values().len() != 4 {
        return Err("final parameter count".into());
    }
    let mut parameter_errors = Vec::new();
    for (value, expected) in before_rejection
        .values()
        .iter()
        .zip(case["final_parameters"].as_array().unwrap())
    {
        parameter_errors.push(close(
            &read(value).await?,
            &data(expected),
            "final parameter",
        )?);
    }
    let forward = model.forward(&input, z.as_ref(), pair.as_ref())?;
    let prediction_error = close(
        &read(forward.prediction()).await?,
        &data(&case["final_prediction"]),
        "final prediction",
    )?;
    let huge = device.upload(
        plan.output_layout().shape(),
        &vec![f32::MAX; plan.output_layout().len()],
    )?;
    let bad = model.backward(&forward, &huge.mul(&huge)?)?;
    let rejected = model.sgd(&bad, 0.03)?;
    let snapshot = rejected.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let rejection = snapshot.read();
    #[cfg(target_arch = "wasm32")]
    let rejection = snapshot.read_async().await;
    if !matches!(rejection, Err(TrainingError::Rejected { .. }))
        || model.parameter_snapshot().revision() != 17
        || !matches!(
            model.sgd(&bad, 0.03),
            Err(InferenceError::Training(TrainingError::ParameterVersion))
        )
    {
        return Err("failed update acceptance/version".into());
    }
    if model.parameter_snapshot().values().len() != 4 {
        return Err("post-rejection parameter count".into());
    }
    for (a, b) in before_rejection
        .values()
        .iter()
        .zip(model.parameter_snapshot().values())
    {
        let a = read(a).await?;
        let b = read(b).await?;
        if a.iter()
            .map(|x| x.to_bits())
            .ne(b.iter().map(|x| x.to_bits()))
        {
            return Err("partial parameter update".into());
        }
    }
    Ok(json!({
        "schema":"spiraltorch.resident_attention_training.v1", "passed":true,
        "adapter":format!("{:?}", runtime.adapter_info()), "oracle":fixture["oracle"],
        "checks":checks, "updates":updates, "final_parameter_errors":parameter_errors,
        "final_prediction_max_abs_error":prediction_error,
        "rejected_update_preserves_all_parameters":true, "rejected_update_invalidates_old_gradients":true,
        "scope":"projection/attention VJP and plain SGD; fixed caller-owned geometry biases; no speed or language-quality claim",
    }))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn fixture_count_guards_reject_missing_and_extra_references() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/resident_attention_training_torch.json"
        ))
        .unwrap();
        validate_fixture(&fixture).unwrap();
        for path in [
            "/cases",
            "/cases/0/projections",
            "/cases/0/parameter_gradients",
            "/training/projections",
            "/training/parameter_gradients",
            "/training/final_parameters",
            "/training/losses",
        ] {
            for extra in [false, true] {
                let mut changed = fixture.clone();
                let values = changed.pointer_mut(path).unwrap().as_array_mut().unwrap();
                if extra {
                    values.push(values[0].clone());
                } else {
                    values.pop();
                }
                assert!(validate_fixture(&changed).is_err(), "{path}, extra={extra}");
            }
        }
    }
}
