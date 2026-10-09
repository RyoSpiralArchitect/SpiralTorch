use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::TrainingError,
    runtime::WgpuRuntime,
};
use st_nn::{
    resident::{
        AttentionInferencePlan, AttentionMask, InferenceError, InferenceOp, InferencePlan,
        ResidualAttentionPlan, ToposResonatorKernel,
    },
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
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(actual) - f64::from(expected)).abs();
        if !actual.is_finite()
            || !expected.is_finite()
            || error > 3e-6 + 5e-5 * f64::from(expected).abs()
        {
            return Err(format!("{name}[{i}]: {actual} != {expected}").into());
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

fn parameter_count(case: &Value) -> usize {
    if case["topos"].as_bool().unwrap() {
        13
    } else {
        12
    }
}

fn plan(case: &Value) -> Result<ResidualAttentionPlan> {
    let dims = shape(&case["input_shape"]);
    let width = dims[2];
    let layout = NdLayout::contiguous(&dims)?;
    let row = |values: &Value| Tensor::from_vec(1, data(values).len(), data(values));
    let norm = |values: &Value| -> Result<InferenceOp> {
        Ok(InferenceOp::LayerNorm {
            gain: row(&values["gain"])?,
            bias: row(&values["bias"])?,
            epsilon: 1e-5,
        })
    };
    let pre = InferencePlan::from_operations(layout.clone(), vec![norm(&case["pre"])?])?;
    let projections = case["projections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let dims = shape(&p["weight_shape"]);
            Ok((
                Tensor::from_vec(dims[0], dims[1], data(&p["weight"]))?,
                row(&p["bias"])?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let attention = AttentionInferencePlan::from_parameters(
        layout.clone(),
        case["heads"].as_u64().unwrap() as usize,
        if case["causal"].as_bool().unwrap() {
            AttentionMask::Causal { query_offset: 0 }
        } else {
            AttentionMask::None
        },
        std::array::from_fn(|i| (&projections[i].0, &projections[i].1)),
    )?;
    let ff = &case["feed_forward"];
    let hidden = ff["hidden"].as_u64().unwrap() as usize;
    let mut operations = vec![
        norm(ff)?,
        InferenceOp::Linear {
            weight: Tensor::from_vec(width, hidden, data(&ff["up_weight"]))?,
            bias: row(&ff["up_bias"])?,
        },
        InferenceOp::Gelu,
    ];
    if case["topos"].as_bool().unwrap() {
        operations.push(InferenceOp::ToposResonator {
            gate: row(&ff["gate"])?,
            kernel: ToposResonatorKernel::new(0.2, 0.12, 0.3, 4)?,
            max_volume: dims[0] * dims[1] * hidden,
        });
    }
    operations.push(InferenceOp::Linear {
        weight: Tensor::from_vec(hidden, width, data(&ff["down_weight"]))?,
        bias: row(&ff["down_bias"])?,
    });
    let feed_forward = InferencePlan::from_operations(layout, operations)?;
    Ok(ResidualAttentionPlan::from_plans(
        &pre,
        &attention,
        &feed_forward,
    )?)
}

fn biases(device: &TensorDevice, case: &Value) -> Result<[Option<ResidentTensor>; 2]> {
    let dims = shape(&case["input_shape"]);
    let shape = [
        dims[0],
        case["heads"].as_u64().unwrap() as usize,
        dims[1],
        dims[1],
    ];
    let bias = |name: &str, shape: &[usize]| -> Result<_> {
        if case[name].is_null() {
            Ok(None)
        } else {
            Ok(Some(device.upload(shape, &data(&case[name]))?))
        }
    };
    Ok([bias("z_bias", &shape[..3])?, bias("pair_bias", &shape)?])
}

fn validate_fixture(fixture: &Value) -> Result<()> {
    if fixture["schema"] != "spiraltorch.residual_attention_torch.v1"
        || fixture["cases"].as_array().map(Vec::len) != Some(60)
        || fixture["training"].as_array().map(Vec::len) != Some(2)
    {
        return Err("wrong residual attention fixture schema/counts".into());
    }
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(fixture["training"].as_array().unwrap())
    {
        if case["projections"].as_array().map(Vec::len) != Some(4) {
            return Err("wrong projection count".into());
        }
        for name in ["initial_parameters", "parameter_gradients"] {
            if case[name].as_array().map(Vec::len) != Some(parameter_count(case)) {
                return Err(format!("wrong {name} count").into());
            }
        }
    }
    for case in fixture["training"].as_array().unwrap() {
        if case["losses"].as_array().map(Vec::len) != Some(32)
            || case["final_parameters"].as_array().map(Vec::len) != Some(parameter_count(case))
        {
            return Err("wrong training reference counts".into());
        }
    }
    Ok(())
}

async fn guard_checks(device: &TensorDevice, case: &Value) -> Result<Value> {
    let plan = plan(case)?;
    let mut model = plan.compile_training_wgpu(device.runtime().clone())?;
    let mut foreign = plan.compile_training_wgpu(device.runtime().clone())?;
    let input = device
        .upload(plan.input_layout().shape(), &data(&case["input"]))?
        .permute(&[0, 2, 1])?
        .contiguous()?
        .permute(&[0, 2, 1])?;
    let seed = device
        .upload(plan.output_layout().shape(), &data(&case["upstream"]))?
        .permute(&[0, 2, 1])?
        .contiguous()?
        .permute(&[0, 2, 1])?;
    let negative = device.upload(
        plan.output_layout().shape(),
        &data(&case["upstream"])
            .into_iter()
            .map(|v| -v)
            .collect::<Vec<_>>(),
    )?;
    let [z, pair] = biases(device, case)?;
    let first = model.forward(&input, z.as_ref(), pair.as_ref())?;
    let latest = model.forward(&input, z.as_ref(), pair.as_ref())?;
    if !matches!(
        model.backward(&first, &seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ) {
        return Err("old forward accepted".into());
    }
    foreign.forward(&input, z.as_ref(), pair.as_ref())?;
    if !matches!(
        foreign.backward(&latest, &seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ) {
        return Err("foreign forward accepted".into());
    }
    let invalid_shape = device.upload(&[1], &[1.])?;
    if model.forward(&invalid_shape, None, None).is_ok()
        || model.backward(&latest, &invalid_shape).is_ok()
    {
        return Err("invalid shape accepted".into());
    }
    let original = model.backward(&latest, &seed)?;
    let opposite = model.backward(&latest, &negative)?;
    if original.parameter_gradients().len() != parameter_count(case)
        || opposite.parameter_gradients().len() != parameter_count(case)
    {
        return Err("guard parameter count".into());
    }
    for (i, reference) in case["parameter_gradients"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        close(
            &read(&original.parameter_gradients()[i]).await?,
            &data(reference),
            "retained gradient",
        )?;
        close(
            &read(&opposite.parameter_gradients()[i]).await?,
            &data(reference).into_iter().map(|v| -v).collect::<Vec<_>>(),
            "opposite gradient",
        )?;
    }
    close(
        &read(original.input_gradient()).await?,
        &data(&case["input_gradient"]),
        "strided input VJP",
    )?;
    for (gradient, name) in [
        (original.z_bias_gradient(), "z_bias_gradient"),
        (original.pair_bias_gradient(), "pair_bias_gradient"),
    ] {
        close(
            &read(gradient.ok_or("missing geometry gradient")?).await?,
            &data(&case[name]),
            "retained geometry gradient",
        )?;
    }
    if !matches!(
        foreign.sgd(&original, 0.02),
        Err(InferenceError::Training(TrainingError::ParameterVersion))
    ) {
        return Err("foreign gradients accepted".into());
    }
    if model.sgd(&original, f32::NAN).is_ok() || model.parameter_snapshot().revision() != 0 {
        return Err("invalid learning rate changed revision".into());
    }
    let before = model.parameter_snapshot();
    let zero = model.sgd(&original, 0.)?;
    let snapshot = zero.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let revision = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let revision = snapshot.read_async().await?;
    if revision != 1
        || !matches!(
            model.sgd(&original, 0.),
            Err(InferenceError::Training(TrainingError::ParameterVersion))
        )
        || !matches!(
            model.backward(&latest, &seed),
            Err(InferenceError::Training(TrainingError::StaleForward))
        )
    {
        return Err("accepted update failed to invalidate tokens".into());
    }
    for (a, b) in before
        .values()
        .iter()
        .zip(model.parameter_snapshot().values())
    {
        if !read(a)
            .await?
            .iter()
            .map(|v| v.to_bits())
            .eq(read(b).await?.iter().map(|v| v.to_bits()))
        {
            return Err("zero-rate changed parameters".into());
        }
    }
    drop((model, foreign, input, seed, negative, z, pair));
    close(
        &read(first.prediction()).await?,
        &data(&case["expected"]),
        "retained first output",
    )?;
    close(
        &read(latest.prediction()).await?,
        &data(&case["expected"]),
        "retained latest output",
    )?;
    close(
        &read(original.input_gradient()).await?,
        &data(&case["input_gradient"]),
        "retained input gradient",
    )?;
    Ok(json!({
        "strided_input_and_cotangent": true, "old_forward_rejected": true,
        "foreign_forward_rejected": true, "foreign_gradients_rejected": true,
        "shape_errors_preserve_latest_tape": true, "repeated_cotangents_independent": true,
        "invalid_rate_preserves_revision": true, "zero_rate_preserves_parameters": true,
        "accepted_update_invalidates_tokens": true, "owned_outputs_survive_drop": true,
    }))
}

async fn terminal_guard_checks(device: &TensorDevice) -> Result<Value> {
    for output_failure in [true, false] {
        let layout = NdLayout::contiguous(&[1, 1, 1])?;
        let scalar = |value| Tensor::from_vec(1, 1, vec![value]);
        let linear = |weight, bias| -> Result<InferenceOp> {
            Ok(InferenceOp::Linear {
                weight: scalar(weight)?,
                bias: scalar(bias)?,
            })
        };
        let pre = InferencePlan::from_operations(layout.clone(), vec![linear(1., 0.)?])?;
        let zero = scalar(0.)?;
        let one = scalar(1.)?;
        let bias = scalar(if output_failure { 2e38 } else { 0. })?;
        let attention = AttentionInferencePlan::from_parameters(
            layout.clone(),
            1,
            AttentionMask::None,
            [(&zero, &zero), (&zero, &zero), (&one, &zero), (&one, &bias)],
        )?;
        let feed = InferencePlan::from_operations(
            layout,
            vec![linear(0., if output_failure { 2e38 } else { 0. })?],
        )?;
        let plan = ResidualAttentionPlan::from_plans(&pre, &attention, &feed)?;
        let mut model = plan.compile_training_wgpu(device.runtime().clone())?;
        let input = device.upload(&[1, 1, 1], &[0.])?;
        let seed = device.upload(&[1, 1, 1], &[if output_failure { 1. } else { 2e38 }])?;
        let before = model.parameter_snapshot();
        if before.values().len() != 8 {
            return Err("terminal guard parameter count".into());
        }
        let z = device.upload(&[1, 1, 1], &[0.])?;
        let pair = device.upload(&[1, 1, 1, 1], &[0.])?;
        let forward = model.forward(&input, Some(&z), Some(&pair))?;
        let snapshot = forward.prediction().snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        let prediction = snapshot.read();
        #[cfg(target_arch = "wasm32")]
        let prediction = snapshot.read_async().await;
        if output_failure {
            if !matches!(
                prediction,
                Err(st_backend_wgpu::resident_tensor::TensorError::NonFinite)
            ) {
                return Err("terminal prediction did not flag overflow".into());
            }
        } else {
            close(&prediction?, &[0.], "finite prediction")?;
        }
        let gradients = model.backward(&forward, &seed)?;
        if gradients.parameter_gradients().len() != 8 {
            return Err("terminal gradient count".into());
        }
        if gradients.z_bias_gradient().is_none() || gradients.pair_bias_gradient().is_none() {
            return Err("terminal geometry gradient missing".into());
        }
        for tensor in std::iter::once(gradients.input_gradient())
            .chain(gradients.parameter_gradients())
            .chain(gradients.z_bias_gradient())
            .chain(gradients.pair_bias_gradient())
        {
            let snapshot = tensor.snapshot()?;
            #[cfg(not(target_arch = "wasm32"))]
            let gradient = snapshot.read();
            #[cfg(target_arch = "wasm32")]
            let gradient = snapshot.read_async().await;
            if !matches!(
                gradient,
                Err(st_backend_wgpu::resident_tensor::TensorError::NonFinite)
            ) {
                return Err("terminal failure missing from whole VJP".into());
            }
        }
        let update = model.sgd(&gradients, 0.01)?;
        let snapshot = update.snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        let acceptance = snapshot.read();
        #[cfg(target_arch = "wasm32")]
        let acceptance = snapshot.read_async().await;
        if !matches!(acceptance, Err(TrainingError::Rejected { .. }))
            || model.parameter_snapshot().revision() != 1
        {
            return Err("terminal failure did not reject update".into());
        }
        for (before, after) in before
            .values()
            .iter()
            .zip(model.parameter_snapshot().values())
        {
            if !read(before)
                .await?
                .iter()
                .map(|v| v.to_bits())
                .eq(read(after).await?.iter().map(|v| v.to_bits()))
            {
                return Err("terminal failure partially updated block".into());
            }
        }
    }
    Ok(
        json!({"terminal_prediction_failure_rejects_vjp_and_sgd": true,
              "terminal_input_gradient_failure_rejects_vjp_and_sgd": true}),
    )
}

/// Same probe executes natively and in browser WebGPU. No readback occurs inside
/// either 32-update loop, including optimizer acceptance flags.
pub async fn run() -> Result<Value> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/residual_attention_torch.json"
    ))?;
    validate_fixture(&fixture)?;
    let runtime = WgpuRuntime::request_headless("residual.attention.fixture").await?;
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
        if gradients.parameter_gradients().len() != parameter_count(case) {
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
    let mut training = Vec::new();
    for case in fixture["training"].as_array().unwrap() {
        let plan = plan(case)?;
        let mut model = plan.compile_training_wgpu(runtime.clone())?;
        let input = device.upload(plan.input_layout().shape(), &data(&case["input"]))?;
        let target = device.upload(plan.output_layout().shape(), &data(&case["target"]))?;
        let [z, pair] = biases(&device, case)?;
        let initial = model.parameter_snapshot();
        if initial.values().len() != parameter_count(case) {
            return Err("initial count".into());
        }
        let mut observations = Vec::new();
        for _ in 0..32 {
            let forward = model.forward(&input, z.as_ref(), pair.as_ref())?;
            let loss = forward.prediction().mean_squared_error(&target)?;
            let gradients = model.backward(&forward, loss.prediction_gradient())?;
            observations.push((
                loss.value().clone(),
                model.sgd(&gradients, case["rate"].as_f64().unwrap() as f32)?,
            ));
        }
        let mut updates = Vec::new();
        for (i, (loss, update)) in observations.iter().enumerate() {
            let snapshot = update.snapshot()?;
            #[cfg(not(target_arch = "wasm32"))]
            let revision = snapshot.read()?;
            #[cfg(target_arch = "wasm32")]
            let revision = snapshot.read_async().await?;
            if revision != (i + 1) as u64 {
                return Err("attempted revision mismatch".into());
            }
            let actual = read(loss).await?;
            let reference = case["losses"][i].as_f64().unwrap() as f32;
            updates.push(
                json!({"revision": revision, "loss": actual[0], "reference_loss": reference,
                "max_abs_error": close(&actual, &[reference], "loss")?}),
            );
        }
        let before_rejection = model.parameter_snapshot();
        if before_rejection.revision() != 32
            || before_rejection.values().len() != parameter_count(case)
        {
            return Err("final parameter count/revision".into());
        }
        let mut parameter_errors = Vec::new();
        for (actual, expected) in before_rejection
            .values()
            .iter()
            .zip(case["final_parameters"].as_array().unwrap())
        {
            parameter_errors.push(close(
                &read(actual).await?,
                &data(expected),
                "final parameter",
            )?);
        }
        for (actual, expected) in initial
            .values()
            .iter()
            .zip(case["initial_parameters"].as_array().unwrap())
        {
            close(
                &read(actual).await?,
                &data(expected),
                "retained initial parameter",
            )?;
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
            || model.parameter_snapshot().revision() != 33
            || !matches!(
                model.sgd(&bad, 0.03),
                Err(InferenceError::Training(TrainingError::ParameterVersion))
            )
        {
            return Err("invalid update accepted or stale gradients usable".into());
        }
        for (before, after) in before_rejection
            .values()
            .iter()
            .zip(model.parameter_snapshot().values())
        {
            if !read(before)
                .await?
                .iter()
                .map(|v| v.to_bits())
                .eq(read(after).await?.iter().map(|v| v.to_bits()))
            {
                return Err("rejected update changed parameters".into());
            }
        }
        let recovered = model.forward(&input, z.as_ref(), pair.as_ref())?;
        close(
            &read(recovered.prediction()).await?,
            &data(&case["final_prediction"]),
            "recovery",
        )?;
        training.push(json!({"topos": case["topos"], "updates": updates,
            "final_parameter_errors": parameter_errors, "final_prediction_error": prediction_error,
            "rejected_update_preserves_all_parameters": true, "recovery": true}));
    }
    let mut guards = guard_checks(&device, &fixture["training"][1]).await?;
    for (name, value) in terminal_guard_checks(&device).await?.as_object().unwrap() {
        guards[name] = value.clone();
    }
    Ok(json!({
        "schema": "spiraltorch.residual_attention_check.v1", "passed": true,
        "adapter": {"name": runtime.adapter_info().name, "backend": format!("{:?}", runtime.adapter_info().backend),
                    "device_type": format!("{:?}", runtime.adapter_info().device_type)},
        "tolerance": {"atol": 3e-6, "rtol": 5e-5}, "checks": checks, "training": training, "guards": guards,
    }))
}
