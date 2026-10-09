use super::*;

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/resident_byte_geometry_torch.json"
    ))?;
    if fixture["schema"] != "spiraltorch.resident_byte_geometry.torch_fixture.v1"
        || fixture["cases"].as_array().map(Vec::len) != Some(2)
    {
        return Err("incomplete initial geometry oracle".into());
    }
    let mut checks = Vec::new();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let p = plan(case, 4)?;
        let count = p.parameter_layout().len();
        let slots = p.parameter_layout().geometry().ok_or("no geometry")?.all();
        let descriptors = case["parameters"].as_array().unwrap();
        if count != [23, 37][index] || descriptors.len() != count {
            return Err("incomplete parameter layout".into());
        }
        let initial = descriptors
            .iter()
            .map(|d| data(&d["values"]))
            .collect::<Vec<_>>();
        let mut rates = vec![0.125; count];
        rates[slots.clone()].fill(0.);
        let windows: Vec<Vec<u8>> = case["windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                sizes(row)
                    .into_iter()
                    .map(|v| u8::try_from(v).unwrap())
                    .collect()
            })
            .collect();
        let mut model = p.compile_training_wgpu(runtime.clone())?;
        let resident = model.prepare_batch(&batch(&windows, 4)?)?;
        let external = biases(model.tensor_device(), case, 4)?;
        let forward = model.forward_with_external_biases(&resident, &borrowed(&external))?;
        let output = read(forward.prediction()).await?;
        close(
            &output,
            &data(&case["output"]),
            "initial frozen geometry logits",
        )?;
        let seed = model
            .tensor_device()
            .upload(model.output_layout().shape(), &data(&case["cotangent"]))?;
        let gradient = model.backward(&forward, &seed)?;
        let gradients = read_many(model.tensor_device(), gradient.parameter_gradients()).await?;
        for (i, actual) in gradients.iter().enumerate() {
            close(
                actual,
                &data(&case["parameter_gradients"][i]),
                "initial parameter VJP",
            )?;
            if slots.contains(&i) {
                geometry_controls::relative_gradient(
                    actual,
                    &data(&case["parameter_gradients"][i]),
                )?;
            }
        }
        let embedding_gradient = read(gradient.embedding_output_gradient()).await?;
        close(
            &embedding_gradient,
            &data(&case["embedding_output_gradient"]),
            "initial embedding VJP",
        )?;
        let metric_controls =
            geometry_controls::run(&runtime, case, &output, &embedding_gradient).await?;

        if !matches!(
            model.sgd_with_rates(&gradient, &rates[..count - 1]),
            Err(InferenceError::Training(TrainingError::ParameterLayout))
        ) {
            return Err("model accepted an incomplete rate vector".into());
        }
        let mut invalid_rates = rates.clone();
        invalid_rates[count - 1] = f32::NAN;
        if !matches!(
            model.sgd_with_rates(&gradient, &invalid_rates),
            Err(InferenceError::Training(TrainingError::LearningRate))
        ) || model.parameter_snapshot().revision() != 0
        {
            return Err("late rate preflight changed the model".into());
        }
        let repeated = model.backward(&forward, &seed)?;
        let repeated = read_many(model.tensor_device(), repeated.parameter_gradients()).await?;
        if repeated.len() != gradients.len()
            || gradients
                .iter()
                .zip(&repeated)
                .any(|(a, b)| !same_bits(a, b))
        {
            return Err("rate preflight invalidated the retained tape".into());
        }

        let mut pending = Vec::new();
        for _ in 0..16 {
            let forward = model.forward_with_external_biases(&resident, &borrowed(&external))?;
            let loss =
                forward.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
            let gradient = model.backward(&forward, loss.prediction_gradient())?;
            let update = model.sgd_with_rates(&gradient, &rates)?;
            pending.push((
                loss,
                update,
                model.parameter_snapshot(),
                gradient.parameter_gradients()[slots.clone()].to_vec(),
                gradient.embedding_output_gradient().clone(),
            ));
        }
        // Keep the full queued learner resident; only the verification phase maps buffers.
        let mut trace = Vec::new();
        let mut first_step_error = 0f64;
        for (step, (loss, update, snapshot, geometry_gradients, embedded)) in
            pending.into_iter().enumerate()
        {
            let receipt = update.snapshot()?;
            #[cfg(target_arch = "wasm32")]
            let revision = receipt.read_async().await?;
            #[cfg(not(target_arch = "wasm32"))]
            let revision = receipt.read()?;
            if revision != (step + 1) as u64 || snapshot.revision() != revision {
                return Err("per-parameter update revision drift".into());
            }
            let parameters = read_many(model.tensor_device(), snapshot.values()).await?;
            if parameters.len() != count {
                return Err("truncated parameter snapshot".into());
            }
            for (i, actual) in parameters.iter().enumerate() {
                if snapshot.values()[i].layout().shape() != sizes(&descriptors[i]["shape"]) {
                    return Err("parameter shape drift".into());
                }
                if slots.contains(&i) {
                    if !same_bits(actual, &initial[i]) {
                        return Err("frozen geometry parameter changed bits".into());
                    }
                } else if step == 0 {
                    first_step_error = first_step_error.max(close(
                        actual,
                        &data(&case["learning"]["trace"][0]["parameters"][i]),
                        "first non-frozen update",
                    )?);
                }
            }
            if same_bits(&parameters[0], &initial[0]) || same_bits(&parameters[1], &initial[1]) {
                return Err("freezing geometry stopped embedding learning".into());
            }
            let geometry_gradients = read_many(model.tensor_device(), &geometry_gradients).await?;
            if geometry_gradients.len() != slots.len()
                || geometry_gradients.iter().any(|g| {
                    g.iter().any(|v| !v.is_finite())
                        || g.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt() <= 1e-8
                })
            {
                return Err("frozen geometry derivative is inert".into());
            }
            let loss = read(loss.value()).await?;
            if loss.len() != 1 || !loss[0].is_finite() {
                return Err("invalid mean CE".into());
            }
            trace.push(json!({"revision":revision, "loss":loss[0], "parameters":parameters,
                "geometry_gradients":geometry_gradients, "embedding_output_gradient":read(&embedded).await?}));
        }
        checks.push(json!({"name":case["name"], "parameter_count":count,
            "parameter_names":descriptors.iter().map(|d| d["name"].clone()).collect::<Vec<_>>(),
            "parameter_shapes":descriptors.iter().map(|d| d["shape"].clone()).collect::<Vec<_>>(),
            "geometry_slots":[slots.start, slots.end], "rates":rates,
            "invalid_rates_preserve_tape":true, "frozen_parameter_bits":true,
            "embedding_parameters_learn":true, "metric_controls":metric_controls,
            "first_step_non_geometry_max_abs_error":first_step_error, "trace":trace}));
    }
    Ok(
        json!({"schema":"spiraltorch.resident_byte_parameter_rates.validation.v1", "passed":true,
        "checks":checks, "scope":"synthetic learner correctness, not quality or performance"}),
    )
}
