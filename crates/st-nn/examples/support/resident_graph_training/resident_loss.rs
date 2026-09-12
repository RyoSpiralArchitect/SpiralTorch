//! The ordinary Rust Loss seeds resident learning; observations follow all updates.
use super::*;
use st_backend_wgpu::resident_training::graph::GraphUpdateReadback;

async fn accepted(value: GraphUpdateReadback) -> Result<u64> {
    #[cfg(not(target_arch = "wasm32"))]
    let value = value.read()?;
    #[cfg(target_arch = "wasm32")]
    let value = value.read_async().await?;
    Ok(value)
}

async fn probes(runtime: WgpuRuntime) -> Result<Value> {
    let device = TensorDevice::new(runtime)?;
    let mut result = Vec::new();
    let mut objective = MeanSquaredError::new();
    for (name, shape) in [
        ("scalar", vec![]),
        ("empty", vec![2, 0, 3]),
        ("tail", vec![257]),
        ("many_partials", vec![65537]),
        ("nd", vec![2, 3, 4, 5]),
        ("strided", vec![3, 4, 5]),
        ("broadcast", vec![2, 3, 5]),
    ] {
        let n = shape.iter().product();
        let data: Vec<_> = (0..n).map(|i| ((i % 23) as f32 - 11.) / 16.).collect();
        let mut prediction = device.upload(&shape, &data)?;
        if name == "strided" {
            prediction = prediction.narrow(1, 1, 2)?.permute(&[2, 0, 1])?;
        }
        let target = if name == "broadcast" {
            device
                .upload(&[], &[0.25])?
                .broadcast_to(prediction.layout().shape())?
        } else {
            device.upload(
                prediction.layout().shape(),
                &vec![0.25; prediction.layout().len()],
            )?
        };
        let pair = objective.evaluate_resident(&prediction, &target)?;
        let a = tensor(prediction.snapshot()?).await?;
        let b = tensor(target.snapshot()?).await?;
        let host_a = Tensor::from_vec(1, a.len(), a.clone())?;
        let host_b = Tensor::from_vec(1, b.len(), b.clone())?;
        let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
            st_core::backend::device_caps::DeviceCaps::cpu(),
        ));
        let value = tensor(pair.value().snapshot()?).await?;
        let gradient = tensor(pair.prediction_gradient().snapshot()?).await?;
        close(&value, objective.forward(&host_a, &host_b)?.data())?;
        close(&gradient, objective.backward(&host_a, &host_b)?.data())?;
        result.push(json!({"name":name, "shape":prediction.layout().shape(),
            "prediction":a, "target":b, "loss":value[0], "gradient":gradient}));
    }
    Ok(json!(result))
}

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    for seed in [17, 29, 43] {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            let shape = vec![2, 3, 4];
            let input: Vec<_> = (0..24)
                .map(|i| ((i * 7 + seed) % 19) as f32 / 16. - 0.5)
                .collect();
            let target: Vec<_> = (0..18)
                .map(|i| ((i * 3 + seed) % 11) as f32 / 16. - 0.25)
                .collect();
            let mut original = model(seed)?;
            let baseline = InferencePlan::from_module(&original, NdLayout::contiguous(&shape)?)?;
            let plan = baseline.graph_definition()?;
            let roles: Vec<_> = plan.parameters().iter().map(|p| p.role).collect();
            let mut reference = model(seed)?;
            let x = Tensor::from_vec(6, 4, input.clone())?;
            let y = Tensor::from_vec(6, 3, target.clone())?;
            let mut learner = baseline.compile_graph_learner_wgpu(runtime.clone(), policy)?;
            let device = learner.tensor_device().clone();
            learner.set_input_tensor(&device.upload(&shape, &input)?)?;
            let target_gpu = device.upload(&[2, 3, 3], &target)?;
            let mut objective = MeanSquaredError::new();
            let rates: Vec<f32> = (0..64)
                .map(|i| if i % 17 == 0 { 0. } else { 0.03 })
                .collect();
            let mut held = Vec::new();
            for &rate in &rates {
                let forward = learner.forward()?;
                let loss = objective.evaluate_resident(forward.prediction(), &target_gpu)?;
                let gradient = learner.backward(&forward, loss.prediction_gradient())?;
                learner.sgd(&gradient, rate)?;
                held.push((
                    forward,
                    loss,
                    gradient,
                    learner.parameter_snapshot()?,
                    learner.update_snapshot()?,
                    cpu_step(&mut reference, &x, &y, rate, policy, &roles)?,
                ));
            }
            let final_forward = learner.forward()?;
            let final_loss =
                objective.evaluate_resident(final_forward.prediction(), &target_gpu)?;
            let updated = learner.parameter_snapshot()?;
            if (
                learner.input_generation(),
                learner.submitted_forwards(),
                learner.submitted_backwards(),
                learner.submitted_updates(),
            ) != (1, 65, 64, 64)
            {
                return Err("resident loss submission counts differ".into());
            }
            drop((learner, device, target_gpu));
            let mut steps = Vec::new();
            for (i, (forward, loss, gradient, values, receipt, expected)) in
                held.into_iter().enumerate()
            {
                if accepted(receipt).await? != i as u64 + 1 {
                    return Err("update receipt differs".into());
                }
                let prediction = tensor(forward.prediction().snapshot()?).await?;
                let loss = tensor(loss.value().snapshot()?).await?[0];
                let dx = tensor(gradient.input_gradient().snapshot()?).await?;
                let mut raw = Vec::new();
                for tensor_value in gradient.parameter_gradients() {
                    raw.push(tensor(tensor_value.snapshot()?).await?);
                }
                let values: Vec<_> = parameters(values)
                    .await?
                    .parameters()
                    .iter()
                    .map(|p| p.values.clone())
                    .collect();
                close(&[loss], &[expected.loss])?;
                close(&prediction, &expected.prediction)?;
                close(&dx, &expected.dx)?;
                for (actual, expected) in raw.iter().zip(&expected.raw) {
                    close(actual, expected)?;
                }
                for (actual, expected) in values.iter().zip(&expected.parameters) {
                    close(actual, expected)?;
                }
                let effective: Vec<Vec<f32>> = raw
                    .iter()
                    .zip(&roles)
                    .map(|(g, role)| {
                        g.iter()
                            .map(|v| {
                                if policy == GraphGradientPolicy::ModuleCompatible
                                    && *role == ParameterRole::Gain
                                {
                                    v / 6.
                                } else {
                                    *v
                                }
                            })
                            .collect()
                    })
                    .collect();
                steps.push(
                    json!({"loss":loss, "prediction":prediction, "input_gradient":dx,
                    "raw_gradients":raw, "effective_gradients":effective, "parameters":values}),
                );
            }
            let final_loss = tensor(final_loss.value().snapshot()?).await?[0];
            if final_loss >= steps[0]["loss"].as_f64().unwrap() as f32 {
                return Err("resident loss training did not improve".into());
            }
            let updated = InferencePlan::from_graph_definition(parameters(updated).await?)?;
            let applied = baseline.apply_parameters_to(
                &mut original,
                &updated,
                st_nn::resident::ModuleOptimizerStatePolicy::Reject,
            )?;
            let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
                st_core::backend::device_caps::DeviceCaps::cpu(),
            ));
            close(
                original.forward(&x)?.data(),
                &tensor(final_forward.prediction().snapshot()?).await?,
            )?;
            cases.push(json!({"seed":seed,"input_shape":shape,"input":input,"target":target,
                "policy":format!("{policy:?}"),"plan":serde_json::from_str::<Value>(&baseline.to_json()?)?,
                "rates":rates,"steps":steps,"final_loss":final_loss,
                "observations_after_updates":64,"module_parameters_applied":applied,
                "optimizer":"explicit_sgd_not_ModuleTrainer"}));
        }
    }
    Ok(json!({"status":"passed","cases":cases,"probes":probes(runtime).await?}))
}
