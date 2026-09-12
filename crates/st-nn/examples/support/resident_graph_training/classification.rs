//! Ordinary classification Loss, N-D probes and resident learning on both clients.
use super::*;
use st_backend_wgpu::resident_tensor::ResidentTensor;
use st_backend_wgpu::resident_training::graph::GraphUpdateReadback;
use st_nn::CrossEntropyWithLogits;
use st_tensor::{CrossEntropyConfig, LossReduction};
async fn accepted(value: GraphUpdateReadback) -> Result<u64> {
    #[cfg(not(target_arch = "wasm32"))]
    let value = value.read()?;
    #[cfg(target_arch = "wasm32")]
    let value = value.read_async().await?;
    Ok(value)
}

async fn probe(
    name: &str,
    prediction: ResidentTensor,
    target: ResidentTensor,
    config: CrossEntropyConfig,
) -> Result<Value> {
    let shape = prediction.layout().shape().to_vec();
    let classes = *shape.last().unwrap();
    let rows = prediction.layout().len() / classes;
    let mut objective = CrossEntropyWithLogits::new(config)?;
    let pair = objective.evaluate_resident(&prediction, &target)?;
    let input = tensor(prediction.snapshot()?).await?;
    let labels = tensor(target.snapshot()?).await?;
    let actual = tensor(pair.value().snapshot()?).await?;
    let gradient = tensor(pair.prediction_gradient().snapshot()?).await?;
    let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
        st_core::backend::device_caps::DeviceCaps::cpu(),
    ));
    let host = Tensor::from_vec(rows, classes, input.clone())?;
    let target_host = Tensor::from_vec(rows, 1, labels.clone())?;
    close(&actual, objective.forward(&host, &target_host)?.data())?;
    close(&gradient, objective.backward(&host, &target_host)?.data())?;
    if name == "tiny_tail" {
        let tail = (-80f64).exp() as f32;
        for (a, b) in actual.iter().chain(&gradient).zip([tail, -tail, tail]) {
            if (a / b - 1.).abs() > 2e-5 {
                return Err("classification tiny tail erased".into());
            }
        }
    }
    Ok(
        json!({"name":name,"shape":shape,"prediction":input,"target":labels,
        "reduction":config.reduction.as_str(),"ignore_index":config.ignore_index,
        "label_smoothing":config.label_smoothing,"loss":actual,"gradient":gradient}),
    )
}

async fn probes(runtime: WgpuRuntime) -> Result<Value> {
    let device = TensorDevice::new(runtime)?;
    let mut out = Vec::new();
    for name in ["nd", "uniform", "strided", "broadcast"] {
        let shape = if name == "uniform" {
            vec![2, 1, 257]
        } else if name == "strided" {
            vec![3, 4, 7]
        } else {
            vec![2, 3, 7]
        };
        let mut values: Vec<_> = (0..shape.iter().product())
            .map(|i| ((i * 7 % 29) as f32 - 14.) / 16.)
            .collect();
        if name == "uniform" {
            values.fill(0.);
        }
        let mut prediction = device.upload(&shape, &values)?;
        if name == "strided" {
            prediction = prediction.narrow(1, 1, 2)?.permute(&[1, 0, 2])?;
        }
        if name == "broadcast" {
            prediction = device.upload(&[7], &values[..7])?.broadcast_to(&shape)?;
        }
        let classes = *prediction.layout().shape().last().unwrap();
        let rows = prediction.layout().len() / classes;
        let labels: Vec<_> = (0..rows)
            .map(|i| {
                if i == 1 {
                    -100.
                } else {
                    ((i * 17) % classes) as f32
                }
            })
            .collect();
        let target_shape = &prediction.layout().shape()[..2];
        let target = device.upload(target_shape, &labels)?;
        for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
            for smoothing in [0., 0.2, 1.] {
                out.push(
                    probe(
                        name,
                        prediction.clone(),
                        target.clone(),
                        CrossEntropyConfig {
                            reduction,
                            label_smoothing: smoothing,
                            ..Default::default()
                        },
                    )
                    .await?,
                );
            }
        }
    }
    let config = |r, s| CrossEntropyConfig {
        reduction: r,
        label_smoothing: s,
        ..Default::default()
    };
    for (name, shape, values, labels, cfg) in [
        (
            "wide_mean",
            vec![2, 2],
            vec![f32::MAX, -f32::MAX, 0., 0.],
            vec![1., 0.],
            config(LossReduction::Mean, 0.),
        ),
        (
            "tiny_smoothing",
            vec![1, 2],
            vec![f32::MAX, -f32::MAX],
            vec![0.],
            config(LossReduction::Mean, 1e-40),
        ),
        (
            "tiny_tail",
            vec![1, 2],
            vec![80., 0.],
            vec![0.],
            config(LossReduction::Mean, 0.),
        ),
        (
            "wide_vocab",
            vec![1, 50257],
            (0..50257).map(|i| ((i % 29) as f32 - 14.) / 16.).collect(),
            vec![50256.],
            config(LossReduction::Mean, 0.2),
        ),
        (
            "single_class",
            vec![3, 1],
            vec![f32::MAX, -f32::MAX, 0.],
            vec![0., 0., 0.],
            config(LossReduction::Mean, 0.2),
        ),
        (
            "empty_none",
            vec![0, 3],
            vec![],
            vec![],
            config(LossReduction::None, 0.),
        ),
        (
            "empty_sum",
            vec![0, 3],
            vec![],
            vec![],
            config(LossReduction::Sum, 0.),
        ),
    ] {
        out.push(
            probe(
                name,
                device.upload(&shape, &values)?,
                device.upload(&[labels.len()], &labels)?,
                cfg,
            )
            .await?,
        );
    }
    Ok(json!(out))
}

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    for (seed, smoothing) in [(17, 0.), (29, 0.2), (43, 0.5)] {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            let shape = vec![2, 3, 4];
            let input: Vec<_> = (0..24)
                .map(|i| ((i * 7 + seed) % 19) as f32 / 16. - 0.5)
                .collect();
            let labels = vec![0., 1., -100., 2., 0., 1.];
            let mut original = model(seed)?;
            let mut reference = model(seed)?;
            let baseline = InferencePlan::from_module(&original, NdLayout::contiguous(&shape)?)?;
            let roles: Vec<_> = baseline
                .graph_definition()?
                .parameters()
                .iter()
                .map(|p| p.role)
                .collect();
            let mut learner = baseline.compile_graph_learner_wgpu(runtime.clone(), policy)?;
            let device = learner.tensor_device().clone();
            learner.set_input_tensor(&device.upload(&shape, &input)?)?;
            let target = device.upload(&[2, 3], &labels)?;
            let x = Tensor::from_vec(6, 4, input.clone())?;
            let y = Tensor::from_vec(6, 1, labels.clone())?;
            let cfg = CrossEntropyConfig {
                label_smoothing: smoothing,
                ..Default::default()
            };
            let mut objective = CrossEntropyWithLogits::new(cfg)?;
            let mut cpu_objective = CrossEntropyWithLogits::new(cfg)?;
            let rates: Vec<f32> = (0..64)
                .map(|i| if i % 17 == 0 { 0. } else { 0.1 })
                .collect();
            let mut held = Vec::new();
            for &rate in &rates {
                let forward = learner.forward()?;
                let pair = objective.evaluate_resident(forward.prediction(), &target)?;
                let gradient = learner.backward(&forward, pair.prediction_gradient())?;
                learner.sgd(&gradient, rate)?;
                held.push((
                    forward,
                    pair,
                    gradient,
                    learner.parameter_snapshot()?,
                    learner.update_snapshot()?,
                    cpu_step_with_loss(
                        &mut reference,
                        &x,
                        &y,
                        rate,
                        policy,
                        &roles,
                        &mut cpu_objective,
                    )?,
                ));
            }
            let final_forward = learner.forward()?;
            let final_pair = objective.evaluate_resident(final_forward.prediction(), &target)?;
            let updated = learner.parameter_snapshot()?;
            if (
                learner.input_generation(),
                learner.submitted_forwards(),
                learner.submitted_backwards(),
                learner.submitted_updates(),
            ) != (1, 65, 64, 64)
            {
                return Err("classification submission counts".into());
            }
            drop((learner, device, target));
            let mut steps = Vec::new();
            for (i, (forward, pair, gradient, values, receipt, expected)) in
                held.into_iter().enumerate()
            {
                if accepted(receipt).await? != i as u64 + 1 {
                    return Err("classification rejected update".into());
                }
                let prediction = tensor(forward.prediction().snapshot()?).await?;
                let value = tensor(pair.value().snapshot()?).await?[0];
                let dx = tensor(gradient.input_gradient().snapshot()?).await?;
                let mut raw = Vec::new();
                for g in gradient.parameter_gradients() {
                    raw.push(tensor(g.snapshot()?).await?);
                }
                let values: Vec<_> = parameters(values)
                    .await?
                    .parameters()
                    .iter()
                    .map(|p| p.values.clone())
                    .collect();
                close(&[value], &[expected.loss])?;
                close(&prediction, &expected.prediction)?;
                close(&dx, &expected.dx)?;
                for (a, b) in raw.iter().zip(&expected.raw) {
                    close(a, b)?;
                }
                for (a, b) in values.iter().zip(&expected.parameters) {
                    close(a, b)?;
                }
                let effective: Vec<Vec<f32>> = raw
                    .iter()
                    .zip(&roles)
                    .map(|(g, r)| {
                        g.iter()
                            .map(|&v| {
                                if policy == GraphGradientPolicy::ModuleCompatible
                                    && *r == ParameterRole::Gain
                                {
                                    v / 6.
                                } else {
                                    v
                                }
                            })
                            .collect()
                    })
                    .collect();
                steps.push(
                    json!({"loss":value,"prediction":prediction,"input_gradient":dx,
                    "raw_gradients":raw,"effective_gradients":effective,"parameters":values}),
                );
            }
            let final_loss = tensor(final_pair.value().snapshot()?).await?[0];
            if final_loss >= steps[0]["loss"].as_f64().unwrap() as f32 {
                return Err("classification did not learn".into());
            }
            let final_prediction = tensor(final_forward.prediction().snapshot()?).await?;
            let updated = InferencePlan::from_graph_definition(parameters(updated).await?)?;
            let applied = baseline.apply_parameters_to(
                &mut original,
                &updated,
                st_nn::resident::ModuleOptimizerStatePolicy::Reject,
            )?;
            let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
                st_core::backend::device_caps::DeviceCaps::cpu(),
            ));
            close(original.forward(&x)?.data(), &final_prediction)?;
            cases.push(json!({"seed":seed,"input_shape":shape,"input":input,"target":labels,
                "policy":format!("{policy:?}"),"plan":serde_json::from_str::<Value>(&baseline.to_json()?)?,
                "rates":rates,"steps":steps,"final_loss":final_loss,"final_prediction":final_prediction,
                "label_smoothing":smoothing,"ignore_index":-100,"reduction":"mean",
                "observations_after_updates":64,"module_parameters_applied":applied,"optimizer":"explicit_sgd_not_ModuleTrainer"}));
        }
    }
    Ok(json!({"status":"passed","cases":cases,"probes":probes(runtime).await?}))
}
