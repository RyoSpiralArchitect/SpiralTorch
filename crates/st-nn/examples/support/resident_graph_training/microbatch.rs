//! Changing input batches, weighted gradient windows, one shared parameter state.
use super::*;
use st_backend_wgpu::{
    resident_tensor::{loss::ResidentLoss, ResidentTensor},
    resident_training::graph::{GraphForward, GraphGradients, GraphUpdateReadback},
};
use st_nn::CrossEntropyWithLogits;
use st_tensor::{CrossEntropyConfig, LossReduction};

struct Micro {
    id: usize,
    weight: f32,
    forward: GraphForward,
    pair: ResidentLoss,
    gradients: GraphGradients,
    expected: Reference,
}
struct Window {
    micros: Vec<Micro>,
    sum: Vec<ResidentTensor>,
    expected_sum: Vec<Vec<f32>>,
    parameters: GraphParameterReadback,
    expected_parameters: Vec<Vec<f32>>,
    receipt: GraphUpdateReadback,
    rate: f32,
    clip: Option<f32>,
    damping: Option<f32>,
    reset_momentum: bool,
    momentum: Option<(Vec<ResidentTensor>, Vec<Vec<f32>>)>,
}
async fn accepted(value: GraphUpdateReadback) -> Result<u64> {
    #[cfg(not(target_arch = "wasm32"))]
    let value = value.read()?;
    #[cfg(target_arch = "wasm32")]
    let value = value.read_async().await?;
    Ok(value)
}
fn dataset(seed: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    [6, 4, 2, 5, 3, 1, 6]
        .into_iter()
        .enumerate()
        .map(|(batch, valid)| {
            let input: Vec<_> = (0..24)
                .map(|i| ((i * 11 + batch * 13 + seed) % 37) as f32 / 18. - 1.)
                .collect();
            let labels = (0..6)
                .map(|row| {
                    if row >= valid {
                        -100.
                    } else {
                        (0..3)
                            .max_by(|&a, &b| input[row * 4 + a].total_cmp(&input[row * 4 + b]))
                            .unwrap() as f32
                    }
                })
                .collect();
            (input, labels)
        })
        .collect()
}

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    run_with_options(runtime, false, false).await
}

pub(super) async fn run_clipped(runtime: WgpuRuntime) -> Result<Value> {
    let mut result = run_with_options(runtime.clone(), true, false).await?;
    result["wide_probes"] = wide_probes(runtime).await?;
    Ok(result)
}

pub(super) async fn run_momentum(runtime: WgpuRuntime) -> Result<Value> {
    let mut result = run_with_options(runtime, true, true).await?;
    result["momentum_rule"] = json!(st_tensor::TOPOS_OPTIMIZER_MOMENTUM_RULE);
    Ok(result)
}

async fn run_with_options(runtime: WgpuRuntime, clipping: bool, momentum: bool) -> Result<Value> {
    let mut cases = Vec::new();
    for (seed, reduction) in [
        (17, LossReduction::Mean),
        (29, LossReduction::Sum),
        (43, LossReduction::Mean),
    ] {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            let shape = vec![2, 3, 4];
            let mut original = model(seed)?;
            let mut reference = model(seed)?;
            let baseline = InferencePlan::from_module(&original, NdLayout::contiguous(&shape)?)?;
            let definition = baseline.graph_definition()?;
            let roles: Vec<_> = definition.parameters().iter().map(|p| p.role).collect();
            let mut learner = baseline.compile_graph_learner_wgpu(runtime.clone(), policy)?;
            let device = learner.tensor_device().clone();
            let mut accumulator = learner.gradient_accumulator()?;
            let data = dataset(seed);
            let batches = data
                .iter()
                .map(|(x, y)| Ok((device.upload(&shape, x)?, device.upload(&[2, 3], y)?)))
                .collect::<Result<Vec<_>>>()?;
            let config = CrossEntropyConfig {
                reduction,
                label_smoothing: 0.1,
                ..Default::default()
            };
            let mut objective = CrossEntropyWithLogits::new(config)?;
            let mut cpu_objective = CrossEntropyWithLogits::new(config)?;
            let mut held = Vec::new();
            let mut submitted_micros = 0;
            let mut history: Vec<Vec<f32>> = definition
                .parameters()
                .iter()
                .map(|p| vec![0.; p.values.len()])
                .collect();
            for window in 0..32 {
                let damping = if momentum {
                    [Some(0.6), Some(0.85), Some(0.), None, Some(0.3)][window % 5]
                } else {
                    None
                };
                if let Some(d) = damping {
                    learner.set_momentum_damping(d)?;
                } else {
                    learner.clear_momentum();
                    for p in &mut history {
                        p.fill(0.);
                    }
                }
                let reset_momentum = momentum && damping.is_some() && window % 13 == 0;
                if reset_momentum {
                    learner.reset_momentum()?;
                    for p in &mut history {
                        p.fill(0.);
                    }
                }
                if learner.momentum_damping() != damping {
                    return Err("momentum setting".into());
                }
                let clip = if clipping {
                    [Some(0.05), None, Some(0.1), Some(2.), Some(0.001)][window % 5]
                } else {
                    None
                };
                if let Some(limit) = clip {
                    learner.set_grad_clip_max_norm(limit)?;
                } else {
                    learner.clear_grad_clip();
                }
                if learner.grad_clip_max_norm() != clip {
                    return Err("clip setting".into());
                }
                learner.zero_accumulator(&mut accumulator)?;
                if accumulator.parameter_generation() != window as u64 || !accumulator.is_empty() {
                    return Err("microbatch window generation".into());
                }
                let ids: Vec<_> = (0..2 + window % 3)
                    .map(|i| (window * 3 + i) % data.len())
                    .collect();
                let count = |id: usize| data[id].1.iter().filter(|&&v| v != -100.).count();
                let total: usize = ids.iter().map(|&id| count(id)).sum();
                let mut sum: Vec<_> = definition
                    .parameters()
                    .iter()
                    .map(|p| vec![0.; p.values.len()])
                    .collect();
                let mut micros = Vec::new();
                for id in ids {
                    let weight = if reduction == LossReduction::Mean {
                        count(id) as f32 / total as f32
                    } else {
                        1. / total as f32
                    };
                    learner.set_input_tensor(&batches[id].0)?;
                    let forward = learner.forward()?;
                    let pair = objective.evaluate_resident(forward.prediction(), &batches[id].1)?;
                    let gradients = learner.backward(&forward, pair.prediction_gradient())?;
                    learner.accumulate(&mut accumulator, &gradients, weight)?;
                    let expected = cpu_step_with_loss(
                        &mut reference,
                        &Tensor::from_vec(6, 4, data[id].0.clone())?,
                        &Tensor::from_vec(6, 1, data[id].1.clone())?,
                        0.,
                        policy,
                        &roles,
                        &mut cpu_objective,
                    )?;
                    for (out, raw) in sum.iter_mut().zip(&expected.raw) {
                        for (a, &b) in out.iter_mut().zip(raw) {
                            *a += weight * b;
                        }
                    }
                    micros.push(Micro {
                        id,
                        weight,
                        forward,
                        pair,
                        gradients,
                        expected,
                    });
                    submitted_micros += 1;
                }
                if accumulator.len() != micros.len() as u64 {
                    return Err("microbatch contribution count".into());
                }
                let sum_gpu = accumulator.parameter_gradients()?;
                let rate = if window % 11 == 0 { 0. } else { 0.1 };
                learner.sgd_accumulated(&accumulator, rate)?;
                let mut effective = sum.clone();
                for (values, &role) in effective.iter_mut().zip(&roles) {
                    if policy == GraphGradientPolicy::ModuleCompatible
                        && role == ParameterRole::Gain
                    {
                        for g in values {
                            *g *= 1. / 6.;
                        }
                    }
                }
                if let Some(limit) = clip {
                    let norm_sq = effective
                        .iter()
                        .flatten()
                        .map(|&g| f64::from(g).powi(2))
                        .sum();
                    let factors = st_kernel_contracts::gradient_clip::GlobalNormClip::new(limit)?
                        .factors(norm_sq)?;
                    for g in effective.iter_mut().flatten() {
                        for &factor in factors.as_slice() {
                            *g *= factor;
                        }
                    }
                }
                let mut expected_parameters = Vec::new();
                if let Some(damping) = damping {
                    // Exercise the ordinary Topos implementation, not a local EMA replica.
                    let control = st_tensor::ToposOptimizerStateControl::new(
                        0.,
                        [0.; st_tensor::TOPOS_OPTIMIZER_GRADIENT_BIAS_BASIS_DIM],
                        1.,
                        damping,
                    )?;
                    for (g, previous) in effective.iter_mut().zip(&mut history) {
                        *g = control.gradient_step(g, previous)?.into_next_momentum();
                        if rate != 0. {
                            previous.clone_from(g);
                        }
                    }
                }
                let mut id = 0;
                reference.visit_parameters_mut(&mut |p| {
                    if rate != 0. {
                        for (v, &g) in p.value_mut().data_mut().iter_mut().zip(&effective[id]) {
                            *v -= rate * g;
                        }
                    }
                    expected_parameters.push(p.value().data().to_vec());
                    id += 1;
                    Ok(())
                })?;
                held.push(Window {
                    micros,
                    sum: sum_gpu,
                    expected_sum: sum,
                    parameters: learner.parameter_snapshot()?,
                    expected_parameters,
                    receipt: learner.update_snapshot()?,
                    rate,
                    clip,
                    damping,
                    reset_momentum,
                    momentum: if damping.is_some() {
                        Some((learner.momentum_tensors()?, history.clone()))
                    } else {
                        None
                    },
                });
            }
            let mut evaluation = Vec::new();
            let mut mean_objective = CrossEntropyWithLogits::new(CrossEntropyConfig {
                label_smoothing: 0.1,
                ..Default::default()
            })?;
            for (x, y) in &batches {
                learner.set_input_tensor(x)?;
                let f = learner.forward()?;
                let pair = mean_objective.evaluate_resident(f.prediction(), y)?;
                evaluation.push((f, pair));
            }
            if (
                learner.input_generation(),
                learner.submitted_forwards(),
                learner.submitted_backwards(),
                learner.submitted_updates(),
            ) != (102, 102, 95, 32)
            {
                return Err("microbatch submission counts".into());
            }
            let updated = learner.parameter_snapshot()?;
            drop((learner, accumulator, device, batches));
            let mut windows = Vec::new();
            for (i, window) in held.into_iter().enumerate() {
                if accepted(window.receipt).await? != i as u64 + 1 {
                    return Err("microbatch update rejected".into());
                }
                let mut micros = Vec::new();
                for m in window.micros {
                    let prediction = tensor(m.forward.prediction().snapshot()?).await?;
                    let value = tensor(m.pair.value().snapshot()?).await?[0];
                    let dx = tensor(m.gradients.input_gradient().snapshot()?).await?;
                    let mut raw = Vec::new();
                    for g in m.gradients.parameter_gradients() {
                        raw.push(tensor(g.snapshot()?).await?);
                    }
                    close(&prediction, &m.expected.prediction)?;
                    close(&[value], &[m.expected.loss])?;
                    close(&dx, &m.expected.dx)?;
                    for (a, b) in raw.iter().zip(&m.expected.raw) {
                        close(a, b)?;
                    }
                    micros.push(json!({"batch":m.id,"weight":m.weight,"prediction":prediction,"loss":value,"input_gradient":dx,"raw_gradients":raw}));
                }
                let mut sum = Vec::new();
                for g in window.sum {
                    sum.push(tensor(g.snapshot()?).await?);
                }
                let values: Vec<_> = parameters(window.parameters)
                    .await?
                    .parameters()
                    .iter()
                    .map(|p| p.values.clone())
                    .collect();
                for (a, b) in sum.iter().zip(&window.expected_sum) {
                    close(a, b)?;
                }
                for (a, b) in values.iter().zip(&window.expected_parameters) {
                    close(a, b)?;
                }
                let observed_momentum = if let Some((snapshots, expected)) = window.momentum {
                    let mut observed = Vec::new();
                    for (t, expected) in snapshots.into_iter().zip(expected) {
                        let value = tensor(t.snapshot()?).await?;
                        close(&value, &expected)?;
                        observed.push(value);
                    }
                    Some(observed)
                } else {
                    None
                };
                windows.push(json!({"microbatches":micros,"accumulated_gradients":sum,"parameters":values,"rate":window.rate,"grad_clip_max_norm":window.clip,
                    "momentum_damping":window.damping,"reset_momentum":window.reset_momentum,"momentum":observed_momentum}));
            }
            let mut evaluated = Vec::new();
            let mut initial = 0.;
            let mut final_loss = 0.;
            let mut sample_count = 0;
            for (id, (f, pair)) in evaluation.into_iter().enumerate() {
                let prediction = tensor(f.prediction().snapshot()?).await?;
                let value = tensor(pair.value().snapshot()?).await?[0];
                let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
                    st_core::backend::device_caps::DeviceCaps::cpu(),
                ));
                let x = Tensor::from_vec(6, 4, data[id].0.clone())?;
                let y = Tensor::from_vec(6, 1, data[id].1.clone())?;
                let original_output = original.forward(&x)?;
                let initial_loss = mean_objective.forward(&original_output, &y)?.data()[0];
                let expected = reference.forward(&x)?;
                close(&prediction, expected.data())?;
                close(&[value], mean_objective.forward(&expected, &y)?.data())?;
                let count = data[id].1.iter().filter(|&&v| v != -100.).count();
                initial += initial_loss * count as f32;
                final_loss += value * count as f32;
                sample_count += count;
                evaluated.push(json!({"batch":id,"initial_loss":initial_loss,"prediction":prediction,"loss":value}));
            }
            let updated = InferencePlan::from_graph_definition(parameters(updated).await?)?;
            let applied = baseline.apply_parameters_to(
                &mut original,
                &updated,
                st_nn::resident::ModuleOptimizerStatePolicy::Reject,
            )?;
            let original_plan =
                InferencePlan::from_module(&original, NdLayout::contiguous(&shape)?)?;
            for (a, b) in original_plan
                .graph_definition()?
                .parameters()
                .iter()
                .zip(updated.graph_definition()?.parameters())
            {
                close(&a.values, &b.values)?;
            }
            cases.push(json!({"seed":seed,"input_shape":shape,"policy":format!("{policy:?}"),
                "plan":serde_json::from_str::<Value>(&baseline.to_json()?)?,"dataset":data.iter().map(|(x,y)|json!({"input":x,"target":y})).collect::<Vec<_>>(),
                "windows":windows,"evaluation":evaluated,"initial_loss":initial/sample_count as f32,"final_loss":final_loss/sample_count as f32,
                "label_smoothing":0.1,"ignore_index":-100,"reduction":reduction.as_str(),"observations_after_updates":32,
                "microbatches":submitted_micros,"module_parameters_applied":applied,"optimizer":"explicit_sgd_not_ModuleTrainer","gradient_clip":clipping,"topos_momentum":momentum}));
        }
    }
    Ok(json!({"status":"passed","cases":cases}))
}

async fn wide_probes(runtime: WgpuRuntime) -> Result<Value> {
    let mut probes = Vec::new();
    for limit in [1., 1e-20] {
        let mut model = Sequential::new();
        model.push(Scaler::new("wide", 513)?);
        model.visit_parameters_mut(&mut |p| {
            p.value_mut().data_mut().fill(0.);
            Ok(())
        })?;
        let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[1, 1, 513])?)?;
        let mut learner =
            plan.compile_graph_learner_wgpu(runtime.clone(), GraphGradientPolicy::Exact)?;
        learner.set_grad_clip_max_norm(limit)?;
        learner.upload(&[1.; 513])?;
        let f = learner.forward()?;
        let gradient = learner.backward(
            &f,
            &learner.tensor_device().upload(&[1, 1, 513], &[1e38; 513])?,
        )?;
        learner.sgd(&gradient, 1.)?;
        accepted(learner.update_snapshot()?).await?;
        let result = parameters(learner.parameter_snapshot()?).await?;
        let values = &result.parameters()[0].values;
        let expected = -f64::from(limit) / 513f64.sqrt();
        for &value in values {
            if !value.is_finite() || (f64::from(value) / expected - 1.).abs() > 1e-5 {
                return Err("wide gradient clipping".into());
            }
        }
        probes.push(json!({"limit":limit,"magnitude":1e38f32,"columns":513,"parameters":values}));
    }
    Ok(json!(probes))
}
