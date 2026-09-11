//! Real custom-objective updates: weighted quadratic/quartic VJPs, no host seeds.
use super::*;
use st_backend_wgpu::resident_training::graph::{
    GraphForward, GraphGradients, GraphUpdateReadback, ResidentGraphLearner,
};

async fn accepted(value: GraphUpdateReadback) -> Result<u64> {
    #[cfg(not(target_arch = "wasm32"))]
    let result = value.read();
    #[cfg(target_arch = "wasm32")]
    let result = value.read_async().await;
    Ok(result?)
}

fn objective(prediction: &[f32], target: &[f32]) -> f32 {
    prediction
        .iter()
        .zip(target)
        .map(|(p, t)| {
            let e = p - t;
            (0.75 * 0.5 * e * e + 0.25 * 0.25 * e * e * e * e) / target.len() as f32
        })
        .sum()
}

fn cpu_step(
    reference: &mut Sequential,
    x: &Tensor,
    target: &[f32],
    policy: GraphGradientPolicy,
) -> Result<Value> {
    let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
        st_core::backend::device_caps::DeviceCaps::cpu(),
    ));
    let prediction = reference.forward(x)?.data().to_vec();
    let e: Vec<_> = prediction.iter().zip(target).map(|(p, t)| p - t).collect();
    let plan = InferencePlan::from_module(reference, NdLayout::contiguous(&[x.shape().0, 4])?)?;
    let definition = plan.graph_definition()?;
    let mut all = Vec::new();
    for degree in [1, 3] {
        reference.zero_accumulators()?;
        let seed: Vec<_> = e
            .iter()
            .map(|v| (if degree == 1 { *v } else { v * v * v }) / e.len() as f32)
            .collect();
        let dx = reference.backward(x, &Tensor::from_vec(x.shape().0, 3, seed)?)?;
        let mut raw = Vec::new();
        let mut slot = 0;
        reference.visit_parameters(&mut |p| {
            let scale = if definition.parameters()[slot].role == ParameterRole::Gain {
                x.shape().0 as f32
            } else {
                1.
            };
            raw.push(
                p.gradient()
                    .unwrap()
                    .data()
                    .iter()
                    .map(|v| v * scale)
                    .collect::<Vec<_>>(),
            );
            slot += 1;
            Ok(())
        })?;
        all.push((dx.data().to_vec(), raw));
    }
    let mut combined = Vec::new();
    let mut values = Vec::new();
    let mut slot = 0;
    reference.visit_parameters_mut(&mut |p| {
        let scale = if policy == GraphGradientPolicy::ModuleCompatible
            && definition.parameters()[slot].role == ParameterRole::Gain
        {
            1. / x.shape().0 as f32
        } else {
            1.
        };
        let grad: Vec<_> = all[0].1[slot]
            .iter()
            .zip(&all[1].1[slot])
            .map(|(a, b)| 0.75 * a + 0.25 * b)
            .collect();
        for (value, gradient) in p.value_mut().data_mut().iter_mut().zip(&grad) {
            *value -= 0.1 * (gradient * scale);
        }
        combined.push(grad);
        values.push(p.value().data().to_vec());
        slot += 1;
        Ok(())
    })?;
    Ok(
        json!({"prediction":prediction,"loss":objective(&prediction,target),"parameters":values,
        "input_gradients":[all[0].0,all[1].0],"raw_gradients":[all[0].1,all[1].1],"combined":combined}),
    )
}

struct Capture {
    step: usize,
    forward: GraphForward,
    gradients: [GraphGradients; 2],
    parameters: GraphParameterReadback,
}

fn seeds(
    gpu: &mut ResidentGraphLearner,
    target: &st_backend_wgpu::resident_tensor::ResidentTensor,
    norm: &st_backend_wgpu::resident_tensor::ResidentTensor,
) -> Result<(GraphForward, [GraphGradients; 2])> {
    let forward = gpu.forward()?;
    let error = forward.prediction().add(target)?;
    let a = error.mul(norm)?;
    let b = error.mul(&error)?.mul(&error)?.mul(norm)?;
    let first = gpu.backward(&forward, &a)?;
    let second = gpu.backward(&forward, &b)?;
    Ok((forward, [first, second]))
}

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    for (seed, shape) in [(17, vec![4]), (29, vec![3, 4]), (43, vec![2, 129, 4])] {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            for fused in [false, true] {
                let mut reference = model(seed)?;
                let rows = shape.iter().product::<usize>() / 4;
                let x =
                    Tensor::from_fn(rows, 4, |r, c| ((r * 7 + c * 11) % 29) as f32 / 16. - 0.8)?;
                let target: Vec<_> = (0..rows * 3)
                    .map(|i| {
                        0.4 * x.data()[i / 3 * 4 + i % 3] - 0.2 * x.data()[i / 3 * 4 + 3]
                            + (i % 3) as f32 / 10.
                    })
                    .collect();
                let mut plan =
                    InferencePlan::from_module(&reference, NdLayout::contiguous(&shape)?)?;
                if fused {
                    plan = plan.fuse_pointwise()?;
                }
                let mut gpu = plan.compile_graph_learner_wgpu(runtime.clone(), policy)?;
                let device = gpu.tensor_device().clone();
                let neg = device.upload(
                    gpu.output_layout().shape(),
                    &target.iter().map(|v| -v).collect::<Vec<_>>(),
                )?;
                let norm = device.upload(&[], &[1. / target.len() as f32])?;
                gpu.upload(x.data())?;
                let mut receipts = Vec::new();
                let mut captures = Vec::new();
                for step in 0..64 {
                    let (forward, gradients) = seeds(&mut gpu, &neg, &norm)?;
                    gpu.sgd_weighted(&[(&gradients[0], 0.75), (&gradients[1], 0.25)], 0.1)?;
                    receipts.push(gpu.update_snapshot()?);
                    if [0, 1, 7, 31, 63].contains(&step) {
                        captures.push(Capture {
                            step,
                            forward,
                            gradients,
                            parameters: gpu.parameter_snapshot()?,
                        });
                    }
                }
                let final_forward = gpu.forward()?;
                if gpu.submitted_updates() != 64 {
                    return Err("missing learner updates".into());
                }
                // No mapped readback or host loss/seed construction during these updates.
                for (i, receipt) in receipts.into_iter().enumerate() {
                    if accepted(receipt).await? != i as u64 + 1 {
                        return Err("receipt counter drift".into());
                    }
                }
                let mut expected = Vec::new();
                for step in 0..64 {
                    let v = cpu_step(&mut reference, &x, &target, policy)?;
                    if [0, 1, 7, 31, 63].contains(&step) {
                        expected.push(v);
                    }
                }
                let mut records = Vec::new();
                let mut maximum = 0f32;
                for (c, expected) in captures.into_iter().zip(expected) {
                    let prediction = tensor(c.forward.prediction().snapshot()?).await?;
                    let parameters = parameters(c.parameters)
                        .await?
                        .parameters()
                        .iter()
                        .map(|p| p.values.clone())
                        .collect::<Vec<_>>();
                    let mut raw = Vec::new();
                    let mut dx = Vec::new();
                    for g in c.gradients {
                        dx.push(tensor(g.input_gradient().snapshot()?).await?);
                        let mut values = Vec::new();
                        for p in g.parameter_gradients() {
                            values.push(tensor(p.snapshot()?).await?);
                        }
                        raw.push(values);
                    }
                    let actual = json!({"step":c.step,"prediction":prediction,"parameters":parameters,"input_gradients":dx,"raw_gradients":raw});
                    fn compare(a: &Value, b: &Value) -> Result<f32> {
                        if let (Some(a), Some(b)) = (a.as_array(), b.as_array()) {
                            if a.len() != b.len() {
                                return Err("learning shape differs".into());
                            }
                            let mut max = 0f32;
                            for (a, b) in a.iter().zip(b) {
                                max = max.max(compare(a, b)?);
                            }
                            Ok(max)
                        } else {
                            close(
                                &[a.as_f64().ok_or("numeric fixture")? as f32],
                                &[b.as_f64().ok_or("numeric reference")? as f32],
                            )
                        }
                    }
                    for field in [
                        "prediction",
                        "parameters",
                        "input_gradients",
                        "raw_gradients",
                    ] {
                        maximum = maximum.max(compare(&actual[field], &expected[field])?);
                    }
                    records.push(actual);
                }
                let final_prediction = tensor(final_forward.prediction().snapshot()?).await?;
                let first: Vec<f32> = serde_json::from_value(records[0]["prediction"].clone())?;
                let initial_loss = objective(&first, &target);
                let final_loss = objective(&final_prediction, &target);
                if !final_loss.is_finite() || final_loss >= initial_loss {
                    return Err("custom objective did not improve".into());
                }
                let exported = InferencePlan::from_graph_definition(
                    parameters(gpu.parameter_snapshot()?).await?,
                )?;
                let restored = InferencePlan::from_json(&exported.to_json()?)?;
                let mut resumed = restored.compile_graph_learner_wgpu(runtime.clone(), policy)?;
                resumed.upload(x.data())?;
                for learner in [&mut gpu, &mut resumed] {
                    let (_, g) = seeds(learner, &neg, &norm)?;
                    learner.sgd_weighted(&[(&g[0], 0.75), (&g[1], 0.25)], 0.1)?;
                    accepted(learner.update_snapshot()?).await?;
                }
                let continued = parameters(gpu.parameter_snapshot()?).await?;
                let resumed_parameters = parameters(resumed.parameter_snapshot()?).await?;
                if bits(&continued) != bits(&resumed_parameters) {
                    return Err("weight-only learner resume drifted".into());
                }
                cases.push(json!({"seed":seed,"shape":shape,"fused":fused,"policy":policy.as_str(),
                    "plan":serde_json::from_str::<Value>(&plan.to_json()?)?,"input":x.data(),"target":target,
                    "steps":64,"rate":0.1,"coefficients":[0.75,0.25],"captures":records,
                    "initial_loss":initial_loss,"final_loss":final_loss,"final_prediction":final_prediction,
                    "max_abs_error":maximum,"accepted_updates":64,"resume":"bit_identical"}));
            }
        }
    }
    Ok(json!({"cases":cases,"guards":guards(runtime).await?}))
}

async fn guards(runtime: WgpuRuntime) -> Result<Vec<Value>> {
    let mut model = Sequential::new();
    let mut linear = Linear::new("linear", 1, 1)?;
    linear.visit_parameters_mut(&mut |p| {
        let value = if p.name().ends_with("weight") { 1. } else { 0. };
        p.value_mut().data_mut().fill(value);
        Ok(())
    })?;
    model.push(linear);
    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[1, 1])?)?;
    let original = bits(&plan.graph_definition()?);
    let mut records = Vec::new();
    for (name, seed_value, weights, rate) in [
        ("late_bias_candidate", 2., vec![1.], f32::MAX),
        (
            "weighted_product_cancellation",
            2.,
            vec![f32::MAX, -f32::MAX],
            0.1,
        ),
        (
            "weighted_sum_cancellation",
            f32::MAX,
            vec![1., 1., -1.],
            0.1,
        ),
        (
            "chunk_product_cancellation",
            2.,
            vec![f32::MAX, 0., 0., 0., -f32::MAX],
            0.1,
        ),
        (
            "chunk_sum_cancellation",
            f32::MAX,
            vec![0.5, 0.5, 0., 0., 0.5, -0.5],
            0.1,
        ),
    ] {
        let mut gpu =
            plan.compile_graph_learner_wgpu(runtime.clone(), GraphGradientPolicy::Exact)?;
        gpu.upload(&[0.25])?;
        let f = gpu.forward()?;
        let device = gpu.tensor_device().clone();
        let g = gpu.backward(&f, &device.upload(&[1, 1], &[seed_value])?)?;
        gpu.sgd_weighted(&weights.iter().map(|w| (&g, *w)).collect::<Vec<_>>(), rate)?;
        let receipt = gpu.update_snapshot()?;
        let checkpoint = gpu.parameter_snapshot()?;
        if gpu.sgd(&g, 0.1).is_ok() || gpu.backward(&f, &device.upload(&[1, 1], &[0.])?).is_ok() {
            return Err("updated parameter state retained old tape".into());
        }
        let fresh = gpu.forward()?;
        let zero = gpu.backward(&fresh, &device.upload(&[1, 1], &[0.])?)?;
        gpu.sgd(&zero, 0.1)?;
        let success = gpu.update_snapshot()?;
        drop(gpu);
        if accepted(receipt).await.is_ok() || bits(&parameters(checkpoint).await?) != original {
            return Err("weighted update partially committed or masked failure".into());
        }
        accepted(success).await?;
        records.push(json!({"case":name,"passed":true}));
    }
    let mut gpu = plan.compile_graph_learner_wgpu(runtime.clone(), GraphGradientPolicy::Exact)?;
    let mut other = plan.compile_graph_learner_wgpu(runtime.clone(), GraphGradientPolicy::Exact)?;
    gpu.upload(&[0.25])?;
    other.upload(&[0.25])?;
    let device = gpu.tensor_device().clone();
    let seed = device.upload(&[1, 1], &[1.])?;
    let f = gpu.forward()?;
    let g = gpu.backward(&f, &seed)?;
    let foreign = other.forward()?;
    let alien = other.backward(&foreign, &seed)?;
    if gpu.update_snapshot().is_ok()
        || gpu.sgd(&alien, 0.1).is_ok()
        || gpu.sgd(&g, f32::NAN).is_ok()
        || gpu.sgd_weighted(&[], 0.1).is_ok()
        || gpu.sgd_weighted(&[(&g, f32::INFINITY)], 0.1).is_ok()
        || gpu.sgd_weighted(&vec![(&g, 1.); 257], 0.1).is_ok()
        || gpu.submitted_updates() != 0
    {
        return Err("invalid update mutated state".into());
    }
    let poison = seed
        .mul(&device.upload(&[], &[f32::MAX])?)?
        .mul(&device.upload(&[], &[2.])?)?;
    let bad = gpu.backward(&f, &poison)?;
    gpu.sgd_weighted(&[(&bad, 0.), (&g, 1.)], 0.)?;
    let rejected = gpu.update_snapshot()?;
    if accepted(rejected).await.is_ok()
        || bits(&parameters(gpu.parameter_snapshot()?).await?) != original
    {
        return Err("zero weights/rate erased a bad VJP".into());
    }
    let f = gpu.forward()?;
    let g = gpu.backward(&f, &seed)?;
    gpu.sgd(&g, 0.1)?;
    let before = gpu.update_snapshot()?;
    gpu.upload(&[0.])?;
    gpu.forward()?;
    if (
        before.input_generation(),
        before.submitted_forward(),
        accepted(before).await?,
    ) != (1, 2, 2)
    {
        return Err("update receipt drifted across input reuse".into());
    }
    records.push(json!({"case":"identity_validation_zero_guard_receipt","passed":true}));
    // Parameterless graphs still validate every source, including zero-weight ones.
    let plan = InferencePlan::from_module(&Relu::new(), NdLayout::contiguous(&[1, 1])?)?;
    let mut gpu = plan.compile_graph_learner_wgpu(runtime, GraphGradientPolicy::Exact)?;
    gpu.upload(&[1.])?;
    let f = gpu.forward()?;
    let g = gpu.backward(&f, &poison)?;
    let good = gpu.backward(&f, &seed)?;
    let mut terms = vec![(&good, 0.); 255];
    terms.push((&g, 0.));
    gpu.sgd_weighted(&terms, 0.1)?;
    if accepted(gpu.update_snapshot()?).await.is_ok() {
        return Err("parameterless update lost guard".into());
    }
    records.push(json!({"case":"parameterless_invalid_source","passed":true}));
    let runtime = gpu.tensor_device().runtime().clone();
    records.push(composition_reuse(runtime).await?);
    Ok(records)
}

async fn composition_reuse(runtime: WgpuRuntime) -> Result<Value> {
    let shape = [2, 3, 257];
    let plan = InferencePlan::from_module(
        &Scaler::new("composition_gain", 257)?,
        NdLayout::contiguous(&shape)?,
    )?;
    let mut gpu = plan.compile_graph_learner_wgpu(runtime, GraphGradientPolicy::Exact)?;
    let d = gpu.tensor_device().clone();
    let input: Vec<_> = (0..1542).map(|i| ((i % 7) as f32 - 3.) * 0.125).collect();
    gpu.upload(&input)?;
    let seeds = [
        d.upload(&shape, &vec![0.5; 1542])?,
        d.upload(&shape, &vec![-0.25; 1542])?,
    ];
    let raw: Vec<Vec<f32>> = [0.5, -0.25]
        .iter()
        .map(|seed| {
            (0..257)
                .map(|c| (0..6).map(|r| input[r * 257 + c] * seed).sum())
                .collect()
        })
        .collect();
    let mut expected = plan.graph_definition()?.parameters()[0].values.clone();
    let counts = [1, 2, 3, 4, 5, 17, 256, 1, 5];
    let mut captures = Vec::new();
    for (step, count) in counts.into_iter().enumerate() {
        let f = gpu.forward()?;
        let g = [gpu.backward(&f, &seeds[0])?, gpu.backward(&f, &seeds[1])?];
        let terms: Vec<_> = (0..count)
            .map(|i| (&g[i % 2], [1., 0.125, -0.5, 0.][(i + step) % 4]))
            .collect();
        for (c, value) in expected.iter_mut().enumerate() {
            let mut gradient = raw[0][c] * terms[0].1;
            for (i, (_, weight)) in terms.iter().enumerate().skip(1) {
                gradient += raw[i % 2][c] * weight;
            }
            *value -= 0.03125 * gradient;
        }
        gpu.sgd_weighted(&terms, 0.03125)?;
        captures.push((
            gpu.update_snapshot()?,
            gpu.parameter_snapshot()?,
            expected.clone(),
        ));
    }
    drop(gpu);
    let mut maximum = 0f32;
    for (i, (receipt, snapshot, reference)) in captures.into_iter().enumerate() {
        if accepted(receipt).await? != i as u64 + 1 {
            return Err("composition receipt reuse".into());
        }
        maximum = maximum.max(close(
            &parameters(snapshot).await?.parameters()[0].values,
            &reference,
        )?);
    }
    Ok(
        json!({"case":"composition_reuse","passed":true,"terms":counts,"width":257,"max_abs_error":maximum}),
    )
}
