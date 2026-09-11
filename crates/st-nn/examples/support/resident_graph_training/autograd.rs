//! Loss-independent VJPs, identical on native Metal and browser WebGPU.
use super::*;
use st_backend_wgpu::resident_training::graph::GraphGradients;

async fn gradients(g: &GraphGradients) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let dx = tensor(g.input_gradient().snapshot()?).await?;
    let mut parameters = Vec::new();
    for p in g.parameter_gradients() {
        parameters.push(tensor(p.snapshot()?).await?);
    }
    Ok((dx, parameters))
}

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    for (seed, shape) in [(17, vec![4]), (29, vec![3, 4]), (43, vec![2, 129, 4])] {
        for fused in [false, true] {
            let mut reference = model(seed)?;
            let layout = NdLayout::contiguous(&shape)?;
            let rows = layout.len() / 4;
            let x = Tensor::from_fn(rows, 4, |r, c| ((r * 7 + c * 11) % 29) as f32 / 16. - 0.8)?;
            let mut plan = InferencePlan::from_module(&reference, layout)?;
            if fused {
                plan = plan.fuse_pointwise()?;
            }
            let mut gpu = plan.compile_graph_autograd_wgpu(runtime.clone())?;
            let device = gpu.tensor_device().clone();
            gpu.set_input_tensor(&device.upload(&shape, x.data())?)?;
            let forward = gpu.forward()?;
            let prediction = tensor(forward.prediction().snapshot()?).await?;
            let mut replays = Vec::new();
            let mut retained = Vec::new();
            for trial in 0..4 {
                let shape = gpu.output_layout().shape();
                let len = gpu.output_layout().len();
                let values: Vec<_> = (0..len)
                    .map(|i| ((i * 7 + trial * 3) % 13) as f32 / 8. - 0.75)
                    .collect();
                let cotangent = match trial {
                    0 => device.upload(shape, &values)?,
                    1 => {
                        if shape.len() == 1 {
                            let packed: Vec<_> =
                                std::iter::once(99.).chain(values.iter().copied()).collect();
                            device.upload(&[len + 1], &packed)?.narrow(0, 1, len)?
                        } else {
                            let reversed: Vec<_> = shape.iter().copied().rev().collect();
                            let axes: Vec<_> = (0..shape.len()).rev().collect();
                            device.upload(&reversed, &values)?.permute(&axes)?
                        }
                    }
                    2 => device.upload(&[], &[0.])?.broadcast_to(shape)?,
                    _ => forward.prediction().mul(&device.upload(&[], &[0.5])?)?,
                };
                let actual_seed = tensor(cotangent.snapshot()?).await?;
                let result = gpu.backward(&forward, &cotangent)?;
                let (dx, raw) = gradients(&result).await?;
                let mut maximum = 0f32;
                {
                    let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
                        st_core::backend::device_caps::DeviceCaps::cpu(),
                    ));
                    reference.zero_accumulators()?;
                    maximum = maximum.max(close(&prediction, reference.forward(&x)?.data())?);
                    maximum = maximum.max(close(
                        &dx,
                        reference
                            .backward(&x, &Tensor::from_vec(rows, 3, actual_seed.clone())?)?
                            .data(),
                    )?);
                    let definition = plan.graph_definition()?;
                    let mut slot = 0;
                    let mut expected_gradients = Vec::new();
                    reference.visit_parameters(&mut |p| {
                        let factor = if definition.parameters()[slot].role == ParameterRole::Gain {
                            rows as f32
                        } else {
                            1.
                        };
                        let expected: Vec<_> = p
                            .gradient()
                            .unwrap()
                            .data()
                            .iter()
                            .map(|v| v * factor)
                            .collect();
                        // Preserve the Module's legacy gain normalization only in the reference conversion.
                        expected_gradients.push(expected);
                        slot += 1;
                        Ok(())
                    })?;
                    for (actual, expected) in raw.iter().zip(&expected_gradients) {
                        maximum = maximum.max(close(actual, expected)?);
                    }
                }
                if (
                    result.input_generation(),
                    result.submitted_forward(),
                    result.submitted_backward(),
                ) != (1, 1, trial as u64 + 1)
                {
                    return Err("incorrect VJP identity/counter".into());
                }
                replays.push(json!({"cotangent":actual_seed,"input_gradient":dx,"raw_gradients":raw,"max_abs_error":maximum}));
                retained.push(result);
            }
            // Reusing the tape never updates parameters or mutates prior outputs.
            let fresh = gpu.forward()?;
            close(&prediction, &tensor(fresh.prediction().snapshot()?).await?)?;
            if !matches!(
                gpu.backward(&forward, fresh.prediction()),
                Err(TrainingError::StaleForward)
            ) {
                return Err("stale forward accepted".into());
            }
            drop(gpu);
            for (result, expected) in retained.iter().zip(&replays) {
                let (dx, raw) = gradients(result).await?;
                if json!(dx) != expected["input_gradient"]
                    || json!(raw) != expected["raw_gradients"]
                {
                    return Err("VJP ownership lost on workspace reuse/drop".into());
                }
            }
            cases.push(json!({"seed":seed,"fused":fused,"input_shape":shape,
                "plan":serde_json::from_str::<Value>(&plan.to_json()?)?,"input":x.data(),
                "prediction":prediction,"replays":replays,"ownership":"passed"}));
        }
    }
    Ok(json!({"cases":cases,"guards":guards(runtime).await?}))
}

async fn guards(runtime: WgpuRuntime) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    for fused in [false, true] {
        for (name, gain, x, seed) in [
            ("masked_forward", f32::MAX, -2., 0.),
            ("backward_overflow", f32::MAX, 0., 2.),
            ("gain_unbroadcast", 0., f32::MAX / 2., 3.),
        ] {
            let mut model = Sequential::new();
            model.push(Scaler::from_gain(
                "gain",
                Tensor::from_vec(1, 1, vec![gain])?,
            )?);
            if name == "masked_forward" {
                model.push(Relu::new());
            }
            let mut plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 1])?)?;
            if fused {
                plan = plan.fuse_pointwise()?;
            }
            let mut gpu = plan.compile_graph_autograd_wgpu(runtime.clone())?;
            let device = gpu.tensor_device().clone();
            gpu.upload(&[x; 2])?;
            let forward = gpu.forward()?;
            let bad = gpu.backward(&forward, &device.upload(&[2, 1], &[seed; 2])?)?;
            let zero = device.upload(&[2, 1], &[0.; 2])?;
            let recovered = gpu.backward(&forward, &zero)?;
            if name == "masked_forward" {
                if gradients(&recovered).await.is_ok()
                    || tensor(forward.prediction().snapshot()?).await.is_ok()
                {
                    return Err("cotangent masked invalid forward".into());
                }
            } else {
                let (dx, raw) = gradients(&recovered).await?;
                close(&dx, &[0.; 2])?;
                close(&raw[0], &[0.])?;
                tensor(forward.prediction().snapshot()?).await?;
            }
            gpu.upload(&[0.; 2])?;
            let fresh = gpu.forward()?;
            gradients(&gpu.backward(&fresh, &zero)?).await?;
            drop(gpu);
            for tensor_value in
                std::iter::once(bad.input_gradient()).chain(bad.parameter_gradients())
            {
                if tensor(tensor_value.snapshot()?).await.is_ok() {
                    return Err("partial VJP escaped whole-backward guard".into());
                }
            }
            records.push(json!({"case":name,"fused":fused,"passed":true}));
        }
    }
    // Wrong workspace, malformed inputs/seeds, stale tokens and inherited guards.
    let plan = InferencePlan::from_module(
        &Scaler::from_gain("g", Tensor::from_vec(1, 1, vec![2.])?)?,
        NdLayout::contiguous(&[2, 1])?,
    )?;
    let mut gpu = plan.compile_graph_autograd_wgpu(runtime.clone())?;
    let mut other = plan.compile_graph_autograd_wgpu(runtime.clone())?;
    let device = gpu.tensor_device().clone();
    if gpu.forward().is_ok() {
        return Err("missing input accepted".into());
    }
    gpu.upload(&[1., 2.])?;
    other.upload(&[1., 2.])?;
    let forward = gpu.forward()?;
    let alien = other.forward()?;
    let seed = device.upload(&[2, 1], &[1., 1.])?;
    if gpu.backward(&alien, &seed).is_ok()
        || gpu.upload(&[]).is_ok()
        || gpu.upload(&[f32::NAN; 2]).is_ok()
        || gpu
            .set_input_tensor(&device.upload(&[1, 2], &[1., 1.])?)
            .is_ok()
        || gpu
            .backward(&forward, &device.upload(&[2], &[1., 1.])?)
            .is_ok()
        || (
            gpu.input_generation(),
            gpu.submitted_forwards(),
            gpu.submitted_backwards(),
        ) != (1, 1, 0)
    {
        return Err("host validation mutated tape/counters or accepted bad identity/shape".into());
    }
    let poisoned = device
        .upload(&[2, 1], &[f32::MAX; 2])?
        .mul(&device.upload(&[], &[2.])?)?
        .mul(&device.upload(&[], &[0.])?)?;
    let bad = gpu.backward(&forward, &poisoned)?;
    let good = gpu.backward(&forward, &seed)?;
    close(&gradients(&good).await?.0, &[2.; 2])?;
    if gradients(&bad).await.is_ok() {
        return Err("inherited cotangent failure lost".into());
    }
    gpu.set_input_tensor(&poisoned)?;
    if gpu.backward(&forward, &seed).is_ok() {
        return Err("input replacement kept old tape valid".into());
    }
    let invalid_forward = gpu.forward()?;
    if gradients(&gpu.backward(&invalid_forward, &seed)?)
        .await
        .is_ok()
    {
        return Err("inherited input guard lost".into());
    }
    gpu.upload(&[1., 2.])?;
    let new_forward = gpu.forward()?;
    close(
        &gradients(&gpu.backward(&new_forward, &seed)?).await?.0,
        &[2.; 2],
    )?;
    records.push(json!({"case":"identity_atomicity_inheritance_recovery","passed":true}));
    let private =
        TensorDevice::new(WgpuRuntime::request_headless("graph.autograd.device_guard").await?)?;
    let alien_input = private.upload(&[2, 1], &[1.; 2])?;
    let before = (
        gpu.input_generation(),
        gpu.submitted_forwards(),
        gpu.submitted_backwards(),
    );
    if gpu.set_input_tensor(&alien_input).is_ok()
        || gpu.backward(&new_forward, &alien_input).is_ok()
        || before
            != (
                gpu.input_generation(),
                gpu.submitted_forwards(),
                gpu.submitted_backwards(),
            )
    {
        return Err("cross-context tensor admitted or mutated the tape".into());
    }
    gradients(&gpu.backward(&new_forward, &seed)?).await?;
    records.push(json!({"case":"device_context_atomicity","passed":true}));
    // Dense guards share the same tape restoration, not just the pointwise mask.
    for x in [-2., 0.] {
        let mut linear = Linear::new("dense", 1, 1)?;
        linear.visit_parameters_mut(&mut |p| {
            let value = if p.name().ends_with("weight") {
                f32::MAX
            } else {
                0.
            };
            p.value_mut().data_mut().fill(value);
            Ok(())
        })?;
        let mut model = Sequential::new();
        model.push(linear);
        let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 1])?)?;
        let mut dense = plan.compile_graph_autograd_wgpu(runtime.clone())?;
        dense.upload(&[x; 2])?;
        let tape = dense.forward()?;
        let invalid = dense.backward(&tape, &device.upload(&[2, 1], &[2.; 2])?)?;
        let recovery = dense.backward(&tape, &device.upload(&[2, 1], &[0.; 2])?)?;
        if (x == 0.) != gradients(&recovery).await.is_ok() {
            return Err("dense forward/backward guards restored incorrectly".into());
        }
        for tensor_value in
            std::iter::once(invalid.input_gradient()).chain(invalid.parameter_gradients())
        {
            if tensor(tensor_value.snapshot()?).await.is_ok() {
                return Err("dense partial VJP escaped".into());
            }
        }
        records.push(json!({"case":"dense_guards","invalid_forward":x!=0.,"passed":true}));
    }
    // No parameters, repeated rhs=0 residuals: y=(2*x)*x, dy/dx=4*x.
    let residual = InferencePlan::from_graph_definition(GraphDefinition::new(
        NdLayout::contiguous(&[2, 2])?,
        vec![GraphStage::Pointwise {
            chain: PointwiseChain::new(
                1,
                vec![
                    PointwiseStep {
                        op: ElementwiseOp::Add,
                        rhs: Some(0),
                    },
                    PointwiseStep {
                        op: ElementwiseOp::Multiply,
                        rhs: Some(0),
                    },
                ],
            )?,
            parameters: vec![],
        }],
        vec![],
    )?)?;
    let mut residual = residual.compile_graph_autograd_wgpu(runtime)?;
    residual.upload(&[-2., -1., 1., 2.])?;
    let tape = residual.forward()?;
    let g = residual.backward(&tape, &device.upload(&[2, 2], &[0.5, -1., 2., -0.5])?)?;
    let (dx, p) = gradients(&g).await?;
    close(&dx, &[-4., 4., 8., -4.])?;
    close(
        &tensor(tape.prediction().snapshot()?).await?,
        &[8., 2., 2., 8.],
    )?;
    if !p.is_empty() {
        return Err("parameterless residual acquired parameter slots".into());
    }
    records.push(json!({"case":"parameterless_repeated_residual","passed":true}));
    Ok(records)
}
