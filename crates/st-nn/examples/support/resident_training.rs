//! Shared native/browser fixtures. Model semantics live in st-nn, not JavaScript.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_dense::{DenseActivation, DenseLayer, DenseReadback},
    resident_matmul::{MatmulAccumulation, MatmulKernel},
    resident_training::{
        ParameterReadback, StepReadback, TrainingError, TrainingState, TrainingStateReadback,
    },
    runtime::WgpuRuntime,
};
use st_nn::{
    layers::Gelu, loss::Loss, module::Module, resident::InferencePlan, Linear, MeanSquaredError,
    Sequential,
};
use st_tensor::{NdLayout, Tensor};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

macro_rules! readback {
    ($name:ident, $input:ty, $output:ty) => {
        async fn $name(value: $input) -> Result<$output> {
            #[cfg(target_arch = "wasm32")]
            let result = value.read_async().await;
            #[cfg(not(target_arch = "wasm32"))]
            let result = value.read();
            Ok(result?)
        }
    };
}
readback!(state, TrainingStateReadback, TrainingState);
readback!(parameters, ParameterReadback, Vec<DenseLayer>);
readback!(loss, StepReadback, f32);
readback!(prediction, DenseReadback, Vec<f32>);

fn cpu_policy() -> st_nn::BackendPolicyGuard {
    st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
        st_core::backend::device_caps::DeviceCaps::cpu(),
    ))
}

fn model(seed: usize) -> Result<Sequential> {
    let mut model = Sequential::new();
    model.push(Linear::new("up", 4, 7)?);
    model.push(Gelu::new());
    model.push(Linear::new("down", 7, 3)?);
    let mut counter = seed;
    model.visit_parameters_mut(&mut |p| {
        for v in p.value_mut().data_mut() {
            *v = ((counter * 17 % 23) as f32 - 11.) / 32.;
            counter += 1;
        }
        Ok(())
    })?;
    Ok(model)
}

fn batch(rows: usize) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_fn(rows, 4, |r, c| ((r * 7 + c * 11) % 29) as f32 / 16. - 0.8)?;
    let y = Tensor::from_fn(rows, 3, |r, c| {
        0.4 * x.data()[r * 4 + c] - 0.2 * x.data()[r * 4 + 3] + c as f32 / 10.
    })?;
    Ok((x, y))
}

fn close(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("tensor lengths differ".into());
    }
    let mut maximum = 0f32;
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-5 + 1e-4 * b.abs() {
            return Err(format!("element {index}: GPU {a} != reference {b}").into());
        }
        maximum = maximum.max((a - b).abs());
    }
    Ok(maximum)
}

struct CpuState {
    loss: f32,
    prediction: Tensor,
    dx: Tensor,
    gradients: Vec<Tensor>,
    parameters: Vec<Tensor>,
}

fn cpu_step(model: &mut Sequential, x: &Tensor, y: &Tensor, rate: f32) -> Result<CpuState> {
    let _cpu = cpu_policy();
    model.zero_accumulators()?;
    let prediction = model.forward(x)?;
    let mut objective = MeanSquaredError::new();
    let loss = objective.forward(&prediction, y)?.data()[0];
    let seed = objective.backward(&prediction, y)?;
    let dx = model.backward(x, &seed)?;
    let mut gradients = Vec::new();
    model.visit_parameters(&mut |p| {
        gradients.push(p.gradient().expect("MSE gradient").clone());
        Ok(())
    })?;
    if rate > 0. {
        model.apply_step(rate)?;
    }
    let mut parameters = Vec::new();
    model.visit_parameters(&mut |p| {
        parameters.push(p.value().clone());
        Ok(())
    })?;
    Ok(CpuState {
        loss,
        prediction,
        dx,
        gradients,
        parameters,
    })
}

fn compare(actual: &TrainingState, expected: &CpuState) -> Result<f32> {
    if actual.parameters.len() * 2 != expected.parameters.len()
        || actual.parameter_gradients.len() * 2 != expected.gradients.len()
    {
        return Err("parameter count differs".into());
    }
    let mut error = close(&[actual.loss], &[expected.loss])?
        .max(close(&actual.prediction, expected.prediction.data())?)
        .max(close(&actual.input_gradient, expected.dx.data())?);
    for (i, (p, g)) in actual
        .parameters
        .iter()
        .zip(&actual.parameter_gradients)
        .enumerate()
    {
        for (a, b) in [
            (&p.weights, &expected.parameters[2 * i]),
            (&p.bias, &expected.parameters[2 * i + 1]),
            (&g.weights, &expected.gradients[2 * i]),
            (&g.bias, &expected.gradients[2 * i + 1]),
        ] {
            error = error.max(close(a, b.data())?);
        }
    }
    Ok(error)
}

fn layer_json(l: &DenseLayer) -> Value {
    json!({"inner":l.inner,"cols":l.cols,"weights":l.weights,"bias":l.bias,
        "gelu":l.activation == DenseActivation::Gelu})
}

fn state_json(s: &TrainingState) -> Value {
    json!({"loss":s.loss,"prediction":s.prediction,"input_gradient":s.input_gradient,
        "parameters":s.parameters.iter().map(layer_json).collect::<Vec<_>>(),
        "parameter_gradients":s.parameter_gradients.iter().map(|g|
            json!({"weights":g.weights,"bias":g.bias})).collect::<Vec<_>>(),
        "submitted_step":s.submitted_step,"batch_generation":s.batch_generation})
}

fn tensor_parameters(layers: &[DenseLayer]) -> Result<Vec<(Tensor, Tensor)>> {
    layers
        .iter()
        .map(|l| {
            Ok((
                Tensor::from_vec(l.inner, l.cols, l.weights.clone())?,
                Tensor::from_vec(1, l.cols, l.bias.clone())?,
            ))
        })
        .collect()
}

/// All kernel policies, two distinct cotangents, leading axes and partial MSE groups.
pub async fn check_vjps(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    let mut torch_cases = Vec::new();
    for (seed, shape) in [(17, vec![4]), (29, vec![3, 4]), (43, vec![2, 47, 4])] {
        for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
            for accumulation in [
                MatmulAccumulation::Sequential,
                MatmulAccumulation::Tiled,
                MatmulAccumulation::Compensated,
            ] {
                let mut model = model(seed)?;
                let input_layout = NdLayout::contiguous(&shape)?;
                let (x, y) = batch(input_layout.len() / 4)?;
                let plan = InferencePlan::from_module(&model, input_layout)?;
                let mut gpu = plan.compile_training_wgpu_with_options(
                    runtime.clone(),
                    Default::default(),
                    kernel,
                    accumulation,
                )?;
                let initial = parameters(gpu.parameter_snapshot()?).await?;
                gpu.upload_batch(x.data(), y.data())?;
                gpu.step(0.)?;
                let probe = gpu.state_snapshot()?;
                let reference_probe = cpu_step(&mut model, &x, &y, 0.)?;
                gpu.step(0.125)?;
                let updated = gpu.state_snapshot()?;
                let reference_updated = cpu_step(&mut model, &x, &y, 0.125)?;
                drop(gpu);
                let probe = state(probe).await?;
                let updated = state(updated).await?;
                if probe.submitted_step != 1
                    || updated.submitted_step != 2
                    || probe.batch_generation != 1
                    || updated.input_layout.shape() != shape
                    || updated.output_layout != *plan.output_layout()
                {
                    return Err("owned step metadata changed".into());
                }
                let error =
                    compare(&probe, &reference_probe)?.max(compare(&updated, &reference_updated)?);
                cases.push(
                    json!({"shape":shape,"seed":seed,"kernel":format!("{kernel:?}"),
                    "accumulation":format!("{accumulation:?}"),"max_abs_error":error}),
                );
                if kernel == MatmulKernel::Scalar && accumulation == MatmulAccumulation::Sequential
                {
                    torch_cases.push(json!({"shape":shape,"input":x.data(),"target":y.data(),
                        "initial_parameters":initial.iter().map(layer_json).collect::<Vec<_>>(),
                        "learning_rate":0.125,"probe":state_json(&probe),"updated":state_json(&updated)}));
                }
            }
        }
    }
    Ok(json!({"cases":cases,"torch_cases":torch_cases}))
}

fn scalar_plan(weights: &[f32], gelu: bool) -> Result<InferencePlan> {
    let mut model = Sequential::new();
    for i in 0..weights.len() {
        model.push(Linear::new(format!("layer_{i}"), 1, 1)?);
        if gelu {
            model.push(Gelu::new());
        }
    }
    let mut i = 0;
    model.visit_parameters_mut(&mut |p| {
        p.value_mut().data_mut()[0] = if i % 2 == 0 { weights[i / 2] } else { 0. };
        i += 1;
        Ok(())
    })?;
    Ok(InferencePlan::from_module(
        &model,
        NdLayout::contiguous(&[1])?,
    )?)
}

fn unchanged(a: &[DenseLayer], b: &[DenseLayer]) -> Result<()> {
    if a.len() != b.len() {
        return Err("parameter count changed".into());
    }
    for (a, b) in a.iter().zip(b) {
        for (&a, &b) in a
            .weights
            .iter()
            .chain(&a.bias)
            .zip(b.weights.iter().chain(&b.bias))
        {
            if a.to_bits() != b.to_bits() {
                return Err("rejected step mutated parameters".into());
            }
        }
    }
    Ok(())
}

pub async fn check_guards(runtime: WgpuRuntime) -> Result<Value> {
    let plan = scalar_plan(&[1., 0.001], false)?;
    let mut gpu = plan.compile_training_wgpu(runtime.clone())?;
    if !matches!(gpu.step(0.1), Err(TrainingError::MissingBatch))
        || !matches!(gpu.loss_snapshot(), Err(TrainingError::StaleStep))
        || !matches!(gpu.state_snapshot(), Err(TrainingError::StaleStep))
    {
        return Err("missing batch/step was accepted".into());
    }
    let before = parameters(gpu.parameter_snapshot()?).await?;
    gpu.upload_batch(&[0.5], &[-0.9995])?;
    let generation = gpu.batch_generation();
    for (x, y) in [
        (vec![], vec![0.]),
        (vec![0.], vec![]),
        (vec![f32::NAN], vec![0.]),
        (vec![0.], vec![f32::INFINITY]),
    ] {
        if gpu.upload_batch(&x, &y).is_ok() || gpu.batch_generation() != generation {
            return Err("invalid upload changed the batch".into());
        }
    }
    for rate in [-1., f32::NAN, f32::INFINITY] {
        if !matches!(gpu.step(rate), Err(TrainingError::LearningRate)) || gpu.submitted_steps() != 0
        {
            return Err("invalid rate changed the step counter".into());
        }
    }
    // Earlier-layer candidates are finite, but the final bias update overflows.
    // No layer may commit before the global decision, including those earlier layers.
    gpu.step(f32::MAX)?;
    let rejected = gpu.loss_snapshot()?;
    let after_rejection = gpu.parameter_snapshot()?;
    gpu.step(0.1)?;
    let recovered = gpu.loss_snapshot()?;
    drop(gpu);
    unchanged(&before, &parameters(after_rejection).await?)?;
    let error = loss(rejected).await.expect_err("overflow must reject");
    if !matches!(
        error.downcast_ref::<TrainingError>(),
        Some(TrainingError::Rejected { .. })
    ) {
        return Err(error);
    }
    loss(recovered).await?;

    let mut failures = Vec::new();
    for (label, weight, gelu, input, target) in [
        ("input_gradient_overflow", f32::MAX, false, 0., -1.),
        ("loss_square_overflow", 1., false, 0., f32::MAX),
        ("gelu_positive_overflow", 1., true, 1e20, 0.),
        ("gelu_negative_overflow", 1., true, -1e20, 0.),
        ("gelu_positive_square_cubic_overflow", 1., true, 1e13, 0.),
        ("gelu_negative_square_cubic_overflow", 1., true, -1e13, 0.),
    ] {
        let mut gpu = scalar_plan(&[weight], gelu)?.compile_training_wgpu(runtime.clone())?;
        let before = parameters(gpu.parameter_snapshot()?).await?;
        gpu.upload_batch(&[input], &[target])?;
        gpu.step(0.)?;
        let failed = gpu.state_snapshot()?;
        unchanged(&before, &parameters(gpu.parameter_snapshot()?).await?)?;
        let error = state(failed)
            .await
            .expect_err("non-finite intermediate must reject");
        match error.downcast_ref::<TrainingError>() {
            Some(TrainingError::Rejected { stage, flags }) => {
                failures.push(json!({"case":label,"stage":stage,"flags":flags}));
            }
            _ => return Err(error),
        }
    }

    // Positive/negative GELU saturation must keep a finite derivative, not inf * 0.
    let mut model = Sequential::new();
    model.push(Linear::new("up", 4, 7)?);
    model.push(Gelu::new());
    model.push(Linear::new("down", 7, 3)?);
    model.visit_parameters_mut(&mut |p| {
        let value = if p.name() == "up::bias" { 12. } else { 0.125 };
        p.value_mut().data_mut().fill(value);
        Ok(())
    })?;
    for bias in [12., -12., 8., -8.] {
        model.visit_parameters_mut(&mut |p| {
            if p.name() == "up::bias" {
                p.value_mut().data_mut().fill(bias);
            }
            Ok(())
        })?;
        let (x, y) = batch(3)?;
        let reference = cpu_step(&mut model, &x, &y, 0.)?;
        let mut gpu = InferencePlan::from_module(&model, NdLayout::contiguous(&[3, 4])?)?
            .compile_training_wgpu(runtime.clone())?;
        gpu.upload_batch(x.data(), y.data())?;
        gpu.step(0.)?;
        compare(&state(gpu.state_snapshot()?).await?, &reference)?;
    }
    Ok(
        json!({"transactional_rollback":"bitwise unchanged across all layers",
        "recovery":"valid step succeeds before rejected snapshot is read",
        "nonfinite_cases":failures,"saturated_gelu_cases":4,
        "host_rejection_cases":10}),
    )
}

pub async fn check_learning(runtime: WgpuRuntime) -> Result<Value> {
    let mut runs = Vec::new();
    for seed in [17, 29, 43] {
        let mut model = model(seed)?;
        let (x, y) = batch(32)?;
        let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 16, 4])?)?;
        let mut gpu = plan.compile_training_wgpu(runtime.clone())?;
        let initial_parameters = parameters(gpu.parameter_snapshot()?).await?;
        gpu.upload_batch(x.data(), y.data())?;
        gpu.step(0.)?;
        let initial = gpu.state_snapshot()?;
        let reference_initial = cpu_step(&mut model, &x, &y, 0.)?;
        for _ in 0..128 {
            gpu.step(0.2)?;
        }
        // No intermediate host readback or parameter upload in the learning loop.
        gpu.step(0.)?;
        let final_state = gpu.state_snapshot()?;
        for _ in 0..128 {
            cpu_step(&mut model, &x, &y, 0.2)?;
        }
        let reference_final = cpu_step(&mut model, &x, &y, 0.)?;
        drop(gpu);
        let initial = state(initial).await?;
        let final_state = state(final_state).await?;
        let error =
            compare(&initial, &reference_initial)?.max(compare(&final_state, &reference_final)?);
        if final_state.loss >= initial.loss * 0.5 {
            return Err(format!(
                "synthetic fit did not improve enough: {} -> {}",
                initial.loss, final_state.loss
            )
            .into());
        }
        let trained = plan.with_parameters(tensor_parameters(&final_state.parameters)?)?;
        let mut inference = trained.compile_wgpu(runtime.clone())?;
        inference.upload(x.data())?;
        inference.dispatch()?;
        close(
            &prediction(inference.snapshot()?).await?,
            &final_state.prediction,
        )?;
        let original: Vec<_> = plan.parameter_snapshots().collect();
        for (i, (w, b)) in original.iter().enumerate() {
            close(w.data(), &initial_parameters[i].weights)?;
            close(b.data(), &initial_parameters[i].bias)?;
        }
        runs.push(json!({"seed":seed,"shape":[2,16,4],"steps":128,"learning_rate":0.2,
            "input":x.data(),"target":y.data(),"initial_parameters":initial_parameters.iter().map(layer_json).collect::<Vec<_>>(),
            "initial":state_json(&initial),"final":state_json(&final_state),"max_abs_error":error,
            "intermediate_host_readbacks":0,"parameter_reuploads":0,"exported_inference":"passed"}));
    }
    // Mean loss makes repeating identical samples leave the update unchanged.
    let mut reference = None;
    for copies in [1, 2, 5] {
        let model = model(17)?;
        let (x, y) = batch(3)?;
        let mut gpu = InferencePlan::from_module(&model, NdLayout::contiguous(&[copies, 3, 4])?)?
            .compile_training_wgpu(runtime.clone())?;
        gpu.upload_batch(&x.data().repeat(copies), &y.data().repeat(copies))?;
        gpu.step(0.125)?;
        let actual = state(gpu.state_snapshot()?).await?;
        if let Some(expected) = &reference {
            let expected: &TrainingState = expected;
            close(&[actual.loss], &[expected.loss])?;
            for (a, b) in actual.parameters.iter().zip(&expected.parameters) {
                close(&a.weights, &b.weights)?;
                close(&a.bias, &b.bias)?;
            }
        } else {
            reference = Some(actual);
        }
    }
    Ok(json!({"runs":runs,"batch_duplication_factors":[1,2,5],
        "boundary":"bounded synthetic training correctness, not model quality or throughput"}))
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("a CPU adapter is not GPU evidence".into());
    }
    let vjps = check_vjps(runtime.clone())
        .await
        .map_err(|e| format!("VJP fixture: {e}"))?;
    let guards = check_guards(runtime.clone())
        .await
        .map_err(|e| format!("guard fixture: {e}"))?;
    let learning = check_learning(runtime.clone())
        .await
        .map_err(|e| format!("learning fixture: {e}"))?;
    Ok(
        json!({"schema":"spiraltorch.resident_training.fixture.v1","status":"passed",
        "adapter":{"name":runtime.adapter_info().name,"backend":format!("{:?}", runtime.adapter_info().backend)},
        "build_manifest":serde_json::from_str::<Value>(st_core::build_manifest_json())?,
        "vjps":vjps,"guards":guards,"learning":learning}),
    )
}
