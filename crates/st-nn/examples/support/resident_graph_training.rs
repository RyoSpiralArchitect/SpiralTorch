//! Identical native/browser graph fixture, checked against ordinary Rust Modules.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{TensorDevice, TensorReadback},
    resident_training::{
        graph::{GraphParameterReadback, GraphState, GraphStateReadback},
        StepReadback, TrainingError,
    },
    runtime::WgpuRuntime,
};
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    pointwise::{PointwiseChain, PointwiseStep},
};
use st_nn::{
    layers::{Gelu, Relu, Scaler},
    loss::Loss,
    module::Module,
    resident::{
        GraphDefinition, GraphGradientPolicy, GraphParameter, GraphStage, InferenceError,
        InferencePlan, ParameterRole,
    },
    Linear, MeanSquaredError, Sequential,
};
use st_tensor::{NdLayout, Tensor};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[path = "resident_graph_training/autograd.rs"]
mod autograd;
#[path = "resident_graph_training/fusion.rs"]
mod fusion;
#[path = "resident_graph_training/learning.rs"]
mod learning;
macro_rules! readback {
    ($name:ident,$input:ty,$output:ty) => {
        async fn $name(value: $input) -> Result<$output> {
            #[cfg(target_arch = "wasm32")]
            let result = value.read_async().await;
            #[cfg(not(target_arch = "wasm32"))]
            let result = value.read();
            Ok(result?)
        }
    };
}
readback!(state, GraphStateReadback, GraphState);
readback!(parameters, GraphParameterReadback, GraphDefinition);
readback!(loss, StepReadback, f32);
readback!(tensor, TensorReadback, Vec<f32>);

fn close(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("length mismatch".into());
    }
    let mut error = 0f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-5 + 2e-4 * b.abs() {
            return Err(format!("element {i}: resident {a} != reference {b}").into());
        }
        error = error.max((a - b).abs());
    }
    Ok(error)
}
fn bits(graph: &GraphDefinition) -> Vec<u32> {
    graph
        .parameters()
        .iter()
        .flat_map(|p| p.values.iter().map(|v| v.to_bits()))
        .collect()
}
fn model(seed: usize) -> Result<Sequential> {
    let mut model = Sequential::new();
    model.push(Scaler::new("input", 4)?);
    model.push(Linear::new("up", 4, 7)?);
    model.push(Gelu::new());
    model.push(Scaler::new("hidden", 7)?);
    model.push(Relu::new());
    model.push(Linear::new("down", 7, 3)?);
    model.push(Scaler::new("output", 3)?);
    let mut counter = seed;
    let roles = [
        ParameterRole::Gain,
        ParameterRole::Weight,
        ParameterRole::Bias,
        ParameterRole::Gain,
        ParameterRole::Weight,
        ParameterRole::Bias,
        ParameterRole::Gain,
    ];
    let mut slot = 0;
    model.visit_parameters_mut(&mut |p| {
        for v in p.value_mut().data_mut() {
            *v = if roles[slot] == ParameterRole::Gain {
                0.75 + (counter % 7) as f32 / 16.
            } else {
                ((counter * 17 % 23) as f32 - 11.) / 32.
            };
            counter += 1;
        }
        slot += 1;
        Ok(())
    })?;
    Ok(model)
}
struct Reference {
    loss: f32,
    prediction: Vec<f32>,
    dx: Vec<f32>,
    raw: Vec<Vec<f32>>,
    effective: Vec<Vec<f32>>,
    parameters: Vec<Vec<f32>>,
}
fn cpu_step(
    model: &mut Sequential,
    x: &Tensor,
    y: &Tensor,
    rate: f32,
    policy: GraphGradientPolicy,
    roles: &[ParameterRole],
) -> Result<Reference> {
    let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
        st_core::backend::device_caps::DeviceCaps::cpu(),
    ));
    model.zero_accumulators()?;
    let output = model.forward(x)?;
    let mut objective = MeanSquaredError::new();
    let loss = objective.forward(&output, y)?.data()[0];
    let seed = objective.backward(&output, y)?;
    let dx = model.backward(x, &seed)?;
    let mut raw = Vec::new();
    let mut effective = Vec::new();
    let mut slot = 0;
    model.visit_parameters(&mut |p| {
        let gradient = p.gradient().expect("fixture gradient").data();
        let scale = if roles[slot] == ParameterRole::Gain {
            x.shape().0 as f32
        } else {
            1.
        };
        raw.push(gradient.iter().map(|v| v * scale).collect::<Vec<_>>());
        effective.push(if policy == GraphGradientPolicy::Exact {
            raw.last().unwrap().clone()
        } else {
            gradient.to_vec()
        });
        slot += 1;
        Ok(())
    })?;
    let mut parameters = Vec::new();
    slot = 0;
    model.visit_parameters_mut(&mut |p| {
        if rate != 0. {
            for (value, gradient) in p.value_mut().data_mut().iter_mut().zip(&effective[slot]) {
                *value -= rate * gradient;
            }
        }
        parameters.push(p.value().data().to_vec());
        slot += 1;
        Ok(())
    })?;
    Ok(Reference {
        loss,
        prediction: output.data().to_vec(),
        dx: dx.data().to_vec(),
        raw,
        effective,
        parameters,
    })
}
fn compare(actual: &GraphState, expected: &Reference) -> Result<f32> {
    let mut error = close(&[actual.loss], &[expected.loss])?
        .max(close(&actual.prediction, &expected.prediction)?)
        .max(close(&actual.input_gradient, &expected.dx)?);
    for (i, p) in actual.graph.parameters().iter().enumerate() {
        error = error
            .max(close(&p.values, &expected.parameters[i])?)
            .max(close(&actual.raw_gradients[i], &expected.raw[i])?)
            .max(close(
                &actual.effective_gradients[i],
                &expected.effective[i],
            )?);
    }
    Ok(error)
}
fn state_json(s: &GraphState) -> Value {
    json!({"loss":s.loss,"prediction":s.prediction,"input_gradient":s.input_gradient,
        "parameters":s.graph.parameters().iter().map(|p|&p.values).collect::<Vec<_>>(),
        "raw_gradients":s.raw_gradients,"effective_gradients":s.effective_gradients,
        "submitted_step":s.submitted_step,"batch_generation":s.batch_generation})
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    for (seed, shape) in [(17, vec![4]), (29, vec![3, 4]), (43, vec![2, 129, 4])] {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            let mut model = model(seed)?;
            let layout = NdLayout::contiguous(&shape)?;
            let rows = layout.len() / 4;
            let x = Tensor::from_fn(rows, 4, |r, c| ((r * 7 + c * 11) % 29) as f32 / 16. - 0.8)?;
            let y = Tensor::from_fn(rows, 3, |r, c| {
                0.4 * x.data()[r * 4 + c] - 0.2 * x.data()[r * 4 + 3] + c as f32 / 10.
            })?;
            let plan = InferencePlan::from_module(&model, layout)?;
            let portable = plan.to_json()?;
            let restored = InferencePlan::from_json(&portable)?;
            if !matches!(
                restored.compile_training_wgpu(runtime.clone()),
                Err(InferenceError::RequiresGraph)
            ) {
                return Err("rich graph silently accepted by dense compiler".into());
            }
            let initial = plan.graph_definition()?;
            let roles = initial
                .parameters()
                .iter()
                .map(|p| p.role)
                .collect::<Vec<_>>();
            let mut gpu = restored.compile_graph_training_wgpu(runtime.clone(), policy)?;
            if !matches!(gpu.step(0.1), Err(TrainingError::MissingBatch)) {
                return Err("missing batch admitted".into());
            }
            gpu.upload_batch(x.data(), y.data())?;
            let rates = [0., 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125];
            let mut snapshots = Vec::new();
            let mut references = Vec::new();
            for rate in rates {
                gpu.step(rate)?;
                snapshots.push(gpu.state_snapshot()?);
                references.push(cpu_step(&mut model, &x, &y, rate, policy, &roles)?);
            }
            let frozen_prediction = gpu.prediction_tensor()?.snapshot()?;
            let exported = gpu.parameter_snapshot()?;
            // Validate-before-mutation on the queue, including generation and prior snapshot.
            if gpu.upload_batch(x.data(), &[]).is_ok() || gpu.batch_generation() != 1 {
                return Err("partial batch mutated graph".into());
            }
            if gpu.step(f32::NAN).is_ok() || gpu.submitted_steps() != 9 {
                return Err("invalid rate mutated step".into());
            }
            drop(gpu);
            let mut steps = Vec::new();
            let mut error = 0f32;
            for (snapshot, reference) in snapshots.into_iter().zip(&references) {
                let actual = state(snapshot).await?;
                if actual.gradient_policy != policy || actual.batch_generation != 1 {
                    return Err("snapshot metadata drift".into());
                }
                error = error.max(compare(&actual, reference)?);
                steps.push(state_json(&actual));
            }
            close(
                &tensor(frozen_prediction).await?,
                &references.last().unwrap().prediction,
            )?;
            let exported = parameters(exported).await?;
            for (i, p) in exported.parameters().iter().enumerate() {
                close(&p.values, &references.last().unwrap().parameters[i])?;
                if roles[i] == ParameterRole::Gain && p.values == initial.parameters()[i].values {
                    return Err("gain never trained".into());
                }
            }
            // Resume the exact exported graph through v2, not a separately defined model.
            let resumed = InferencePlan::from_graph_definition(exported)?;
            let resumed = InferencePlan::from_json(&resumed.to_json()?)?;
            let mut gpu = resumed.compile_graph_training_wgpu(runtime.clone(), policy)?;
            let device = TensorDevice::new(runtime.clone())?;
            gpu.upload_batch_tensors(
                &device.upload(&shape, x.data())?,
                &device.upload(plan.output_layout().shape(), y.data())?,
            )?;
            gpu.step(0.)?;
            error = error.max(compare(
                &state(gpu.state_snapshot()?).await?,
                &cpu_step(&mut model, &x, &y, 0., policy, &roles)?,
            )?);
            cases.push(json!({"seed":seed,"input_shape":shape,"policy":format!("{policy:?}"),"plan":serde_json::from_str::<Value>(&portable)?,
                "input":x.data(),"target":y.data(),"rates":rates,"steps":steps,"max_abs_error":error,"resume":"passed"}));
        }
    }
    let guards = guards(runtime.clone()).await?;
    let workspace_reuse = workspace_reuse(runtime.clone()).await?;
    let mut primitive_checks = Vec::new();
    for name in ["relu", "gelu", "scaler"] {
        let mut model = Sequential::new();
        match name {
            "relu" => model.push(Relu::new()),
            "gelu" => model.push(Gelu::new()),
            _ => model.push(Scaler::new("gain", 3)?),
        }
        let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[3])?)?;
        let initial = plan.graph_definition()?;
        let roles = initial
            .parameters()
            .iter()
            .map(|p| p.role)
            .collect::<Vec<_>>();
        let mut gpu =
            plan.compile_graph_training_wgpu(runtime.clone(), GraphGradientPolicy::Exact)?;
        if bits(&parameters(gpu.parameter_snapshot()?).await?) != bits(&initial) {
            return Err("pre-step snapshot differs".into());
        }
        let x = Tensor::from_vec(1, 3, vec![-1., 0., 2.])?;
        let y = Tensor::zeros(1, 3)?;
        gpu.upload_batch(x.data(), y.data())?;
        gpu.step(0.05)?;
        let actual = state(gpu.state_snapshot()?).await?;
        let error = compare(
            &actual,
            &cpu_step(&mut model, &x, &y, 0.05, GraphGradientPolicy::Exact, &roles)?,
        )?;
        primitive_checks.push(json!({"case":name,"max_abs_error":error,"parameters":roles.len()}));
    }
    Ok(
        json!({"schema":"spiraltorch.resident_graph_training_fixture.v1","status":"passed",
        "build_manifest":serde_json::from_str::<Value>(st_core::build_manifest_json())?,"primitive_checks":primitive_checks,
        "adapter":format!("{:?}",runtime.adapter_info()),"cases":cases,"guards":guards,
        "workspace_reuse":workspace_reuse,"pointwise_fusion":fusion::run(runtime.clone()).await?,
        "autograd":autograd::run(runtime.clone()).await?,
        "learning":learning::run(runtime.clone()).await?,
        "scope":"Sequential with owned gains; mean-MSE plain SGD; no intermediate host readbacks; not a throughput claim"}),
    )
}

async fn workspace_reuse(runtime: WgpuRuntime) -> Result<Value> {
    let mut records = Vec::new();
    for shape in [vec![4], vec![3, 4], vec![2, 129, 4]] {
        let layout = NdLayout::contiguous(&shape)?;
        let rows = layout.len() / 4;
        let policy = GraphGradientPolicy::Exact;
        let mut models = [model(17)?, model(43)?];
        let plans = models
            .iter()
            .map(|m| InferencePlan::from_module(m, layout.clone()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let roles = plans[0]
            .graph_definition()?
            .parameters()
            .iter()
            .map(|p| p.role)
            .collect::<Vec<_>>();
        let mut graphs = plans
            .iter()
            .map(|p| p.compile_graph_training_wgpu(runtime.clone(), policy))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut captures = Vec::new();
        // Same topology, distinct parameters/batches, alternating writes, deferred reads.
        for round in 0..12 {
            for lane in [round % 2, (round + 1) % 2] {
                let x = Tensor::from_fn(rows, 4, |r, c| {
                    ((r * 7 + c * 3 + round * 11 + lane * 5) % 23) as f32 / 16. - 0.5
                })?;
                let y = Tensor::from_fn(rows, 3, |r, c| x.data()[r * 4 + c] * 0.3 - 0.1)?;
                let rate = if round % 3 == 0 { 0. } else { 0.025 };
                graphs[lane].upload_batch(x.data(), y.data())?;
                graphs[lane].step(rate)?;
                captures.push((
                    graphs[lane].state_snapshot()?,
                    cpu_step(&mut models[lane], &x, &y, rate, policy, &roles)?,
                ));
            }
        }
        drop(graphs);
        let mut maximum = 0f32;
        for (snapshot, reference) in captures {
            maximum = maximum.max(compare(&state(snapshot).await?, &reference)?);
        }
        records.push(
            json!({"input_shape":shape,"independent_graphs":2,"snapshots":24,
            "steps_per_graph":12,"read_after_graph_drop":true,"max_abs_error":maximum}),
        );
    }
    Ok(json!(records))
}

fn guard_plan(rows: usize, weight: f32, gain: f32) -> Result<InferencePlan> {
    let parameters = vec![
        GraphParameter {
            role: ParameterRole::Weight,
            shape: vec![1, 1],
            values: vec![weight],
        },
        GraphParameter {
            role: ParameterRole::Bias,
            shape: vec![1],
            values: vec![0.],
        },
        GraphParameter {
            role: ParameterRole::Gain,
            shape: vec![1],
            values: vec![gain],
        },
    ];
    let stages = vec![
        GraphStage::Linear {
            weight: 0,
            bias: 1,
            gelu: false,
        },
        GraphStage::Pointwise {
            chain: PointwiseChain::new(
                2,
                vec![PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(1),
                }],
            )?,
            parameters: vec![2],
        },
    ];
    Ok(InferencePlan::from_graph_definition(GraphDefinition::new(
        NdLayout::contiguous(&[rows, 1])?,
        stages,
        parameters,
    )?)?)
}
async fn guards(runtime: WgpuRuntime) -> Result<Value> {
    let mut records = masked_forward_guards(runtime.clone()).await?;
    for (name, rows, w, g, x, y, rate) in [
        ("late_gain_candidate", 1, 1., 0.0625, 16., 0., f32::MAX / 8.),
        (
            "gain_unbroadcast_overflow",
            2,
            f32::MAX / 2.,
            0.,
            1.,
            -1.5,
            0.1,
        ),
    ] {
        let plan = guard_plan(rows, w, g)?;
        let initial = plan.graph_definition()?;
        let mut gpu =
            plan.compile_graph_training_wgpu(runtime.clone(), GraphGradientPolicy::Exact)?;
        gpu.upload_batch(&vec![x; rows], &vec![y; rows])?;
        gpu.step(rate)?;
        let loss = loss(gpu.loss_snapshot()?)
            .await
            .expect_err("nonfinite graph accepted");
        let error = loss
            .downcast_ref::<TrainingError>()
            .ok_or("wrong guard error")?;
        if !matches!(error, TrainingError::Rejected { .. }) {
            return Err("guard not a numerical rejection".into());
        }
        if tensor(gpu.prediction_tensor()?.snapshot()?).await.is_ok() {
            return Err("prediction dropped whole-step guard".into());
        }
        if tensor(gpu.input_gradient_tensor()?.snapshot()?)
            .await
            .is_ok()
        {
            return Err("input VJP dropped whole-step guard".into());
        }
        if bits(&parameters(gpu.parameter_snapshot()?).await?) != bits(&initial) {
            return Err("partial commit on rejected gain".into());
        }
        records.push(
            json!({"case":name,"error":error.to_string(),"all_parameter_bits_unchanged":true}),
        );
        // Reuse the exact failed workspace with a finite batch: no stale scratch/flags.
        gpu.upload_batch(&vec![0.; rows], &vec![0.; rows])?;
        gpu.step(0.1)?;
        let recovered = state(gpu.state_snapshot()?).await?;
        if recovered.loss != 0. || bits(&recovered.graph) != bits(&initial) {
            return Err("failed workspace did not recover without a partial update".into());
        }
    }
    let plan = guard_plan(1, 0., 0.)?;
    let initial = plan.graph_definition()?;
    let mut gpu = plan.compile_graph_training_wgpu(runtime.clone(), GraphGradientPolicy::Exact)?;
    let device = TensorDevice::new(runtime)?;
    let maximum = device.upload(&[1, 1], &[f32::MAX])?;
    let bad = maximum.apply(
        ElementwiseOp::Multiply,
        Some(&device.upload(&[1, 1], &[2.])?),
    )?;
    let target = device.upload(&[1, 1], &[0.])?;
    gpu.upload_batch_tensors(&bad, &target)?;
    for _ in 0..2 {
        gpu.step(0.1)?;
        if loss(gpu.loss_snapshot()?).await.is_ok()
            || bits(&parameters(gpu.parameter_snapshot()?).await?) != bits(&initial)
        {
            return Err("inherited invalid input bypassed transaction".into());
        }
    }
    records.push(
        json!({"case":"inherited_input_failure_each_step","all_parameter_bits_unchanged":true}),
    );
    gpu.upload_batch(&[1.], &[0.])?;
    gpu.step(0.1)?;
    loss(gpu.loss_snapshot()?).await?;
    records.push(json!({"case":"fresh_batch_clears_prior_failure","passed":true}));
    Ok(json!(records))
}

async fn masked_forward_guards(runtime: WgpuRuntime) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    for fault in 0..8 {
        let mut model = Sequential::new();
        for stage in 0..8 {
            let coefficient = if stage == fault { f32::MAX } else { 1. };
            if stage % 2 == 0 {
                let mut linear = Linear::new(format!("dense_{stage}"), 1, 1)?;
                linear.visit_parameters_mut(&mut |p| {
                    let value = if p.name().ends_with("weight") {
                        coefficient
                    } else {
                        0.
                    };
                    p.value_mut().data_mut().fill(value);
                    Ok(())
                })?;
                model.push(linear);
            } else {
                model.push(Scaler::from_gain(
                    format!("gain_{stage}"),
                    Tensor::from_vec(1, 1, vec![coefficient])?,
                )?);
            }
        }
        model.push(Relu::new());
        let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 1, 1])?)?;
        let initial = plan.graph_definition()?;
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            for rate in [0., 0.01] {
                let mut gpu = plan.compile_graph_training_wgpu(runtime.clone(), policy)?;
                gpu.upload_batch(&[-2.; 2], &[0.; 2])?;
                gpu.step(rate)?;
                let rejected = gpu.loss_snapshot()?;
                let rejected_state = gpu.state_snapshot()?;
                let prediction = gpu.prediction_tensor()?;
                let gradient = gpu.input_gradient_tensor()?;
                let rollback = gpu.parameter_snapshot()?;
                // Recovery must not clear guards on owning captures of the bad step.
                gpu.upload_batch(&[0.; 2], &[0.; 2])?;
                gpu.step(0.01)?;
                let recovered = gpu.state_snapshot()?;
                drop(gpu);
                let error = loss(rejected)
                    .await
                    .expect_err("masked overflow was accepted");
                if !matches!(
                    error.downcast_ref::<TrainingError>(),
                    Some(TrainingError::Rejected { .. })
                ) || state(rejected_state).await.is_ok()
                    || tensor(prediction.relu()?.snapshot()?).await.is_ok()
                    || tensor(gradient.snapshot()?).await.is_ok()
                {
                    return Err("masked forward overflow lost the transaction guard".into());
                }
                let recovered = state(recovered).await?;
                if bits(&parameters(rollback).await?) != bits(&initial)
                    || bits(&recovered.graph) != bits(&initial)
                    || recovered.loss != 0.
                {
                    return Err(
                        "masked forward overflow partially committed or recovery failed".into(),
                    );
                }
                records.push(json!({"case":"masked_forward_overflow","fault_stage":fault,
                    "policy":format!("{policy:?}"),"rate":rate,"error":error.to_string(),
                    "all_parameter_bits_unchanged":true,"retained_guard_after_reuse_and_drop":true}));
            }
        }
    }
    Ok(records)
}
