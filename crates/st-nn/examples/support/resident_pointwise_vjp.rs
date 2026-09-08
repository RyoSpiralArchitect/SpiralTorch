//! Shared native/browser VJP fixtures, including a resident NN input-gradient bridge.
use serde_json::{json, Value};
use st_backend_wgpu::runtime::WgpuRuntime;
use st_nn::{layers::Gelu, module::Module, resident::InferencePlan, Linear, Sequential};
use st_tensor::{
    ElementwiseOp, NdLayout, NdPointwiseVjpPlan, NdTensor, PointwiseChain, PointwiseExecution,
    PointwiseStep, WgpuTensorDevice,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
async fn values(t: &NdTensor) -> Result<Vec<f32>> {
    #[cfg(not(target_arch = "wasm32"))]
    let value = t.read_values();
    #[cfg(target_arch = "wasm32")]
    let value = t.read_values_async().await;
    Ok(value?)
}
fn close(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("VJP length mismatch".into());
    }
    let mut error = 0f32;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-5 + 2e-4 * b.abs() {
            return Err(format!("VJP mismatch {a} vs {b}").into());
        }
        error = error.max((a - b).abs());
    }
    Ok(error)
}
fn recipe() -> Vec<PointwiseStep> {
    vec![
        PointwiseStep {
            op: ElementwiseOp::Multiply,
            rhs: Some(1),
        },
        PointwiseStep {
            op: ElementwiseOp::Gelu,
            rhs: None,
        },
        PointwiseStep {
            op: ElementwiseOp::Add,
            rhs: Some(0),
        },
        PointwiseStep {
            op: ElementwiseOp::Multiply,
            rhs: Some(2),
        },
        PointwiseStep {
            op: ElementwiseOp::Relu,
            rhs: None,
        },
    ]
}
fn chain_json(steps: &[PointwiseStep]) -> Value {
    json!(steps
        .iter()
        .map(|step| json!({"op":format!("{:?}",step.op).to_lowercase(),"rhs":step.rhs}))
        .collect::<Vec<_>>())
}

async fn case(
    device: &WgpuTensorDevice,
    shape: &[usize],
    gain_shape: &[usize],
    seed: usize,
) -> Result<Value> {
    let len = NdLayout::contiguous(shape)?.len();
    let x = NdTensor::from_vec(
        shape,
        (0..len)
            .map(|i| ((i * 13 + seed) % 61) as f32 / 32. - 0.875)
            .collect(),
    )?;
    let gain_len = NdLayout::contiguous(gain_shape)?.len();
    let gain = NdTensor::from_vec(
        gain_shape,
        (0..gain_len).map(|i| 0.5 + (i % 7) as f32 / 8.).collect(),
    )?;
    let scale = NdTensor::from_vec(&[], vec![0.75])?;
    let cotangent = NdTensor::from_vec(
        shape,
        (0..len)
            .map(|i| ((i * 7 + seed) % 17) as f32 / 16. - 0.5)
            .collect(),
    )?;
    let steps = recipe();
    let chain = PointwiseChain::new(3, steps.clone())?;
    let host = NdPointwiseVjpPlan::new(chain.clone(), &[&x, &gain, &scale])?;
    let gpu_x = x.to_wgpu(device)?;
    let gpu_gain = gain.to_wgpu(device)?;
    let gpu_scale = scale.to_wgpu(device)?;
    let gpu_seed = cotangent.to_wgpu(device)?;
    let gpu = NdPointwiseVjpPlan::new(chain, &[&gpu_x, &gpu_gain, &gpu_scale])?;
    let output = gpu.forward(&[&gpu_x, &gpu_gain, &gpu_scale], PointwiseExecution::Fused)?;
    let actual = gpu.vjp(&[&gpu_x, &gpu_gain, &gpu_scale], &gpu_seed)?;
    let expected = host.vjp(&[&x, &gain, &scale], &cotangent)?;
    let mut gradients = Vec::new();
    let mut error = close(
        &values(&output).await?,
        &values(&host.forward(&[&x, &gain, &scale], PointwiseExecution::Fused)?).await?,
    )?;
    for (a, b) in actual.iter().zip(&expected) {
        let a_values = values(a).await?;
        error = error.max(close(&a_values, &values(b).await?)?);
        gradients.push(json!({"shape":a.shape(),"values":a_values}));
    }
    Ok(
        json!({"shape":shape,"gain_shape":gain_shape,"seed":seed,"steps":chain_json(&steps),
        "inputs":[{"shape":shape,"values":values(&x).await?},
            {"shape":gain_shape,"values":values(&gain).await?},
            {"shape":[],"values":[0.75]}],
        "cotangent":values(&cotangent).await?,"output":values(&output).await?,
        "gradients":gradients,"max_abs_error":error}),
    )
}

async fn bridge(runtime: &WgpuRuntime, device: &WgpuTensorDevice) -> Result<Value> {
    let x = NdTensor::from_vec(&[3, 2, 4], (0..24).map(|i| i as f32 / 16. - 0.5).collect())?
        .permute(&[1, 0, 2])?
        .narrow(1, 1, 2)?;
    let gain = NdTensor::from_vec(
        &[2, 4],
        vec![0.; 4]
            .into_iter()
            .chain([0.5, 1.25, -0.5, 0.75])
            .collect(),
    )?
    .narrow(0, 1, 1)?
    .reshape(&[4])?;
    let scale = NdTensor::from_vec(&[], vec![1.])?;
    let gx = NdTensor::from_vec(&[3, 2, 4], (0..24).map(|i| i as f32 / 16. - 0.5).collect())?
        .to_wgpu(device)?
        .permute(&[1, 0, 2])?
        .narrow(1, 1, 2)?;
    let gg = NdTensor::from_vec(
        &[2, 4],
        vec![0.; 4]
            .into_iter()
            .chain([0.5, 1.25, -0.5, 0.75])
            .collect(),
    )?
    .to_wgpu(device)?
    .narrow(0, 1, 1)?
    .reshape(&[4])?;
    let gs = scale.to_wgpu(device)?;
    let vjp = NdPointwiseVjpPlan::new(PointwiseChain::new(3, recipe())?, &[&gx, &gg, &gs])?;
    let processed = vjp.forward(&[&gx, &gg, &gs], PointwiseExecution::Fused)?;
    let mut model = Sequential::new();
    model.push(Linear::new("up", 4, 5)?);
    model.push(Gelu::new());
    model.push(Linear::new("down", 5, 2)?);
    let mut cursor = 0;
    model.visit_parameters_mut(&mut |p| {
        for value in p.value_mut().data_mut() {
            *value = ((cursor * 11 % 23) as f32 - 11.) / 32.;
            cursor += 1;
        }
        Ok(())
    })?;
    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(processed.shape())?)?;
    let mut training = plan.compile_training_wgpu(runtime.clone())?;
    if training.input_gradient_tensor(device).is_ok() {
        return Err("gradient before step".into());
    }
    let targets = NdTensor::from_vec(
        plan.output_layout().shape(),
        vec![0.125; plan.output_layout().len()],
    )?
    .to_wgpu(device)?;
    training.upload_batch_tensors(processed.as_wgpu().unwrap(), targets.as_wgpu().unwrap())?;
    // Probe only. External gain is not part of the dense transactional SGD group.
    training.step(0.)?;
    let incoming = NdTensor::from_wgpu(training.input_gradient_tensor(device)?);
    let gradients = vjp.vjp(&[&gx, &gg, &gs], &incoming)?;
    let frozen = incoming.clone();
    let host_plan =
        NdPointwiseVjpPlan::new(PointwiseChain::new(3, recipe())?, &[&x, &gain, &scale])?;
    let incoming_values = values(&incoming).await?;
    let expected = host_plan.vjp(
        &[&x, &gain, &scale],
        &NdTensor::from_vec(x.shape(), incoming_values.clone())?,
    )?;
    let mut error = 0f32;
    let mut gradient_json = Vec::new();
    for (a, b) in gradients.iter().zip(&expected) {
        let av = values(a).await?;
        error = error.max(close(&av, &values(b).await?)?);
        gradient_json.push(json!({"shape":a.shape(),"values":av}));
    }
    training.upload_batch(&vec![0.; processed.len()], &vec![0.25; targets.len()])?;
    if training.input_gradient_tensor(device).is_ok() {
        return Err("stale input gradient accepted".into());
    }
    training.step(0.01)?;
    let huge = NdTensor::from_vec(&[], vec![f32::MAX])?.to_wgpu(device)?;
    let two = NdTensor::from_vec(&[], vec![2.])?.to_wgpu(device)?;
    let invalid = huge.mul(&two)?.broadcast_to(processed.shape())?;
    training.upload_batch_tensors(invalid.as_wgpu().unwrap(), targets.as_wgpu().unwrap())?;
    training.step(0.)?;
    let failed_seed = NdTensor::from_wgpu(training.input_gradient_tensor(device)?);
    if values(&failed_seed).await.is_ok() {
        return Err("rejected NN step exposed a valid input gradient".into());
    }
    for gradient in vjp.vjp(&[&gx, &gg, &gs], &failed_seed)? {
        if values(&gradient).await.is_ok() {
            return Err("upstream VJP erased the NN step rejection".into());
        }
    }
    drop(training);
    close(&values(&frozen).await?, &incoming_values)?;
    let layers:Vec<_>=plan.parameter_snapshots().enumerate().map(|(i,(w,b))|
        json!({"inner":w.shape().0,"cols":w.shape().1,"weights":w.data(),"bias":b.data(),"gelu":i==0})).collect();
    Ok(
        json!({"status":"passed","shape":x.shape(),"steps":chain_json(&recipe()),
        "inputs":[{"shape":x.shape(),"values":values(&x).await?},{"shape":gain.shape(),"values":values(&gain).await?},{"shape":[],"values":[1.]}],
        "processed":values(&processed).await?,"layers":layers,"targets":values(&targets).await?,
        "output_shape":plan.output_layout().shape(),"input_cotangent":incoming_values,"gradients":gradient_json,
        "max_abs_error":error,"snapshots_survive_reuse":true,"stale_gradient_rejected":true,
        "rejected_nn_step_invalidates_upstream_gradients":true,
        "boundary":"mean-MSE dense probe at rate zero -> frozen resident input VJP -> pointwise/gain VJP; no external gain update or cross-graph transaction claim"}),
    )
}

async fn guards(device: &WgpuTensorDevice) -> Result<Value> {
    let scalar =
        |x| Ok::<_, Box<dyn std::error::Error>>(NdTensor::from_vec(&[], vec![x])?.to_wgpu(device)?);
    let huge = scalar(f32::MAX)?;
    let two = scalar(2.)?;
    let zero = scalar(0.)?;
    let one = scalar(1.)?;
    let multiply = PointwiseChain::new(
        2,
        vec![PointwiseStep {
            op: ElementwiseOp::Multiply,
            rhs: Some(1),
        }],
    )?;
    let plan = NdPointwiseVjpPlan::new(multiply.clone(), &[&one, &two])?;
    for (inputs, seed) in [([&huge, &two], &zero), ([&one, &two], &huge)] {
        for gradient in plan.vjp(&inputs, seed)? {
            if values(&gradient).await.is_ok() {
                return Err("overflow or zero-seed masking accepted".into());
            }
        }
    }
    let root = NdTensor::from_vec(&[2], vec![1., 1.])?.to_wgpu(device)?;
    let seed = NdTensor::from_vec(&[2], vec![f32::MAX; 2])?.to_wgpu(device)?;
    let reduce = NdPointwiseVjpPlan::new(multiply.clone(), &[&root, &one])?;
    for gradient in reduce.vjp(&[&root, &one], &seed)? {
        if values(&gradient).await.is_ok() {
            return Err("late reduction failed to invalidate all gradients".into());
        }
    }
    let empty = NdTensor::from_vec(&[0, 3], vec![])?.to_wgpu(device)?;
    let invalid = huge.mul(&two)?.broadcast_to(&[0, 3])?;
    let empty_plan = NdPointwiseVjpPlan::new(multiply, &[&empty, &one])?;
    for gradient in empty_plan.vjp(&[&empty, &one], &invalid)? {
        if values(&gradient).await.is_ok() {
            return Err("empty cotangent erased inherited failure".into());
        }
    }
    if plan.vjp(&[&root, &one], &one).is_ok()
        || plan.vjp(&[&one], &one).is_ok()
        || plan
            .vjp(&[&one, &two], &NdTensor::from_vec(&[], vec![1.])?)
            .is_ok()
    {
        return Err("VJP layout/count/device contract accepted invalid inputs".into());
    }
    let foreign_device =
        WgpuTensorDevice::new(WgpuRuntime::request_headless("vjp.foreign").await?)?;
    let foreign = NdTensor::from_vec(&[], vec![1.])?.to_wgpu(&foreign_device)?;
    if plan.vjp(&[&foreign, &two], &one).is_ok() || plan.vjp(&[&one, &two], &foreign).is_ok() {
        return Err("VJP accepted a foreign device or queue".into());
    }
    let host = NdTensor::from_vec(&[2, 3], vec![1., -2., 3., 4., -5., 6.])?.permute(&[1, 0])?;
    let gpu = NdTensor::from_vec(&[2, 3], vec![1., -2., 3., 4., -5., 6.])?
        .to_wgpu(device)?
        .permute(&[1, 0])?;
    let chain = PointwiseChain::new(
        2,
        vec![PointwiseStep {
            op: ElementwiseOp::Multiply,
            rhs: Some(1),
        }],
    )?;
    let host_two = NdTensor::from_vec(&[], vec![2.])?;
    let host_plan = NdPointwiseVjpPlan::new(chain.clone(), &[&host, &host_two])?;
    let gpu_plan = NdPointwiseVjpPlan::new(chain, &[&gpu, &two])?;
    let old = gpu_plan.vjp(&[&gpu, &two], &gpu)?;
    for (a, b) in old.iter().zip(host_plan.vjp(&[&host, &host_two], &host)?) {
        close(&values(a).await?, &values(&b).await?)?;
    }
    let frozen = values(&old[0]).await?;
    let _new = gpu_plan.vjp(&[&gpu, &two], &gpu.mul(&two)?)?;
    close(&values(&old[0]).await?, &frozen)?;
    Ok(
        json!({"status":"passed","zero_seed_does_not_hide_bad_forward":true,
        "derivative_overflow_rejected":true,"late_reduction_invalidates_all_slots":true,
        "empty_failed_seed_preserved":true,"shape_count_mixed_host_rejected":true,
        "foreign_device_rejected":true,"strided_cotangent":true,"immutable_gradients":true}),
    )
}

pub async fn run(runtime: &WgpuRuntime, device: &WgpuTensorDevice) -> Result<Value> {
    let mut cases = Vec::new();
    for (shape, gain, seed) in [
        (vec![2, 3, 4], vec![4], 17),
        (vec![2, 3, 4], vec![1, 3, 1], 29),
        (vec![513, 3], vec![3], 43),
        (vec![65537, 1], vec![], 47),
        (vec![], vec![], 53),
        (vec![2, 0, 3], vec![3], 59),
    ] {
        cases.push(case(device, &shape, &gain, seed).await?);
    }
    Ok(
        json!({"schema":"spiraltorch.pointwise_vjp.fixture.v1","status":"passed","cases":cases,
        "nn_bridge":bridge(runtime,device).await?,"guards":guards(device).await?}),
    )
}
