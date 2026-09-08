//! Same Rust N-D -> existing NN -> N-D fixture for native and real WebGPU.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_dense::{DenseActivation, DenseLayer},
    resident_tensor::{TensorError, INVALID_TENSOR_FLAG},
    resident_training::{TrainingError, TrainingState, TrainingStateReadback},
    runtime::WgpuRuntime,
};
use st_nn::{layers::Gelu, module::Module, resident::InferencePlan, Linear, Sequential};
use st_tensor::{NdLayout, NdTensor, Tensor, WgpuTensorDevice};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn values(tensor: &NdTensor) -> Result<Vec<f32>> {
    #[cfg(not(target_arch = "wasm32"))]
    let values = tensor.read_values();
    #[cfg(target_arch = "wasm32")]
    let values = tensor.read_values_async().await;
    Ok(values?)
}

async fn state(
    snapshot: TrainingStateReadback,
) -> std::result::Result<TrainingState, TrainingError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        snapshot.read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        snapshot.read_async().await
    }
}

fn close(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("length mismatch".into());
    }
    let mut error = 0f32;
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-5 + 1e-4 * b.abs() {
            return Err(format!("numeric mismatch {a} vs {b}").into());
        }
        error = error.max((a - b).abs());
    }
    Ok(error)
}

fn model(seed: usize, width: usize) -> Result<Sequential> {
    let mut model = Sequential::new();
    model.push(Linear::new("up", width, width + 3)?);
    model.push(Gelu::new());
    model.push(Linear::new("down", width + 3, width - 1)?);
    let mut counter = seed;
    model.visit_parameters_mut(&mut |p| {
        for v in p.value_mut().data_mut() {
            *v = ((counter * 17 % 23) as f32 - 11.) / 64.;
            counter += 1;
        }
        Ok(())
    })?;
    Ok(model)
}

fn layer_json(layer: &DenseLayer) -> Value {
    json!({"inner":layer.inner,"cols":layer.cols,"weights":layer.weights,"bias":layer.bias,"gelu":layer.activation==DenseActivation::Gelu})
}
fn state_json(state: &TrainingState) -> Value {
    json!({"loss":state.loss,"prediction":state.prediction,"input_gradient":state.input_gradient,
        "parameters":state.parameters.iter().map(layer_json).collect::<Vec<_>>(),
        "parameter_gradients":state.parameter_gradients.iter().map(|g| json!({"weights":g.weights,"bias":g.bias})).collect::<Vec<_>>()})
}

fn pipeline(
    root: NdTensor,
    bias: &NdTensor,
    gain: &NdTensor,
    iterations: usize,
) -> Result<NdTensor> {
    let length = root.shape()[1] - 1;
    let mut value = root.permute(&[1, 0, 2])?.narrow(0, 1, length)?;
    for _ in 0..iterations {
        value = value.add(bias)?.mul(gain)?.gelu()?;
    }
    Ok(value)
}

async fn case(
    runtime: &WgpuRuntime,
    device: &WgpuTensorDevice,
    shape: [usize; 3],
    seed: usize,
    iterations: usize,
) -> Result<Value> {
    let width = shape[2];
    let data: Vec<_> = (0..shape.iter().product())
        .map(|i| ((i * 13 + seed) % 61) as f32 / 64. - 0.46875)
        .collect();
    let bias: Vec<_> = (0..width).map(|i| (i % 5) as f32 / 32. - 0.0625).collect();
    let root = Tensor::from_vec(shape[0] * shape[1], width, data.clone())?
        .into_nd()?
        .reshape(&shape)?;
    let b = NdTensor::from_vec(&[width], bias.clone())?;
    let g = NdTensor::from_vec(&[], vec![0.75])?;
    let gpu_g = g.to_wgpu(device)?;
    let host = pipeline(root.clone(), &b, &g, iterations)?;
    let input = pipeline(
        root.to_wgpu(device)?,
        &b.to_wgpu(device)?,
        &gpu_g,
        iterations,
    )?;
    if !input.is_wgpu() {
        return Err("resident pipeline changed device".into());
    }
    let host_values = values(&host).await?;
    let model = model(seed, width)?;
    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(input.shape())?)?;
    let layers: Vec<_> = plan.parameter_snapshots().enumerate().map(|(i,(w,b))| json!({"inner":w.shape().0,"cols":w.shape().1,"weights":w.data(),"bias":b.data(),"gelu":i==0})).collect();
    let mut nn = plan.compile_wgpu(runtime.clone())?;
    nn.set_input_tensor(input.as_wgpu().unwrap())?;
    nn.dispatch()?;
    let prediction = NdTensor::from_wgpu(nn.tensor_snapshot(device)?);
    let output = prediction.mul(&gpu_g)?.permute(&[1, 0, 2])?.relu()?;
    let generation = nn.generation();
    let wrong_shape = input.contiguous()?.reshape(&[input.len()])?;
    if nn.set_input_tensor(wrong_shape.as_wgpu().unwrap()).is_ok() || nn.generation() != generation
    {
        return Err("bad inference shape changed state".into());
    }
    nn.upload(&vec![0.; input.len()])?;
    nn.dispatch()?;
    drop(nn);
    let cpu_prediction = {
        let _policy = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
            st_core::backend::device_caps::DeviceCaps::cpu(),
        ));
        model.forward(&Tensor::from_vec(
            input.len() / width,
            width,
            host_values.clone(),
        )?)?
    };
    let expected = cpu_prediction
        .into_nd()?
        .reshape(plan.output_layout().shape())?
        .mul(&g)?
        .permute(&[1, 0, 2])?
        .relu()?;
    let output_values = values(&output).await?;
    let mut error = close(&output_values, &values(&expected).await?)?;
    error = error.max(close(&values(&input).await?, &host_values)?);

    let target_data: Vec<_> = (0..plan.output_layout().len())
        .map(|i| (i % 7) as f32 / 32. - 0.09375)
        .collect();
    let target =
        NdTensor::from_vec(plan.output_layout().shape(), target_data.clone())?.to_wgpu(device)?;
    let mut training = plan.compile_training_wgpu(runtime.clone())?;
    let mut baseline = plan.compile_training_wgpu(runtime.clone())?;
    training.upload_batch_tensors(input.as_wgpu().unwrap(), target.as_wgpu().unwrap())?;
    baseline.upload_batch(&host_values, &target_data)?;
    let generation = training.batch_generation();
    if training
        .upload_batch_tensors(wrong_shape.as_wgpu().unwrap(), target.as_wgpu().unwrap())
        .is_ok()
        || training.batch_generation() != generation
    {
        return Err("bad batch shape changed state".into());
    }
    for _ in 0..8 {
        training.step(0.02)?;
        baseline.step(0.02)?;
    }
    let prediction = NdTensor::from_wgpu(training.prediction_tensor(device)?);
    let trained = state(training.state_snapshot()?).await?;
    let reference = state(baseline.state_snapshot()?).await?;
    error = error
        .max(close(&trained.prediction, &reference.prediction)?)
        .max(close(&trained.input_gradient, &reference.input_gradient)?);
    for (a, b) in trained.parameters.iter().zip(&reference.parameters) {
        error = error
            .max(close(&a.weights, &b.weights)?)
            .max(close(&a.bias, &b.bias)?);
    }
    for (a, b) in trained
        .parameter_gradients
        .iter()
        .zip(&reference.parameter_gradients)
    {
        error = error
            .max(close(&a.weights, &b.weights)?)
            .max(close(&a.bias, &b.bias)?);
    }
    error = error.max(close(&[trained.loss], &[reference.loss])?);
    let frozen = prediction.relu()?;
    training.step(0.02)?;
    drop(training);
    close(
        &values(&frozen).await?,
        &trained
            .prediction
            .iter()
            .map(|x| x.max(0.))
            .collect::<Vec<_>>(),
    )?;
    Ok(
        json!({"shape":shape,"seed":seed,"iterations":iterations,"input":data,"bias":bias,"gain":0.75,
        "processed_shape":input.shape(),"processed":values(&input).await?,"layers":layers,
        "output_shape":output.shape(),"output":output_values,"targets":target_data,
        "steps":8,"learning_rate":0.02,"training":state_json(&trained),"max_abs_error":error,
        "invalid_shapes_preserve_state":true,"snapshots_survive_reuse_and_drop":true}),
    )
}

async fn guards(runtime: &WgpuRuntime, device: &WgpuTensorDevice) -> Result<Value> {
    let model = model(7, 4)?;
    let shape = [1, 2, 4];
    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&shape)?)?;
    let scalar = |x| device.upload(&[], &[x]);
    // ReLU hides the overflow in values, but must never erase the inherited error.
    let invalid = scalar(-f32::MAX)?.mul(&scalar(2.)?)?.relu()?;
    let input = scalar(0.25)?.broadcast_to(&shape)?;
    let target = scalar(0.1)?.broadcast_to(plan.output_layout().shape())?;
    let bad_input = invalid.broadcast_to(&shape)?;
    let bad_target = invalid.broadcast_to(plan.output_layout().shape())?;
    let mut nn = plan.compile_wgpu(runtime.clone())?;
    nn.set_input_tensor(&bad_input)?;
    nn.dispatch()?;
    let bad_output = NdTensor::from_wgpu(nn.tensor_snapshot(device)?.relu()?);
    if values(&bad_output).await.is_ok() {
        return Err("inference erased an upstream failure".into());
    }
    let mut training = plan.compile_training_wgpu(runtime.clone())?;
    training.upload_batch_tensors(&input, &target)?;
    training.step(0.)?;
    let before = state(training.state_snapshot()?).await?;
    for (x, y) in [(&bad_input, &target), (&input, &bad_target)] {
        training.upload_batch_tensors(x, y)?;
        for _ in 0..2 {
            training.step(0.05)?;
            match state(training.state_snapshot()?).await {
                Err(TrainingError::Rejected { flags, .. }) if flags & INVALID_TENSOR_FLAG != 0 => {}
                other => return Err(format!("wrong upstream rejection: {other:?}").into()),
            }
            let rejected = NdTensor::from_wgpu(training.prediction_tensor(device)?.relu()?);
            if values(&rejected).await.is_ok() {
                return Err("training prediction erased rejection".into());
            }
        }
    }
    training.upload_batch(&[0.25; 8], &[0.1; 6])?;
    training.step(0.)?;
    let after = state(training.state_snapshot()?).await?;
    for (a, b) in before.parameters.iter().zip(&after.parameters) {
        if a.weights
            .iter()
            .chain(&a.bias)
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
            != b.weights
                .iter()
                .chain(&b.bias)
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        {
            return Err("rejected input changed weights".into());
        }
    }
    let other_device =
        WgpuTensorDevice::new(WgpuRuntime::request_headless("tensor.nd.foreign").await?)?;
    let foreign = other_device.upload(&shape, &[1.; 8])?;
    let generation = training.batch_generation();
    if !matches!(
        training.upload_batch_tensors(&foreign, &target),
        Err(TrainingError::Tensor(TensorError::DeviceMismatch))
    ) || training.batch_generation() != generation
    {
        return Err("foreign queue was accepted".into());
    }
    if input.add(&foreign).is_ok()
        || nn.set_input_tensor(&foreign).is_ok()
        || nn.tensor_snapshot(&other_device).is_ok()
    {
        return Err("foreign tensor bridge accepted".into());
    }
    let snapshot = invalid.broadcast_to(&[0, 3])?.relu()?.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let empty_failed = snapshot.read().is_err();
    #[cfg(target_arch = "wasm32")]
    let empty_failed = snapshot.read_async().await.is_err();
    if !empty_failed {
        return Err("empty output erased a failure".into());
    }
    Ok(
        json!({"masked_overflow_rejected":true,"empty_failure_preserved":true,"input_and_target_rejections":4,
        "all_weights_bitwise_unchanged":true,"host_upload_clears_upstream_flags":true,"foreign_device_rejected":true}),
    )
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("software adapter is not GPU evidence".into());
    }
    let device = WgpuTensorDevice::new(runtime.clone())?;
    let mut cases = Vec::new();
    for (shape, seed, iterations) in [([2, 3, 4], 17, 0), ([3, 5, 7], 29, 1), ([2, 4, 11], 43, 20)]
    {
        cases.push(case(&runtime, &device, shape, seed, iterations).await?);
    }
    Ok(
        json!({"schema":"spiraltorch.resident_nd_tensor.fixture.v1","status":"passed",
        "adapter":format!("{:?}",runtime.adapter_info()),"cases":cases,"guards":guards(&runtime,&device).await?,
        "build_fingerprint":st_core::build_fingerprint(),
        "build_manifest":serde_json::from_str::<Value>(st_core::build_manifest_json())?,
        "boundary":"Rust CPU snapshot / WGPU immutable tensor; preprocessing and NN bridges stay on device; explicit terminal reads; no general autograd claim"}),
    )
}
