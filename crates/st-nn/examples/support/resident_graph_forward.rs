//! One fixture for native WGPU and browser WebGPU, including deferred failures.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_graph::{GraphInferenceError, GraphReadback},
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{TensorError, TensorReadback},
    runtime::WgpuRuntime,
};
use st_nn::{
    layers::{Gelu, Relu, Scaler},
    module::Module,
    resident::InferencePlan,
    Linear, Sequential,
};
use st_tensor::{NdLayout, Tensor};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn read(value: GraphReadback) -> Result<Vec<f32>> {
    #[cfg(target_arch = "wasm32")]
    let result = value.read_async().await;
    #[cfg(not(target_arch = "wasm32"))]
    let result = value.read();
    Ok(result?)
}
async fn tensor(value: TensorReadback) -> Result<Vec<f32>> {
    #[cfg(target_arch = "wasm32")]
    let result = value.read_async().await;
    #[cfg(not(target_arch = "wasm32"))]
    let result = value.read();
    Ok(result?)
}
fn close(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("comparison length".into());
    }
    let mut max = 0f32;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-5 + 2e-4 * b.abs() {
            return Err(format!("forward mismatch: {a} != {b}").into());
        }
        max = max.max((a - b).abs());
    }
    Ok(max)
}
fn model(seed: usize) -> Result<Sequential> {
    let mut model = Sequential::new();
    model.push(Scaler::new("input", 4)?);
    model.push(Linear::new("up", 4, 7)?);
    model.push(Gelu::new());
    model.push(Relu::new());
    model.push(Scaler::new("hidden", 7)?);
    model.push(Linear::new("down", 7, 3)?);
    let mut counter = seed;
    model.visit_parameters_mut(&mut |p| {
        for value in p.value_mut().data_mut() {
            *value = ((counter * 17 % 23) as f32 - 11.) / 16.;
            counter += 1;
        }
        Ok(())
    })?;
    Ok(model)
}
fn cpu(model: &mut impl Module, input: &[f32], width: usize) -> Result<Vec<f32>> {
    let _policy = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
        st_core::backend::device_caps::DeviceCaps::cpu(),
    ));
    Ok(model
        .forward(&Tensor::from_vec(
            input.len() / width,
            width,
            input.to_vec(),
        )?)?
        .data()
        .to_vec())
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    for seed in [17, 29] {
        for shape in [vec![4], vec![3, 4], vec![2, 129, 4]] {
            for (kernel, accumulation) in [
                (MatmulKernel::Scalar, MatmulAccumulation::Sequential),
                (MatmulKernel::Register2x2, MatmulAccumulation::Compensated),
            ] {
                let mut model = model(seed)?;
                let layout = NdLayout::contiguous(&shape)?;
                let plan = InferencePlan::from_module(&model, layout.clone())?;
                let portable = plan.to_json()?;
                let plan = InferencePlan::from_json(&portable)?;
                let mut graph = plan.compile_graph_wgpu_with_options(
                    runtime.clone(),
                    MatmulTile::default(),
                    kernel,
                    accumulation,
                )?;
                assert!(matches!(
                    graph.dispatch(),
                    Err(GraphInferenceError::MissingInput)
                ));
                assert!(matches!(
                    graph.snapshot(),
                    Err(GraphInferenceError::StaleOutput)
                ));
                assert!(matches!(
                    graph.output_tensor(),
                    Err(GraphInferenceError::StaleOutput)
                ));
                let device = graph.tensor_device().clone();
                let input: Vec<_> = (0..layout.len())
                    .map(|i| ((i * 11 + seed) % 29) as f32 / 16. - 0.8)
                    .collect();
                let reference = cpu(&mut model, &input, 4)?;
                graph.upload(&input)?;
                assert_eq!(graph.dispatch()?, 1);
                let host = graph.snapshot()?;
                assert_eq!((host.generation(), host.dispatch()), (1, 1));
                // Invalid host/device shape replacements do not stale a valid output.
                assert!(graph.upload(&[]).is_err());
                assert!(graph.upload(&vec![f32::NAN; input.len()]).is_err());
                let wrong = device.upload(&[input.len(), 1], &input)?;
                assert!(matches!(
                    graph.set_input_tensor(&wrong),
                    Err(GraphInferenceError::InputShape)
                ));
                assert_eq!(graph.generation(), 1);
                assert!(graph.snapshot().is_ok());

                // A narrow view has a nonzero offset and, for rank > 1, row gaps.
                let mut padded_shape = shape.clone();
                *padded_shape.last_mut().unwrap() += 2;
                let mut padded = vec![91f32; input.len() / 4 * 6];
                for (row, values) in input.chunks_exact(4).enumerate() {
                    padded[row * 6 + 1..row * 6 + 5].copy_from_slice(values);
                }
                let source = device.upload(&padded_shape, &padded)?;
                let view = source.narrow(shape.len() - 1, 1, 4)?;
                graph.set_input_tensor(&view)?;
                assert!(matches!(
                    graph.output_tensor(),
                    Err(GraphInferenceError::StaleOutput)
                ));
                assert_eq!(graph.dispatch()?, 2);
                let strided = graph.snapshot()?;
                let gains = [1., 0.5, -0.75, 1.25];
                let pre = view.mul(&device.upload(&[4], &gains)?)?.relu()?;
                drop(view);
                drop(source);
                let pre_values: Vec<_> = input
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (x * gains[i % 4]).max(0.))
                    .collect();
                let pre_reference = cpu(&mut model, &pre_values, 4)?;
                graph.set_input_tensor(&pre)?;
                for dispatch in 3..=18 {
                    assert_eq!(graph.dispatch()?, dispatch);
                }
                let output = graph.output_tensor()?;
                let pre_snapshot = graph.snapshot()?;
                assert_eq!(
                    (pre_snapshot.generation(), pre_snapshot.dispatch()),
                    (3, 18)
                );
                let shift = [-0.125, 0.25, 0.5];
                let post = output.add(&device.upload(&[3], &shift)?)?.gelu()?;
                let shifted: Vec<_> = pre_reference
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v + shift[i % 3])
                    .collect();
                let mut gelu = Gelu::new();
                let post_reference = cpu(&mut gelu, &shifted, 3)?;
                // Another NN graph consumes the GPU tensor without host observation.
                let mut next =
                    InferencePlan::from_module(&Relu::new(), graph.output_layout().clone())?
                        .compile_graph_wgpu(runtime.clone())?;
                assert_eq!(next.parameter_count(), 0);
                next.set_input_tensor(&post)?;
                next.dispatch()?;
                let chained = next.output_tensor()?.snapshot()?;
                let post_snapshot = post.snapshot()?;
                let frozen = output.snapshot()?;
                graph.upload(&vec![0.; input.len()])?;
                graph.dispatch()?;
                drop(graph);
                drop(next);
                drop(post);
                drop(output);
                drop(pre);
                // Read only after workspace mutation/drop; no intermediate CPU dependency.
                let host = read(host).await?;
                let strided = read(strided).await?;
                let predicted = read(pre_snapshot).await?;
                let frozen = tensor(frozen).await?;
                let post = tensor(post_snapshot).await?;
                let chained = tensor(chained).await?;
                let chained_reference: Vec<_> = post_reference.iter().map(|v| v.max(0.)).collect();
                let error = close(&host, &reference)?
                    .max(close(&strided, &reference)?)
                    .max(close(&predicted, &pre_reference)?)
                    .max(close(&frozen, &pre_reference)?)
                    .max(close(&post, &post_reference)?)
                    .max(close(&chained, &chained_reference)?);
                cases.push(json!({"seed":seed,"shape":shape,"kernel":kernel.as_str(),
                    "accumulation":accumulation.as_str(),"plan":serde_json::from_str::<Value>(&portable)?,
                    "input":input,"pre_gains":gains,"post_shift":shift,"dispatches_before_capture":18,
                    "host":host,"strided":strided,"prediction":predicted,"frozen":frozen,
                    "post":post,"chained":chained,"max_abs_module_error":error}));
            }
        }
    }
    let guards = guard_checks(runtime.clone()).await?;
    let info = runtime.adapter_info();
    Ok(
        json!({"schema":"spiraltorch.resident_graph_forward.v1","status":"passed",
        "adapter":{"name":info.name,"backend":format!("{:?}",info.backend),"device_type":format!("{:?}",info.device_type)},
        "cases":cases,"guards":guards,"scope":"forward-only correctness and GPU ownership; not throughput or automatic Module residency"}),
    )
}

async fn guard_checks(runtime: WgpuRuntime) -> Result<Value> {
    let mut model = Sequential::new();
    model.push(Scaler::new("overflow", 1)?);
    model.push(Relu::new());
    model.visit_parameters_mut(&mut |p| {
        p.value_mut().data_mut()[0] = f32::MAX;
        Ok(())
    })?;
    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[1])?)?;
    let mut graph = plan.compile_graph_wgpu(runtime.clone())?;
    let device = graph.tensor_device().clone();
    graph.upload(&[-2.])?;
    graph.dispatch()?;
    let failed = graph.snapshot()?;
    let failed_tensor = graph.output_tensor()?;
    graph.upload(&[0.])?;
    graph.dispatch()?;
    close(&read(graph.snapshot()?).await?, &[0.])?;
    assert!(read(failed).await.is_err());
    assert!(tensor(failed_tensor.snapshot()?).await.is_err());
    let mut consumer = InferencePlan::from_module(&Relu::new(), NdLayout::contiguous(&[1])?)?
        .compile_graph_wgpu(runtime.clone())?;
    consumer.set_input_tensor(&failed_tensor)?;
    consumer.dispatch()?;
    let inherited = consumer.snapshot()?;
    consumer.dispatch()?;
    assert!(tensor(consumer.output_tensor()?.snapshot()?).await.is_err());
    consumer.upload(&[0.])?;
    consumer.dispatch()?;
    close(&read(consumer.snapshot()?).await?, &[0.])?;
    assert!(read(inherited).await.is_err());
    // The legacy no-tape dense kernel is reused and its intermediate flags survive ReLU.
    let mut linear = Linear::new("overflow", 1, 1)?;
    let mut index = 0;
    linear.visit_parameters_mut(&mut |p| {
        p.value_mut().data_mut()[0] = if index == 0 { f32::MAX } else { 0. };
        index += 1;
        Ok(())
    })?;
    let mut dense_model = Sequential::new();
    dense_model.push(linear);
    dense_model.push(Relu::new());
    let mut dense = InferencePlan::from_module(&dense_model, NdLayout::contiguous(&[1])?)?
        .compile_graph_wgpu(runtime.clone())?;
    dense.upload(&[-2.])?;
    dense.dispatch()?;
    assert!(read(dense.snapshot()?).await.is_err());
    assert!(tensor(dense.output_tensor()?.relu()?.snapshot()?)
        .await
        .is_err());
    dense.upload(&[0.])?;
    dense.dispatch()?;
    close(&read(dense.snapshot()?).await?, &[0.])?;
    // Non-finite GPU inputs are not sanitized by a finite downstream activation.
    let bad = device
        .upload(&[1], &[-f32::MAX])?
        .mul(&device.upload(&[1], &[2.])?)?
        .relu()?;
    consumer.set_input_tensor(&bad)?;
    consumer.dispatch()?;
    assert!(read(consumer.snapshot()?).await.is_err());
    // Same shape is insufficient: handles must belong to the same device AND queue.
    let other = WgpuRuntime::request_headless("graph.forward.foreign").await?;
    let foreign =
        st_backend_wgpu::resident_tensor::TensorDevice::new(other)?.upload(&[1], &[0.])?;
    let generation = consumer.generation();
    assert!(matches!(
        consumer.set_input_tensor(&foreign),
        Err(GraphInferenceError::Tensor(TensorError::DeviceMismatch))
    ));
    assert_eq!(consumer.generation(), generation);
    consumer.upload(&[0.])?;
    consumer.dispatch()?;
    close(&read(consumer.snapshot()?).await?, &[0.])?;
    let broadcast_view = device
        .upload(&[3, 1], &[-1., 2., 3.])?
        .broadcast_to(&[2, 3, 1])?
        .permute(&[1, 0, 2])?;
    let mut view_graph =
        InferencePlan::from_module(&Relu::new(), NdLayout::contiguous(&[3, 2, 1])?)?
            .compile_graph_wgpu(runtime.clone())?;
    view_graph.set_input_tensor(&broadcast_view)?;
    view_graph.dispatch()?;
    close(
        &read(view_graph.snapshot()?).await?,
        &[0., 0., 2., 2., 3., 3.],
    )?;
    let mut linear = Linear::new("portable", 1, 1)?;
    let mut slot = 0;
    linear.visit_parameters_mut(&mut |p| {
        p.value_mut().data_mut()[0] = if slot == 0 { 0.25 } else { 0.125 };
        slot += 1;
        Ok(())
    })?;
    let v1 = InferencePlan::from_module(&linear, NdLayout::contiguous(&[2, 3, 1])?)?;
    assert!(v1.is_dense());
    let v2 = InferencePlan::from_graph_definition(v1.graph_definition()?)?;
    assert!(!v2.is_dense());
    let input = [0., 1., 2., 3., 4., 5.];
    let expected = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375];
    for plan in [&v1, &v2] {
        let restored = InferencePlan::from_json(&plan.to_json()?)?;
        let mut compiled = restored.compile_graph_wgpu(runtime.clone())?;
        compiled.upload(&input)?;
        compiled.dispatch()?;
        close(&read(compiled.snapshot()?).await?, &expected)?;
    }
    let mut specialized = v1.compile_wgpu(runtime)?;
    specialized.upload(&input)?;
    specialized.dispatch()?;
    let snapshot = specialized.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    close(&values, &expected)?;
    Ok(
        json!({"pointwise_masked_overflow":true,"dense_masked_overflow":true,
        "whole_graph_guard_capture":true,"repeated_inherited_guard":true,
        "gpu_input_guard":true,"device_mismatch_atomic":true,"recovery":true,
        "broadcast_permute":true,"dense_v1_v2_specialized":true}),
    )
}
