use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice, TensorError},
    runtime::{WgpuRuntime, WgpuRuntimeError},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn values(outputs: [ResidentTensor; 3]) -> Result<Vec<Vec<f32>>> {
    let mut values = Vec::new();
    for tensor in outputs {
        let snapshot = tensor.snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        values.push(snapshot.read()?);
        #[cfg(target_arch = "wasm32")]
        values.push(snapshot.read_async().await?);
    }
    Ok(values)
}

pub async fn run() -> Result<serde_json::Value> {
    let ordinary =
        TensorDevice::new(WgpuRuntime::request_headless("conv.profile.ordinary").await?)?;
    let unavailable =
        ordinary.profile_convolution_vjps(|| -> std::result::Result<(), TensorError> {
            panic!("unsupported device ran capture closure")
        });
    assert!(matches!(
        unavailable,
        Err(TensorError::Runtime(
            WgpuRuntimeError::TimestampQueriesUnavailable
        ))
    ));
    drop(ordinary);
    let runtime = WgpuRuntime::request_profiled_headless("conv.profile.fixture").await?;
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let adapter = format!("{:?}", runtime.adapter_info());
    let device = TensorDevice::new(runtime.clone())?;
    let graph_device = TensorDevice::new(runtime)?;
    let x = graph_device
        .upload(&[1, 2, 2, 2], &[0.5, 1., 1.5, 2., -0.5, -1., -1.5, -2.])?
        .permute(&[0, 1, 3, 2])?;
    assert!(!x.layout().is_contiguous());
    let dense_w = graph_device.upload(&[3, 2, 1, 1], &[0.1, 0.2, 0.3, -0.1, -0.2, -0.3])?;
    let depth_w = graph_device.upload(&[2, 1, 1], &[0.25, -0.5])?;
    let dense_u = graph_device.upload(&[1, 3, 2, 2], &[0.125; 12])?;
    let depth_u = graph_device.upload(&[1, 2, 2, 2], &[0.25; 8])?;
    let dense = || x.conv2d_vjp(&dense_w, &dense_u, (1, 1), (0, 0), (1, 1));
    let depth = || x.depthwise_conv2d_vjp(&depth_w, &depth_u, (1, 1), (0, 0), (1, 1));
    let expected = [values(dense()?).await?, values(depth()?).await?];
    let (_, cancelled) = device.profile_convolution_vjps(dense)?;
    drop(cancelled);
    let (actual, pending) = device.profile_convolution_vjps(|| {
        assert!(device
            .profile_convolution_vjps(|| Ok::<_, TensorError>(()))
            .is_err());
        Ok::<_, TensorError>([dense()?, depth()?])
    })?;
    let [dense_actual, depth_actual] = actual;
    assert_eq!(
        [values(dense_actual).await?, values(depth_actual).await?],
        expected
    );
    let bad = device.upload(&[1, 2, 2, 2], &[f32::MAX; 8])?;
    let seed = device.upload(&[1, 3, 2, 2], &[4.; 12])?;
    let (_, rejected) = device
        .profile_convolution_vjps(|| bad.conv2d_vjp(&dense_w, &seed, (1, 1), (0, 0), (1, 1)))?;
    #[cfg(not(target_arch = "wasm32"))]
    let rejected = rejected.read();
    #[cfg(target_arch = "wasm32")]
    let rejected = rejected.read_async().await;
    assert!(matches!(rejected, Err(TensorError::NonFinite)));
    drop((
        x,
        dense_w,
        depth_w,
        dense_u,
        depth_u,
        bad,
        seed,
        graph_device,
        device,
    ));
    #[cfg(not(target_arch = "wasm32"))]
    let captured = pending.read()?;
    #[cfg(target_arch = "wasm32")]
    let captured = pending.read_async().await?;
    assert_eq!(captured.operations.len(), 2);
    assert_eq!(captured.operations[0].geometry.kind, "dense");
    assert_eq!(captured.operations[1].geometry.kind, "depthwise");
    assert!(captured
        .operations
        .iter()
        .all(|op| op.timestamps.passes.len() == 3
            && op
                .timestamps
                .passes
                .iter()
                .all(|p| p.elapsed_ns.is_finite() && p.elapsed_ns >= 0.)));
    Ok(
        serde_json::json!({"schema": "spiraltorch.convolution_profile_fixture.v1", "passed": true,
        "adapter": adapter, "checks": ["unsupported_before_closure", "runtime_clone_capture",
        "dense_depthwise_value_parity", "view_packing", "nested_capture_rejected",
        "nonfinite_guard_rejected", "retained_after_drop", "cancelled_reuse", "written_prefix_only"],
        "profile": captured.to_json_value()}),
    )
}
