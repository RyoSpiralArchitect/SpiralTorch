// SPDX-License-Identifier: AGPL-3.0-or-later

use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{
    current_tensor_util_route, push_backend_policy, AcceleratorFallback, BackendPolicy,
    ExecutionConfig,
};
use st_nn::layers::conv::DepthwiseConv2d;
use st_nn::module::Module;
use st_tensor::backend::wgpu_dense;
use st_tensor::Tensor;
use st_tensor::TensorUtilBackend;
use std::time::Instant;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn bench_shape(
    batch: usize,
    channels: usize,
    side: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let layer = DepthwiseConv2d::new(
        "vision.depthwise.bench",
        channels,
        (7, 7),
        (1, 1),
        (3, 3),
        (1, 1),
        (side, side),
    )?;
    let input = Tensor::from_fn(batch, channels * side * side, |row, col| {
        ((row * 37 + col * 17) % 257) as f32 / 256.0 - 0.5
    })?;
    let cpu_policy = BackendPolicy::from_device_caps_with_config(
        DeviceCaps::cpu(),
        ExecutionConfig::new(AcceleratorFallback::Forbid, 1),
    );
    let gpu_policy = BackendPolicy::from_device_caps_with_config(
        DeviceCaps::wgpu(32, true, 256),
        ExecutionConfig::new(AcceleratorFallback::Forbid, 1),
    );
    let mut cpu_ms = Vec::new();
    let mut gpu_ms = Vec::new();
    let mut max_abs_diff = 0.0f32;

    for iteration in 0..9 {
        let mut outputs = [None, None];
        for offset in 0..2 {
            let route = (iteration + offset) % 2;
            let policy = if route == 0 { cpu_policy } else { gpu_policy };
            let _guard = push_backend_policy(policy);
            if route == 1 && iteration == 0 {
                let work = batch * channels * side * side * 7 * 7;
                if !matches!(
                    current_tensor_util_route(work).selected_backend,
                    TensorUtilBackend::GpuWgpu
                ) {
                    return Err("WGPU benchmark policy did not select WGPU".into());
                }
            }
            let start = Instant::now();
            outputs[route] = Some(layer.forward(&input)?);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if iteration >= 2 {
                if route == 0 {
                    cpu_ms.push(elapsed_ms);
                } else {
                    gpu_ms.push(elapsed_ms);
                }
            }
        }
        let cpu = outputs[0].as_ref().unwrap();
        let gpu = outputs[1].as_ref().unwrap();
        assert_eq!(cpu.shape(), gpu.shape());
        for (&left, &right) in cpu.data().iter().zip(gpu.data()) {
            max_abs_diff = max_abs_diff.max((left - right).abs());
        }
    }
    if max_abs_diff > 1e-4 {
        return Err(format!("depthwise parity failed: max_abs_diff={max_abs_diff}").into());
    }
    let cpu_median = median(&mut cpu_ms);
    let gpu_median = median(&mut gpu_ms);
    println!(
        "shape={batch}x{channels}x{side}x{side} kernel=7x7 warmup=2 samples=7 max_abs_diff={max_abs_diff:.8}"
    );
    println!("cpu_ms={cpu_ms:?}");
    println!("wgpu_upload_compute_readback_ms={gpu_ms:?}");
    println!(
        "cpu_median_ms={cpu_median:.3} wgpu_median_ms={gpu_median:.3} wgpu_over_cpu={:.3}",
        gpu_median / cpu_median
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
        "vision.depthwise.conv2d.benchmark",
    )?;
    if !wgpu_dense::is_available() {
        return Err("native WGPU runtime unavailable; benchmark cannot fall back to CPU".into());
    }
    println!("adapter={:?}", runtime.adapter_info());
    bench_shape(1, 4, 32)?;
    bench_shape(1, 8, 48)?;
    bench_shape(1, 8, 64)?;
    bench_shape(2, 16, 64)?;
    bench_shape(4, 32, 128)?;
    Ok(())
}
