// SPDX-License-Identifier: AGPL-3.0-or-later

use st_backend_wgpu::resident_tensor::TensorDevice;
use st_backend_wgpu::runtime::ensure_default_runtime_blocking;
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig};
use st_nn::layers::conv::DepthwiseConv2d;
use st_nn::module::Module;
use st_tensor::Tensor;
use std::time::Instant;

const WARMUPS: usize = 3;
const SAMPLES: usize = 11;

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn bench_case(
    device: &TensorDevice,
    batch: usize,
    channels: usize,
    side: usize,
    kernel: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut layer = DepthwiseConv2d::new(
        "vision.depthwise.vjp.bench",
        channels,
        (kernel, kernel),
        (1, 1),
        (kernel / 2, kernel / 2),
        (1, 1),
        (side, side),
    )?;
    let input = Tensor::from_fn(batch, channels * side * side, |row, col| {
        ((row * 41 + col * 17) % 101) as f32 / 101.0 - 0.5
    })?;
    let seed = Tensor::from_fn(batch, channels * side * side, |row, col| {
        ((row * 13 + col * 7) % 89) as f32 / 89.0 - 0.5
    })?;
    let shape = [batch, channels, side, side];
    let resident_input = device.upload(&shape, input.data())?;
    let resident_seed = device.upload(&shape, seed.data())?;
    layer.forward_resident(&resident_input)?;
    let cpu_policy = BackendPolicy::from_device_caps_with_config(
        DeviceCaps::cpu(),
        ExecutionConfig::new(AcceleratorFallback::Forbid, 1),
    );
    let mut times = [Vec::new(), Vec::new()];
    let mut max_abs_diff = 0.0f32;
    for iteration in 0..WARMUPS + SAMPLES {
        let mut outputs: [Option<[Vec<f32>; 3]>; 2] = [None, None];
        for offset in 0..2 {
            let route = (iteration + offset) % 2;
            let (values, elapsed_ms) = if route == 0 {
                let _policy = push_backend_policy(cpu_policy);
                layer.visit_parameters_mut(&mut |parameter| {
                    parameter.zero_gradient();
                    Ok(())
                })?;
                let start = Instant::now();
                let input_gradient = layer.backward(&input, &seed)?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                let mut gradients = vec![input_gradient.data().to_vec()];
                layer.visit_parameters(&mut |parameter| {
                    gradients.push(parameter.gradient().unwrap().data().to_vec());
                    Ok(())
                })?;
                (gradients.try_into().unwrap(), elapsed_ms)
            } else {
                let start = Instant::now();
                let gradients = layer.vjp_resident(&resident_input, &resident_seed)?;
                let values = [
                    gradients[0].snapshot()?.read()?,
                    gradients[1].snapshot()?.read()?,
                    gradients[2].snapshot()?.read()?,
                ];
                (values, start.elapsed().as_secs_f64() * 1000.0)
            };
            if iteration >= WARMUPS {
                times[route].push(elapsed_ms);
            }
            outputs[route] = Some(values);
        }
        let cpu = outputs[0].as_ref().unwrap();
        let gpu = outputs[1].as_ref().unwrap();
        for (left, right) in cpu.iter().zip(gpu) {
            if left.len() != right.len() {
                return Err("depthwise VJP gradient length mismatch".into());
            }
            for (&expected, &actual) in left.iter().zip(right) {
                if !expected.is_finite() || !actual.is_finite() {
                    return Err("non-finite depthwise VJP gradient".into());
                }
                let error = (expected - actual).abs();
                max_abs_diff = max_abs_diff.max(error);
                if error > 2e-3 * (1.0 + expected.abs()) {
                    return Err("depthwise VJP CPU/WGPU parity failed".into());
                }
            }
        }
    }
    let cpu_ms = median(&times[0]);
    let gpu_ms = median(&times[1]);
    println!(
        "shape={batch}x{channels}x{side}x{side} kernel={kernel}x{kernel} warmups={WARMUPS} samples={SAMPLES}"
    );
    println!("cpu_backward_ms={:?}", times[0]);
    println!("resident_vjp_to_readback_ms={:?}", times[1]);
    println!(
        "median_ms cpu_backward={cpu_ms:.3} resident_vjp_to_readback={gpu_ms:.3} resident_over_cpu={:.3} max_abs_diff={max_abs_diff:.6}",
        gpu_ms / cpu_ms
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (runtime, _) = ensure_default_runtime_blocking("vision.depthwise_vjp.bench")?;
    println!("adapter={:?}", runtime.adapter_info());
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("depthwise VJP benchmark requires a non-CPU adapter".into());
    }
    let device = TensorDevice::new(runtime)?;
    bench_case(&device, 1, 4, 16, 3)?;
    bench_case(&device, 2, 16, 32, 7)?;
    Ok(())
}
