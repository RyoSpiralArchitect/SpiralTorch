// SPDX-License-Identifier: AGPL-3.0-or-later
// Part of SpiralTorch - Licensed under AGPL-3.0-or-later.

use st_backend_wgpu::resident_tensor::TensorDevice;
use st_backend_wgpu::runtime::ensure_default_runtime_blocking;
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig};
use st_nn::module::Module;
use st_tensor::Tensor;
use st_vision::models::{ConvNeXtBackbone, ConvNeXtConfig};
use std::time::Instant;

const WARMUPS: usize = 3;
const SAMPLES: usize = 11;

struct Case {
    name: &'static str,
    batch: usize,
    side: usize,
    stage_dims: &'static [usize],
    stage_depths: &'static [usize],
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn bench_case(case: Case, device: &TensorDevice) -> Result<(), Box<dyn std::error::Error>> {
    let model = ConvNeXtBackbone::new(ConvNeXtConfig {
        input_channels: 3,
        input_hw: (case.side, case.side),
        stage_dims: case.stage_dims.to_vec(),
        stage_depths: case.stage_depths.to_vec(),
        patch_size: (2, 2),
        curvature: -1.0,
        epsilon: 1e-6,
    })?;
    let host = Tensor::from_fn(case.batch, 3 * case.side * case.side, |row, col| {
        ((row * 41 + col * 17) % 101) as f32 / 101.0 - 0.5
    })?;
    let shape = [case.batch, 3, case.side, case.side];
    let resident = device.upload(&shape, host.data())?;
    let cpu_policy = BackendPolicy::from_device_caps_with_config(
        DeviceCaps::cpu(),
        ExecutionConfig::new(AcceleratorFallback::Forbid, 1),
    );
    let mut times = [Vec::new(), Vec::new(), Vec::new()];
    let mut max_abs_diff = [0.0f32; 2];

    for iteration in 0..WARMUPS + SAMPLES {
        let mut outputs: [Option<Vec<f32>>; 3] = [None, None, None];
        for offset in 0..3 {
            let route = (iteration + offset) % 3;
            let (values, elapsed_ms) = match route {
                0 => {
                    let _policy = push_backend_policy(cpu_policy);
                    let start = Instant::now();
                    let output = model.forward(&host)?;
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    (output.data().to_vec(), elapsed_ms)
                }
                1 => {
                    let start = Instant::now();
                    let output = model.forward_resident(&resident)?.snapshot()?.read()?;
                    (output, start.elapsed().as_secs_f64() * 1000.0)
                }
                2 => {
                    let start = Instant::now();
                    let uploaded = device.upload(&shape, host.data())?;
                    let output = model.forward_resident(&uploaded)?.snapshot()?.read()?;
                    (output, start.elapsed().as_secs_f64() * 1000.0)
                }
                _ => unreachable!(),
            };
            if iteration >= WARMUPS {
                times[route].push(elapsed_ms);
            }
            outputs[route] = Some(values);
        }
        let cpu = outputs[0].as_ref().unwrap();
        for route in 1..3 {
            let gpu = outputs[route].as_ref().unwrap();
            if cpu.len() != gpu.len() {
                return Err(
                    format!("{}: output length mismatch on route {route}", case.name).into(),
                );
            }
            for (&expected, &actual) in cpu.iter().zip(gpu) {
                if !expected.is_finite() || !actual.is_finite() {
                    return Err(format!(
                        "{}: non-finite CPU/WGPU output on route {route}",
                        case.name
                    )
                    .into());
                }
                let error = (expected - actual).abs();
                max_abs_diff[route - 1] = max_abs_diff[route - 1].max(error);
                if error > 2e-3 * (1.0 + expected.abs()) {
                    return Err(format!(
                        "{}: CPU/WGPU parity failed on route {route}: {expected} != {actual}",
                        case.name
                    )
                    .into());
                }
            }
        }
    }

    let cpu_ms = median(&times[0]);
    let resident_ms = median(&times[1]);
    let upload_ms = median(&times[2]);
    if !cpu_ms.is_finite() || cpu_ms <= 0.0 {
        return Err(format!("{}: invalid CPU median {cpu_ms}", case.name).into());
    }
    println!(
        "case={} shape={}x3x{}x{} dims={:?} depths={:?} warmups={WARMUPS} samples={SAMPLES}",
        case.name, case.batch, case.side, case.side, case.stage_dims, case.stage_depths
    );
    println!("cpu_ms={:?}", times[0]);
    println!("wgpu_resident_input_to_readback_ms={:?}", times[1]);
    println!("wgpu_upload_to_readback_ms={:?}", times[2]);
    println!(
        "median_ms cpu={cpu_ms:.3} resident={resident_ms:.3} upload={upload_ms:.3} resident_over_cpu={:.3} upload_over_cpu={:.3} max_abs_diff={max_abs_diff:?}",
        resident_ms / cpu_ms,
        upload_ms / cpu_ms
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (runtime, _) = ensure_default_runtime_blocking("vision.convnext_resident_bench")?;
    println!("adapter={:?}", runtime.adapter_info());
    let device = TensorDevice::new(runtime)?;
    for case in [
        Case {
            name: "small",
            batch: 1,
            side: 16,
            stage_dims: &[4, 8],
            stage_depths: &[1, 1],
        },
        Case {
            name: "medium_no_blocks",
            batch: 1,
            side: 32,
            stage_dims: &[8, 16],
            stage_depths: &[0, 0],
        },
        Case {
            name: "medium_first_block",
            batch: 1,
            side: 32,
            stage_dims: &[8, 16],
            stage_depths: &[1, 0],
        },
        Case {
            name: "medium_full",
            batch: 1,
            side: 32,
            stage_dims: &[8, 16],
            stage_depths: &[1, 1],
        },
        Case {
            name: "large_no_blocks",
            batch: 2,
            side: 64,
            stage_dims: &[16, 32],
            stage_depths: &[0, 0],
        },
        Case {
            name: "large_first_block",
            batch: 2,
            side: 64,
            stage_dims: &[16, 32],
            stage_depths: &[1, 0],
        },
        Case {
            name: "large_full",
            batch: 2,
            side: 64,
            stage_dims: &[16, 32],
            stage_depths: &[1, 1],
        },
    ] {
        bench_case(case, &device)?;
    }
    Ok(())
}
