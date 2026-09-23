//! Exploratory serial versus grouped terminal readback for resident LayerNorm.
#[cfg(all(feature = "wgpu_dense", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{
        resident_tensor::{ResidentTensor, TensorDevice},
        runtime,
    };
    use st_tensor::{LayerNormBackend, Tensor, TensorUtilBackend};
    use std::time::Instant;

    fn resident_outputs(
        device: &TensorDevice,
        rows: usize,
        cols: usize,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        seed: &[f32],
    ) -> Result<[ResidentTensor; 4], Box<dyn std::error::Error>> {
        let x = device.upload(&[rows, cols], x)?;
        let gamma = device.upload(&[cols], gamma)?;
        let beta = device.upload(&[cols], beta)?;
        let seed = device.upload(&[rows, cols], seed)?;
        let tape = x.layer_norm_affine(&gamma, &beta, 1e-5)?;
        let [dx, dg, db] = tape.backward(&seed, 0.5, [true; 3])?;
        Ok([tape.value().clone(), dx.unwrap(), dg.unwrap(), db.unwrap()])
    }

    fn read(
        device: &TensorDevice,
        outputs: &[ResidentTensor; 4],
        grouped: bool,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if grouped {
            Ok(device
                .snapshot_many(&[&outputs[0], &outputs[1], &outputs[2], &outputs[3]])?
                .read()?)
        } else {
            outputs
                .iter()
                .map(|tensor| Ok(tensor.snapshot()?.read()?))
                .collect()
        }
    }

    let (runtime, _) = runtime::ensure_default_runtime_blocking("layer_norm.readback_bench")?;
    let device = TensorDevice::new(runtime.clone())?;
    let mut cases = vec![];
    for (rows, cols) in [
        (2, 3),
        (8, 257),
        (32, 256),
        (64, 768),
        (128, 1025),
        (256, 256),
    ] {
        let values = |n, multiplier, modulus, divisor| {
            (0..n)
                .map(|i| {
                    (((i * multiplier + 17) % modulus) as f32 - (modulus / 2) as f32) / divisor
                })
                .collect::<Vec<_>>()
        };
        let x = values(rows * cols, 37, 257, 64.);
        let gamma = values(cols, 19, 127, 64.);
        let beta = values(cols, 7, 31, 64.);
        let seed = values(rows * cols, 29, 193, 128.);
        let cpu_x = Tensor::from_vec(rows, cols, x.clone())?;
        let cpu_gamma = Tensor::from_vec(1, cols, gamma.clone())?;
        let cpu_beta = Tensor::from_vec(1, cols, beta.clone())?;
        let cpu_seed = Tensor::from_vec(rows, cols, seed.clone())?;
        let y = cpu_x.layer_norm_affine_with_backend(
            &cpu_gamma,
            &cpu_beta,
            1e-5,
            LayerNormBackend::Cpu,
        )?;
        let (dx, dg, db) = cpu_x.layer_norm_affine_backward_with_backend(
            &cpu_gamma,
            &cpu_seed,
            1e-5,
            0.5,
            TensorUtilBackend::Cpu,
        )?;
        let oracle = [y, dx, dg, db].map(|tensor| tensor.data().to_vec());
        let precomputed = resident_outputs(&device, rows, cols, &x, &gamma, &beta, &seed)?;
        let mut scopes = vec![];
        for scope in ["terminal_readback_only", "upload_forward_backward_readback"] {
            let mut intervals = vec![];
            let mut max_scaled_error = [0.0f64; 2];
            for iteration in 0..21 {
                for j in 0..2 {
                    let route = (iteration + j) % 2;
                    let start = Instant::now();
                    let outputs = if scope == "terminal_readback_only" {
                        None
                    } else {
                        Some(resident_outputs(
                            &device, rows, cols, &x, &gamma, &beta, &seed,
                        )?)
                    };
                    let actual = read(
                        &device,
                        outputs.as_ref().unwrap_or(&precomputed),
                        route == 1,
                    )?;
                    let ms = start.elapsed().as_secs_f64() * 1000.;
                    for (result, expected) in actual.iter().zip(&oracle) {
                        assert_eq!(result.len(), expected.len());
                        for (&a, &b) in result.iter().zip(expected) {
                            let scaled = (f64::from(a) - f64::from(b)).abs()
                                / (2e-5 * (1. + f64::from(b).abs()));
                            assert!(
                                a.is_finite() && scaled <= 1.,
                                "{scope} {rows}x{cols}: {a} != {b}"
                            );
                            max_scaled_error[route] = max_scaled_error[route].max(scaled);
                        }
                    }
                    if iteration >= 3 {
                        intervals.push(serde_json::json!({
                            "iteration": iteration - 3, "route": route, "ms": ms,
                        }));
                    }
                    std::hint::black_box(actual);
                }
            }
            scopes.push(serde_json::json!({
                "scope": scope, "max_scaled_error": max_scaled_error, "intervals": intervals,
            }));
        }
        cases.push(serde_json::json!({"rows": rows, "cols": cols, "scopes": scopes}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "spiraltorch.layer_norm.readback_exploratory.v1",
            "adapter": format!("{:?}", runtime.adapter_info()),
            "scope": "Four CPU-owned outputs. Serial maps versus one grouped map; fixed inputs; shared-machine exploratory timings, not a speed guarantee or PyTorch comparison.",
            "routes": ["serial_snapshot_read", "grouped_snapshot_many_read"],
            "epsilon": 1e-5, "parameter_gradient_scale": 0.5,
            "warmup": 3, "iterations": 18, "cases": cases,
        }))?
    );
    Ok(())
}

#[cfg(any(not(feature = "wgpu_dense"), target_arch = "wasm32"))]
fn main() {
    eprintln!("build with --features wgpu_dense");
}
