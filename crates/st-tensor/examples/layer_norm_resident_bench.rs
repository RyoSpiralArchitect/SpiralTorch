//! Exploratory host-to-host LayerNorm forward + all three VJPs. No routing change.
#[cfg(all(feature = "wgpu_dense", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime};
    use st_tensor::{LayerNormBackend, Tensor, TensorUtilBackend};
    use std::time::Instant;
    let _strict = st_tensor::execution::push_accelerator_fallback(
        st_tensor::execution::AcceleratorFallback::Forbid,
    );
    let (runtime, _) = runtime::ensure_default_runtime_blocking("layer_norm.exploratory_bench")?;
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
        let x = Tensor::from_vec(rows, cols, values(rows * cols, 37, 257, 64.))?;
        let g = Tensor::from_vec(1, cols, values(cols, 19, 127, 64.))?;
        let b = Tensor::from_vec(1, cols, values(cols, 7, 31, 64.))?;
        let seed = Tensor::from_vec(rows, cols, values(rows * cols, 29, 193, 128.))?;
        let evaluate = |route: usize| -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
            if route < 2 {
                let backend = if route == 0 {
                    TensorUtilBackend::Cpu
                } else {
                    TensorUtilBackend::GpuWgpu
                };
                let forward = if route == 0 {
                    LayerNormBackend::Cpu
                } else {
                    LayerNormBackend::GpuWgpu
                };
                let y = x.layer_norm_affine_with_backend(&g, &b, 1e-5, forward)?;
                let (dx, dg, db) =
                    x.layer_norm_affine_backward_with_backend(&g, &seed, 1e-5, 0.5, backend)?;
                Ok([y, dx, dg, db].iter().map(|v| v.data().to_vec()).collect())
            } else {
                let x = device.upload(&[rows, cols], x.data())?;
                let g = device.upload(&[cols], g.data())?;
                let b = device.upload(&[cols], b.data())?;
                let seed = device.upload(&[rows, cols], seed.data())?;
                let tape = x.layer_norm_affine(&g, &b, 1e-5)?;
                let grads = tape.backward(&seed, 0.5, [true; 3])?;
                let mut output = vec![tape.value().snapshot()?.read()?];
                for grad in grads {
                    output.push(grad.unwrap().snapshot()?.read()?);
                }
                Ok(output)
            }
        };
        let oracle = evaluate(0)?;
        let mut max_scaled = [0.0f64; 3];
        let mut intervals = vec![];
        for iteration in 0..21 {
            for j in 0..3 {
                let route = (iteration + j) % 3;
                let start = Instant::now();
                let output = evaluate(route)?;
                let ms = start.elapsed().as_secs_f64() * 1000.;
                for (actual, expected) in output.iter().zip(&oracle) {
                    for (&a, &b) in actual.iter().zip(expected) {
                        let scaled = (f64::from(a) - f64::from(b)).abs()
                            / (2e-5 * (1. + f64::from(b).abs()));
                        assert!(
                            a.is_finite() && scaled <= 1.,
                            "route={route}, shape={rows}x{cols}: {a} != {b}"
                        );
                        max_scaled[route] = max_scaled[route].max(scaled);
                    }
                }
                if iteration >= 3 {
                    intervals.push(
                        serde_json::json!({"iteration": iteration - 3, "route": route, "ms": ms}),
                    );
                }
                std::hint::black_box(output);
            }
        }
        cases.push(serde_json::json!({"rows": rows, "cols": cols, "max_scaled_error": max_scaled, "intervals": intervals}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "spiraltorch.layer_norm.exploratory.v1", "adapter": format!("{:?}", runtime.adapter_info()),
            "scope": "host-to-host; input uploads + affine forward + all VJPs + four CPU-owned outputs; exploratory shared-machine timings, not a speed guarantee",
            "routes": ["cpu_centered_f64", "legacy_hybrid_wgpu", "resident_wide_wgpu"],
            "epsilon": 1e-5, "parameter_gradient_scale": 0.5, "warmup": 3, "iterations": 18, "cases": cases,
        }))?
    );
    Ok(())
}

#[cfg(any(not(feature = "wgpu_dense"), target_arch = "wasm32"))]
fn main() {
    eprintln!("build with --features wgpu_dense");
}
