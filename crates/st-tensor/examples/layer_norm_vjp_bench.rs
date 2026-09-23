//! Exploratory host-to-host LayerNorm VJP: CPU, hybrid WGPU, resident WGPU.
#[cfg(all(feature = "wgpu_dense", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::runtime;
    use st_tensor::{Tensor, TensorUtilBackend};
    use std::time::Instant;

    let _strict = st_tensor::execution::push_accelerator_fallback(
        st_tensor::execution::AcceleratorFallback::Forbid,
    );
    let (runtime, _) = runtime::ensure_default_runtime_blocking("layer_norm.vjp_bench")?;
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
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
        let input = Tensor::from_vec(rows, cols, values(rows * cols, 37, 257, 64.))?;
        let gamma = Tensor::from_vec(1, cols, values(cols, 19, 127, 64.))?;
        let seed = Tensor::from_vec(rows, cols, values(rows * cols, 29, 193, 128.))?;
        let run = |route: usize| -> Result<[Tensor; 3], Box<dyn std::error::Error>> {
            if route == 2 {
                Ok(input
                    .layer_norm_affine_backward_resident(&gamma, &seed, 1e-5, 0.5, [true; 3])?
                    .map(Option::unwrap))
            } else {
                let backend = if route == 0 {
                    TensorUtilBackend::Cpu
                } else {
                    TensorUtilBackend::GpuWgpu
                };
                let (dx, dg, db) = input
                    .layer_norm_affine_backward_with_backend(&gamma, &seed, 1e-5, 0.5, backend)?;
                Ok([dx, dg, db])
            }
        };
        let oracle = run(0)?;
        let mut max_scaled_error = [0.0f64; 3];
        let mut intervals = vec![];
        for iteration in 0..21 {
            for j in 0..3 {
                let route = (iteration + j) % 3;
                let start = Instant::now();
                let output = run(route)?;
                let ms = start.elapsed().as_secs_f64() * 1000.;
                for (actual, expected) in output.iter().zip(&oracle) {
                    assert_eq!(actual.shape(), expected.shape());
                    for (&a, &b) in actual.data().iter().zip(expected.data()) {
                        let scaled = (f64::from(a) - f64::from(b)).abs()
                            / (2e-5 * (1. + f64::from(b).abs()));
                        assert!(
                            a.is_finite() && scaled <= 1.,
                            "route={route}, {rows}x{cols}: {a} != {b}"
                        );
                        max_scaled_error[route] = max_scaled_error[route].max(scaled);
                    }
                }
                if iteration >= 3 {
                    intervals.push(serde_json::json!({
                        "iteration": iteration - 3, "route": route, "ms": ms,
                    }));
                }
                std::hint::black_box(output);
            }
        }
        cases.push(serde_json::json!({
            "rows": rows, "cols": cols, "max_scaled_error": max_scaled_error,
            "intervals": intervals,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "spiraltorch.layer_norm.vjp_exploratory.v1",
            "adapter": format!("{:?}", runtime.adapter_info()),
            "scope": "host-to-host with input uploads for WGPU routes, all three requested VJPs and CPU-owned outputs; no forward affine output; shared-machine exploratory timings, not a speed guarantee",
            "routes": ["cpu_centered_f64", "hybrid_wgpu", "resident_statistics_wgpu"],
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
