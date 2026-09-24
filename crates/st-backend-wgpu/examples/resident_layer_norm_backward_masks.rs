//! Compare GPU-resident LayerNorm input and affine backward paths independently.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime};
    use std::time::Instant;

    const STEPS: usize = 32;
    const WARMUP: usize = 2;
    const ITERATIONS: usize = 5;
    const ROUTES: [(&str, [bool; 3]); 3] = [
        ("input", [true, false, false]),
        ("affine", [false, true, true]),
        ("both", [true, true, true]),
    ];

    let (runtime, _) = runtime::ensure_default_runtime_blocking("layer_norm.backward.masks")?;
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("non-CPU GPU adapter required".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let mut cases = Vec::new();
    for (rows, cols) in [(2, 3), (32, 256), (128, 1025)] {
        let input: Vec<_> = (0..rows * cols)
            .map(|i| (((i * 37 + 17) % 257) as f32 - 128.) / 64.)
            .collect();
        let seed: Vec<_> = (0..rows * cols)
            .map(|i| (((i * 11 + 17) % 67) as f32 - 33.) / 64.)
            .collect();
        let input = device.upload(&[rows, cols], &input)?;
        let gain = device.upload(&[cols], &vec![1.; cols])?;
        let bias = device.upload(&[cols], &vec![0.; cols])?;
        let upstream = device.upload(&[rows, cols], &seed)?;
        let tape = input.layer_norm_affine(&gain, &bias, 1e-5)?;
        let reference = tape.backward(&upstream, 1., [true; 3])?;
        let reference = device
            .snapshot_many(&[
                reference[0].as_ref().unwrap(),
                reference[1].as_ref().unwrap(),
                reference[2].as_ref().unwrap(),
            ])?
            .read()?;
        let mut intervals = [Vec::new(), Vec::new(), Vec::new()];
        let mut max_scaled_error = 0f64;
        for iteration in 0..WARMUP + ITERATIONS {
            for offset in 0..ROUTES.len() {
                let route = (iteration + offset) % ROUTES.len();
                let mask = ROUTES[route].1;
                let start = Instant::now();
                let mut result = None;
                for _ in 0..STEPS {
                    result = Some(tape.backward(&upstream, 1., mask)?);
                }
                // Identical terminal readback fences every route without adding
                // route-dependent gradient transfer to the timed interval.
                device.snapshot_many(&[tape.value()])?.read()?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                let result = result.unwrap();
                let requested: Vec<_> = result.iter().filter_map(Option::as_ref).collect();
                let actual = device.snapshot_many(&requested)?.read()?;
                for ((index, _), values) in mask
                    .iter()
                    .enumerate()
                    .filter(|(_, enabled)| **enabled)
                    .zip(actual)
                {
                    for (&actual, &expected) in values.iter().zip(&reference[index]) {
                        let scaled = (f64::from(actual) - f64::from(expected)).abs()
                            / (5e-4 * (1. + f64::from(expected).abs()));
                        assert!(
                            scaled <= 1.,
                            "shape={rows}x{cols}, route={}, gradient={index}: {actual} != {expected}",
                            ROUTES[route].0
                        );
                        max_scaled_error = max_scaled_error.max(scaled);
                    }
                    assert_eq!(values.len(), reference[index].len());
                }
                if iteration >= WARMUP {
                    intervals[route].push(elapsed_ms);
                }
            }
        }
        let mut sorted = intervals.clone();
        for times in &mut sorted {
            times.sort_by(f64::total_cmp);
        }
        cases.push(serde_json::json!({
            "rows": rows,
            "cols": cols,
            "routes": ROUTES.iter().enumerate().map(|(i, (name, _))| serde_json::json!({
                "name": name,
                "intervals_ms": intervals[i],
                "median_ms": sorted[i][ITERATIONS / 2],
            })).collect::<Vec<_>>(),
            "max_scaled_error": max_scaled_error,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "spiraltorch.layer_norm.backward_masks.v1",
            "adapter": format!("{:?}", runtime.adapter_info()),
            "boundary": "32 repeated backward calls on one prepared tape; forward/upload excluded, identical terminal readback included; host wall time, not GPU-only",
            "steps": STEPS,
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "cases": cases,
        }))?
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("native GPU benchmark only");
}
