//! Matched 32-step affine LayerNorm + MSE SGD with and without host transfers.
#[cfg(all(feature = "wgpu_dense", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{
        resident_tensor::{ResidentTensor, TensorDevice},
        runtime,
    };
    use st_tensor::{LayerNormBackend, Tensor, TensorUtilBackend};
    use std::time::Instant;

    const STEPS: usize = 32;
    const WARMUP: usize = 3;
    const ITERATIONS: usize = 9;
    const EPSILON: f32 = 1e-5;
    const SCALED_TOLERANCE: f64 = 5e-4;
    type BenchResult = Result<(Vec<Vec<f32>>, usize), Box<dyn std::error::Error>>;

    fn values(n: usize, multiplier: usize, modulus: usize, divisor: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * multiplier + 17) % modulus) as f32 - (modulus / 2) as f32) / divisor)
            .collect()
    }

    fn digest(values: &[f32]) -> String {
        let mut hash = 0xcbf29ce484222325u64;
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        format!("{hash:016x}")
    }

    fn run_cpu(
        input: &Tensor,
        target: &Tensor,
        rows: usize,
        cols: usize,
        rate: f32,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut gamma = Tensor::from_vec(1, cols, vec![1.; cols])?;
        let mut beta = Tensor::from_vec(1, cols, vec![0.; cols])?;
        let mut final_result = Vec::new();
        for _ in 0..STEPS {
            let prediction = input.layer_norm_affine_with_backend(
                &gamma,
                &beta,
                EPSILON,
                LayerNormBackend::Cpu,
            )?;
            let difference: Vec<_> = prediction
                .data()
                .iter()
                .zip(target.data())
                .map(|(&value, &expected)| value - expected)
                .collect();
            let loss = (difference
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>()
                / (rows * cols) as f64) as f32;
            let seed = Tensor::from_vec(
                rows,
                cols,
                difference
                    .iter()
                    .map(|&value| 2. * value / (rows * cols) as f32)
                    .collect(),
            )?;
            let (_, dg, db) = input.layer_norm_affine_backward_with_backend(
                &gamma,
                &seed,
                EPSILON,
                1.,
                TensorUtilBackend::Cpu,
            )?;
            let next_gamma: Vec<_> = gamma
                .data()
                .iter()
                .zip(dg.data())
                .map(|(&value, &gradient)| value + rate * gradient)
                .collect();
            let next_beta: Vec<_> = beta
                .data()
                .iter()
                .zip(db.data())
                .map(|(&value, &gradient)| value + rate * gradient)
                .collect();
            gamma = Tensor::from_vec(1, cols, next_gamma)?;
            beta = Tensor::from_vec(1, cols, next_beta)?;
            final_result = vec![
                vec![loss],
                gamma.data().to_vec(),
                beta.data().to_vec(),
                dg.data().to_vec(),
                db.data().to_vec(),
            ];
        }
        Ok(final_result)
    }

    fn run_gpu(
        input: ResidentTensor,
        target: ResidentTensor,
        mut gamma: ResidentTensor,
        mut beta: ResidentTensor,
        rate: &ResidentTensor,
        requested: [bool; 3],
    ) -> BenchResult {
        let mut final_loss = None;
        let mut final_dg = None;
        let mut final_db = None;
        for _ in 0..STEPS {
            let tape = input.layer_norm_affine(&gamma, &beta, EPSILON)?;
            let loss = tape.value().mean_squared_error(&target)?;
            let [_, dg, db] = tape.backward(loss.prediction_gradient(), 1., requested)?;
            let dg = dg.unwrap();
            let db = db.unwrap();
            gamma = gamma.add(&dg.mul(rate)?)?;
            beta = beta.add(&db.mul(rate)?)?;
            final_loss = Some(loss);
            final_dg = Some(dg);
            final_db = Some(db);
        }
        let pending = input.device().snapshot_many(&[
            final_loss.as_ref().unwrap().value(),
            &gamma,
            &beta,
            final_dg.as_ref().unwrap(),
            final_db.as_ref().unwrap(),
        ])?;
        let maps = pending.staging_buffer_count();
        Ok((pending.read()?, maps))
    }

    let _strict = st_tensor::execution::push_accelerator_fallback(
        st_tensor::execution::AcceleratorFallback::Forbid,
    );
    let (runtime, _) = runtime::ensure_default_runtime_blocking("layer_norm.training_residency")?;
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime.clone())?;
    let mut cases = Vec::new();
    for (rows, cols) in [
        (2, 3),
        (8, 257),
        (32, 256),
        (64, 768),
        (128, 1025),
        (256, 256),
    ] {
        let input_values = values(rows * cols, 37, 257, 64.);
        let target_values = values(rows * cols, 11, 67, 64.);
        let input = Tensor::from_vec(rows, cols, input_values.clone())?;
        let target = Tensor::from_vec(rows, cols, target_values.clone())?;
        let initial_prediction = input.layer_norm_affine_with_backend(
            &Tensor::from_vec(1, cols, vec![1.; cols])?,
            &Tensor::from_vec(1, cols, vec![0.; cols])?,
            EPSILON,
            LayerNormBackend::Cpu,
        )?;
        let initial_loss = (initial_prediction
            .data()
            .iter()
            .zip(target.data())
            .map(|(&value, &expected)| f64::from(value - expected).powi(2))
            .sum::<f64>()
            / (rows * cols) as f64) as f32;
        let rate = -0.1 * cols as f32;
        let gpu_input = device.upload(&[rows, cols], &input_values)?;
        let gpu_target = device.upload(&[rows, cols], &target_values)?;
        let gpu_gamma = device.upload(&[cols], &vec![1.; cols])?;
        let gpu_beta = device.upload(&[cols], &vec![0.; cols])?;
        let gpu_rate = device.upload(&[], &[rate])?;
        let run = |route| -> BenchResult {
            match route {
                0 => Ok((run_cpu(&input, &target, rows, cols, rate)?, 0)),
                1 => run_gpu(
                    device.upload(&[rows, cols], &input_values)?,
                    device.upload(&[rows, cols], &target_values)?,
                    device.upload(&[cols], &vec![1.; cols])?,
                    device.upload(&[cols], &vec![0.; cols])?,
                    &device.upload(&[], &[rate])?,
                    [true; 3],
                ),
                2 => run_gpu(
                    gpu_input.clone(),
                    gpu_target.clone(),
                    gpu_gamma.clone(),
                    gpu_beta.clone(),
                    &gpu_rate,
                    [true; 3],
                ),
                3 => run_gpu(
                    gpu_input.clone(),
                    gpu_target.clone(),
                    gpu_gamma.clone(),
                    gpu_beta.clone(),
                    &gpu_rate,
                    [false, true, true],
                ),
                _ => unreachable!(),
            }
        };
        let (oracle, _) = run(0)?;
        assert!(oracle[0][0] < initial_loss);
        let mut intervals = Vec::new();
        let mut max_scaled_error = [0f64; 4];
        let mut terminal_maps = [0usize; 4];
        let mut final_outputs = [None, None, None, None];
        for iteration in 0..WARMUP + ITERATIONS {
            for j in 0..4 {
                let route = (iteration + j) % 4;
                let start = Instant::now();
                let (result, maps) = run(route)?;
                let ms = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(result.len(), oracle.len());
                for (actual, expected) in result.iter().zip(&oracle) {
                    assert_eq!(actual.len(), expected.len());
                    for (&value, &reference) in actual.iter().zip(expected) {
                        let scaled = (f64::from(value) - f64::from(reference)).abs()
                            / (SCALED_TOLERANCE * (1. + f64::from(reference).abs()));
                        assert!(
                            value.is_finite() && scaled <= 1.,
                            "route={route}, shape={rows}x{cols}: {value} != {reference}"
                        );
                        max_scaled_error[route] = max_scaled_error[route].max(scaled);
                    }
                }
                assert_eq!(maps, usize::from(route != 0));
                if iteration >= WARMUP {
                    intervals.push(serde_json::json!({
                        "iteration": iteration - WARMUP, "route": route, "ms": ms,
                    }));
                }
                terminal_maps[route] = maps;
                final_outputs[route] = Some(result);
            }
        }
        let profile_tape = gpu_input.layer_norm_affine(&gpu_gamma, &gpu_beta, EPSILON)?;
        let profile_loss = profile_tape.value().mean_squared_error(&gpu_target)?;
        let profile_gradients =
            profile_tape.backward(profile_loss.prediction_gradient(), 1., [true; 3])?;
        device
            .snapshot_many(&[profile_loss.value(), profile_gradients[0].as_ref().unwrap()])?
            .read()?;
        let profile_stage = |stage| -> Result<(), Box<dyn std::error::Error>> {
            let values = match stage {
                0 => device.snapshot_many(&[&gpu_input])?.read()?,
                1 => {
                    let tape = gpu_input.layer_norm_affine(&gpu_gamma, &gpu_beta, EPSILON)?;
                    device.snapshot_many(&[tape.value()])?.read()?
                }
                2 => {
                    let loss = profile_tape.value().mean_squared_error(&gpu_target)?;
                    device
                        .snapshot_many(&[loss.prediction_gradient()])?
                        .read()?
                }
                3 => {
                    let gradients =
                        profile_tape.backward(profile_loss.prediction_gradient(), 1., [true; 3])?;
                    device
                        .snapshot_many(&gradients.iter().flatten().collect::<Vec<_>>())?
                        .read()?
                }
                4 => {
                    let gradients = profile_tape.backward(
                        profile_loss.prediction_gradient(),
                        1.,
                        [true, false, false],
                    )?;
                    device
                        .snapshot_many(&[gradients[0].as_ref().unwrap()])?
                        .read()?
                }
                5 => {
                    let gradients = profile_tape.backward(
                        profile_loss.prediction_gradient(),
                        1.,
                        [false, true, true],
                    )?;
                    device
                        .snapshot_many(&[
                            &gpu_input,
                            gradients[1].as_ref().unwrap(),
                            gradients[2].as_ref().unwrap(),
                        ])?
                        .read()?
                }
                6 => {
                    let next_gamma =
                        gpu_gamma.add(&profile_gradients[1].as_ref().unwrap().mul(&gpu_rate)?)?;
                    let next_beta =
                        gpu_beta.add(&profile_gradients[2].as_ref().unwrap().mul(&gpu_rate)?)?;
                    device
                        .snapshot_many(&[&gpu_input, &next_gamma, &next_beta])?
                        .read()?
                }
                _ => unreachable!(),
            };
            assert!(values.iter().flatten().all(|v| v.is_finite()));
            std::hint::black_box(values);
            Ok(())
        };
        let mut stage_intervals = Vec::new();
        for iteration in 0..WARMUP + ITERATIONS {
            for j in 0..7 {
                let stage = (iteration + j) % 7;
                let start = Instant::now();
                profile_stage(stage)?;
                let ms = start.elapsed().as_secs_f64() * 1000.;
                if iteration >= WARMUP {
                    stage_intervals.push(serde_json::json!({
                        "iteration": iteration - WARMUP, "stage": stage, "ms": ms,
                    }));
                }
            }
        }
        cases.push(serde_json::json!({
            "rows": rows, "cols": cols,
            "initial_loss": initial_loss,
            "input_fnv64": digest(&input_values), "target_fnv64": digest(&target_values),
            "max_scaled_error": max_scaled_error, "terminal_maps": terminal_maps,
            "intervals": intervals, "final_outputs": final_outputs,
            "stage_intervals": stage_intervals,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "spiraltorch.layer_norm.training_residency_exploratory.v1",
            "adapter": format!("{:?}", runtime.adapter_info()),
            "scope": "32-step forward+MSE+all VJPs+SGD; host-to-host includes initial uploads and one terminal batch, preloaded excludes initial uploads; all routes return CPU-owned final loss, parameters and affine gradients",
        "routes": ["rust_cpu_all", "wgpu_host_to_host_all", "wgpu_preloaded_all", "wgpu_preloaded_affine_only"],
        "stage_routes": ["readback_only", "forward", "mse", "backward_all", "backward_input", "backward_affine", "update"],
            "steps": STEPS, "warmup": WARMUP, "iterations": ITERATIONS,
            "epsilon": EPSILON, "rate": "-0.1 * cols", "scaled_tolerance": SCALED_TOLERANCE,
            "cases": cases,
        }))?
    );
    Ok(())
}

#[cfg(any(not(feature = "wgpu_dense"), target_arch = "wasm32"))]
fn main() {
    eprintln!("build with --features wgpu_dense");
}
