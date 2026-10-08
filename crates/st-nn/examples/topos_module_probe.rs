// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded CPU NN forward/both-VJP probe, reusable against old and new binaries.

use serde_json::json;
use sha2::{Digest, Sha256};
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Module, OpenCartesianTopos, Tensor, ToposResonator, ToposResonatorConfig};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: topos_module_probe ROWS FEATURES ITERATIONS COUPLING ROUNDS NEW_JSON".into(),
        );
    }
    let rows: usize = args[1].parse()?;
    let features: usize = args[2].parse()?;
    let iterations: usize = args[3].parse()?;
    let coupling: f32 = args[4].parse()?;
    let rounds: usize = args[5].parse()?;
    let volume = rows.checked_mul(features).ok_or("shape overflow")?;
    if rows == 0
        || features == 0
        || volume > 1_048_576
        || !(1..=64).contains(&iterations)
        || !(2..=256).contains(&rounds)
        || rounds % 2 != 0
    {
        return Err("invalid shape or balanced round budget".into());
    }
    let config = ToposResonatorConfig::new(coupling, iterations)?;
    let report_path = PathBuf::from(&args[6]);
    let vectors_path = report_path.with_extension("f32le");
    if report_path.exists() || vectors_path.exists() {
        return Err("output already exists".into());
    }
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let topos =
        OpenCartesianTopos::new(-1.0, 1e-6, 1.0, iterations + 1, volume)?.with_porosity(0.2)?;
    let mut layer = ToposResonator::with_config_and_topos("gate", rows, features, config, topos)?;
    let make = |mul: usize, modulo: usize, scale: f32, shift: f32| {
        Tensor::from_vec(
            rows,
            features,
            (0..volume)
                .map(|i| ((i * mul) % modulo) as f32 * scale - shift)
                .collect(),
        )
    };
    let input = make(17, 31, 0.1, 1.5)?;
    let gate = make(7, 19, 0.1, 0.9)?;
    let upstream = make(13, 23, 0.07, 0.77)?;
    *layer.parameter_mut().value_mut() = gate.clone();
    // Match the zero-reset accumulation state used in every measured round,
    // including the signed-zero behavior of adding a gradient to +0.
    layer
        .parameter_mut()
        .accumulate_euclidean(&Tensor::zeros(rows, features)?)?;
    let output = layer.forward(&input)?;
    let grad_input = layer.backward(&input, &upstream)?;
    let grad_gate = layer
        .parameter()
        .gradient()
        .ok_or("missing gate gradient")?
        .clone();
    let forward_audit = layer.latest_audit().ok_or("missing forward audit")?;
    let backward_audit = layer
        .latest_backward_audit()
        .ok_or("missing backward audit")?;
    let expected = [
        bits(output.data()),
        bits(grad_input.data()),
        bits(grad_gate.data()),
    ];

    let mut timings = [Vec::new(), Vec::new()];
    let mut orders = Vec::new();
    for round in 0..(rounds + 2) {
        let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        if round >= 2 {
            orders.push(order);
        }
        for route in order {
            // Exclude gradient reset on both revisions; it also invalidates the cache.
            layer.parameter_mut().zero_gradient();
            let start = Instant::now();
            let actual = std::hint::black_box(layer.forward(&input)?);
            let dx = if route == 1 {
                Some(std::hint::black_box(layer.backward(&input, &upstream)?))
            } else {
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if round >= 2 {
                timings[route].push(elapsed);
            }
            assert_eq!(bits(actual.data()), expected[0]);
            assert_eq!(layer.latest_audit(), Some(forward_audit));
            if let Some(dx) = dx {
                assert_eq!(bits(dx.data()), expected[1]);
                assert_eq!(
                    bits(layer.parameter().gradient().unwrap().data()),
                    expected[2]
                );
                assert_eq!(layer.latest_backward_audit(), Some(backward_audit));
            }
        }
    }
    let mut vectors = Vec::with_capacity(volume * 6 * 4);
    let mut hashes = Vec::new();
    for tensor in [&input, &gate, &upstream, &output, &grad_input, &grad_gate] {
        let mut hash = Sha256::new();
        for value in tensor.data() {
            let bytes = value.to_le_bytes();
            vectors.extend_from_slice(&bytes);
            hash.update(bytes);
        }
        hashes.push(format!("{:x}", hash.finalize()));
    }
    let report = json!({
        "schema": "spiraltorch.topos_nn_probe.v1", "status": "measured",
        "shape": [rows, features], "dtype": "float32", "backend": "cpu",
        "config": {"iterations": iterations, "coupling": coupling, "saturation": 1.0, "porosity": 0.2_f32},
        "warmup_per_route": 2, "round_order": orders,
        "gradient_storage": "preallocated_zero_accumulator_for_reference_and_all_rounds",
        "measurements_ms": {"forward": timings[0], "forward_backward": timings[1]},
        "forward_audit": forward_audit, "backward_audit": backward_audit,
        "vector_order": ["input", "gate", "upstream", "output", "grad_input", "grad_gate"],
        "vector_sha256": hashes, "vectors_sha256": format!("{:x}", Sha256::digest(&vectors)),
        "scope": "Actual Rust NN CPU module, full per-element gate, forward or forward + both VJPs and audits. No optimizer update in timings; no throughput, accelerator or quality claim."
    });
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&vectors_path)?
        .write_all(&vectors)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    println!(
        "{}",
        json!({"status":"measured", "shape":[rows,features], "iterations":iterations})
    );
    Ok(())
}
