// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic variable-batch shared-gate learning, not a performance benchmark.

use serde_json::json;
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Module, OpenCartesianTopos, Sequential, Tensor, ToposResonator, ToposResonatorConfig};
use st_tensor::Layout;
use std::fs::OpenOptions;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: topos_shared_gate_probe NEW_JSON".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let features = 5;
    let learning_rate = 0.03_f32;
    let initial_gate = [0.7_f32, -0.8, 0.4, 1.2, -1.1];
    let mut cases = Vec::new();
    for porosity in [0.0_f32, 0.3] {
        let config = ToposResonatorConfig::new(0.2, 5)?;
        let topos =
            OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 16, 8 * features)?.with_porosity(porosity)?;
        let mut layer = ToposResonator::with_shared_gate("gate", features, config, topos)?;
        *layer.parameter_mut().value_mut() = Tensor::from_vec(1, features, initial_gate.to_vec())?;
        let mut model = Sequential::new();
        model.push(st_nn::layers::Identity);
        model.push(layer);
        model.push(st_nn::layers::Identity);
        let mut records = Vec::new();
        for step in 0..100 {
            let rows = [1, 3, 8, 2][step % 4];
            let layout = if step % 2 == 0 {
                Layout::RowMajor
            } else {
                Layout::ColMajor
            };
            let input = Tensor::from_fn(rows, features, |r, c| {
                ((r * 13 + c * 7 + step * 3) % 29) as f32 * 0.11 - 1.5
            })?;
            let target = Tensor::from_fn(rows, features, |r, c| {
                ((r * 5 + c * 11) % 23) as f32 * 0.02 - 0.22
            })?;
            let logical_input = input.to_layout(layout)?;
            let output = model.forward(&logical_input)?;
            let loss = output
                .data()
                .iter()
                .zip(target.data())
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>()
                / (rows * features) as f64;
            let upstream = Tensor::from_vec(
                rows,
                features,
                output
                    .data()
                    .iter()
                    .zip(target.data())
                    .map(|(a, b)| 2.0 * (a - b) / (rows * features) as f32)
                    .collect(),
            )?;
            let dx = model.backward(&logical_input, &upstream.to_layout(layout)?)?;
            let mut grad_gate = Vec::new();
            let mut gate_after = Vec::new();
            model.visit_parameters_mut(&mut |parameter| {
                assert_eq!(parameter.value().shape(), (1, features));
                grad_gate = parameter
                    .gradient()
                    .expect("shared gate gradient")
                    .data()
                    .to_vec();
                gate_after = parameter
                    .value()
                    .data()
                    .iter()
                    .zip(&grad_gate)
                    .map(|(gate, gradient)| gate - learning_rate * gradient)
                    .collect();
                *parameter.value_mut() = Tensor::from_vec(1, features, gate_after.clone())?;
                parameter.zero_gradient();
                Ok(())
            })?;
            records.push(json!({
                "step": step, "shape": [rows, features],
                "input_layout": if layout == Layout::RowMajor { "row_major" } else { "col_major" },
                "input": input.data(), "target": target.data(), "loss": loss,
                "output": output.data(), "grad_input": dx.data(),
                "grad_gate": grad_gate, "gate_after": gate_after,
            }));
        }
        cases.push(json!({
            "config": {"coupling": config.coupling(), "iterations": config.iterations(),
                "saturation": 1.0, "porosity": porosity},
            "initial_gate": initial_gate, "records": records,
        }));
    }
    let report = json!({
        "schema": "spiraltorch.topos_shared_gate_learning.v1", "status": "executed",
        "backend": "cpu", "dtype": "float32", "learning_rate": learning_rate,
        "gate_layout": "shared_rows", "gate_gradient_reduction": "sum_without_additional_mean",
        "cases": cases,
        "scope": "Rust NN Sequential, two synthetic 100-update SGD trajectories with one gate per feature and variable rows. Sequential backward recaptures its forward activations. No timing, GPU residency, pretrained-model or quality claim."
    });
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    println!(
        "{}",
        json!({"status": "executed", "cases": 2, "updates": 200})
    );
    Ok(())
}
