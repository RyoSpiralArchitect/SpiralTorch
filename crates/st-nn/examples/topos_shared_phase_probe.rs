// SPDX-License-Identifier: AGPL-3.0-or-later
//! Diagnostic timings of overlapping public scopes, not additive phase accounting.

use serde_json::json;
use st_core::backend::device_caps::DeviceCaps;
use st_core::dynamics::topos_resonator::{
    validate_topos_resonator_state_with_layout, ToposResonatorGateLayout, ToposResonatorOperator,
    ToposResonatorRequest,
};
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Module, OpenCartesianTopos, Tensor, ToposResonator, ToposResonatorConfig};
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: topos_shared_phase_probe ROWS FEATURES ITERATIONS NEW_JSON".into());
    }
    let rows: usize = args[1].parse()?;
    let features: usize = args[2].parse()?;
    let iterations: usize = args[3].parse()?;
    let volume = rows.checked_mul(features).ok_or("shape overflow")?;
    if rows == 0 || features == 0 || volume > 1_048_576 || !(1..=64).contains(&iterations) {
        return Err("shape/iteration budget".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let config = ToposResonatorConfig::new(0.25, iterations)?;
    let topos =
        OpenCartesianTopos::new(-1.0, 1e-6, 1.0, iterations + 1, volume)?.with_porosity(0.2)?;
    let make = |r: usize, mul: usize, modulo: usize, scale: f32, shift: f32| {
        Tensor::from_fn(r, features, |r, c| {
            (((r * features + c) * mul) % modulo) as f32 * scale - shift
        })
    };
    let input = make(rows, 17, 31, 0.1, 1.5)?;
    let gate = make(1, 7, 19, 0.1, 0.9)?;
    let upstream = make(rows, 13, 23, 0.07, 0.77)?;
    let operator = ToposResonatorOperator::new(config, topos.clone())?;
    let captured = operator.capture_shared_rows(input.data(), gate.data(), rows, features)?;
    let mut layer = ToposResonator::from_shared_gate("gate", gate.clone(), config, topos.clone())?;
    let output = layer.forward(&input)?;
    let dx = layer.backward(&input, &upstream)?;
    let expected = [
        bits(output.data()),
        bits(dx.data()),
        bits(layer.parameter().gradient().unwrap().data()),
    ];
    assert_eq!(bits(captured.output()), expected[0]);
    assert_eq!(Some(captured.step().audit), layer.latest_audit());
    let expected_audit = layer.latest_backward_audit().unwrap();
    let request = ToposResonatorRequest {
        input: input.data(),
        gate: gate.data(),
        rows,
        features,
        config,
        topos: &topos,
    };
    let mut timings = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut orders = Vec::new();
    for round in 0..26 {
        let order: Vec<_> = (0..4).map(|offset| (round + offset) % 4).collect();
        if round >= 2 {
            orders.push(order.clone());
        }
        for route in order {
            layer.parameter_mut().zero_gradient();
            let start = Instant::now();
            let elapsed = match route {
                0 => {
                    validate_topos_resonator_state_with_layout(
                        black_box(request),
                        ToposResonatorGateLayout::SharedRows,
                    )?;
                    start.elapsed()
                }
                1 => {
                    let batch = black_box(operator.capture_shared_rows(
                        input.data(),
                        gate.data(),
                        rows,
                        features,
                    )?);
                    let elapsed = start.elapsed();
                    assert_eq!(bits(batch.output()), expected[0]);
                    assert_eq!(batch.step().audit, captured.step().audit);
                    elapsed
                }
                2 => {
                    let actual = black_box(layer.forward(&input)?);
                    let elapsed = start.elapsed();
                    assert_eq!(bits(actual.data()), expected[0]);
                    assert_eq!(layer.latest_audit(), Some(captured.step().audit));
                    elapsed
                }
                _ => {
                    let (backward, audit) =
                        black_box(captured.vjp_audited_elementwise(upstream.data())?);
                    let elapsed = start.elapsed();
                    assert_eq!(bits(&backward.grad_input), expected[1]);
                    assert_eq!(bits(&backward.grad_gate), expected[2]);
                    assert_eq!(audit, expected_audit);
                    elapsed
                }
            };
            if round >= 2 {
                timings[route].push(elapsed.as_secs_f64() * 1000.0);
            }
        }
    }
    let report = json!({
        "schema": "spiraltorch.topos_shared_phase_probe.v1", "status": "measured",
        "shape": [rows, features], "iterations": iterations, "coupling": 0.25,
        "porosity": 0.2_f32, "backend": "cpu", "dtype": "float32", "warmup_per_route": 2,
        "round_order": orders,
        "routes": ["shared_state_validation", "core_shared_capture", "nn_shared_forward", "core_captured_vjp_audited"],
        "measurements_ms": timings,
        "scope": "Overlapping public CPU scopes, not additive phases: core capture already includes validation; NN forward includes core capture. Core VJP is separate, with no gate accumulation. Reset/drop/comparisons excluded. Diagnostic only, not a Torch comparison or causal cost decomposition.",
        "parity": "Every forward and both VJPs are bit-identical to the prepared native NN reference; both audits equal. Checked after each measured call."
    });
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    Ok(())
}
