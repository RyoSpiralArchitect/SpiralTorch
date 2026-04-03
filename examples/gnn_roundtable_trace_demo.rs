// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Small end-to-end demo that traces how a roundtable signal changes
//! message-passing coefficients inside a `ZSpaceGraphNetwork`.

use serde_json::{json, Value};
use st_core::backend::device_caps::DeviceCaps;
use st_core::telemetry::xai::{GraphFlowTracer, GraphLayerReport, GraphRoundtableTrace};
use st_nn::{
    GraphContext, GraphLayerSpec, Module, ModuleTrainer, NeighborhoodAggregation,
    RoundtableBandInfluence, RoundtableBandSignal, RoundtableConfig, Tensor,
    ZSpaceGraphNetworkBuilder,
};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

fn build_context() -> st_nn::PureResult<GraphContext> {
    let adjacency = Tensor::from_vec(
        4,
        4,
        vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )?;
    GraphContext::from_adjacency(adjacency)
}

fn roundtable_trace_json(trace: &GraphRoundtableTrace) -> Value {
    json!({
        "signal": {
            "above": trace.signal.above,
            "here": trace.signal.here,
            "beneath": trace.signal.beneath,
            "drift": trace.signal.drift,
            "band_sizes": [
                trace.signal.band_sizes.0,
                trace.signal.band_sizes.1,
                trace.signal.band_sizes.2,
            ],
            "sheet_index": trace.signal.sheet_index,
            "sheet_confidence": trace.signal.sheet_confidence,
            "curvature": trace.signal.curvature,
            "spin": trace.signal.spin,
            "energy": trace.signal.energy,
        },
        "influence": {
            "above_multiplier": trace.influence.above_multiplier,
            "here_multiplier": trace.influence.here_multiplier,
            "beneath_multiplier": trace.influence.beneath_multiplier,
            "drift_bias": trace.influence.drift_bias,
        },
        "aggregation": {
            "base_coefficients": trace.aggregation.base_coefficients,
            "step_scales": trace.aggregation.step_scales,
            "effective_coefficients": trace.aggregation.effective_coefficients,
        },
        "band_pass": trace.band_pass.as_ref().map(|pass| json!({
            "band": pass.band.as_str(),
            "gradient_l1": pass.gradient_l1,
            "gradient_l2": pass.gradient_l2,
        })),
    })
}

fn report_json(report: &GraphLayerReport) -> Value {
    json!({
        "layer": report.layer,
        "curvature": report.curvature,
        "total_flow_energy": report.total_flow_energy(),
        "weight_update_magnitude": report.weight_update_magnitude,
        "bias_update_magnitude": report.bias_update_magnitude,
        "node_flows": report.node_flows.iter().map(|flow| {
            json!({
                "node_index": flow.node_index,
                "incoming_weight": flow.incoming_weight,
                "aggregated_norm": flow.aggregated_norm,
                "energy": flow.energy(),
            })
        }).collect::<Vec<_>>(),
        "roundtable": report.roundtable.as_ref().map(roundtable_trace_json),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let curvature = -1.0f32;
    let learning_rate = 0.05f32;
    let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));

    let mut builder = ZSpaceGraphNetworkBuilder::new(
        build_context()?,
        NonZeroUsize::new(2).unwrap(),
        curvature,
        learning_rate,
    )
    .with_tracer(tracer.clone());
    builder.push_layer(
        GraphLayerSpec::new(NonZeroUsize::new(3).unwrap()).with_aggregation(
            NeighborhoodAggregation::multi_hop_sum(NonZeroUsize::new(2).unwrap())
                .with_include_self(true)
                .with_attenuation(0.6),
        ),
    );
    builder.push_layer(GraphLayerSpec::new(NonZeroUsize::new(2).unwrap()));
    let mut network = builder.build("zgnn_roundtable")?;

    let mut trainer =
        ModuleTrainer::new(DeviceCaps::cpu(), curvature, learning_rate, learning_rate);
    let schedule = trainer.roundtable(
        4,
        2,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let input = Tensor::from_vec(4, 2, vec![1.0, 0.25, 0.5, -0.75, -0.5, 1.0, 0.75, -0.25])?;
    let grad_probe = Tensor::from_vec(
        4,
        2,
        vec![0.18, -0.04, 0.11, 0.06, -0.07, 0.09, 0.03, -0.12],
    )?;
    let signal = RoundtableBandSignal::from_schedule(&schedule, schedule.band_energy(&grad_probe)?);
    let influence = RoundtableBandInfluence::from_signal(&signal);
    network.apply_roundtable_band(&signal)?;

    let output = network.forward(&input)?;
    let grad_output = Tensor::from_vec(
        output.shape().0,
        output.shape().1,
        vec![0.05, -0.03, 0.08, 0.02, -0.04, 0.06, 0.03, -0.01],
    )?;
    let _ = network.backward(&input, &grad_output)?;

    let reports = tracer
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .drain();

    let payload = json!({
        "signal": {
            "above": signal.energy().above,
            "here": signal.energy().here,
            "beneath": signal.energy().beneath,
            "drift": signal.energy().drift,
            "band_sizes": [signal.band_sizes().0, signal.band_sizes().1, signal.band_sizes().2],
            "spectral": {
                "sheet_index": signal.energy().spectral.sheet_index,
                "sheet_confidence": signal.energy().spectral.sheet_confidence,
                "curvature": signal.energy().spectral.curvature,
                "spin": signal.energy().spectral.spin,
                "energy": signal.energy().spectral.energy,
            }
        },
        "influence": {
            "above_multiplier": influence.multipliers().0,
            "here_multiplier": influence.multipliers().1,
            "beneath_multiplier": influence.multipliers().2,
            "drift_bias": influence.drift_bias(),
        },
        "reports": reports.iter().map(report_json).collect::<Vec<_>>(),
    });

    let trace_path = std::env::temp_dir().join("spiraltorch_gnn_roundtable_trace.json");
    std::fs::write(&trace_path, serde_json::to_string_pretty(&payload)?)?;

    println!("trace_json={}", trace_path.display());
    println!(
        "signal: above={:.4} here={:.4} beneath={:.4} drift={:.4}",
        signal.energy().above,
        signal.energy().here,
        signal.energy().beneath,
        signal.energy().drift
    );
    println!(
        "influence: above={:.4} here={:.4} beneath={:.4} drift_bias={:.4}",
        influence.multipliers().0,
        influence.multipliers().1,
        influence.multipliers().2,
        influence.drift_bias()
    );
    for report in &reports {
        if let Some(trace) = report.roundtable.as_ref() {
            println!(
                "layer={} band={} coeffs={:?} scales={:?} effective={:?}",
                report.layer,
                trace
                    .band_pass
                    .as_ref()
                    .map(|pass| pass.band.as_str())
                    .unwrap_or("none"),
                trace.aggregation.base_coefficients,
                trace.aggregation.step_scales,
                trace.aggregation.effective_coefficients
            );
        }
    }

    Ok(())
}
