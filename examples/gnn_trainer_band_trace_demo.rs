// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Runs a single `ModuleTrainer::train_epoch()` step and exports
//! band-labelled GNN replay telemetry produced during `backward_bands()`.

use serde_json::{json, Value};
use st_core::backend::device_caps::DeviceCaps;
use st_core::telemetry::xai::{GraphFlowTracer, GraphLayerReport, GraphRoundtableTrace};
use st_nn::{
    EpochStats, GraphActivation, GraphContext, GraphLayerSpec, MeanSquaredError, ModuleTrainer,
    NeighborhoodAggregation, RoundtableConfig, Tensor, ZSpaceGraphNetworkBuilder,
};
use std::collections::BTreeMap;
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
            "band_pass_scales": trace.aggregation.band_pass_scales,
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

fn signal_json(signal: &st_nn::RoundtableBandSignal) -> Value {
    json!({
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
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));
    let mut builder =
        ZSpaceGraphNetworkBuilder::new(build_context()?, NonZeroUsize::new(2).unwrap(), -1.0, 0.05)
            .with_tracer(tracer.clone());
    builder.push_layer(
        GraphLayerSpec::new(NonZeroUsize::new(3).unwrap()).with_aggregation(
            NeighborhoodAggregation::multi_hop_sum(NonZeroUsize::new(2).unwrap())
                .with_include_self(true)
                .with_attenuation(0.6),
        ),
    );
    builder.push_layer(
        GraphLayerSpec::new(NonZeroUsize::new(2).unwrap()).with_activation(GraphActivation::Relu),
    );
    let mut model = builder.build("trainer_band_trace")?;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 0.05, 0.01);
    trainer.prepare(&mut model)?;
    let schedule = trainer.roundtable(
        4,
        2,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let dataset = vec![(
        Tensor::from_vec(4, 2, vec![1.0, 0.25, 0.5, -0.75, -0.5, 1.0, 0.75, -0.25])?,
        Tensor::from_vec(4, 2, vec![0.2, -0.1, 0.3, 0.15, -0.15, 0.25, 0.1, -0.2])?,
    )];
    let mut loss = MeanSquaredError::new();
    let EpochStats {
        batches,
        average_loss,
        ..
    } = trainer.train_epoch(&mut model, &mut loss, dataset, &schedule)?;

    let signal = trainer
        .gnn_roundtable_signal()
        .expect("trainer should emit a roundtable signal");
    let reports = tracer
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .drain();

    let mut by_band = BTreeMap::<String, Vec<Value>>::new();
    for report in &reports {
        let Some(trace) = report.roundtable.as_ref() else {
            continue;
        };
        let Some(pass) = trace.band_pass.as_ref() else {
            continue;
        };
        by_band
            .entry(pass.band.as_str().to_string())
            .or_default()
            .push(json!({
                "layer": report.layer,
                "gradient_l1": pass.gradient_l1,
                "gradient_l2": pass.gradient_l2,
                "effective_coefficients": trace.aggregation.effective_coefficients,
                "step_scales": trace.aggregation.step_scales,
                "band_pass_scales": trace.aggregation.band_pass_scales,
                "total_flow_energy": report.total_flow_energy(),
            }));
    }

    let payload = json!({
        "trainer": {
            "batches": batches,
            "average_loss": average_loss,
        },
        "signal": signal_json(&signal),
        "reports": reports.iter().map(report_json).collect::<Vec<_>>(),
        "band_replays": by_band,
    });

    let trace_path = std::env::temp_dir().join("spiraltorch_gnn_trainer_band_trace.json");
    std::fs::write(&trace_path, serde_json::to_string_pretty(&payload)?)?;

    println!("trace_json={}", trace_path.display());
    println!(
        "trainer: batches={} average_loss={:.6}",
        batches, average_loss
    );
    println!(
        "signal: above={:.4} here={:.4} beneath={:.4} drift={:.4}",
        signal.energy().above,
        signal.energy().here,
        signal.energy().beneath,
        signal.energy().drift
    );
    for (band, entries) in &by_band {
        println!("band={}", band);
        for entry in entries {
            println!(
                "  layer={} grad_l2={:.6} effective={:?}",
                entry["layer"].as_str().unwrap_or("layer"),
                entry["gradient_l2"].as_f64().unwrap_or(0.0),
                entry["effective_coefficients"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
            );
        }
    }

    Ok(())
}
