// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =============================================================================

//! Telemetry primitives that expose Z-space explainability artefacts.
//!
//! The [`GraphFlowTracer`] collects per-layer graph flow summaries so higher
//! level tooling can surface where hypergrad energy travelled during a forward
//! and backward pass.

use crate::ops::zspace_round::RoundtableBand;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

/// Records explainability artefacts emitted by GNN-style layers.
#[derive(Debug, Default)]
pub struct GraphFlowTracer {
    reports: Vec<GraphLayerReport>,
}

impl GraphFlowTracer {
    /// Creates an empty tracer.
    pub fn new() -> Self {
        Self {
            reports: Vec::new(),
        }
    }

    /// Begins a new report for a graph layer, capturing the instantaneous
    /// forward flow statistics.
    pub fn begin_layer(
        &mut self,
        layer: impl Into<String>,
        curvature: f32,
        node_flows: Vec<NodeFlowSample>,
    ) {
        let layer = layer.into();
        let node_count = node_flows.len();
        let total_energy: f32 = node_flows.iter().map(NodeFlowSample::energy).sum();
        let max_node_energy = node_flows
            .iter()
            .map(NodeFlowSample::energy)
            .filter(|energy| energy.is_finite())
            .fold(0.0f32, f32::max);
        self.reports.push(GraphLayerReport {
            layer: layer.clone(),
            curvature,
            node_flows,
            weight_update_magnitude: None,
            bias_update_magnitude: None,
            elliptic: None,
            roundtable: None,
        });
        emit_tensor_op("graph_flow_layer_begin", &[node_count], &[1, 3]);
        emit_tensor_op_meta("graph_flow_layer_begin", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "kind": "st_core_graph_flow_layer_begin",
                "layer": layer,
                "reports": self.reports.len(),
                "nodes": node_count,
                "curvature": if curvature.is_finite() { curvature } else { 0.0 },
                "curvature_finite": curvature.is_finite(),
                "total_flow_energy": total_energy,
                "max_node_energy": max_node_energy,
            })
        });
    }

    /// Records the magnitude of the weight/bias updates once the backward pass
    /// completes.
    pub fn record_weight_update(&mut self, weight: f32, bias: Option<f32>) {
        if let Some(report) = self
            .reports
            .iter_mut()
            .rev()
            .find(|report| report.weight_update_magnitude.is_none())
        {
            report.weight_update_magnitude = Some(weight);
            report.bias_update_magnitude = bias;
            emit_tensor_op(
                "graph_flow_weight_update",
                &[report.node_flows.len()],
                &[1, 2],
            );
            emit_tensor_op_meta("graph_flow_weight_update", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": "auto",
                    "kind": "st_core_graph_flow_weight_update",
                    "layer": report.layer,
                    "nodes": report.node_flows.len(),
                    "weight_update_magnitude": if weight.is_finite() { weight } else { 0.0 },
                    "weight_update_finite": weight.is_finite(),
                    "has_bias_update": bias.is_some(),
                    "bias_update_magnitude": bias.unwrap_or(0.0),
                    "total_flow_energy": report.total_flow_energy(),
                })
            });
        }
    }

    /// Annotates the most recent layer with elliptic curvature telemetry.
    pub fn annotate_elliptic(&mut self, sample: EllipticLayerSample) {
        if let Some(report) = self.reports.last_mut() {
            report.elliptic = Some(sample);
            emit_tensor_op(
                "graph_flow_elliptic_annotation",
                &[report.node_flows.len()],
                &[1, 4],
            );
            emit_tensor_op_meta("graph_flow_elliptic_annotation", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": "auto",
                    "kind": "st_core_graph_flow_elliptic_annotation",
                    "layer": report.layer,
                    "nodes": report.node_flows.len(),
                    "curvature_radius": sample.curvature_radius,
                    "mean_geodesic": sample.mean_geodesic,
                    "normalized_radius": sample.normalized_radius(),
                    "sheet_bias": sample.sheet_bias,
                    "spin_alignment": sample.spin_alignment,
                })
            });
        }
    }

    /// Annotates the most recent layer with roundtable-aware aggregation telemetry.
    pub fn annotate_roundtable(&mut self, sample: GraphRoundtableTrace) {
        if let Some(report) = self.reports.last_mut() {
            let hop_count = sample.aggregation.hop_count();
            let effective_sum: f32 = sample.aggregation.effective_coefficients.iter().sum();
            let max_effective = sample
                .aggregation
                .effective_coefficients
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .fold(0.0f32, f32::max);
            let signal = sample.signal;
            let influence = sample.influence;
            let has_band_pass = sample.band_pass.is_some();
            let band_pass_band = sample
                .band_pass
                .as_ref()
                .map(|pass| band_label(pass.band))
                .unwrap_or("none");
            let band_pass_rms = sample
                .band_pass
                .as_ref()
                .map(|pass| pass.gradient_rms)
                .unwrap_or(0.0);
            report.roundtable = Some(sample);
            emit_tensor_op(
                "graph_flow_roundtable_annotation",
                &[report.node_flows.len(), hop_count],
                &[1, hop_count.max(1)],
            );
            emit_tensor_op_meta("graph_flow_roundtable_annotation", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": "auto",
                    "kind": "st_core_graph_flow_roundtable_annotation",
                    "layer": report.layer,
                    "nodes": report.node_flows.len(),
                    "hops": hop_count,
                    "signal_above": signal.above,
                    "signal_here": signal.here,
                    "signal_beneath": signal.beneath,
                    "signal_drift": signal.drift,
                    "band_size_above": signal.band_sizes.0,
                    "band_size_here": signal.band_sizes.1,
                    "band_size_beneath": signal.band_sizes.2,
                    "sheet_index": signal.sheet_index,
                    "sheet_confidence": signal.sheet_confidence,
                    "influence_above": influence.above_multiplier,
                    "influence_here": influence.here_multiplier,
                    "influence_beneath": influence.beneath_multiplier,
                    "drift_bias": influence.drift_bias,
                    "effective_coeff_sum": effective_sum,
                    "max_effective_coeff": max_effective,
                    "has_band_pass": has_band_pass,
                    "band_pass_band": band_pass_band,
                    "band_pass_rms": band_pass_rms,
                })
            });
        }
    }

    /// Returns an immutable slice of all accumulated reports.
    pub fn layers(&self) -> &[GraphLayerReport] {
        &self.reports
    }

    /// Returns the cumulative energy transported across all recorded layers.
    pub fn total_energy(&self) -> f32 {
        self.reports
            .iter()
            .map(GraphLayerReport::total_flow_energy)
            .sum()
    }

    /// Consumes all accumulated reports, returning them in insertion order.
    pub fn drain(&mut self) -> Vec<GraphLayerReport> {
        let reports = self.reports.len();
        let nodes: usize = self
            .reports
            .iter()
            .map(|report| report.node_flows.len())
            .sum();
        let total_energy = self.total_energy();
        let updated_layers = self
            .reports
            .iter()
            .filter(|report| report.weight_update_magnitude.is_some())
            .count();
        let elliptic_layers = self
            .reports
            .iter()
            .filter(|report| report.elliptic.is_some())
            .count();
        let roundtable_layers = self
            .reports
            .iter()
            .filter(|report| report.roundtable.is_some())
            .count();
        emit_tensor_op("graph_flow_drain", &[reports, nodes], &[reports]);
        emit_tensor_op_meta("graph_flow_drain", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "kind": "st_core_graph_flow_drain",
                "reports": reports,
                "nodes": nodes,
                "total_flow_energy": total_energy,
                "updated_layers": updated_layers,
                "elliptic_layers": elliptic_layers,
                "roundtable_layers": roundtable_layers,
            })
        });
        core::mem::take(&mut self.reports)
    }
}

/// Summary emitted for a single graph layer invocation.
#[derive(Debug, Clone)]
pub struct GraphLayerReport {
    /// Name of the layer that produced the report.
    pub layer: String,
    /// Hyperbolic curvature used by the layer's hypergrad tape.
    pub curvature: f32,
    /// Per-node flow summaries captured during the forward pass.
    pub node_flows: Vec<NodeFlowSample>,
    /// Optional magnitude of the weight update emitted during the backward pass.
    pub weight_update_magnitude: Option<f32>,
    /// Optional magnitude of the bias update emitted during the backward pass.
    pub bias_update_magnitude: Option<f32>,
    /// Optional elliptic geometry summary attached to the layer.
    pub elliptic: Option<EllipticLayerSample>,
    /// Optional roundtable-aware aggregation trace for the layer.
    pub roundtable: Option<GraphRoundtableTrace>,
}

impl GraphLayerReport {
    /// Total energy transported by all node flows in this layer.
    pub fn total_flow_energy(&self) -> f32 {
        self.node_flows.iter().map(NodeFlowSample::energy).sum()
    }
}

/// Summary describing elliptic geometry captured during a graph layer.
#[derive(Debug, Clone, Copy, Default)]
pub struct EllipticLayerSample {
    pub curvature_radius: f32,
    pub mean_geodesic: f32,
    pub sheet_bias: f32,
    pub spin_alignment: f32,
}

/// Snapshot of the roundtable signal consumed by a graph layer.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphRoundtableSignalSample {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
    pub drift: f32,
    pub band_sizes: (u32, u32, u32),
    pub sheet_index: usize,
    pub sheet_confidence: f32,
    pub curvature: f32,
    pub spin: f32,
    pub energy: f32,
}

/// Influence multipliers derived from the roundtable signal.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphRoundtableInfluenceSample {
    pub above_multiplier: f32,
    pub here_multiplier: f32,
    pub beneath_multiplier: f32,
    pub drift_bias: f32,
}

/// Identifies which gradient band triggered a replay inside `backward_bands`.
#[derive(Debug, Clone)]
pub struct GraphRoundtableBandPassSample {
    pub band: RoundtableBand,
    pub gradient_l1: f32,
    pub gradient_l2: f32,
    pub gradient_rms: f32,
}

/// Aggregation coefficients emitted by a graph layer after applying roundtable scaling.
#[derive(Debug, Clone, Default)]
pub struct GraphAggregationSample {
    pub base_coefficients: Vec<f32>,
    pub step_scales: Vec<f32>,
    pub band_pass_scales: Vec<f32>,
    pub effective_coefficients: Vec<f32>,
}

impl GraphAggregationSample {
    /// Returns the number of hop/self coefficients tracked by the sample.
    pub fn hop_count(&self) -> usize {
        self.effective_coefficients.len()
    }
}

/// Combined trace describing how roundtable state altered message passing.
#[derive(Debug, Clone, Default)]
pub struct GraphRoundtableTrace {
    pub signal: GraphRoundtableSignalSample,
    pub influence: GraphRoundtableInfluenceSample,
    pub aggregation: GraphAggregationSample,
    pub band_pass: Option<GraphRoundtableBandPassSample>,
}

impl EllipticLayerSample {
    /// Returns the normalised geodesic radius within \([0, 1]\).
    pub fn normalized_radius(&self) -> f32 {
        if self.curvature_radius <= 0.0 {
            0.0
        } else {
            (self.mean_geodesic / (self.curvature_radius * std::f32::consts::PI)).clamp(0.0, 1.0)
        }
    }
}

/// Aggregated contribution for a single node inside a graph layer.
#[derive(Debug, Clone)]
pub struct NodeFlowSample {
    /// Index of the node in the processed batch.
    pub node_index: usize,
    /// Normalised incoming weight after adjacency scaling.
    pub incoming_weight: f32,
    /// Absolute sum of the aggregated feature vector.
    pub aggregated_norm: f32,
}

impl NodeFlowSample {
    /// Energy carried by this node during propagation.
    pub fn energy(&self) -> f32 {
        self.aggregated_norm.abs() * self.incoming_weight.abs()
    }
}

fn band_label(band: RoundtableBand) -> &'static str {
    match band {
        RoundtableBand::Above => "above",
        RoundtableBand::Here => "here",
        RoundtableBand::Beneath => "beneath",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::telemetry::tensor_observer_lock()
    }

    #[test]
    fn tracer_records_reports() {
        let mut tracer = GraphFlowTracer::new();
        tracer.begin_layer(
            "layer",
            -1.0,
            vec![NodeFlowSample {
                node_index: 0,
                incoming_weight: 1.0,
                aggregated_norm: 0.5,
            }],
        );
        tracer.record_weight_update(0.25, Some(0.1));
        let reports = tracer.layers();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].layer, "layer");
        assert_eq!(reports[0].curvature, -1.0);
        assert_eq!(reports[0].weight_update_magnitude, Some(0.25));
        assert_eq!(reports[0].bias_update_magnitude, Some(0.1));
        assert_eq!(reports[0].node_flows.len(), 1);
        assert_eq!(reports[0].node_flows[0].node_index, 0);
    }

    #[test]
    fn tracer_drain_clears_reports() {
        let mut tracer = GraphFlowTracer::new();
        tracer.begin_layer("layer", -1.0, Vec::new());
        assert_eq!(tracer.layers().len(), 1);
        let drained = tracer.drain();
        assert_eq!(drained.len(), 1);
        assert!(tracer.layers().is_empty());
    }

    #[test]
    fn node_flow_energy_is_product_of_weight_and_norm() {
        let sample = NodeFlowSample {
            node_index: 0,
            incoming_weight: 2.0,
            aggregated_norm: -0.5,
        };
        assert!((sample.energy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tracer_tracks_total_energy() {
        let mut tracer = GraphFlowTracer::new();
        tracer.begin_layer(
            "layer",
            -1.0,
            vec![NodeFlowSample {
                node_index: 0,
                incoming_weight: 0.5,
                aggregated_norm: 0.4,
            }],
        );
        assert!(tracer.total_energy() > 0.0);
        let report = tracer.layers()[0].clone();
        assert!((report.total_flow_energy() - tracer.total_energy()).abs() < 1e-6);
    }

    #[test]
    fn tracer_accepts_elliptic_annotations() {
        let mut tracer = GraphFlowTracer::new();
        tracer.begin_layer("layer", 1.0, Vec::new());
        tracer.annotate_elliptic(EllipticLayerSample {
            curvature_radius: 1.5,
            mean_geodesic: 1.2,
            sheet_bias: 0.4,
            spin_alignment: 0.1,
        });
        let report = tracer.layers()[0].clone();
        let sample = report.elliptic.expect("elliptic sample");
        assert!((sample.sheet_bias - 0.4).abs() < 1e-6);
        assert!(sample.normalized_radius() > 0.0);
    }

    #[test]
    fn tracer_accepts_roundtable_annotations() {
        let mut tracer = GraphFlowTracer::new();
        tracer.begin_layer("layer", -1.0, Vec::new());
        tracer.annotate_roundtable(GraphRoundtableTrace {
            signal: GraphRoundtableSignalSample {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
                drift: 0.2,
                band_sizes: (2, 1, 1),
                sheet_index: 1,
                sheet_confidence: 0.8,
                curvature: 0.5,
                spin: 0.4,
                energy: 0.25,
            },
            influence: GraphRoundtableInfluenceSample {
                above_multiplier: 1.2,
                here_multiplier: 0.9,
                beneath_multiplier: 0.7,
                drift_bias: 1.1,
            },
            aggregation: GraphAggregationSample {
                base_coefficients: vec![1.0, 0.5, 0.25],
                step_scales: vec![0.9, 1.2, 0.7],
                band_pass_scales: vec![1.0, 1.1, 0.94],
                effective_coefficients: vec![0.9, 0.6, 0.175],
            },
            band_pass: Some(GraphRoundtableBandPassSample {
                band: RoundtableBand::Above,
                gradient_l1: 0.6,
                gradient_l2: 0.4,
                gradient_rms: 0.2,
            }),
        });
        let report = tracer.layers()[0].clone();
        let trace = report.roundtable.expect("roundtable trace");
        assert_eq!(trace.signal.band_sizes, (2, 1, 1));
        assert_eq!(trace.aggregation.hop_count(), 3);
        assert!((trace.influence.drift_bias - 1.1).abs() < 1e-6);
        assert_eq!(
            trace.band_pass.expect("band pass").band,
            RoundtableBand::Above
        );
    }

    #[test]
    fn tracer_backfills_weight_updates_in_reverse_order() {
        let mut tracer = GraphFlowTracer::new();
        tracer.begin_layer("layer0", -1.0, Vec::new());
        tracer.begin_layer("layer1", -1.0, Vec::new());
        tracer.record_weight_update(0.2, Some(0.1));
        tracer.record_weight_update(0.4, Some(0.3));
        let reports = tracer.layers();
        assert_eq!(reports[0].weight_update_magnitude, Some(0.4));
        assert_eq!(reports[1].weight_update_magnitude, Some(0.2));
        assert_eq!(reports[0].bias_update_magnitude, Some(0.3));
        assert_eq!(reports[1].bias_update_magnitude, Some(0.1));
    }

    #[test]
    fn tracer_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut tracer = GraphFlowTracer::new();
        tracer.begin_layer(
            "roundtable-layer",
            -0.8,
            vec![
                NodeFlowSample {
                    node_index: 0,
                    incoming_weight: 0.5,
                    aggregated_norm: 0.4,
                },
                NodeFlowSample {
                    node_index: 1,
                    incoming_weight: 1.5,
                    aggregated_norm: 0.2,
                },
            ],
        );
        tracer.annotate_elliptic(EllipticLayerSample {
            curvature_radius: 1.5,
            mean_geodesic: 1.2,
            sheet_bias: 0.4,
            spin_alignment: 0.1,
        });
        tracer.annotate_roundtable(GraphRoundtableTrace {
            signal: GraphRoundtableSignalSample {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
                drift: 0.2,
                band_sizes: (2, 1, 1),
                sheet_index: 1,
                sheet_confidence: 0.8,
                curvature: 0.5,
                spin: 0.4,
                energy: 0.25,
            },
            influence: GraphRoundtableInfluenceSample {
                above_multiplier: 1.2,
                here_multiplier: 0.9,
                beneath_multiplier: 0.7,
                drift_bias: 1.1,
            },
            aggregation: GraphAggregationSample {
                base_coefficients: vec![1.0, 0.5, 0.25],
                step_scales: vec![0.9, 1.2, 0.7],
                band_pass_scales: vec![1.0, 1.1, 0.94],
                effective_coefficients: vec![0.9, 0.66, 0.1645],
            },
            band_pass: Some(GraphRoundtableBandPassSample {
                band: RoundtableBand::Here,
                gradient_l1: 0.8,
                gradient_l2: 0.5,
                gradient_rms: 0.25,
            }),
        });
        tracer.record_weight_update(0.4, Some(0.2));
        let drained = tracer.drain();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(drained.len(), 1);
        let events = events.lock().unwrap();
        let begin = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "graph_flow_layer_begin"
                    && data["layer"] == "roundtable-layer"
                    && data["nodes"] == 2
            })
            .expect("graph_flow_layer_begin metadata event");
        assert_eq!(begin.1["kind"], "st_core_graph_flow_layer_begin");

        let roundtable = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "graph_flow_roundtable_annotation"
                    && data["layer"] == "roundtable-layer"
                    && data["has_band_pass"] == true
            })
            .expect("graph_flow_roundtable_annotation metadata event");
        assert_eq!(
            roundtable.1["kind"],
            "st_core_graph_flow_roundtable_annotation"
        );
        assert_eq!(roundtable.1["band_pass_band"], "here");

        let update = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "graph_flow_weight_update"
                    && data["layer"] == "roundtable-layer"
                    && data["has_bias_update"] == true
            })
            .expect("graph_flow_weight_update metadata event");
        assert_eq!(update.1["kind"], "st_core_graph_flow_weight_update");

        let drain = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "graph_flow_drain"
                    && data["reports"] == 1
                    && data["roundtable_layers"] == 1
            })
            .expect("graph_flow_drain metadata event");
        assert_eq!(drain.1["kind"], "st_core_graph_flow_drain");
        assert!(drain.1["total_flow_energy"].as_f64().unwrap_or(0.0) > 0.0);
    }
}
