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
        self.reports.push(GraphLayerReport {
            layer: layer.into(),
            curvature,
            node_flows,
            weight_update_magnitude: None,
            bias_update_magnitude: None,
            elliptic: None,
        });
    }

    /// Records the magnitude of the weight/bias updates once the backward pass
    /// completes.
    pub fn record_weight_update(&mut self, weight: f32, bias: Option<f32>) {
        if let Some(report) = self.reports.last_mut() {
            report.weight_update_magnitude = Some(weight);
            report.bias_update_magnitude = bias;
        }
    }

    /// Annotates the most recent layer with elliptic curvature telemetry.
    pub fn annotate_elliptic(&mut self, sample: EllipticLayerSample) {
        if let Some(report) = self.reports.last_mut() {
            report.elliptic = Some(sample);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
