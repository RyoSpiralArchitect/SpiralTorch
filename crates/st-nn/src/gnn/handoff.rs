// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::GraphContext;
use crate::schedule::{BandEnergy, RoundtableSchedule};
use crate::{PureResult, Tensor, TensorError};
use st_core::telemetry::xai::{GraphLayerReport, NodeFlowSample};
use st_tensor::topos::TensorBiome;
use st_tensor::{OpenCartesianTopos, RewriteMonad};

/// Result of embedding a graph propagation step inside an open-cartesian topos biome.
#[derive(Debug)]
pub struct GraphMonadExport {
    /// Tensor biome that absorbed every propagated node embedding.
    pub biome: TensorBiome,
    /// Captured node flows emitted during propagation.
    pub flows: Vec<NodeFlowSample>,
}

impl GraphMonadExport {
    /// Total accumulated weight after absorbing every node embedding into the biome.
    pub fn total_weight(&self) -> f32 {
        self.biome.total_weight()
    }
}

/// Projects graph propagation onto a topos biome so RewriteMonad consumers can
/// fold the graph activations into their canvases without leaving Z-space.
pub fn embed_into_biome(
    context: &GraphContext,
    features: &Tensor,
    topos: OpenCartesianTopos,
) -> PureResult<GraphMonadExport> {
    let (propagated, flows) = context.propagate_with_trace(features)?;
    let feature_dim = propagated.shape().1;
    if feature_dim == 0 {
        return Err(TensorError::EmptyInput("graph_features"));
    }

    let mut biome = TensorBiome::new(topos);
    let data = propagated.data();
    for (idx, flow) in flows.iter().enumerate() {
        let start = idx * feature_dim;
        let end = start + feature_dim;
        if end > data.len() {
            break;
        }
        let mut row = Vec::with_capacity(feature_dim);
        row.extend_from_slice(&data[start..end]);
        let mut node = Tensor::from_vec(1, feature_dim, row)?;
        let monad = RewriteMonad::new(biome.topos());
        monad.rewrite_tensor("graph_node", &mut node)?;
        let weight = flow.energy().max(1e-6);
        biome.absorb_weighted("graph_node", node, weight)?;
    }

    Ok(GraphMonadExport { biome, flows })
}

fn flows_to_grid(flows: &[NodeFlowSample], width: usize, height: usize) -> PureResult<Tensor> {
    if width == 0 || height == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: height,
            cols: width,
        });
    }
    let mut data = vec![0.0f32; width * height];
    let len = data.len().max(1);
    for (idx, flow) in flows.iter().enumerate() {
        let slot = idx % len;
        data[slot] += flow.energy();
    }
    Tensor::from_vec(height, width, data)
}

/// Tensor built from node flows that can be painted directly onto a canvas transformer.
pub fn flows_to_canvas_tensor(flows: &[NodeFlowSample], width: usize) -> PureResult<Tensor> {
    if width == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: flows.len().max(1),
            cols: width,
        });
    }
    let height = ((flows.len().max(1) + width - 1) / width).max(1);
    flows_to_grid(flows, width, height)
}

/// Tensor built from node flows that fits an explicit canvas shape.
pub fn flows_to_canvas_tensor_with_shape(
    flows: &[NodeFlowSample],
    width: usize,
    height: usize,
) -> PureResult<Tensor> {
    flows_to_grid(flows, width, height)
}

/// Roundtable energy extended with a graph channel so A/B/C negotiations can
/// include the graph manifold as a fourth participant.
#[derive(Debug, Clone, Copy)]
pub struct QuadBandEnergy {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
    pub graph: f32,
    pub drift: f32,
}

impl QuadBandEnergy {
    /// Normalises the four bands so they form barycentric weights.
    pub fn barycentric(self) -> [f32; 4] {
        let sum = self.above.abs() + self.here.abs() + self.beneath.abs() + self.graph.abs();
        if sum <= f32::EPSILON {
            [0.25, 0.25, 0.25, 0.25]
        } else {
            [
                (self.above / sum).clamp(0.0, 1.0),
                (self.here / sum).clamp(0.0, 1.0),
                (self.beneath / sum).clamp(0.0, 1.0),
                (self.graph / sum).clamp(0.0, 1.0),
            ]
        }
    }

    /// Returns the classic band energy component for compatibility with existing consumers.
    pub fn roundtable(&self) -> BandEnergy {
        BandEnergy {
            above: self.above,
            here: self.here,
            beneath: self.beneath,
            drift: self.drift,
        }
    }
}

impl Default for QuadBandEnergy {
    fn default() -> Self {
        Self {
            above: 0.0,
            here: 0.0,
            beneath: 0.0,
            graph: 0.0,
            drift: 0.0,
        }
    }
}

/// Folds graph flow energy into the roundtable split, returning a fourth band
/// that can be consumed by four-party consensus drivers.
pub fn fold_into_roundtable(
    schedule: &RoundtableSchedule,
    gradient: &Tensor,
    report: &GraphLayerReport,
) -> PureResult<QuadBandEnergy> {
    let base = schedule.band_energy(gradient)?;
    let graph_energy = report.total_flow_energy() * (1.0 + report.curvature.abs());
    Ok(QuadBandEnergy {
        above: base.above,
        here: base.here,
        beneath: base.beneath,
        graph: graph_energy,
        drift: base.drift,
    })
}

/// Combines an existing band energy measurement with a graph flow report without
/// recomputing the schedule split. Useful when the baseline energy has already
/// been measured for the current training step.
pub fn fold_with_band_energy(base: &BandEnergy, report: &GraphLayerReport) -> QuadBandEnergy {
    QuadBandEnergy {
        above: base.above,
        here: base.here,
        beneath: base.beneath,
        graph: report.total_flow_energy() * (1.0 + report.curvature.abs()),
        drift: base.drift,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::GraphContext;
    use crate::plan::RankPlanner;
    use st_core::backend::device_caps::DeviceCaps;

    fn sample_context() -> GraphContext {
        let adjacency =
            Tensor::from_vec(3, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        GraphContext::from_adjacency(adjacency).unwrap()
    }

    #[test]
    fn embed_into_biome_absorbs_nodes() {
        let context = sample_context();
        let features = Tensor::from_vec(3, 2, vec![1.0, 0.5, -0.5, 0.2, 0.3, -0.1]).unwrap();
        let topos = OpenCartesianTopos::new(-1.0, 1e-3, 2.0, 8, 64).unwrap();
        let export = embed_into_biome(&context, &features, topos).unwrap();
        assert_eq!(export.flows.len(), 3);
        assert_eq!(export.biome.len(), 3);
        assert!(export.total_weight() > 0.0);
    }

    #[test]
    fn canvas_tensor_respects_width() {
        let flows = vec![
            NodeFlowSample {
                node_index: 0,
                incoming_weight: 1.0,
                aggregated_norm: 0.5,
            },
            NodeFlowSample {
                node_index: 1,
                incoming_weight: 0.8,
                aggregated_norm: 0.25,
            },
            NodeFlowSample {
                node_index: 2,
                incoming_weight: 1.2,
                aggregated_norm: 0.75,
            },
            NodeFlowSample {
                node_index: 3,
                incoming_weight: 0.4,
                aggregated_norm: 0.5,
            },
        ];
        let canvas = flows_to_canvas_tensor(&flows, 2).unwrap();
        assert_eq!(canvas.shape(), (2, 2));
        let total: f32 = canvas.data().iter().sum();
        assert!(total > 0.0);
        let shaped = flows_to_canvas_tensor_with_shape(&flows, 2, 3).unwrap();
        assert_eq!(shaped.shape(), (3, 2));
    }

    #[test]
    fn quad_band_energy_combines_roundtable() {
        let planner = RankPlanner::new(DeviceCaps::cpu());
        let schedule =
            RoundtableSchedule::new(&planner, 1, 4, crate::schedule::RoundtableConfig::default());
        let gradient = Tensor::from_vec(1, 4, vec![0.5, -0.2, 0.1, -0.3]).unwrap();
        let report = GraphLayerReport {
            layer: "graph".into(),
            curvature: -1.0,
            node_flows: vec![
                NodeFlowSample {
                    node_index: 0,
                    incoming_weight: 1.0,
                    aggregated_norm: 0.5,
                },
                NodeFlowSample {
                    node_index: 1,
                    incoming_weight: 0.5,
                    aggregated_norm: 0.3,
                },
            ],
            weight_update_magnitude: Some(0.1),
            bias_update_magnitude: Some(0.05),
        };
        let quad = fold_into_roundtable(&schedule, &gradient, &report).unwrap();
        let weights = quad.barycentric();
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-3);
        assert!(quad.graph > 0.0);
    }

    #[test]
    fn fold_with_band_energy_reuses_baseline() {
        let base = BandEnergy {
            above: 0.4,
            here: 0.3,
            beneath: 0.2,
            drift: 0.1,
        };
        let report = GraphLayerReport {
            layer: "graph".into(),
            curvature: -1.0,
            node_flows: vec![NodeFlowSample {
                node_index: 0,
                incoming_weight: 1.0,
                aggregated_norm: 0.5,
            }],
            weight_update_magnitude: Some(0.1),
            bias_update_magnitude: Some(0.05),
        };
        let quad = fold_with_band_energy(&base, &report);
        assert_eq!(quad.above, base.above);
        assert!(quad.graph > 0.0);
        assert_eq!(quad.drift, base.drift);
    }
}
