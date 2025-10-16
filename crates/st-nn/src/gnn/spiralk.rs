// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::handoff::{fold_with_band_energy, QuadBandEnergy};
use crate::schedule::BandEnergy;
use crate::PureResult;
use st_core::telemetry::xai::GraphFlowTracer;
use std::cell::RefCell;
use std::rc::Rc;

/// Bridge that translates graph flow telemetry into SpiralK-friendly hints and
/// consensus multipliers that can be consumed during training.
#[derive(Clone)]
pub struct GraphConsensusBridge {
    tracer: Rc<RefCell<GraphFlowTracer>>,
    blend: f32,
    base_program: Option<String>,
}

impl GraphConsensusBridge {
    /// Builds a bridge that drains the provided tracer each time a digest is
    /// requested. The bridge is intentionally `Clone` so the same tracer can be
    /// shared across trainer instances when required.
    pub fn new(tracer: Rc<RefCell<GraphFlowTracer>>) -> Self {
        Self {
            tracer,
            blend: 0.35,
            base_program: None,
        }
    }

    /// Sets the blending factor applied when converting barycentric weights
    /// into Above/Here/Beneath multipliers.
    pub fn with_blend(mut self, blend: f32) -> Self {
        self.blend = blend.clamp(0.0, 1.0);
        self
    }

    /// Prepends an existing SpiralK program to the generated hints.
    pub fn with_base_program(mut self, program: impl Into<String>) -> Self {
        let script = program.into();
        self.base_program = if script.trim().is_empty() {
            None
        } else {
            Some(script)
        };
        self
    }

    /// Produces a digest of the currently recorded graph flows. Returns
    /// `Ok(None)` if the tracer has not recorded any layers since the previous
    /// call.
    pub fn digest(&self, baseline: &BandEnergy) -> PureResult<Option<GraphConsensusDigest>> {
        let reports = self.tracer.borrow_mut().drain();
        if reports.is_empty() {
            return Ok(None);
        }

        let mut layer_shares = Vec::with_capacity(reports.len());
        let mut script_lines = Vec::new();
        let mut total_graph_energy = 0.0f32;

        for report in &reports {
            let quad = fold_with_band_energy(baseline, report);
            if quad.graph <= f32::EPSILON {
                continue;
            }
            total_graph_energy += quad.graph;
            layer_shares.push((report.layer.clone(), quad.graph));
        }

        if total_graph_energy <= f32::EPSILON {
            return Ok(Some(GraphConsensusDigest {
                quad_energy: QuadBandEnergy {
                    above: baseline.above,
                    here: baseline.here,
                    beneath: baseline.beneath,
                    graph: 0.0,
                    drift: baseline.drift,
                },
                barycentric: [0.25, 0.25, 0.25, 0.25],
                multipliers: (1.0, 1.0, 1.0),
                graph_energy: 0.0,
                layer_shares: Vec::new(),
                spiralk_script: None,
            }));
        }

        for (layer, energy) in layer_shares.iter_mut() {
            let share = *energy / total_graph_energy;
            *energy = share;
            let line = build_spiralk_hint(layer, share, total_graph_energy);
            script_lines.push(line);
        }

        let quad_energy = QuadBandEnergy {
            above: baseline.above,
            here: baseline.here,
            beneath: baseline.beneath,
            graph: total_graph_energy,
            drift: baseline.drift,
        };
        let barycentric = quad_energy.barycentric();
        let base_sum = baseline.above.abs() + baseline.here.abs() + baseline.beneath.abs();
        let base_bary = if base_sum <= f32::EPSILON {
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        } else {
            [
                (baseline.above.abs() / base_sum).clamp(0.0, 1.0),
                (baseline.here.abs() / base_sum).clamp(0.0, 1.0),
                (baseline.beneath.abs() / base_sum).clamp(0.0, 1.0),
            ]
        };

        let graph_boost = 1.0 + self.blend * barycentric[3];
        let mut multipliers = (
            graph_boost * (1.0 + self.blend * (barycentric[0] - base_bary[0])),
            graph_boost * (1.0 + self.blend * (barycentric[1] - base_bary[1])),
            graph_boost * (1.0 + self.blend * (barycentric[2] - base_bary[2])),
        );
        multipliers.0 = multipliers.0.clamp(0.5, 1.5);
        multipliers.1 = multipliers.1.clamp(0.5, 1.5);
        multipliers.2 = multipliers.2.clamp(0.5, 1.5);

        let spiralk_script = if script_lines.is_empty() {
            None
        } else {
            let mut combined = String::new();
            if let Some(base) = &self.base_program {
                combined.push_str(base.trim());
                if !combined.is_empty() && !combined.trim_end().ends_with(';') {
                    combined.push(';');
                }
                combined.push('\n');
            }
            for line in &script_lines {
                combined.push_str(line);
                combined.push('\n');
            }
            Some(combined.trim().to_string())
        };

        Ok(Some(GraphConsensusDigest {
            quad_energy,
            barycentric,
            multipliers,
            graph_energy: total_graph_energy,
            layer_shares,
            spiralk_script,
        }))
    }
}

/// Snapshot of the consensus blending derived from the latest graph flow
/// reports.
#[derive(Clone, Debug)]
pub struct GraphConsensusDigest {
    pub quad_energy: QuadBandEnergy,
    pub barycentric: [f32; 4],
    pub multipliers: (f32, f32, f32),
    pub graph_energy: f32,
    pub layer_shares: Vec<(String, f32)>,
    pub spiralk_script: Option<String>,
}

impl GraphConsensusDigest {
    /// Number of layers that contributed to the digest.
    pub fn layer_count(&self) -> usize {
        self.layer_shares.len()
    }
}

fn build_spiralk_hint(layer: &str, share: f32, total_energy: f32) -> String {
    let name = sanitize_layer(layer);
    let weight = (0.55 + 0.45 * share).clamp(0.55, 1.0);
    format!(
        "soft (graph.band, {:.6}, {:.3}, layer == \"{}\");",
        total_energy * share,
        weight,
        name
    )
}

fn sanitize_layer(layer: &str) -> String {
    let mut out = String::with_capacity(layer.len());
    for ch in layer.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if ch.is_whitespace() || matches!(ch, ':' | '-' | '_' | '/') {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("layer");
    }
    if out
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        out.insert(0, 'L');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::telemetry::xai::NodeFlowSample;

    fn sample_flows(energy: f32) -> Vec<NodeFlowSample> {
        vec![NodeFlowSample {
            node_index: 0,
            incoming_weight: 1.0,
            aggregated_norm: energy,
        }]
    }

    #[test]
    fn digest_builds_script_and_multipliers() {
        let tracer = Rc::new(RefCell::new(GraphFlowTracer::new()));
        let bridge = GraphConsensusBridge::new(tracer.clone());
        let baseline = BandEnergy {
            above: 0.4,
            here: 0.3,
            beneath: 0.2,
            drift: 0.1,
        };
        tracer
            .borrow_mut()
            .begin_layer("gnn::conv1", -1.0, sample_flows(0.5));
        tracer.borrow_mut().record_weight_update(0.1, Some(0.05));
        let digest = bridge.digest(&baseline).unwrap().unwrap();
        assert!(digest.graph_energy > 0.0);
        assert_eq!(digest.layer_count(), 1);
        assert!(digest
            .spiralk_script
            .as_ref()
            .unwrap()
            .contains("graph.band"));
        assert!(digest.multipliers.0 > 0.0);
    }

    #[test]
    fn sanitize_layer_handles_non_alphanumeric() {
        assert_eq!(sanitize_layer("0-main/branch"), "L0_main_branch");
        assert_eq!(sanitize_layer("layer"), "layer");
    }
}
