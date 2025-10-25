// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Contextual Lagrangian gating for vision pipelines.
//!
//! This module adapts the contextual observation machinery so vision
//! components can project latent arrangements into Z-space pulses.  It
//! mirrors the textual bridge while embracing image-centric affordances
//! such as 2D adjacency and explicit field dimensions.

use st_core::theory::zpulse::ZPulse;
use st_logic::contextual_observation::{
    Arrangement, LagrangianGate, MeaningProjection, OrientationGauge, PureAtom,
};
use st_tensor::{PureResult, Tensor, TensorError};

/// Human-readable story describing what a contextual projection encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionContextualNarrative {
    pub summary: String,
    pub highlights: Vec<String>,
}

impl VisionContextualNarrative {
    fn from_projection(projection: &MeaningProjection, dimensions: Option<(usize, usize)>) -> Self {
        let support = projection.support;
        let mut summary = match (projection.label, &projection.signature) {
            (Some(label), Some(signature)) => format!(
                "Gauge {} traced boundary {} across {} sites",
                label.as_str(),
                signature.boundary_edges,
                support
            ),
            (Some(label), None) => format!(
                "Gauge {} covered {} sites without boundary signature",
                label.as_str(),
                support
            ),
            (None, Some(signature)) => format!(
                "Boundary {} emerged over {} sites",
                signature.boundary_edges, support
            ),
            (None, None) => format!("Context spans {} sites without orientation", support),
        };
        if let Some((height, width)) = dimensions {
            summary.push_str(&format!(" within a {}×{} field", height, width));
        }
        summary.push('.');

        let mut highlights = Vec::new();
        if let Some(signature) = &projection.signature {
            highlights.push(format!("boundary {}", signature.boundary_edges));
            highlights.push(format!(
                "population |Δ| {}",
                signature.absolute_population_imbalance
            ));
            highlights.push(format!("cluster imbalance {}", signature.cluster_imbalance));
        }
        if let Some(label) = projection.label {
            highlights.push(format!("gauge {}", label.as_str()));
        }
        if let Some((bin, magnitude)) = projection.dominant_frequency_bin() {
            highlights.push(format!("dominant freq bin {} {:.3}", bin, magnitude));
        }
        highlights.push(format!("lexical weight {:.3}", projection.lexical_weight()));
        if let Some((height, width)) = dimensions {
            highlights.push(format!("extent {}x{}", height, width));
        }

        Self {
            summary,
            highlights,
        }
    }
}

/// Pulse record tying together the descriptive narrative, projection, and Z-space emission.
#[derive(Clone, Debug)]
pub struct VisionContextualPulse {
    pub narrative: VisionContextualNarrative,
    pub projection: MeaningProjection,
    pub pulse: ZPulse,
    pub dimensions: Option<(usize, usize)>,
}

/// Adapter that pushes contextual meaning into the vision domain and emits pulses via a
/// Lagrangian gate.
#[derive(Clone, Debug)]
pub struct VisionContextualGate {
    gate: LagrangianGate,
}

impl VisionContextualGate {
    pub fn new(gate: LagrangianGate) -> Self {
        Self { gate }
    }

    pub fn gate(&self) -> &LagrangianGate {
        &self.gate
    }

    pub fn gate_from_arrangement(
        &self,
        arrangement: &Arrangement,
        gauge: OrientationGauge,
        ts: u64,
    ) -> PureResult<VisionContextualPulse> {
        self.gate_from_arrangement_with_dims(arrangement, gauge, ts, None)
    }

    pub fn gate_from_arrangement_with_dims(
        &self,
        arrangement: &Arrangement,
        gauge: OrientationGauge,
        ts: u64,
        dimensions: Option<(usize, usize)>,
    ) -> PureResult<VisionContextualPulse> {
        let projection = MeaningProjection::from_arrangement(arrangement, gauge)?;
        let narrative = VisionContextualNarrative::from_projection(&projection, dimensions);
        let pulse = self.gate.emit(&projection, ts);
        Ok(VisionContextualPulse {
            narrative,
            projection,
            pulse,
            dimensions,
        })
    }

    pub fn gate_from_tensor(
        &self,
        tensor: &Tensor,
        pivot: f32,
        gauge: OrientationGauge,
        ts: u64,
    ) -> PureResult<VisionContextualPulse> {
        let (height, width) = tensor.shape();
        let arrangement = arrangement_from_tensor(tensor, pivot)?;
        self.gate_from_arrangement_with_dims(&arrangement, gauge, ts, Some((height, width)))
    }
}

/// Converts a dense tensor slice into a contextual arrangement using 4-neighbour adjacency.
pub fn arrangement_from_tensor(tensor: &Tensor, pivot: f32) -> PureResult<Arrangement> {
    let (height, width) = tensor.shape();
    if height == 0 || width == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: height,
            cols: width,
        });
    }

    let placements: Vec<PureAtom> = tensor
        .data()
        .iter()
        .map(|value| {
            if *value >= pivot {
                PureAtom::B
            } else {
                PureAtom::A
            }
        })
        .collect();

    let mut edges =
        Vec::with_capacity(height.saturating_sub(1) * width + width.saturating_sub(1) * height);
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if col + 1 < width {
                edges.push((idx, idx + 1));
            }
            if row + 1 < height {
                edges.push((idx, idx + width));
            }
        }
    }

    Ok(Arrangement::new(placements, edges))
}

/// Builds a rectangular arrangement directly from boolean mask data.
pub fn arrangement_from_mask(
    mask: &[bool],
    height: usize,
    width: usize,
) -> PureResult<Arrangement> {
    if height * width != mask.len() {
        return Err(TensorError::ShapeMismatch {
            left: (height, width),
            right: (mask.len(), 1),
        });
    }
    let placements = mask
        .iter()
        .map(|flag| if *flag { PureAtom::B } else { PureAtom::A })
        .collect();
    let mut edges =
        Vec::with_capacity(height.saturating_sub(1) * width + width.saturating_sub(1) * height);
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if col + 1 < width {
                edges.push((idx, idx + 1));
            }
            if row + 1 < height {
                edges.push((idx, idx + width));
            }
        }
    }
    Ok(Arrangement::new(placements, edges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_logic::contextual_observation::{LagrangianGateConfig, OrientationGauge};

    #[test]
    fn tensor_to_arrangement_tracks_boundaries() {
        let tensor = Tensor::from_vec(2, 3, vec![0.1, 0.6, 0.4, 0.9, 0.2, 0.8]).unwrap();
        let arrangement = arrangement_from_tensor(&tensor, 0.5).unwrap();
        assert_eq!(arrangement.len(), 6);
        assert_eq!(arrangement.boundary_edges(), 7);
    }

    #[test]
    fn gate_emits_pulse_with_dimensions() {
        let tensor = Tensor::from_vec(2, 2, vec![0.2, 0.8, 0.4, 0.9]).unwrap();
        let gate = VisionContextualGate::new(LagrangianGate::new(
            LagrangianGateConfig::default().energy_gain(1.5),
        ));
        let pulse = gate
            .gate_from_tensor(&tensor, 0.5, OrientationGauge::Preserve, 144)
            .unwrap();
        assert_eq!(pulse.projection.support, 4);
        assert_eq!(pulse.pulse.ts, 144);
        assert_eq!(pulse.dimensions, Some((2, 2)));
        assert!(pulse.narrative.summary.contains("4 sites"));
        assert!(pulse
            .narrative
            .highlights
            .iter()
            .any(|entry| entry.starts_with("extent")));
    }
}
