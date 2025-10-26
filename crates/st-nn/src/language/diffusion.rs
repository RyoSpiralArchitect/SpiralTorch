// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::info_geometry::InformationGeometryMetric;
use super::maxwell::NarrativeHint;
use crate::PureResult;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Diffusion model that evolves narrative concepts across the Z-space atlas.
///
/// The model discretises the diffusion equation `∂P/∂t = ∇·(D ∇P)` using the
/// Laplacian emitted by [`InformationGeometryMetric`]. The diffusion tensor is
/// modulated by a Z-bias vector that can bend the probability flow toward or
/// away from particular tags.
#[derive(Clone, Debug)]
pub struct ConceptDiffusion {
    metric: InformationGeometryMetric,
    diffusion_tensor: DMatrix<f64>,
    z_bias: DVector<f64>,
    state: DVector<f64>,
    timestep: f64,
}

impl ConceptDiffusion {
    /// Builds a diffusion process rooted in the supplied metric.
    pub fn new(metric: InformationGeometryMetric) -> Self {
        let dim = metric.dimension();
        let diffusion_tensor = DMatrix::identity(dim, dim);
        let z_bias = DVector::from_element(dim, 0.0);
        let state = DVector::from_element(dim, 1.0 / dim.max(1) as f64);
        Self {
            metric,
            diffusion_tensor,
            z_bias,
            state,
            timestep: 1e-2,
        }
    }

    /// Applies a custom diffusion tensor. The tensor must be symmetric positive
    /// semi-definite. The function validates dimensionality at runtime.
    pub fn with_diffusion_tensor(mut self, tensor: DMatrix<f64>) -> PureResult<Self> {
        if tensor.nrows() != self.metric.dimension() || tensor.ncols() != self.metric.dimension() {
            return Err(st_tensor::TensorError::InvalidValue {
                label: "diffusion tensor dimensionality mismatch",
            });
        }
        self.diffusion_tensor = tensor;
        Ok(self)
    }

    /// Sets the simulation timestep.
    pub fn with_timestep(mut self, timestep: f64) -> Self {
        self.timestep = timestep.max(1e-4);
        self
    }

    /// Injects a Z-bias vector which warps the diffusion. Positive entries push
    /// probability mass toward a tag while negative entries absorb it.
    pub fn set_z_bias_map(&mut self, bias: HashMap<String, f64>) {
        let mut vector = DVector::from_element(self.metric.dimension(), 0.0);
        for (tag, value) in bias {
            if let Some(index) = self.metric.index_of(&tag) {
                vector[index] = value;
            }
        }
        self.z_bias = vector;
    }

    /// Blends the internal state with a new narrative observation.
    pub fn observe(&mut self, hint: &NarrativeHint, weight: f64) {
        let encoded = self.metric.encode(hint);
        let weight = weight.clamp(0.0, 1.0);
        self.state = (&self.state * (1.0 - weight)) + encoded * weight;
        self.normalise_state();
    }

    /// Evolves the diffusion process by one timestep and returns a snapshot of
    /// the updated state.
    pub fn step(&mut self) -> DiffusionStep {
        let laplacian = self.metric.laplacian_matrix();
        let gradient = &laplacian * &self.state;
        let drift = &self.diffusion_tensor * gradient + &self.z_bias;
        self.state += drift * self.timestep;
        self.state.iter_mut().for_each(|value| {
            if !value.is_finite() {
                *value = 0.0;
            }
        });
        self.normalise_state();
        DiffusionStep {
            state: self.state.clone(),
            tags: self.metric.tags().into_values().collect(),
        }
    }

    fn normalise_state(&mut self) {
        let sum: f64 = self.state.iter().sum();
        if sum > 0.0 {
            self.state /= sum;
        }
    }

    /// Returns the current state as a probability map over tags.
    pub fn state_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        for (index, tag) in self.metric.tags() {
            map.insert(tag, self.state[index]);
        }
        map
    }
}

/// Snapshot produced by a single diffusion step.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionStep {
    pub state: DVector<f64>,
    pub tags: Vec<String>,
}

impl DiffusionStep {
    /// Returns the state as an ordered probability list.
    pub fn as_pairs(&self) -> Vec<(String, f64)> {
        self.tags
            .iter()
            .enumerate()
            .map(|(idx, tag)| (tag.clone(), self.state[idx]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::info_geometry::InformationGeometryMetric;
    use super::*;

    fn hint(channel: &str, tags: &[&str], intensity: f32) -> NarrativeHint {
        NarrativeHint::new(
            channel,
            tags.iter().map(|t| t.to_string()).collect(),
            intensity,
        )
    }

    #[test]
    fn diffusion_updates_state() {
        let hints = vec![
            hint("alpha", &["spiral", "torch"], 1.0),
            hint("beta", &["spiral", "narrative"], 0.8),
        ];
        let metric = InformationGeometryMetric::from_narratives(&hints);
        let mut diffusion = ConceptDiffusion::new(metric);
        diffusion.observe(&hints[0], 0.5);
        diffusion.set_z_bias_map(HashMap::from([(String::from("spiral"), 0.2)]));
        let step = diffusion.step();
        assert_eq!(step.tags.len(), step.state.len());
        let pairs = step.as_pairs();
        assert!(pairs.iter().any(|(tag, _)| tag == "spiral"));
        assert!((step.state.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
