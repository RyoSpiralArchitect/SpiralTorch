// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::maxwell::NarrativeHint;
use crate::PureResult;
use nalgebra::{DMatrix, DVector};
use std::collections::{BTreeMap, HashMap};

/// Bridges Z-space narrative hints with an information-geometric metric.
///
/// The metric treats the co-occurrence structure of narrative tags as a
/// Riemannian slice equipped with a Fisher information tensor. Curvature is
/// estimated through symmetric KL divergence so downstream planners can reason
/// about how "close" two narratives are inside the Z-space atlas.
#[derive(Clone, Debug)]
pub struct InformationGeometryMetric {
    tag_index: HashMap<String, usize>,
    affinity: DMatrix<f64>,
    smoothing: f64,
}

impl InformationGeometryMetric {
    /// Builds the metric from a collection of narrative hints. Tags are
    /// gathered across all hints to generate a shared atlas. Co-occurrence
    /// intensities become entries in the affinity tensor.
    pub fn from_narratives(hints: &[NarrativeHint]) -> Self {
        let mut tag_index = HashMap::new();
        for hint in hints {
            for tag in hint.tags() {
                let next_index = tag_index.len();
                tag_index.entry(tag.clone()).or_insert(next_index);
            }
        }
        let dim = tag_index.len().max(1);
        let mut affinity = DMatrix::from_element(dim, dim, 1e-6);

        for hint in hints {
            let mut indices = Vec::new();
            for tag in hint.tags() {
                if let Some(&idx) = tag_index.get(tag) {
                    indices.push(idx);
                }
            }
            if indices.is_empty() {
                continue;
            }
            let contribution = (hint.intensity() as f64).max(0.0) / indices.len() as f64;
            for &i in &indices {
                affinity[(i, i)] += contribution;
                for &j in &indices {
                    if i != j {
                        affinity[(i, j)] += contribution;
                    }
                }
            }
        }

        Self {
            tag_index,
            affinity,
            smoothing: 1e-5,
        }
    }

    /// Returns the dimensionality of the atlas.
    pub fn dimension(&self) -> usize {
        self.affinity.nrows()
    }

    /// Retrieves the stable ordering of tags used by the metric.
    pub fn tags(&self) -> BTreeMap<usize, String> {
        let mut reverse = BTreeMap::new();
        for (tag, idx) in &self.tag_index {
            reverse.insert(*idx, tag.clone());
        }
        reverse
    }

    /// Returns the index of a tag if it is registered in the atlas.
    pub fn index_of(&self, tag: &str) -> Option<usize> {
        self.tag_index.get(tag).copied()
    }

    /// Encodes a hint into the probability simplex defined by the atlas.
    pub fn encode(&self, hint: &NarrativeHint) -> DVector<f64> {
        let dim = self.dimension();
        let mut vector = DVector::from_element(dim, self.smoothing);
        if dim == 0 {
            return vector;
        }
        let mut present = 0;
        for tag in hint.tags() {
            if let Some(&idx) = self.tag_index.get(tag) {
                let increment = (hint.intensity() as f64).max(0.0);
                vector[idx] += increment;
                present += 1;
            }
        }
        if present == 0 {
            // fall back to a diffuse encoding when the hint has no registered tags
            let spread = (hint.intensity() as f64).max(0.0);
            if spread > 0.0 {
                for value in vector.iter_mut() {
                    *value += spread / dim as f64;
                }
            }
        }
        let sum: f64 = vector.iter().sum();
        if sum > 0.0 {
            vector /= sum;
        }
        vector
    }

    /// Returns the Fisher information tensor (diagonal approximation) for a
    /// single hint.
    pub fn fisher_metric(&self, hint: &NarrativeHint) -> DMatrix<f64> {
        let encoded = self.encode(hint);
        let mut fisher = DMatrix::zeros(self.dimension(), self.dimension());
        for (i, value) in encoded.iter().enumerate() {
            let clamped = value.max(self.smoothing);
            fisher[(i, i)] = 1.0 / clamped;
        }
        fisher
    }

    /// KL divergence between two narratives living in the atlas.
    pub fn kl_divergence(&self, a: &NarrativeHint, b: &NarrativeHint) -> f64 {
        let pa = self.encode(a);
        let pb = self.encode(b);
        kl_divergence(&pa, &pb, self.smoothing)
    }

    /// Symmetric KL divergence providing a curvature proxy.
    pub fn symmetric_kl(&self, a: &NarrativeHint, b: &NarrativeHint) -> f64 {
        let forward = self.kl_divergence(a, b);
        let backward = self.kl_divergence(b, a);
        0.5 * (forward + backward)
    }

    /// Sectional curvature estimate derived from the symmetric KL divergence.
    /// Larger curvature indicates narratives occupy distant "Z-space sheets".
    pub fn sectional_curvature(&self, a: &NarrativeHint, b: &NarrativeHint) -> f64 {
        let sym_kl = self.symmetric_kl(a, b);
        let laplacian = self.laplacian_matrix();
        if self.dimension() == 0 {
            return 0.0;
        }
        // Normalise curvature against the atlas energy so the value is scale
        // invariant and comparable across batches.
        let trace = laplacian.trace().max(self.smoothing);
        sym_kl / trace
    }

    /// Computes the Laplacian induced by the affinity tensor.
    pub fn laplacian_matrix(&self) -> DMatrix<f64> {
        let mut degree = DMatrix::zeros(self.dimension(), self.dimension());
        for i in 0..self.dimension() {
            let mut row_sum = 0.0;
            for j in 0..self.dimension() {
                row_sum += self.affinity[(i, j)];
            }
            degree[(i, i)] = row_sum;
        }
        degree - &self.affinity
    }

    /// Computes an atlas geodesic velocity between two narratives. The value is
    /// bounded to `[0, 1]` and increases when narratives align.
    pub fn geodesic_velocity(&self, a: &NarrativeHint, b: &NarrativeHint) -> f64 {
        let curvature = self.sectional_curvature(a, b);
        (1.0 + (-curvature).exp()).recip()
    }

    /// Summarises the bridge between two narratives into a ready-to-log
    /// descriptor that exposes curvature, symmetric KL, and Fisher traces.
    pub fn bridge_slice(&self, a: &NarrativeHint, b: &NarrativeHint) -> NarrativeBridgeCurvature {
        let fisher_a = self.fisher_metric(a);
        let fisher_b = self.fisher_metric(b);
        NarrativeBridgeCurvature {
            channel_a: a.channel().to_string(),
            channel_b: b.channel().to_string(),
            symmetric_kl: self.symmetric_kl(a, b),
            sectional_curvature: self.sectional_curvature(a, b),
            fisher_trace_a: fisher_a.trace(),
            fisher_trace_b: fisher_b.trace(),
            geodesic_velocity: self.geodesic_velocity(a, b),
        }
    }

    /// Normalises a batch of narrative hints into probability vectors.
    pub fn encode_batch(&self, hints: &[NarrativeHint]) -> Vec<DVector<f64>> {
        hints.iter().map(|hint| self.encode(hint)).collect()
    }
}

fn kl_divergence(p: &DVector<f64>, q: &DVector<f64>, smoothing: f64) -> f64 {
    let mut divergence = 0.0;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let pi = pi.max(smoothing);
        let qi = qi.max(smoothing);
        divergence += pi * (pi / qi).ln();
    }
    divergence
}

/// Snapshot of the geometric relationship between two narratives.
#[derive(Clone, Debug, PartialEq)]
pub struct NarrativeBridgeCurvature {
    pub channel_a: String,
    pub channel_b: String,
    pub symmetric_kl: f64,
    pub sectional_curvature: f64,
    pub fisher_trace_a: f64,
    pub fisher_trace_b: f64,
    pub geodesic_velocity: f64,
}

impl NarrativeBridgeCurvature {
    /// Convenience helper that expresses the curvature as a bounded affinity.
    pub fn affinity(&self) -> f64 {
        (1.0 + self.sectional_curvature.exp()).recip()
    }
}

/// Computes a dense curvature table for telemetry dashboards.
pub fn compute_curvature_table(
    metric: &InformationGeometryMetric,
    hints: &[NarrativeHint],
) -> PureResult<Vec<NarrativeBridgeCurvature>> {
    let mut results = Vec::new();
    for i in 0..hints.len() {
        for j in i + 1..hints.len() {
            results.push(metric.bridge_slice(&hints[i], &hints[j]));
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hint(channel: &str, tags: &[&str], intensity: f32) -> NarrativeHint {
        NarrativeHint::new(
            channel,
            tags.iter().map(|t| t.to_string()).collect(),
            intensity,
        )
    }

    #[test]
    fn builds_metric_and_curvature() {
        let hints = vec![
            hint("alpha", &["spiral", "torch"], 0.6),
            hint("beta", &["spiral", "z-space"], 0.8),
            hint("gamma", &["torch", "narrative"], 0.5),
        ];
        let metric = InformationGeometryMetric::from_narratives(&hints);
        assert_eq!(metric.dimension(), 4);
        let curvature = metric.bridge_slice(&hints[0], &hints[1]);
        assert!(curvature.symmetric_kl >= 0.0);
        assert!(curvature.sectional_curvature >= 0.0);
        assert!(curvature.geodesic_velocity <= 1.0);
        assert!(curvature.fisher_trace_a > 0.0);
        assert!(curvature.fisher_trace_b > 0.0);
    }

    #[test]
    fn curvature_table_has_pairs() {
        let hints = vec![
            hint("alpha", &["spiral"], 1.0),
            hint("beta", &["torch"], 1.0),
            hint("gamma", &["spiral", "torch"], 1.5),
        ];
        let metric = InformationGeometryMetric::from_narratives(&hints);
        let table = compute_curvature_table(&metric, &hints).unwrap();
        assert_eq!(table.len(), 3);
        for entry in table {
            assert!(entry.affinity() > 0.0);
        }
    }
}
