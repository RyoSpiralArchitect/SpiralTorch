// SPDX-License-Identifier: AGPL-3.0-or-later
// (c) 2025 Ryo SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch - Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL Section 13.

use super::maxwell::NarrativeHint;
use crate::PureResult;
use nalgebra::{DMatrix, DVector};
use st_core::inference::concept_diffusion::{
    compare_fisher_rao, fisher_information_diagonal, ConceptDiffusionError, FisherRaoComparison,
};
use st_tensor::TensorError;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(super) fn concept_diffusion_error_to_tensor(error: ConceptDiffusionError) -> TensorError {
    match error {
        ConceptDiffusionError::EmptyState => {
            TensorError::EmptyInput("concept diffusion probability state")
        }
        ConceptDiffusionError::NonFinite { field, value }
        | ConceptDiffusionError::NonFiniteDerived { field, value } => TensorError::NonFiniteValue {
            label: field,
            value: value as f32,
        },
        error => TensorError::Generic(format!("concept diffusion failed: {error}")),
    }
}

fn validate_intensity(hint: &NarrativeHint) -> PureResult<f64> {
    let intensity = f64::from(hint.intensity());
    if !intensity.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "narrative intensity",
            value: hint.intensity(),
        });
    }
    if intensity < 0.0 {
        return Err(TensorError::InvalidValue {
            label: "narrative intensity must be non-negative",
        });
    }
    Ok(intensity)
}

/// A categorical Fisher-Rao atlas over narrative tags.
///
/// Co-occurrence defines the graph used by diffusion, while probability
/// comparisons use the exact square-root embedding of the categorical simplex.
#[derive(Clone, Debug)]
pub struct InformationGeometryMetric {
    tags: Vec<String>,
    tag_index: HashMap<String, usize>,
    affinity: DMatrix<f64>,
    smoothing: f64,
}

impl InformationGeometryMetric {
    /// Builds a deterministic atlas from validated narrative hints.
    pub fn from_narratives(hints: &[NarrativeHint]) -> PureResult<Self> {
        let mut tag_set = BTreeSet::new();
        for hint in hints {
            validate_intensity(hint)?;
            for tag in hint.tags() {
                if tag.trim().is_empty() {
                    return Err(TensorError::InvalidValue {
                        label: "narrative tag must not be empty",
                    });
                }
                tag_set.insert(tag.clone());
            }
        }
        if tag_set.is_empty() {
            return Err(TensorError::EmptyInput("information geometry tags"));
        }

        let tags = tag_set.into_iter().collect::<Vec<_>>();
        let tag_index = tags
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, tag)| (tag, index))
            .collect::<HashMap<_, _>>();
        let dimension = tags.len();
        let smoothing = 1.0e-5;
        let mut affinity = DMatrix::from_element(dimension, dimension, smoothing);

        for hint in hints {
            let intensity = validate_intensity(hint)?;
            let indices = hint
                .tags()
                .iter()
                .filter_map(|tag| tag_index.get(tag).copied())
                .collect::<BTreeSet<_>>();
            if indices.is_empty() {
                continue;
            }
            let contribution = intensity / indices.len() as f64;
            for &row in &indices {
                affinity[(row, row)] += contribution;
                for &col in &indices {
                    if row != col {
                        affinity[(row, col)] += contribution;
                    }
                }
            }
        }

        Ok(Self {
            tags,
            tag_index,
            affinity,
            smoothing,
        })
    }

    pub fn dimension(&self) -> usize {
        self.tags.len()
    }

    pub fn tags(&self) -> BTreeMap<usize, String> {
        self.tags.iter().cloned().enumerate().collect()
    }

    pub fn tag_labels(&self) -> &[String] {
        &self.tags
    }

    pub fn index_of(&self, tag: &str) -> Option<usize> {
        self.tag_index.get(tag).copied()
    }

    pub(crate) fn affinity_rows(&self) -> Vec<Vec<f64>> {
        (0..self.dimension())
            .map(|row| {
                (0..self.dimension())
                    .map(|col| self.affinity[(row, col)])
                    .collect()
            })
            .collect()
    }

    /// Encodes a hint into the atlas simplex and rejects invalid intensities.
    pub fn encode(&self, hint: &NarrativeHint) -> PureResult<DVector<f64>> {
        let intensity = validate_intensity(hint)?;
        let dimension = self.dimension();
        let mut vector = DVector::from_element(dimension, self.smoothing);
        let indices = hint
            .tags()
            .iter()
            .filter_map(|tag| self.tag_index.get(tag).copied())
            .collect::<BTreeSet<_>>();
        for &index in &indices {
            vector[index] += intensity;
        }
        if indices.is_empty() && intensity > 0.0 {
            let spread = intensity / dimension as f64;
            for value in vector.iter_mut() {
                *value += spread;
            }
        }
        let sum = vector.iter().sum::<f64>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "information geometry encoding sum",
                value: sum as f32,
            });
        }
        vector /= sum;
        Ok(vector)
    }

    /// Returns the exact ambient diagonal representation of the categorical Fisher metric.
    pub fn fisher_metric(&self, hint: &NarrativeHint) -> PureResult<DMatrix<f64>> {
        let encoded = self.encode(hint)?;
        let diagonal = fisher_information_diagonal(encoded.as_slice())
            .map_err(concept_diffusion_error_to_tensor)?;
        Ok(DMatrix::from_diagonal(&DVector::from_vec(diagonal)))
    }

    fn compare(
        &self,
        left: &NarrativeHint,
        right: &NarrativeHint,
    ) -> PureResult<FisherRaoComparison> {
        let left = self.encode(left)?;
        let right = self.encode(right)?;
        compare_fisher_rao(left.as_slice(), right.as_slice())
            .map_err(concept_diffusion_error_to_tensor)
    }

    pub fn kl_divergence(&self, left: &NarrativeHint, right: &NarrativeHint) -> PureResult<f64> {
        Ok(self.compare(left, right)?.forward_kl)
    }

    pub fn symmetric_kl(&self, left: &NarrativeHint, right: &NarrativeHint) -> PureResult<f64> {
        Ok(self.compare(left, right)?.symmetric_kl)
    }

    pub fn fisher_rao_distance(
        &self,
        left: &NarrativeHint,
        right: &NarrativeHint,
    ) -> PureResult<f64> {
        Ok(self.compare(left, right)?.fisher_rao_distance)
    }

    /// Returns the exact constant curvature of the categorical Fisher simplex.
    /// Curvature is undefined for simplex manifolds below two dimensions.
    pub fn sectional_curvature(
        &self,
        left: &NarrativeHint,
        right: &NarrativeHint,
    ) -> PureResult<Option<f64>> {
        Ok(self.compare(left, right)?.sectional_curvature)
    }

    pub fn laplacian_matrix(&self) -> DMatrix<f64> {
        let mut degree = DMatrix::zeros(self.dimension(), self.dimension());
        for row in 0..self.dimension() {
            let row_sum = (0..self.dimension())
                .filter(|col| *col != row)
                .map(|col| self.affinity[(row, col)])
                .sum();
            degree[(row, row)] = row_sum;
        }
        let mut adjacency = self.affinity.clone();
        for index in 0..self.dimension() {
            adjacency[(index, index)] = 0.0;
        }
        degree - adjacency
    }

    /// Returns the speed of the constant-speed Fisher-Rao geodesic over unit time.
    pub fn geodesic_velocity(
        &self,
        left: &NarrativeHint,
        right: &NarrativeHint,
    ) -> PureResult<f64> {
        Ok(self.compare(left, right)?.fisher_rao_distance)
    }

    pub fn bridge_slice(
        &self,
        left: &NarrativeHint,
        right: &NarrativeHint,
    ) -> PureResult<NarrativeBridgeCurvature> {
        let fisher_left = self.fisher_metric(left)?;
        let fisher_right = self.fisher_metric(right)?;
        let comparison = self.compare(left, right)?;
        Ok(NarrativeBridgeCurvature {
            channel_a: left.channel().to_owned(),
            channel_b: right.channel().to_owned(),
            symmetric_kl: comparison.symmetric_kl,
            fisher_rao_distance: comparison.fisher_rao_distance,
            bhattacharyya_coefficient: comparison.bhattacharyya_coefficient,
            sectional_curvature: comparison.sectional_curvature,
            fisher_trace_a: fisher_left.trace(),
            fisher_trace_b: fisher_right.trace(),
            geodesic_velocity: comparison.fisher_rao_distance,
        })
    }

    pub fn encode_batch(&self, hints: &[NarrativeHint]) -> PureResult<Vec<DVector<f64>>> {
        hints.iter().map(|hint| self.encode(hint)).collect()
    }
}

/// Auditable Fisher-Rao relationship between two narrative observations.
#[derive(Clone, Debug, PartialEq)]
pub struct NarrativeBridgeCurvature {
    pub channel_a: String,
    pub channel_b: String,
    pub symmetric_kl: f64,
    pub fisher_rao_distance: f64,
    pub bhattacharyya_coefficient: f64,
    pub sectional_curvature: Option<f64>,
    pub fisher_trace_a: f64,
    pub fisher_trace_b: f64,
    pub geodesic_velocity: f64,
}

impl NarrativeBridgeCurvature {
    pub fn affinity(&self) -> f64 {
        self.bhattacharyya_coefficient
    }
}

pub fn compute_curvature_table(
    metric: &InformationGeometryMetric,
    hints: &[NarrativeHint],
) -> PureResult<Vec<NarrativeBridgeCurvature>> {
    let mut results = Vec::new();
    for left in 0..hints.len() {
        for right in left + 1..hints.len() {
            results.push(metric.bridge_slice(&hints[left], &hints[right])?);
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
            tags.iter().map(|tag| (*tag).to_owned()).collect(),
            intensity,
        )
    }

    #[test]
    fn builds_a_deterministic_fisher_rao_atlas() {
        let hints = vec![
            hint("alpha", &["spiral", "torch"], 0.6),
            hint("beta", &["spiral", "z-space"], 0.8),
            hint("gamma", &["torch", "narrative"], 0.5),
        ];
        let metric = InformationGeometryMetric::from_narratives(&hints).unwrap();
        assert_eq!(
            metric.tag_labels(),
            &["narrative", "spiral", "torch", "z-space"]
        );
        let comparison = metric.bridge_slice(&hints[0], &hints[1]).unwrap();
        assert!(comparison.symmetric_kl >= 0.0);
        assert!(comparison.fisher_rao_distance >= 0.0);
        assert!((0.0..=1.0).contains(&comparison.bhattacharyya_coefficient));
        assert_eq!(comparison.sectional_curvature, Some(0.25));
        assert_eq!(comparison.geodesic_velocity, comparison.fisher_rao_distance);
        assert!(comparison.fisher_trace_a > 0.0);
        assert!(comparison.fisher_trace_b > 0.0);
    }

    #[test]
    fn identical_narratives_have_unit_affinity_and_zero_distance() {
        let narrative = hint("same", &["spiral", "torch"], 1.0);
        let metric = InformationGeometryMetric::from_narratives(std::slice::from_ref(&narrative))
            .expect("valid atlas");
        let comparison = metric.bridge_slice(&narrative, &narrative).unwrap();

        assert!(comparison.fisher_rao_distance < 1.0e-12);
        assert!(comparison.geodesic_velocity < 1.0e-12);
        assert!((comparison.affinity() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn duplicate_tags_do_not_change_probability_encoding() {
        let atlas = vec![hint("atlas", &["spiral", "torch"], 1.0)];
        let metric = InformationGeometryMetric::from_narratives(&atlas).unwrap();
        let single = metric.encode(&hint("single", &["spiral"], 0.7)).unwrap();
        let duplicate = metric
            .encode(&hint("duplicate", &["spiral", "spiral"], 0.7))
            .unwrap();

        assert_eq!(single, duplicate);
    }

    #[test]
    fn curvature_table_has_all_pairs() {
        let hints = vec![
            hint("alpha", &["spiral"], 1.0),
            hint("beta", &["torch"], 1.0),
            hint("gamma", &["spiral", "torch"], 1.5),
        ];
        let metric = InformationGeometryMetric::from_narratives(&hints).unwrap();
        let table = compute_curvature_table(&metric, &hints).unwrap();
        assert_eq!(table.len(), 3);
        assert!(table
            .iter()
            .all(|entry| (0.0..=1.0).contains(&entry.affinity())));
    }

    #[test]
    fn invalid_or_empty_atlases_fail_closed() {
        assert!(InformationGeometryMetric::from_narratives(&[]).is_err());
        let invalid = hint("bad", &["tag"], f32::NAN);
        assert!(matches!(
            InformationGeometryMetric::from_narratives(&[invalid]),
            Err(TensorError::NonFiniteValue {
                label: "narrative intensity",
                ..
            })
        ));
    }
}
