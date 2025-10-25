// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! General relativity utilities that blend Lorentzian curvature with Z-space.
//!
//! The routines implemented here interpret local Lorentzian geometry as a
//! modulation source for Z-space pulses.  Given a metric tensor and its
//! derivatives, we compute the Christoffel symbols, Ricci tensor, scalar
//! curvature, and Einstein tensor.  The resulting energy-momentum budget can be
//! projected into the cooperative Z runtime as a [`crate::theory::zpulse::ZPulse`]
//! tagged with [`crate::theory::zpulse::ZSource::GW`].

use nalgebra::Matrix4;

use crate::theory::zpulse::{ZPulse, ZSource, ZSupport};

/// Number of spacetime dimensions supported by the GR helpers.
const DIM: usize = 4;

/// Alias describing the metric tensor in local coordinates.
pub type MetricTensor = Matrix4<f64>;

/// Alias describing the inverse metric tensor.
pub type InverseMetricTensor = Matrix4<f64>;

/// First-order partial derivatives of the metric tensor.
///
/// The index ordering follows `∂_k g_{μν}` where `k` is the spatial/temporal
/// derivative index.
pub type MetricDerivatives = [Matrix4<f64>; DIM];

/// Christoffel symbols Γ^μ_{νρ} packed in a 3rd-order tensor.
pub type ChristoffelSymbols = [[[f64; DIM]; DIM]; DIM];

/// Partial derivatives of the Christoffel symbols.
///
/// The index ordering is `∂_σ Γ^μ_{νρ}` where `σ` corresponds to the coordinate
/// derivative.
pub type ChristoffelDerivatives = [[[[f64; DIM]; DIM]; DIM]; DIM];

/// Rank-2 tensor alias used for curvature outputs.
pub type Rank2Tensor = Matrix4<f64>;

/// Relativistic patch describing the local metric and its inverse.
#[derive(Clone, Debug)]
pub struct RelativisticPatch {
    metric: MetricTensor,
    inverse_metric: InverseMetricTensor,
}

impl RelativisticPatch {
    /// Creates a new relativistic patch, ensuring the metric is symmetric and
    /// invertible.  The constructor falls back to the symmetric component when
    /// tiny numerical asymmetries are present so downstream curvature
    /// evaluations remain stable.
    pub fn new(metric: MetricTensor) -> Self {
        let symmetric = 0.5 * (metric.clone() + metric.transpose());
        let inverse_metric = symmetric
            .try_inverse()
            .expect("metric tensor must be invertible");
        Self {
            metric: symmetric,
            inverse_metric,
        }
    }

    /// Returns a Minkowski patch with signature `(+, -, -, -)`.
    pub fn minkowski() -> Self {
        let mut metric = Matrix4::zeros();
        metric[(0, 0)] = 1.0;
        metric[(1, 1)] = -1.0;
        metric[(2, 2)] = -1.0;
        metric[(3, 3)] = -1.0;
        Self::new(metric)
    }

    /// Accessor for the metric tensor.
    pub fn metric(&self) -> &MetricTensor {
        &self.metric
    }

    /// Accessor for the inverse metric tensor.
    pub fn inverse_metric(&self) -> &InverseMetricTensor {
        &self.inverse_metric
    }

    /// Computes the Christoffel symbols Γ^μ_{νρ} from the metric derivatives.
    pub fn christoffel(&self, derivatives: &MetricDerivatives) -> ChristoffelSymbols {
        let mut gamma = [[[0.0; DIM]; DIM]; DIM];
        for mu in 0..DIM {
            for nu in 0..DIM {
                for rho in 0..DIM {
                    let mut sum = 0.0;
                    for alpha in 0..DIM {
                        let term = derivatives[nu][(alpha, rho)] + derivatives[rho][(alpha, nu)]
                            - derivatives[alpha][(nu, rho)];
                        sum += self.inverse_metric[(mu, alpha)] * term;
                    }
                    gamma[mu][nu][rho] = 0.5 * sum;
                }
            }
        }
        gamma
    }

    /// Computes the Ricci tensor using pre-computed Christoffel symbols and
    /// their derivatives.
    pub fn ricci(
        &self,
        christoffel: &ChristoffelSymbols,
        christoffel_derivatives: &ChristoffelDerivatives,
    ) -> Rank2Tensor {
        let mut ricci = Matrix4::zeros();
        for mu in 0..DIM {
            for nu in 0..DIM {
                let mut term1 = 0.0;
                for lambda in 0..DIM {
                    term1 += christoffel_derivatives[lambda][lambda][mu][nu];
                }

                let mut term2 = 0.0;
                for lambda in 0..DIM {
                    term2 += christoffel_derivatives[nu][lambda][mu][lambda];
                }

                let mut term3 = 0.0;
                for lambda in 0..DIM {
                    for sigma in 0..DIM {
                        term3 += christoffel[lambda][mu][nu] * christoffel[sigma][lambda][sigma];
                    }
                }

                let mut term4 = 0.0;
                for lambda in 0..DIM {
                    for sigma in 0..DIM {
                        term4 += christoffel[sigma][mu][lambda] * christoffel[lambda][nu][sigma];
                    }
                }

                ricci[(mu, nu)] = term1 - term2 + term3 - term4;
            }
        }
        ricci
    }

    /// Computes the scalar curvature `R` from the Ricci tensor.
    pub fn scalar_curvature(&self, ricci: &Rank2Tensor) -> f64 {
        let mut scalar = 0.0;
        for mu in 0..DIM {
            for nu in 0..DIM {
                scalar += self.inverse_metric[(mu, nu)] * ricci[(mu, nu)];
            }
        }
        scalar
    }

    /// Computes the Einstein tensor `G_{μν} = R_{μν} - 1/2 g_{μν} R`.
    pub fn einstein_tensor(&self, ricci: &Rank2Tensor) -> Rank2Tensor {
        let scalar = self.scalar_curvature(ricci);
        let mut einstein = Matrix4::zeros();
        for mu in 0..DIM {
            for nu in 0..DIM {
                einstein[(mu, nu)] = ricci[(mu, nu)] - 0.5 * self.metric[(mu, nu)] * scalar;
            }
        }
        einstein
    }

    /// Projects the Einstein tensor into a Z-space pulse observation.
    ///
    /// The projection splits the tensor into three energy bands (Above/Here/
    /// Beneath) using the temporal row and spatial curvature budget.  The
    /// resulting `ZPulse` can be fed into the cooperative scheduler to bias the
    /// Z-space trajectory based on relativistic curvature.
    pub fn to_zpulse(&self, einstein: &Rank2Tensor, timestamp: u64, tempo: f32) -> ZPulse {
        let energy_density = einstein[(0, 0)].max(0.0) as f32;
        let spatial_trace = einstein[(1, 1)] + einstein[(2, 2)] + einstein[(3, 3)];
        let above = (einstein[(0, 1)].abs() + einstein[(0, 2)].abs()) as f32 * 0.5;
        let beneath = einstein[(0, 3)].abs() as f32;
        let here = (energy_density + spatial_trace.abs() as f32) * 0.5;

        let bands = (above, here, beneath);
        let support = ZSupport::from_band_energy(bands);
        let scalar = -self.scalar_curvature(einstein);

        ZPulse {
            source: ZSource::GW,
            ts: timestamp,
            tempo,
            band_energy: bands,
            drift: (above - beneath) / (bands.0 + bands.1 + bands.2 + f32::EPSILON),
            z_bias: scalar as f32,
            support,
            scale: None,
            quality: 1.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minkowski_patch_has_zero_curvature() {
        let patch = RelativisticPatch::minkowski();
        let zero_derivatives = [Matrix4::zeros(); DIM];
        let christoffel = patch.christoffel(&zero_derivatives);
        let zero_derivative_tensor = [[[[0.0; DIM]; DIM]; DIM]; DIM];
        let ricci = patch.ricci(&christoffel, &zero_derivative_tensor);
        let scalar = patch.scalar_curvature(&ricci);
        let einstein = patch.einstein_tensor(&ricci);

        assert!(ricci.iter().all(|x| x.abs() < 1e-12));
        assert!(einstein.iter().all(|x| x.abs() < 1e-12));
        assert!(scalar.abs() < 1e-12);
    }

    #[test]
    fn einstein_projection_produces_pulse() {
        let patch = RelativisticPatch::minkowski();
        let mut ricci = Matrix4::zeros();
        ricci[(0, 0)] = 2.0;
        ricci[(1, 1)] = 1.0;
        let einstein = patch.einstein_tensor(&ricci);
        let pulse = patch.to_zpulse(&einstein, 42, 1.5);

        assert_eq!(pulse.source, ZSource::GW);
        assert_eq!(pulse.ts, 42);
        assert!(!pulse.support.is_empty());
        assert!(pulse.z_bias.is_finite());
    }
}
