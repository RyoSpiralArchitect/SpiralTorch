// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! General relativity helpers for transplanting Lorentzian geometry onto the
//! abstract Z-space manifold described in `docs/general_relativity_zspace.md`.
//!
//! The guide enumerates a concrete workflow:
//! 1. Specify the smooth manifold and its coordinate charts.
//! 2. Introduce a Lorentzian metric and compute its compatible connection.
//! 3. Derive curvature tensors, the Einstein tensor, and diagnostics such as the
//!    Kretschmann invariant.
//! 4. Combine the curvature data with matter models via the Einstein field
//!    equation.
//! 5. Choose coordinate ansätze, topology, and boundary conditions when solving
//!    the system.
//! 6. When augmenting spacetime with extra Z-space parameters, assemble the
//!    product geometry using the helpers in this module.
//!
//! This module mirrors that structure with Rust primitives.  Each step is
//! encoded as a small data type so downstream code can inspect or evolve Z-space
//! geometries without having to re-derive the symbolic machinery.

use core::fmt;
use std::collections::HashMap;
use std::f64::consts::PI;

use nalgebra::{DMatrix, Matrix4, SymmetricEigen};
use thiserror::Error;

const DIM: usize = 4;

/// Error raised when the supplied metric fails the Lorentzian checks.
#[derive(Debug, Error, PartialEq)]
pub enum MetricError {
    /// Metric matrix is not symmetric up to numerical tolerance.
    #[error("metric tensor must be symmetric (|g_{{μν}} - g_{{νμ}}| < {0:?})")]
    NonSymmetric(f64),
    /// Metric tensor cannot be inverted, so it is degenerate.
    #[error("metric tensor is degenerate and cannot be inverted")]
    Degenerate,
    /// Signature is not Lorentzian (one negative, three positive directions).
    #[error("metric tensor is not Lorentzian (expected signature (-,+,+,+))")]
    NonLorentzian,
    /// Metric tensor must be square.
    #[error("metric tensor must be square but had shape {rows}×{cols}")]
    NonSquare { rows: usize, cols: usize },
    /// Internal metric must be positive-definite.
    #[error("internal metric tensor must be positive-definite")]
    NonPositiveDefinite,
    /// Mixed block dimensions must match the base and internal spaces.
    #[error(
        "cross-term block must have shape {expected_rows}×{expected_cols} but had {found_rows}×{found_cols}"
    )]
    CrossTermShape {
        expected_rows: usize,
        expected_cols: usize,
        found_rows: usize,
        found_cols: usize,
    },
    /// Warp factor must be strictly positive.
    #[error("warp factor must be strictly positive")]
    InvalidWarpFactor,
}

/// Coordinate chart covering a region of Z-space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatePatch {
    /// Name of the chart (for diagnostics).
    pub name: String,
    /// Coordinate labels (μ = 0,1,2,3).
    pub coordinates: [String; DIM],
}

impl CoordinatePatch {
    /// Creates a new coordinate patch with the supplied labels.
    pub fn new<N: Into<String>, S: Into<String>>(name: N, coordinates: [S; DIM]) -> Self {
        Self {
            name: name.into(),
            coordinates: coordinates.map(Into::into),
        }
    }
}

/// Abstract 4-dimensional manifold used to represent Z-space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZManifold {
    /// Human-readable label for the manifold instance.
    pub name: String,
    /// Coordinate patches covering the manifold.
    pub patches: Vec<CoordinatePatch>,
}

impl ZManifold {
    /// Returns the canonical Z-space manifold with the standard coordinate chart.
    pub fn canonical() -> Self {
        Self {
            name: "Z".into(),
            patches: vec![CoordinatePatch::new("global", ["t", "x", "y", "z"])],
        }
    }

    /// Inserts an additional chart covering a new region.
    pub fn add_patch(&mut self, patch: CoordinatePatch) {
        self.patches.push(patch);
    }

    /// Number of dimensions of the manifold (fixed to 4 here).
    pub fn dimension(&self) -> usize {
        DIM
    }
}

/// Coordinate chart covering the internal Z-space directions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InternalPatch {
    /// Name of the patch.
    pub name: String,
    /// Coordinate labels for the internal indices A, B, …
    pub coordinates: Vec<String>,
}

impl InternalPatch {
    /// Builds a new internal patch.
    pub fn new<N: Into<String>, S: Into<String>>(name: N, coordinates: Vec<S>) -> Self {
        Self {
            name: name.into(),
            coordinates: coordinates.into_iter().map(Into::into).collect(),
        }
    }
}

/// Internal Z-space regarded as a smooth manifold that augments spacetime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InternalSpace {
    /// Display name of the internal sector.
    pub name: String,
    /// Coordinate charts covering the internal directions.
    pub patches: Vec<InternalPatch>,
}

impl InternalSpace {
    /// Creates a new internal space with a single patch.
    pub fn new<N: Into<String>>(name: N, patch: InternalPatch) -> Self {
        Self {
            name: name.into(),
            patches: vec![patch],
        }
    }

    /// Number of internal dimensions (assumes all patches share the same arity).
    pub fn dimension(&self) -> usize {
        self.patches
            .first()
            .map(|patch| patch.coordinates.len())
            .unwrap_or(0)
    }

    /// Adds an additional patch.
    pub fn add_patch(&mut self, patch: InternalPatch) {
        self.patches.push(patch);
    }
}

/// Lorentzian metric tensor with signature (-,+,+,+).
#[derive(Clone, Debug, PartialEq)]
pub struct LorentzianMetric {
    components: Matrix4<f64>,
    inverse: Matrix4<f64>,
    signature: [i8; DIM],
}

impl LorentzianMetric {
    /// Constructs a metric tensor, verifying symmetry, non-degeneracy, and Lorentzian signature.
    pub fn try_new(components: Matrix4<f64>) -> Result<Self, MetricError> {
        const TOLERANCE: f64 = 1e-12;
        for i in 0..DIM {
            for j in 0..DIM {
                let diff = components[(i, j)] - components[(j, i)];
                if diff.abs() > TOLERANCE {
                    return Err(MetricError::NonSymmetric(TOLERANCE));
                }
            }
        }

        let Some(inverse) = components.try_inverse() else {
            return Err(MetricError::Degenerate);
        };

        let eigen = SymmetricEigen::new(components);
        let mut signature = [0_i8; DIM];
        let mut negative = 0;
        let mut positive = 0;
        let mut zero = 0;
        for (idx, value) in eigen.eigenvalues.iter().enumerate() {
            let entry = if *value > TOLERANCE {
                positive += 1;
                1
            } else if *value < -TOLERANCE {
                negative += 1;
                -1
            } else {
                zero += 1;
                0
            };
            signature[idx] = entry;
        }

        if negative != 1 || positive != 3 || zero != 0 {
            return Err(MetricError::NonLorentzian);
        }

        Ok(Self {
            components,
            inverse,
            signature,
        })
    }

    /// Returns a reference to the metric matrix.
    pub fn components(&self) -> &Matrix4<f64> {
        &self.components
    }

    /// Returns a reference to the inverse metric.
    pub fn inverse(&self) -> &Matrix4<f64> {
        &self.inverse
    }

    /// Determinant of the metric tensor.
    pub fn determinant(&self) -> f64 {
        self.components.determinant()
    }

    /// Computes the invariant volume element \(\sqrt{-g}\).
    pub fn volume_element(&self) -> Option<f64> {
        let det = self.determinant();
        if det >= 0.0 {
            None
        } else {
            Some((-det).sqrt())
        }
    }

    /// Returns the signature encoded as (-1, +1, +1, +1).
    pub fn signature(&self) -> [i8; DIM] {
        self.signature
    }
}

/// Positive-definite metric on the internal Z-space directions.
#[derive(Clone, Debug, PartialEq)]
pub struct InternalMetric {
    components: DMatrix<f64>,
    inverse: DMatrix<f64>,
}

impl InternalMetric {
    /// Constructs an internal metric, enforcing symmetry and positive-definiteness.
    pub fn try_new(components: DMatrix<f64>) -> Result<Self, MetricError> {
        const TOLERANCE: f64 = 1e-12;
        if components.nrows() != components.ncols() {
            return Err(MetricError::NonSquare {
                rows: components.nrows(),
                cols: components.ncols(),
            });
        }

        for i in 0..components.nrows() {
            for j in 0..components.ncols() {
                let diff = components[(i, j)] - components[(j, i)];
                if diff.abs() > TOLERANCE {
                    return Err(MetricError::NonSymmetric(TOLERANCE));
                }
            }
        }

        let Some(inverse) = components.clone().try_inverse() else {
            return Err(MetricError::Degenerate);
        };

        let eigen = SymmetricEigen::new(components.clone());
        if eigen.eigenvalues.iter().any(|value| *value <= TOLERANCE) {
            return Err(MetricError::NonPositiveDefinite);
        }

        Ok(Self {
            components,
            inverse,
        })
    }

    /// Returns the internal dimension.
    pub fn dimension(&self) -> usize {
        self.components.nrows()
    }

    /// Returns the metric components.
    pub fn components(&self) -> &DMatrix<f64> {
        &self.components
    }

    /// Returns the inverse internal metric.
    pub fn inverse(&self) -> &DMatrix<f64> {
        &self.inverse
    }
}

/// Mixed spacetime/internal block g_{μA} capturing gauge-like interactions.
#[derive(Clone, Debug, PartialEq)]
pub struct MixedBlock {
    components: DMatrix<f64>,
}

impl MixedBlock {
    /// Creates a mixed block with dimension checks.
    pub fn new(
        components: DMatrix<f64>,
        base_dim: usize,
        internal_dim: usize,
    ) -> Result<Self, MetricError> {
        if components.nrows() != base_dim || components.ncols() != internal_dim {
            return Err(MetricError::CrossTermShape {
                expected_rows: base_dim,
                expected_cols: internal_dim,
                found_rows: components.nrows(),
                found_cols: components.ncols(),
            });
        }
        Ok(Self { components })
    }

    /// Zero mixed block with the requested dimensions.
    pub fn zeros(base_dim: usize, internal_dim: usize) -> Self {
        Self {
            components: DMatrix::zeros(base_dim, internal_dim),
        }
    }

    /// Returns g_{μA}.
    pub fn component(&self, mu: usize, a: usize) -> f64 {
        self.components[(mu, a)]
    }

    /// Immutable access to the underlying matrix.
    pub fn components(&self) -> &DMatrix<f64> {
        &self.components
    }
}

/// Constant warp factor e^{2A} applied to the spacetime block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarpFactor {
    scale: f64,
}

impl WarpFactor {
    /// Builds a warp factor from its multiplier e^{2A}. The input must be strictly positive.
    pub fn from_multiplier(scale: f64) -> Result<Self, MetricError> {
        if scale <= 0.0 {
            return Err(MetricError::InvalidWarpFactor);
        }
        Ok(Self { scale })
    }

    /// Builds a warp factor from the exponent A so that the multiplier is exp(2A).
    pub fn from_exponent(exponent: f64) -> Result<Self, MetricError> {
        let scale = (2.0 * exponent).exp();
        Self::from_multiplier(scale)
    }

    /// Returns the multiplicative scale applied to the spacetime metric.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

/// Combined metric on the product manifold M × Z.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductMetric {
    base: LorentzianMetric,
    internal: InternalMetric,
    mixed: MixedBlock,
    warp: Option<WarpFactor>,
    block: DMatrix<f64>,
}

impl ProductMetric {
    /// Constructs the block metric according to
    ///
    /// g^IJ =
    /// ┌                 ┐
    /// │ e^{2A} g_{μν}   g_{μB} │
    /// │ g_{Aν}          h_{AB} │
    /// └                 ┘
    pub fn try_new(
        base: LorentzianMetric,
        internal: InternalMetric,
        mixed: Option<MixedBlock>,
        warp: Option<WarpFactor>,
    ) -> Result<Self, MetricError> {
        let internal_dim = internal.dimension();
        let mixed_block = match mixed {
            Some(block) => block,
            None => MixedBlock::zeros(DIM, internal_dim),
        };

        let mut block = DMatrix::zeros(DIM + internal_dim, DIM + internal_dim);
        let scale = warp.map_or(1.0, |factor| factor.scale());
        let base_components = base.components();
        for mu in 0..DIM {
            for nu in 0..DIM {
                block[(mu, nu)] = scale * base_components[(mu, nu)];
            }
        }

        let internal_components = internal.components();
        for a in 0..internal_dim {
            for b in 0..internal_dim {
                block[(DIM + a, DIM + b)] = internal_components[(a, b)];
            }
        }

        let mixed_components = mixed_block.components();
        for mu in 0..DIM {
            for a in 0..internal_dim {
                let value = mixed_components[(mu, a)];
                block[(mu, DIM + a)] = value;
                block[(DIM + a, mu)] = value;
            }
        }

        Ok(Self {
            base,
            internal,
            mixed: mixed_block,
            warp,
            block,
        })
    }

    /// Total dimension of the product manifold.
    pub fn total_dimension(&self) -> usize {
        DIM + self.internal.dimension()
    }

    /// Returns a reference to the Lorentzian spacetime metric.
    pub fn base(&self) -> &LorentzianMetric {
        &self.base
    }

    /// Returns a reference to the internal metric.
    pub fn internal(&self) -> &InternalMetric {
        &self.internal
    }

    /// Returns the mixed block g_{μA}.
    pub fn mixed(&self) -> &MixedBlock {
        &self.mixed
    }

    /// Warp factor applied to the spacetime metric block.
    pub fn warp(&self) -> Option<WarpFactor> {
        self.warp
    }

    /// Full block matrix representing g^IJ.
    pub fn block_matrix(&self) -> &DMatrix<f64> {
        &self.block
    }

    /// Effective 4D Newton constant obtained after compactifying the internal space.
    pub fn effective_newton_constant(
        &self,
        constants: &PhysicalConstants,
        internal_volume: f64,
    ) -> f64 {
        assert!(internal_volume > 0.0, "internal volume must be positive");
        constants.gravitational_constant / internal_volume
    }
}

/// Full product geometry containing the manifolds and their block metric.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductGeometry {
    spacetime: ZManifold,
    internal: InternalSpace,
    metric: ProductMetric,
}

impl ProductGeometry {
    /// Builds a new product geometry from its constituents.
    pub fn new(spacetime: ZManifold, internal: InternalSpace, metric: ProductMetric) -> Self {
        Self {
            spacetime,
            internal,
            metric,
        }
    }

    /// Returns the spacetime manifold M.
    pub fn spacetime(&self) -> &ZManifold {
        &self.spacetime
    }

    /// Returns the internal manifold Z.
    pub fn internal(&self) -> &InternalSpace {
        &self.internal
    }

    /// Returns the block metric on M × Z.
    pub fn metric(&self) -> &ProductMetric {
        &self.metric
    }

    /// Combined dimensionality of the product manifold.
    pub fn total_dimension(&self) -> usize {
        self.metric.total_dimension()
    }
}

/// First derivatives of the metric tensor (∂_ρ g_{μν}).
#[derive(Clone, Debug, PartialEq)]
pub struct MetricDerivatives {
    partials: [[[f64; DIM]; DIM]; DIM],
}

impl MetricDerivatives {
    /// Creates a zero derivative field.
    pub fn zero() -> Self {
        Self {
            partials: [[[0.0; DIM]; DIM]; DIM],
        }
    }

    /// Builds the structure from a closure `f(ρ, μ, ν)`.
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize, usize, usize) -> f64,
    {
        let mut partials = [[[0.0; DIM]; DIM]; DIM];
        for rho in 0..DIM {
            for mu in 0..DIM {
                for nu in 0..DIM {
                    partials[rho][mu][nu] = f(rho, mu, nu);
                }
            }
        }
        Self { partials }
    }

    /// Returns ∂_ρ g_{μν}.
    pub fn partial(&self, rho: usize, mu: usize, nu: usize) -> f64 {
        self.partials[rho][mu][nu]
    }
}

/// Second derivatives of the metric tensor (∂_λ∂_ρ g_{μν}).
#[derive(Clone, Debug, PartialEq)]
pub struct MetricSecondDerivatives {
    partials: [[[[f64; DIM]; DIM]; DIM]; DIM],
}

impl MetricSecondDerivatives {
    /// Zero second-derivative field.
    pub fn zero() -> Self {
        Self {
            partials: [[[[0.0; DIM]; DIM]; DIM]; DIM],
        }
    }

    /// Builds from a closure `f(λ, ρ, μ, ν)`.
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize, usize, usize, usize) -> f64,
    {
        let mut partials = [[[[0.0; DIM]; DIM]; DIM]; DIM];
        for lambda in 0..DIM {
            for rho in 0..DIM {
                for mu in 0..DIM {
                    for nu in 0..DIM {
                        partials[lambda][rho][mu][nu] = f(lambda, rho, mu, nu);
                    }
                }
            }
        }
        Self { partials }
    }

    /// Returns ∂_λ∂_ρ g_{μν}.
    pub fn partial(&self, lambda: usize, rho: usize, mu: usize, nu: usize) -> f64 {
        self.partials[lambda][rho][mu][nu]
    }
}

type Offset = [i8; DIM];

fn sample_metric<F>(
    cache: &mut HashMap<Offset, Matrix4<f64>>,
    metric_fn: &mut F,
    point: [f64; DIM],
    offset: Offset,
    step: f64,
) -> Matrix4<f64>
where
    F: FnMut(&[f64; DIM]) -> Matrix4<f64>,
{
    if let Some(value) = cache.get(&offset) {
        return value.clone();
    }

    let mut coords = point;
    for dim in 0..DIM {
        coords[dim] += (offset[dim] as f64) * step;
    }

    let matrix = metric_fn(&coords);
    cache.insert(offset, matrix.clone());
    matrix
}

/// Approximates metric derivatives via second-order central finite differences.
pub fn finite_difference_metric_data<F>(
    mut metric_fn: F,
    point: [f64; DIM],
    step: f64,
) -> Result<(LorentzianMetric, MetricDerivatives, MetricSecondDerivatives), MetricError>
where
    F: FnMut(&[f64; DIM]) -> Matrix4<f64>,
{
    assert!(step > 0.0, "finite difference step must be positive");

    let base = metric_fn(&point);
    let metric = LorentzianMetric::try_new(base.clone())?;

    let mut cache: HashMap<Offset, Matrix4<f64>> = HashMap::new();
    cache.insert([0_i8; DIM], base);

    let mut first = [[[0.0; DIM]; DIM]; DIM];
    for rho in 0..DIM {
        let mut plus = [0_i8; DIM];
        plus[rho] = 1;
        let g_plus = sample_metric(&mut cache, &mut metric_fn, point, plus, step);

        let mut minus = [0_i8; DIM];
        minus[rho] = -1;
        let g_minus = sample_metric(&mut cache, &mut metric_fn, point, minus, step);

        for mu in 0..DIM {
            for nu in 0..DIM {
                first[rho][mu][nu] = (g_plus[(mu, nu)] - g_minus[(mu, nu)]) / (2.0 * step);
            }
        }
    }

    let base_matrix = metric.components().clone();
    let mut second = [[[[0.0; DIM]; DIM]; DIM]; DIM];
    for lambda in 0..DIM {
        for rho in 0..DIM {
            if lambda == rho {
                let mut plus = [0_i8; DIM];
                plus[lambda] = 1;
                let g_plus = sample_metric(&mut cache, &mut metric_fn, point, plus, step);

                let mut minus = [0_i8; DIM];
                minus[lambda] = -1;
                let g_minus = sample_metric(&mut cache, &mut metric_fn, point, minus, step);

                for mu in 0..DIM {
                    for nu in 0..DIM {
                        second[lambda][rho][mu][nu] =
                            (g_plus[(mu, nu)] - 2.0 * base_matrix[(mu, nu)] + g_minus[(mu, nu)])
                                / (step * step);
                    }
                }
            } else {
                let mut pp = [0_i8; DIM];
                pp[lambda] = 1;
                pp[rho] = 1;
                let g_pp = sample_metric(&mut cache, &mut metric_fn, point, pp, step);

                let mut pm = [0_i8; DIM];
                pm[lambda] = 1;
                pm[rho] = -1;
                let g_pm = sample_metric(&mut cache, &mut metric_fn, point, pm, step);

                let mut mp = [0_i8; DIM];
                mp[lambda] = -1;
                mp[rho] = 1;
                let g_mp = sample_metric(&mut cache, &mut metric_fn, point, mp, step);

                let mut mm = [0_i8; DIM];
                mm[lambda] = -1;
                mm[rho] = -1;
                let g_mm = sample_metric(&mut cache, &mut metric_fn, point, mm, step);

                for mu in 0..DIM {
                    for nu in 0..DIM {
                        second[lambda][rho][mu][nu] =
                            (g_pp[(mu, nu)] - g_pm[(mu, nu)] - g_mp[(mu, nu)] + g_mm[(mu, nu)])
                                / (4.0 * step * step);
                    }
                }
            }
        }
    }

    Ok((
        metric,
        MetricDerivatives { partials: first },
        MetricSecondDerivatives { partials: second },
    ))
}

/// Christoffel symbols Γ^ρ_{μν} of the Levi-Civita connection.
#[derive(Clone, Debug, PartialEq)]
pub struct LeviCivitaConnection {
    coefficients: [[[f64; DIM]; DIM]; DIM],
}

impl LeviCivitaConnection {
    /// Builds the unique torsion-free, metric-compatible connection.
    pub fn from_metric(metric: &LorentzianMetric, derivatives: &MetricDerivatives) -> Self {
        let mut coefficients = [[[0.0; DIM]; DIM]; DIM];
        let g_inv = metric.inverse();

        for rho in 0..DIM {
            for mu in 0..DIM {
                for nu in 0..DIM {
                    let mut sum = 0.0;
                    for sigma in 0..DIM {
                        let term = derivatives.partial(mu, nu, sigma)
                            + derivatives.partial(nu, mu, sigma)
                            - derivatives.partial(sigma, mu, nu);
                        sum += g_inv[(rho, sigma)] * term;
                    }
                    coefficients[rho][mu][nu] = 0.5 * sum;
                }
            }
        }

        Self { coefficients }
    }

    /// Returns Γ^ρ_{μν}.
    pub fn coefficient(&self, rho: usize, mu: usize, nu: usize) -> f64 {
        self.coefficients[rho][mu][nu]
    }

    /// Immutable access to the raw tensor.
    pub fn coefficients(&self) -> &[[[f64; DIM]; DIM]; DIM] {
        &self.coefficients
    }
}

/// Directional derivatives ∂_λ Γ^ρ_{μν}.
#[derive(Clone, Debug, PartialEq)]
pub struct ConnectionDerivatives {
    partials: [[[[f64; DIM]; DIM]; DIM]; DIM],
}

impl ConnectionDerivatives {
    /// Computes ∂_λ Γ^ρ_{μν} from metric derivatives up to second order.
    pub fn from_metric(
        metric: &LorentzianMetric,
        first: &MetricDerivatives,
        second: &MetricSecondDerivatives,
    ) -> Self {
        let mut partials = [[[[0.0; DIM]; DIM]; DIM]; DIM];
        let g_inv = metric.inverse();

        for lambda in 0..DIM {
            // ∂_λ g^{ρσ} = - g^{ρα} g^{σβ} ∂_λ g_{αβ}
            let mut d_inverse = [[0.0; DIM]; DIM];
            for rho in 0..DIM {
                for sigma in 0..DIM {
                    let mut sum = 0.0;
                    for alpha in 0..DIM {
                        for beta in 0..DIM {
                            sum -= g_inv[(rho, alpha)]
                                * g_inv[(sigma, beta)]
                                * first.partial(lambda, alpha, beta);
                        }
                    }
                    d_inverse[rho][sigma] = sum;
                }
            }

            for rho in 0..DIM {
                for mu in 0..DIM {
                    for nu in 0..DIM {
                        let mut sum = 0.0;
                        for sigma in 0..DIM {
                            let first_sym = first.partial(mu, nu, sigma)
                                + first.partial(nu, mu, sigma)
                                - first.partial(sigma, mu, nu);
                            let second_sym = second.partial(lambda, mu, nu, sigma)
                                + second.partial(lambda, nu, mu, sigma)
                                - second.partial(lambda, sigma, mu, nu);
                            sum += d_inverse[rho][sigma] * first_sym
                                + g_inv[(rho, sigma)] * second_sym;
                        }
                        partials[lambda][rho][mu][nu] = 0.5 * sum;
                    }
                }
            }
        }

        Self { partials }
    }

    /// Returns ∂_λ Γ^ρ_{μν}.
    pub fn partial(&self, lambda: usize, rho: usize, mu: usize, nu: usize) -> f64 {
        self.partials[lambda][rho][mu][nu]
    }
}

/// Riemann curvature tensor R^σ_{ μ ν ρ }.
#[derive(Clone, Debug, PartialEq)]
pub struct RiemannTensor {
    components: [[[[f64; DIM]; DIM]; DIM]; DIM],
}

impl RiemannTensor {
    /// Builds the tensor from the Levi-Civita connection.
    pub fn from_connection(
        connection: &LeviCivitaConnection,
        derivatives: &ConnectionDerivatives,
    ) -> Self {
        let mut components = [[[[0.0; DIM]; DIM]; DIM]; DIM];
        let gamma = connection.coefficients();

        for sigma in 0..DIM {
            for mu in 0..DIM {
                for nu in 0..DIM {
                    for rho in 0..DIM {
                        let mut value = derivatives.partial(nu, sigma, mu, rho)
                            - derivatives.partial(rho, sigma, mu, nu);
                        for lambda in 0..DIM {
                            value += gamma[sigma][lambda][nu] * gamma[lambda][mu][rho]
                                - gamma[sigma][lambda][rho] * gamma[lambda][mu][nu];
                        }
                        components[sigma][mu][nu][rho] = value;
                    }
                }
            }
        }

        Self { components }
    }

    /// Returns R^σ_{ μ ν ρ }.
    pub fn component(&self, sigma: usize, mu: usize, nu: usize, rho: usize) -> f64 {
        self.components[sigma][mu][nu][rho]
    }

    /// Immutable access to the raw tensor.
    pub fn components(&self) -> &[[[[f64; DIM]; DIM]; DIM]; DIM] {
        &self.components
    }
}
/// Ricci tensor R_{μν} obtained by contracting the Riemann tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct RicciTensor {
    components: [[f64; DIM]; DIM],
}

impl RicciTensor {
    /// Contracts R^σ_{ μ σ ν }.
    pub fn from_riemann(riemann: &RiemannTensor) -> Self {
        let mut components = [[0.0; DIM]; DIM];
        for mu in 0..DIM {
            for nu in 0..DIM {
                let mut sum = 0.0;
                for sigma in 0..DIM {
                    sum += riemann.component(sigma, mu, sigma, nu);
                }
                components[mu][nu] = sum;
            }
        }
        Self { components }
    }

    /// Returns R_{μν}.
    pub fn component(&self, mu: usize, nu: usize) -> f64 {
        self.components[mu][nu]
    }

    /// Immutable access to the raw matrix.
    pub fn components(&self) -> &[[f64; DIM]; DIM] {
        &self.components
    }

    /// Computes the scalar curvature R = g^{μν} R_{μν}.
    pub fn scalar_curvature(&self, metric: &LorentzianMetric) -> f64 {
        let mut scalar = 0.0;
        let g_inv = metric.inverse();
        for mu in 0..DIM {
            for nu in 0..DIM {
                scalar += g_inv[(mu, nu)] * self.components[mu][nu];
            }
        }
        scalar
    }

    /// Computes R_{μν} R^{μν} for diagnostics.
    pub fn contracted_square(&self, metric: &LorentzianMetric) -> f64 {
        let mut raised = [[0.0; DIM]; DIM];
        let g_inv = metric.inverse();
        for mu in 0..DIM {
            for nu in 0..DIM {
                let mut sum = 0.0;
                for alpha in 0..DIM {
                    sum += g_inv[(mu, alpha)] * self.components[alpha][nu];
                }
                raised[mu][nu] = sum;
            }
        }

        let mut contraction = 0.0;
        for mu in 0..DIM {
            for nu in 0..DIM {
                contraction += self.components[mu][nu] * raised[mu][nu];
            }
        }
        contraction
    }
}

/// Einstein tensor G_{μν} = R_{μν} - ½ R g_{μν}.
#[derive(Clone, Debug, PartialEq)]
pub struct EinsteinTensor {
    components: [[f64; DIM]; DIM],
}

impl EinsteinTensor {
    /// Builds the tensor from Ricci data and the metric.
    pub fn from_ricci(
        ricci: &RicciTensor,
        scalar_curvature: f64,
        metric: &LorentzianMetric,
    ) -> Self {
        let mut components = [[0.0; DIM]; DIM];
        let g = metric.components();
        for mu in 0..DIM {
            for nu in 0..DIM {
                components[mu][nu] = ricci.component(mu, nu) - 0.5 * scalar_curvature * g[(mu, nu)];
            }
        }
        Self { components }
    }

    /// Returns G_{μν}.
    pub fn component(&self, mu: usize, nu: usize) -> f64 {
        self.components[mu][nu]
    }

    /// Immutable access to the components.
    pub fn components(&self) -> &[[f64; DIM]; DIM] {
        &self.components
    }
}

/// Energy-momentum tensor T_{μν}.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyMomentumTensor {
    components: [[f64; DIM]; DIM],
}

impl EnergyMomentumTensor {
    /// Constructs a symmetric energy-momentum tensor.
    pub fn try_new(components: [[f64; DIM]; DIM], tolerance: f64) -> Result<Self, MetricError> {
        for mu in 0..DIM {
            for nu in 0..DIM {
                if (components[mu][nu] - components[nu][mu]).abs() > tolerance {
                    return Err(MetricError::NonSymmetric(tolerance));
                }
            }
        }
        Ok(Self { components })
    }

    /// Zero energy-momentum tensor (vacuum solution).
    pub fn zero() -> Self {
        Self {
            components: [[0.0; DIM]; DIM],
        }
    }

    /// Returns T_{μν}.
    pub fn component(&self, mu: usize, nu: usize) -> f64 {
        self.components[mu][nu]
    }

    /// Immutable access to the components.
    pub fn components(&self) -> &[[f64; DIM]; DIM] {
        &self.components
    }
}

/// Physical constants required by the Einstein field equation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalConstants {
    /// Gravitational constant G.
    pub gravitational_constant: f64,
    /// Speed of light c.
    pub speed_of_light: f64,
}

impl PhysicalConstants {
    /// Creates a new set of constants.
    pub fn new(gravitational_constant: f64, speed_of_light: f64) -> Self {
        Self {
            gravitational_constant,
            speed_of_light,
        }
    }

    /// Prefactor 8πG/c⁴ multiplying the energy-momentum tensor.
    pub fn einstein_prefactor(&self) -> f64 {
        8.0 * PI * self.gravitational_constant / self.speed_of_light.powi(4)
    }
}

/// Left-hand side of the Einstein field equation.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldEquation {
    lhs: [[f64; DIM]; DIM],
}

impl FieldEquation {
    /// Builds `G_{μν} + Λ g_{μν}` for a given cosmological constant Λ.
    pub fn vacuum(
        einstein: &EinsteinTensor,
        cosmological_constant: f64,
        metric: &LorentzianMetric,
    ) -> Self {
        let mut lhs = [[0.0; DIM]; DIM];
        let g = metric.components();
        for mu in 0..DIM {
            for nu in 0..DIM {
                lhs[mu][nu] = einstein.component(mu, nu) + cosmological_constant * g[(mu, nu)];
            }
        }
        Self { lhs }
    }

    /// Residual `G_{μν} + Λ g_{μν} - 8πG/(c⁴) T_{μν}`.
    pub fn residual(
        &self,
        energy_momentum: &EnergyMomentumTensor,
        constants: &PhysicalConstants,
    ) -> [[f64; DIM]; DIM] {
        let prefactor = constants.einstein_prefactor();
        let mut residual = [[0.0; DIM]; DIM];
        for mu in 0..DIM {
            for nu in 0..DIM {
                residual[mu][nu] = self.lhs[mu][nu] - prefactor * energy_momentum.component(mu, nu);
            }
        }
        residual
    }
}

/// Coordinate ansatz capturing Z-space symmetries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymmetryAnsatz {
    /// Static, spherically symmetric configuration (Schwarzschild-like).
    StaticSpherical,
    /// Homogeneous and isotropic configuration (FRW-like).
    HomogeneousIsotropic,
    /// Custom ansatz described via free-form text.
    Custom(String),
}

impl SymmetryAnsatz {
    /// Human-readable description.
    pub fn description(&self) -> &str {
        match self {
            SymmetryAnsatz::StaticSpherical => "Static, spherically symmetric metric ansatz",
            SymmetryAnsatz::HomogeneousIsotropic => "Homogeneous and isotropic cosmological ansatz",
            SymmetryAnsatz::Custom(description) => description.as_str(),
        }
    }

    /// Returns a seed metric consistent with the ansatz (using unit parameters).
    pub fn seed_metric(&self) -> Matrix4<f64> {
        match self {
            SymmetryAnsatz::StaticSpherical => Matrix4::new(
                -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ),
            SymmetryAnsatz::HomogeneousIsotropic => Matrix4::new(
                -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ),
            SymmetryAnsatz::Custom(_) => Matrix4::identity(),
        }
    }
}
/// Topological class of the manifold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Topology {
    /// Simply connected ℝ⁴.
    R4,
    /// Product space ℝ³ × S¹.
    R3xS1,
    /// Custom topology with textual description.
    Custom(String),
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Topology::R4 => write!(f, "ℝ^4"),
            Topology::R3xS1 => write!(f, "ℝ^3 × S^1"),
            Topology::Custom(description) => write!(f, "{description}"),
        }
    }
}

/// Boundary condition kind applied during solution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryConditionKind {
    /// Asymptotic flatness at spatial infinity.
    AsymptoticallyFlat,
    /// Regularity at a selected radius or hypersurface.
    Regularity,
    /// Periodic boundary (useful for compactified directions).
    Periodic,
    /// Custom constraint described textually.
    Custom(String),
}

impl BoundaryConditionKind {
    /// Human-readable description.
    pub fn description(&self) -> &str {
        match self {
            BoundaryConditionKind::AsymptoticallyFlat => "Asymptotically flat",
            BoundaryConditionKind::Regularity => "Regularity condition",
            BoundaryConditionKind::Periodic => "Periodic boundary",
            BoundaryConditionKind::Custom(description) => description.as_str(),
        }
    }
}

/// Boundary condition specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryCondition {
    /// Kind of boundary condition enforced.
    pub kind: BoundaryConditionKind,
    /// Optional location or hypersurface label.
    pub location: Option<String>,
    /// Additional notes describing the constraint.
    pub notes: Option<String>,
}

impl BoundaryCondition {
    /// Creates a new boundary condition.
    pub fn new<K: Into<BoundaryConditionKind>>(kind: K) -> Self {
        Self {
            kind: kind.into(),
            location: None,
            notes: None,
        }
    }

    /// Adds a location descriptor.
    pub fn with_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Adds descriptive notes.
    pub fn with_notes<S: Into<String>>(mut self, notes: S) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

impl From<String> for BoundaryConditionKind {
    fn from(value: String) -> Self {
        BoundaryConditionKind::Custom(value)
    }
}

impl<'a> From<&'a str> for BoundaryConditionKind {
    fn from(value: &'a str) -> Self {
        BoundaryConditionKind::Custom(value.to_owned())
    }
}

/// Curvature diagnostics (Kretschmann invariant, etc.).
#[derive(Clone, Debug, PartialEq)]
pub struct CurvatureDiagnostics {
    /// Scalar curvature R.
    pub scalar_curvature: f64,
    /// Ricci contraction R_{μν} R^{μν}.
    pub ricci_square: f64,
    /// Kretschmann invariant R_{μνρσ} R^{μνρσ}.
    pub kretschmann: f64,
}

impl CurvatureDiagnostics {
    /// Computes invariants from the supplied curvature tensors.
    pub fn from_fields(
        riemann: &RiemannTensor,
        metric: &LorentzianMetric,
        ricci: &RicciTensor,
        scalar_curvature: f64,
    ) -> Self {
        let g = metric.components();
        let g_inv = metric.inverse();

        // Lower the first index of Riemann: R_{μνρσ} = g_{μα} R^α_{νρσ}.
        let mut riemann_lower = [[[[0.0; DIM]; DIM]; DIM]; DIM];
        for mu in 0..DIM {
            for nu in 0..DIM {
                for rho in 0..DIM {
                    for sigma in 0..DIM {
                        let mut sum = 0.0;
                        for alpha in 0..DIM {
                            sum += g[(mu, alpha)] * riemann.component(alpha, nu, rho, sigma);
                        }
                        riemann_lower[mu][nu][rho][sigma] = sum;
                    }
                }
            }
        }

        // Raise all indices to compute R^{μνρσ}.
        let mut riemann_all_up = [[[[0.0; DIM]; DIM]; DIM]; DIM];
        for mu in 0..DIM {
            for nu in 0..DIM {
                for rho in 0..DIM {
                    for sigma in 0..DIM {
                        let mut sum = 0.0;
                        for alpha in 0..DIM {
                            for beta in 0..DIM {
                                for gamma in 0..DIM {
                                    for delta in 0..DIM {
                                        sum += g_inv[(mu, alpha)]
                                            * g_inv[(nu, beta)]
                                            * g_inv[(rho, gamma)]
                                            * g_inv[(sigma, delta)]
                                            * riemann_lower[alpha][beta][gamma][delta];
                                    }
                                }
                            }
                        }
                        riemann_all_up[mu][nu][rho][sigma] = sum;
                    }
                }
            }
        }

        let mut kretschmann = 0.0;
        for mu in 0..DIM {
            for nu in 0..DIM {
                for rho in 0..DIM {
                    for sigma in 0..DIM {
                        kretschmann +=
                            riemann_lower[mu][nu][rho][sigma] * riemann_all_up[mu][nu][rho][sigma];
                    }
                }
            }
        }

        let ricci_square = ricci.contracted_square(metric);

        Self {
            scalar_curvature,
            ricci_square,
            kretschmann,
        }
    }
}

/// Bundle describing the GR configuration on Z-space.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralRelativityModel {
    /// Underlying manifold description.
    pub manifold: ZManifold,
    /// Metric tensor.
    pub metric: LorentzianMetric,
    /// Levi-Civita connection.
    pub connection: LeviCivitaConnection,
    /// Riemann curvature.
    pub riemann: RiemannTensor,
    /// Ricci tensor.
    pub ricci: RicciTensor,
    /// Einstein tensor.
    pub einstein: EinsteinTensor,
    /// Curvature diagnostics.
    pub diagnostics: CurvatureDiagnostics,
    /// Coordinate ansatz employed.
    pub symmetry: SymmetryAnsatz,
    /// Topology of the manifold.
    pub topology: Topology,
    /// Boundary conditions enforced when solving the field equations.
    pub boundary_conditions: Vec<BoundaryCondition>,
}

impl GeneralRelativityModel {
    /// Creates a new model from metric derivatives and matter-independent data.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manifold: ZManifold,
        metric: LorentzianMetric,
        first: MetricDerivatives,
        second: MetricSecondDerivatives,
        symmetry: SymmetryAnsatz,
        topology: Topology,
        boundary_conditions: Vec<BoundaryCondition>,
    ) -> Self {
        let connection = LeviCivitaConnection::from_metric(&metric, &first);
        let connection_derivatives = ConnectionDerivatives::from_metric(&metric, &first, &second);
        let riemann = RiemannTensor::from_connection(&connection, &connection_derivatives);
        let ricci = RicciTensor::from_riemann(&riemann);
        let scalar = ricci.scalar_curvature(&metric);
        let einstein = EinsteinTensor::from_ricci(&ricci, scalar, &metric);
        let diagnostics = CurvatureDiagnostics::from_fields(&riemann, &metric, &ricci, scalar);

        Self {
            manifold,
            metric,
            connection,
            riemann,
            ricci,
            einstein,
            diagnostics,
            symmetry,
            topology,
            boundary_conditions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, Vector4};

    #[test]
    fn minkowski_vacuum_is_flat() {
        let (metric, first, second) = finite_difference_metric_data(
            |_: &[f64; DIM]| Matrix4::from_diagonal(&Vector4::new(-1.0, 1.0, 1.0, 1.0)),
            [0.0; DIM],
            1e-3,
        )
        .unwrap();
        let connection = LeviCivitaConnection::from_metric(&metric, &first);
        for rho in 0..DIM {
            for mu in 0..DIM {
                for nu in 0..DIM {
                    assert_relative_eq!(connection.coefficient(rho, mu, nu), 0.0, epsilon = 1e-12);
                }
            }
        }

        let connection_derivatives = ConnectionDerivatives::from_metric(&metric, &first, &second);
        for lambda in 0..DIM {
            for rho in 0..DIM {
                for mu in 0..DIM {
                    for nu in 0..DIM {
                        assert_relative_eq!(
                            connection_derivatives.partial(lambda, rho, mu, nu),
                            0.0,
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }

        let riemann = RiemannTensor::from_connection(&connection, &connection_derivatives);
        for sigma in 0..DIM {
            for mu in 0..DIM {
                for nu in 0..DIM {
                    for rho in 0..DIM {
                        assert_relative_eq!(
                            riemann.component(sigma, mu, nu, rho),
                            0.0,
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }

        let ricci = RicciTensor::from_riemann(&riemann);
        for mu in 0..DIM {
            for nu in 0..DIM {
                assert_relative_eq!(ricci.component(mu, nu), 0.0, epsilon = 1e-12);
            }
        }

        let scalar = ricci.scalar_curvature(&metric);
        assert_relative_eq!(scalar, 0.0, epsilon = 1e-12);

        let einstein = EinsteinTensor::from_ricci(&ricci, scalar, &metric);
        for mu in 0..DIM {
            for nu in 0..DIM {
                assert_relative_eq!(einstein.component(mu, nu), 0.0, epsilon = 1e-12);
            }
        }

        let diagnostics = CurvatureDiagnostics::from_fields(&riemann, &metric, &ricci, scalar);
        assert_relative_eq!(diagnostics.kretschmann, 0.0, epsilon = 1e-12);
        assert_relative_eq!(diagnostics.ricci_square, 0.0, epsilon = 1e-12);
        assert_relative_eq!(diagnostics.scalar_curvature, 0.0, epsilon = 1e-12);

        let field_equation = FieldEquation::vacuum(&einstein, 0.0, &metric);
        let constants = PhysicalConstants::new(6.67430e-11, 299_792_458.0);
        let residual = field_equation.residual(&EnergyMomentumTensor::zero(), &constants);
        for mu in 0..DIM {
            for nu in 0..DIM {
                assert_relative_eq!(residual[mu][nu], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn topology_and_boundary_metadata_round_trip() {
        let mut manifold = ZManifold::canonical();
        manifold.add_patch(CoordinatePatch::new("polar cap", ["t", "r", "θ", "φ"]));

        let boundary = BoundaryCondition::new(BoundaryConditionKind::AsymptoticallyFlat)
            .with_location("r → ∞")
            .with_notes("Demand Minkowski fall-off");

        assert_eq!(manifold.dimension(), 4);
        assert_eq!(manifold.patches.len(), 2);
        assert_eq!(boundary.kind.description(), "Asymptotically flat");
        assert_eq!(boundary.location.as_deref(), Some("r → ∞"));
        assert_eq!(boundary.notes.as_deref(), Some("Demand Minkowski fall-off"));

        let symmetry = SymmetryAnsatz::StaticSpherical;
        assert_eq!(
            symmetry.description(),
            "Static, spherically symmetric metric ansatz"
        );
        let seed = symmetry.seed_metric();
        assert_relative_eq!(seed[(0, 0)], -1.0);
        assert_relative_eq!(seed[(1, 1)], 1.0);

        let topo = Topology::R3xS1;
        assert_eq!(topo.to_string(), "ℝ^3 × S^1");
    }

    #[test]
    fn product_metric_composes_blocks() {
        let base =
            LorentzianMetric::try_new(Matrix4::from_diagonal(&Vector4::new(-1.0, 1.0, 1.0, 1.0)))
                .unwrap();

        let internal =
            InternalMetric::try_new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 4.0])).unwrap();
        let mixed = MixedBlock::new(
            DMatrix::from_row_slice(
                DIM,
                internal.dimension(),
                &[0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0],
            ),
            DIM,
            internal.dimension(),
        )
        .unwrap();
        let warp = WarpFactor::from_multiplier(4.0).unwrap();

        let geometry =
            ProductMetric::try_new(base.clone(), internal.clone(), Some(mixed), Some(warp))
                .unwrap();
        assert_eq!(geometry.total_dimension(), DIM + internal.dimension());

        let block = geometry.block_matrix();
        assert_eq!(block.nrows(), DIM + internal.dimension());
        assert_eq!(block.ncols(), DIM + internal.dimension());
        assert_relative_eq!(block[(0, 0)], -4.0, epsilon = 1e-12);
        assert_relative_eq!(block[(4, 4)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(block[(5, 5)], 4.0, epsilon = 1e-12);
        assert_relative_eq!(block[(1, 4)], 0.5, epsilon = 1e-12);
        assert_relative_eq!(block[(4, 1)], 0.5, epsilon = 1e-12);
        assert_relative_eq!(block[(2, 5)], -0.5, epsilon = 1e-12);
        assert_relative_eq!(block[(5, 2)], -0.5, epsilon = 1e-12);

        let constants = PhysicalConstants::new(6.67430e-11, 299_792_458.0);
        let volume = (2.0 * PI) * (4.0_f64.sqrt());
        let effective = geometry.effective_newton_constant(&constants, volume);
        assert_relative_eq!(
            effective,
            constants.gravitational_constant / volume,
            epsilon = 1e-16
        );

        let mixed_block = geometry.mixed();
        assert_relative_eq!(mixed_block.component(1, 0), 0.5, epsilon = 1e-12);
        assert_relative_eq!(mixed_block.component(2, 1), -0.5, epsilon = 1e-12);
        assert!(geometry.warp().is_some());
        assert_relative_eq!(geometry.warp().unwrap().scale(), 4.0, epsilon = 1e-12);

        let mut spacetime = ZManifold::canonical();
        spacetime.add_patch(CoordinatePatch::new("static patch", ["t", "r", "θ", "φ"]));
        let mut internal_space = InternalSpace::new(
            "compact Z",
            InternalPatch::new("torus chart", vec!["ψ", "χ"]),
        );
        internal_space.add_patch(InternalPatch::new("alt chart", vec!["u", "v"]));

        let product =
            ProductGeometry::new(spacetime.clone(), internal_space.clone(), geometry.clone());
        assert_eq!(product.total_dimension(), DIM + internal.dimension());
        assert_eq!(product.spacetime().patches.len(), spacetime.patches.len());
        assert_eq!(product.internal().dimension(), internal_space.dimension());
        assert!(product.metric().warp().is_some());
    }
}
