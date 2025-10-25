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
//!
//! This module mirrors that structure with Rust primitives.  Each step is
//! encoded as a small data type so downstream code can inspect or evolve Z-space
//! geometries without having to re-derive the symbolic machinery.

use core::fmt;
use std::f64::consts::PI;

use nalgebra::{Matrix4, SymmetricEigen};
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
    use nalgebra::Vector4;

    #[test]
    fn minkowski_vacuum_is_flat() {
        let metric =
            LorentzianMetric::try_new(Matrix4::from_diagonal(&Vector4::new(-1.0, 1.0, 1.0, 1.0)))
                .unwrap();

        let first = MetricDerivatives::zero();
        let second = MetricSecondDerivatives::zero();
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
}
