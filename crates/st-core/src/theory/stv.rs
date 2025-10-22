// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! SpinoTensorVector kernel algebra.
//!
//! This module implements the numerical core required by the Z-space
//! documentation: construction of the mixed tensor `T = S + W`, evaluation of
//! its determinant, closed-form computation of the scalar invariants `α` and
//! `β`, and causal classification of the resulting kernel vector.

use std::fmt;

use nalgebra as na;
use num_complex::Complex;

/// Alias for a spatial three-vector.
pub type Vec3 = [f64; 3];

/// Alias for a Minkowski four-vector with signature `(+, -, -, -)`.
pub type Vec4 = [f64; 4];

/// Alias for a two-component complex Pauli spinor.
pub type SpinorComponents = [Complex<f64>; 2];

/// Numerical tolerance used when checking symmetry or degeneracy.
const DEFAULT_TOLERANCE: f64 = 1e-9;

/// Errors that can be produced by the SpinoTensorVector algebra routines.
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq)]
pub enum StvError {
    /// The spatial tensor `A` must be symmetric within a small tolerance.
    #[error("spatial tensor must be symmetric (max deviation {max_deviation:.3e} exceeds tolerance {tolerance:.3e})")]
    NonSymmetric { max_deviation: f64, tolerance: f64 },
    /// The matrix `D = A + Ω(ω)` became singular.
    #[error("matrix D is singular (determinant ≈ 0)")]
    SingularD,
    /// The matrix `D D^⊤` became singular while evaluating β.
    #[error("matrix D·Dᵀ is singular (determinant ≈ 0)")]
    SingularDdT,
    /// The whitening step yielded a non-positive eigenvalue for `(D Dᵀ)⁻¹`.
    #[error(
        "intersection requires positive definite D·Dᵀ (encountered eigenvalue {eigenvalue:.3e})"
    )]
    NonPositiveGamma { eigenvalue: f64 },
    /// The supplied Pauli spinor had zero norm.
    #[error("spinor must have non-zero norm")]
    ZeroSpinor,
}

/// Normalised Pauli spinor with unit probability mass.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliSpinor {
    components: SpinorComponents,
}

impl PauliSpinor {
    /// Constructs a new Pauli spinor, normalising the components so that the
    /// probability amplitude sums to one.
    pub fn new(components: SpinorComponents) -> Result<Self, StvError> {
        let norm_squared = components[0].norm_sqr() + components[1].norm_sqr();
        if norm_squared <= DEFAULT_TOLERANCE {
            return Err(StvError::ZeroSpinor);
        }
        let norm = norm_squared.sqrt();
        Ok(Self {
            components: [components[0] / norm, components[1] / norm],
        })
    }

    /// Returns the complex components of the spinor.
    pub fn components(&self) -> SpinorComponents {
        self.components
    }

    /// Applies a global phase rotation, preserving the Bloch current.
    pub fn phase_shift(&self, phase: f64) -> Self {
        let factor = Complex::from_polar(1.0, phase);
        Self {
            components: [self.components[0] * factor, self.components[1] * factor],
        }
    }

    /// Computes the Minkowski four-current `(j⁰, **j**)` induced by the Pauli
    /// spinor. The components are the expectation values of the Pauli matrices.
    pub fn bloch_current(&self) -> Vec4 {
        let [a, b] = self.components;
        let j0 = a.norm_sqr() + b.norm_sqr();
        let coherence = a * b.conj();
        let jx = 2.0 * coherence.re;
        let jy = -2.0 * coherence.im;
        let jz = a.norm_sqr() - b.norm_sqr();
        [j0, jx, jy, jz]
    }
}

/// Aggregates the Pauli spinor, kernel tensor, and induced vector according to
/// the SpinoTensorVector formalism.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinoTensorVector {
    spinor: PauliSpinor,
    kernel: SpinoTensorKernel,
}

impl SpinoTensorVector {
    /// Constructs a SpinoTensorVector from raw spinor components.
    pub fn from_components(
        spinor: SpinorComponents,
        kernel: SpinoTensorKernel,
    ) -> Result<Self, StvError> {
        Ok(Self {
            spinor: PauliSpinor::new(spinor)?,
            kernel,
        })
    }

    /// Creates a new SpinoTensorVector from a pre-normalised spinor.
    pub fn new(spinor: PauliSpinor, kernel: SpinoTensorKernel) -> Self {
        Self { spinor, kernel }
    }

    /// Returns the stored Pauli spinor.
    pub fn spinor(&self) -> &PauliSpinor {
        &self.spinor
    }

    /// Returns the kernel descriptor.
    pub fn kernel(&self) -> &SpinoTensorKernel {
        &self.kernel
    }

    /// Computes the Bloch current associated with the spinor.
    pub fn bloch_current(&self) -> Vec4 {
        self.spinor.bloch_current()
    }

    /// Applies the kernel tensor to the Bloch current, returning the induced
    /// Minkowski vector `v = T(j(ψ))`.
    pub fn induced_vector(&self) -> Vec4 {
        let current = self.bloch_current();
        self.kernel.apply_tensor(&current)
    }

    /// Produces a new SpinoTensorVector with a phase-rotated spinor.
    pub fn with_phase_shift(&self, phase: f64) -> Self {
        Self {
            spinor: self.spinor.phase_shift(phase),
            kernel: self.kernel.clone(),
        }
    }

    /// Produces a new SpinoTensorVector with an updated kernel descriptor.
    pub fn with_kernel(&self, kernel: SpinoTensorKernel) -> Self {
        Self {
            spinor: self.spinor.clone(),
            kernel,
        }
    }
}

/// Causal classification of the kernel vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalClass {
    Timelike,
    Lightlike,
    Spacelike,
}

/// Outcome of attempting to normalise the kernel vector with the Minkowski
/// metric of signature `(+, -, -, -)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelNormalization {
    /// Time-like kernel: provides the unit-norm Minkowski vector together with
    /// the value of `β` used for normalisation.
    Timelike { vector: Vec4, beta: f64 },
    /// Light-like kernel: returns the direction and the computed `β` (which is
    /// numerically close to 1).
    Lightlike { direction: Vec4, beta: f64 },
    /// Space-like kernel: returns the raw direction and `β` (> 1).
    Spacelike { direction: Vec4, beta: f64 },
}

/// Branch selection for the intersection curve between the kernel hyperplane
/// and the lightlike ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionBranch {
    Positive,
    Negative,
}

impl IntersectionBranch {
    fn sign(self) -> f64 {
        match self {
            IntersectionBranch::Positive => 1.0,
            IntersectionBranch::Negative => -1.0,
        }
    }
}

/// Elliptic intersection curve of the kernel hyperplane with the lightlike
/// ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntersectionEllipse {
    rotation: Matrix3,
    gamma_inv_sqrt: Matrix3,
    q: Matrix3,
    a1: f64,
    a2: f64,
}

impl IntersectionEllipse {
    /// Returns the semi-axes parameters `(a₁, a₂)` appearing in the parametric
    /// description of the ellipse.
    pub fn parameters(&self) -> (f64, f64) {
        (self.a1, self.a2)
    }

    /// Evaluates the electric field on the intersection curve for the supplied
    /// parameter `t ∈ [0, 2π)` and branch sign.
    pub fn electric_field(&self, t: f64, branch: IntersectionBranch) -> Vec3 {
        let cos_t = t.cos();
        let sin_t = t.sin();
        let z1 = self.a1 * cos_t;
        let z2 = self.a2 * sin_t;
        let mut inside =
            1.0 - self.a1 * self.a1 * cos_t * cos_t - self.a2 * self.a2 * sin_t * sin_t;
        if inside < 0.0 {
            inside = inside.max(-DEFAULT_TOLERANCE);
        }
        let z3 = inside.max(0.0).sqrt() * branch.sign();
        let z = [z1, z2, z3];
        let y = self.q.mul_vec(&z);
        let whitened = self.gamma_inv_sqrt.mul_vec(&y);
        self.rotation.transpose().mul_vec(&whitened)
    }
}

/// Possible shapes of the intersection between `𝒬_α` and `𝒠_β`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntersectionCurve {
    /// No real intersection exists or it is degenerate beyond the supported
    /// model (e.g. hyperbolic).
    None,
    /// Elliptic intersection with parametric accessors.
    Ellipse(IntersectionEllipse),
}

/// Minimal-norm solution for either the electric field or the vorticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimalVectorSolution {
    /// Squared magnitude of the extremal vector.
    pub magnitude_squared: f64,
    /// The extremal vector achieving the minimal norm.
    pub vector: Vec3,
}

impl MinimalVectorSolution {
    /// Returns the unit direction of the extremal vector.
    pub fn direction(&self) -> Vec3 {
        let norm = self.magnitude_squared.sqrt();
        if norm <= 0.0 {
            [0.0, 0.0, 0.0]
        } else {
            [
                self.vector[0] / norm,
                self.vector[1] / norm,
                self.vector[2] / norm,
            ]
        }
    }
}

/// Container for a 3×3 real matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    data: [[f64; 3]; 3],
}

impl Matrix3 {
    pub const fn new(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }

    pub const fn zero() -> Self {
        Self {
            data: [[0.0; 3]; 3],
        }
    }

    pub const fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub const fn from_diagonal(diagonal: Vec3) -> Self {
        Self {
            data: [
                [diagonal[0], 0.0, 0.0],
                [0.0, diagonal[1], 0.0],
                [0.0, 0.0, diagonal[2]],
            ],
        }
    }

    pub fn from_columns(columns: [Vec3; 3]) -> Self {
        let mut data = [[0.0; 3]; 3];
        for col in 0..3 {
            for row in 0..3 {
                data[row][col] = columns[col][row];
            }
        }
        Self { data }
    }

    pub fn transpose(&self) -> Self {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.data[j][i];
            }
        }
        Self { data: result }
    }

    pub fn determinant(&self) -> f64 {
        let m = &self.data;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn adjugate(&self) -> Self {
        let m = &self.data;
        let mut adj = [[0.0; 3]; 3];

        adj[0][0] = minor(m, 0, 0);
        adj[0][1] = -minor(m, 1, 0);
        adj[0][2] = minor(m, 2, 0);

        adj[1][0] = -minor(m, 0, 1);
        adj[1][1] = minor(m, 1, 1);
        adj[1][2] = -minor(m, 2, 1);

        adj[2][0] = minor(m, 0, 2);
        adj[2][1] = -minor(m, 1, 2);
        adj[2][2] = minor(m, 2, 2);

        Self { data: adj }.transpose()
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() <= DEFAULT_TOLERANCE {
            return None;
        }
        let adj = self.adjugate();
        Some(adj.scale(1.0 / det))
    }

    pub fn scale(&self, scalar: f64) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] * scalar;
            }
        }
        Self { data: out }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        Self { data: out }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        Self { data: out }
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += self.data[i][k] * rhs.data[k][j];
                }
                out[i][j] = acc;
            }
        }
        Self { data: out }
    }

    pub fn mul_vec(&self, v: &Vec3) -> Vec3 {
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = self.data[i][0] * v[0] + self.data[i][1] * v[1] + self.data[i][2] * v[2];
        }
        out
    }

    pub fn symmetrize(&self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = 0.5 * (self.data[i][j] + self.data[j][i]);
            }
        }
        Self { data: out }
    }

    pub fn outer(a: &Vec3, b: &Vec3) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i] * b[j];
            }
        }
        Self { data: out }
    }

    pub fn column(&self, index: usize) -> Vec3 {
        [
            self.data[0][index],
            self.data[1][index],
            self.data[2][index],
        ]
    }

    pub fn as_array(&self) -> [[f64; 3]; 3] {
        self.data
    }

    pub fn to_na(&self) -> na::Matrix3<f64> {
        na::Matrix3::from_row_slice(&[
            self.data[0][0],
            self.data[0][1],
            self.data[0][2],
            self.data[1][0],
            self.data[1][1],
            self.data[1][2],
            self.data[2][0],
            self.data[2][1],
            self.data[2][2],
        ])
    }

    pub fn from_na(matrix: na::Matrix3<f64>) -> Self {
        let mut data = [[0.0; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                data[row][col] = matrix[(row, col)];
            }
        }
        Self { data }
    }

    pub fn symmetric_eigendecomposition(&self) -> ([f64; 3], Matrix3) {
        let eig = na::SymmetricEigen::new(self.to_na());
        let mut pairs: Vec<(f64, Vec3)> = (0..3)
            .map(|idx| {
                (
                    eig.eigenvalues[idx],
                    [
                        eig.eigenvectors[(0, idx)],
                        eig.eigenvectors[(1, idx)],
                        eig.eigenvectors[(2, idx)],
                    ],
                )
            })
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut values = [0.0; 3];
        let mut columns = [[0.0; 3]; 3];
        for (idx, (value, column)) in pairs.into_iter().enumerate() {
            values[idx] = value;
            columns[idx] = column;
        }
        (values, Matrix3::from_columns(columns))
    }
}

impl fmt::Display for Matrix3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..3 {
            writeln!(
                f,
                "[{:.6}, {:.6}, {:.6}]",
                self.data[i][0], self.data[i][1], self.data[i][2]
            )?;
        }
        Ok(())
    }
}

fn minor(m: &[[f64; 3]; 3], row: usize, col: usize) -> f64 {
    let mut vals = [0.0; 4];
    let mut idx = 0;
    for i in 0..3 {
        if i == row {
            continue;
        }
        for j in 0..3 {
            if j == col {
                continue;
            }
            vals[idx] = m[i][j];
            idx += 1;
        }
    }
    vals[0] * vals[3] - vals[1] * vals[2]
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn omega_matrix(omega: &Vec3) -> Matrix3 {
    Matrix3::new([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])
}

fn max_skew_symmetry(a: &Matrix3) -> f64 {
    let mut max_dev = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            if i < j {
                max_dev = max_dev.max((a.data[i][j] - a.data[j][i]).abs());
            }
        }
    }
    max_dev
}

/// SpinoTensorVector kernel descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinoTensorKernel {
    s0: f64,
    a: Matrix3,
    e: Vec3,
    omega: Vec3,
}

impl SpinoTensorKernel {
    /// Constructs a new kernel, validating that `A` is symmetric.
    pub fn new(s0: f64, a: [[f64; 3]; 3], e: Vec3, omega: Vec3) -> Result<Self, StvError> {
        let matrix_a = Matrix3::new(a);
        let deviation = max_skew_symmetry(&matrix_a);
        if deviation > DEFAULT_TOLERANCE {
            return Err(StvError::NonSymmetric {
                max_deviation: deviation,
                tolerance: DEFAULT_TOLERANCE,
            });
        }

        Ok(Self {
            s0,
            a: matrix_a,
            e,
            omega,
        })
    }

    pub fn s0(&self) -> f64 {
        self.s0
    }

    pub fn spatial_tensor(&self) -> Matrix3 {
        self.a
    }

    pub fn electric_field(&self) -> Vec3 {
        self.e
    }

    pub fn spin_frequency(&self) -> Vec3 {
        self.omega
    }

    /// Returns the matrix `D = A + Ω(ω)`.
    pub fn d(&self) -> Matrix3 {
        self.a.add(&omega_matrix(&self.omega))
    }

    /// Returns the full 4×4 block tensor `T = \begin{pmatrix}s₀ & E^⊤\\ E & D\end{pmatrix}`.
    pub fn block_tensor(&self) -> [[f64; 4]; 4] {
        let d = self.d().as_array();
        [
            [self.s0, self.e[0], self.e[1], self.e[2]],
            [self.e[0], d[0][0], d[0][1], d[0][2]],
            [self.e[1], d[1][0], d[1][1], d[1][2]],
            [self.e[2], d[2][0], d[2][1], d[2][2]],
        ]
    }

    /// Returns the matrix `D D^⊤`.
    pub fn dd_t(&self) -> Matrix3 {
        let d = self.d();
        d.mul(&d.transpose())
    }

    /// Applies the SpinoTensor kernel tensor to a Minkowski vector.
    pub fn apply_tensor(&self, vector: &Vec4) -> Vec4 {
        let spatial = [vector[1], vector[2], vector[3]];
        let rotated = self.d().mul_vec(&spatial);
        let time = self.s0 * vector[0] + dot(&self.e, &spatial);
        [
            time,
            self.e[0] * vector[0] + rotated[0],
            self.e[1] * vector[0] + rotated[1],
            self.e[2] * vector[0] + rotated[2],
        ]
    }

    /// Determinant of `D`.
    pub fn det_d(&self) -> f64 {
        self.d().determinant()
    }

    /// Adjugate of `D`.
    pub fn adj_d(&self) -> Matrix3 {
        self.d().adjugate()
    }

    /// Determinant of the full block matrix `T`.
    pub fn det_t(&self) -> f64 {
        let d = self.d();
        let det_d = d.determinant();
        let adj_d = d.adjugate();
        let adj_e = adj_d.mul_vec(&self.e);
        self.s0 * det_d - dot(&self.e, &adj_e)
    }

    /// Computes `α = Eᵀ D⁻¹ E` when `D` is invertible.
    pub fn alpha(&self) -> Result<f64, StvError> {
        let d = self.d();
        let det_d = d.determinant();
        if det_d.abs() <= DEFAULT_TOLERANCE {
            return Err(StvError::SingularD);
        }
        let adj_d = d.adjugate();
        let adj_e = adj_d.mul_vec(&self.e);
        Ok(dot(&self.e, &adj_e) / det_d)
    }

    /// Computes `β = Eᵀ (D Dᵀ)⁻¹ E` when `D Dᵀ` is invertible.
    pub fn beta(&self) -> Result<f64, StvError> {
        let d = self.d();
        let dd_t = d.mul(&d.transpose());
        let det_dd_t = dd_t.determinant();
        if det_dd_t.abs() <= DEFAULT_TOLERANCE {
            return Err(StvError::SingularDdT);
        }
        let inv = dd_t.adjugate().scale(1.0 / det_dd_t);
        let projected = inv.mul_vec(&self.e);
        Ok(dot(&self.e, &projected))
    }

    /// Returns a parametric description of `𝒬_α ∩ 𝒠_β` when it forms an
    /// ellipse. Degenerate or hyperbolic cases yield `IntersectionCurve::None`.
    pub fn intersection_curve(&self, tolerance: f64) -> Result<IntersectionCurve, StvError> {
        let d = self.d();
        let dd_t = self.dd_t();
        let det_dd_t = dd_t.determinant();
        if det_dd_t.abs() <= DEFAULT_TOLERANCE {
            return Err(StvError::SingularDdT);
        }
        let inv_dd_t = match dd_t.inverse() {
            Some(inv) => inv.symmetrize(),
            None => return Err(StvError::SingularDdT),
        };
        let (gamma_vals, gamma_vecs) = inv_dd_t.symmetric_eigendecomposition();
        for value in gamma_vals.iter() {
            if *value <= tolerance {
                return Err(StvError::NonPositiveGamma { eigenvalue: *value });
            }
        }
        let gamma_inv_sqrt = Matrix3::from_diagonal([
            1.0 / gamma_vals[0].sqrt(),
            1.0 / gamma_vals[1].sqrt(),
            1.0 / gamma_vals[2].sqrt(),
        ]);
        let rotation = gamma_vecs.transpose();
        let adj_d = d.adjugate();
        let b = adj_d.add(&adj_d.transpose()).scale(0.5);
        let temp = rotation.mul(&b.mul(&rotation.transpose()));
        let c_matrix = gamma_inv_sqrt.mul(&temp.mul(&gamma_inv_sqrt)).symmetrize();
        let (c_vals, q_vectors) = c_matrix.symmetric_eigendecomposition();

        let c1 = c_vals[0];
        let c2 = c_vals[1];
        let c3 = c_vals[2];
        let den1 = c1 - c3;
        let den2 = c2 - c3;
        if den1.abs() <= tolerance || den2.abs() <= tolerance {
            return Ok(IntersectionCurve::None);
        }

        let kappa = self.s0 * d.determinant();
        let a1_sq = (kappa - c3) / den1;
        let a2_sq = (kappa - c3) / den2;
        if !a1_sq.is_finite() || !a2_sq.is_finite() {
            return Ok(IntersectionCurve::None);
        }
        if a1_sq < -tolerance || a2_sq < -tolerance {
            return Ok(IntersectionCurve::None);
        }
        let a1_sq = a1_sq.clamp(0.0, 1.0);
        let a2_sq = a2_sq.clamp(0.0, 1.0);
        if 1.0 - a1_sq < -tolerance || 1.0 - a2_sq < -tolerance {
            return Ok(IntersectionCurve::None);
        }

        Ok(IntersectionCurve::Ellipse(IntersectionEllipse {
            rotation,
            gamma_inv_sqrt,
            q: q_vectors,
            a1: a1_sq.sqrt(),
            a2: a2_sq.sqrt(),
        }))
    }

    /// Computes the kernel direction `(1, -D⁻¹E)` if `D` is invertible.
    pub fn kernel_direction(&self) -> Result<Vec4, StvError> {
        let d = self.d();
        let inv_d = match d.inverse() {
            Some(inv) => inv,
            None => return Err(StvError::SingularD),
        };
        let spatial = inv_d.mul_vec(&self.e);
        Ok([1.0, -spatial[0], -spatial[1], -spatial[2]])
    }

    /// Causal classification derived from `β`.
    pub fn classify(&self, tolerance: f64) -> Result<CausalClass, StvError> {
        let beta = self.beta()?;
        if (beta - 1.0).abs() <= tolerance {
            Ok(CausalClass::Lightlike)
        } else if beta < 1.0 {
            Ok(CausalClass::Timelike)
        } else {
            Ok(CausalClass::Spacelike)
        }
    }

    /// Attempts to normalise the kernel according to equation (12).
    pub fn normalise_kernel(&self, tolerance: f64) -> Result<KernelNormalization, StvError> {
        let direction = self.kernel_direction()?;
        let beta = self.beta()?;
        let norm_sq = 1.0 - beta;
        if norm_sq.abs() <= tolerance {
            Ok(KernelNormalization::Lightlike { direction, beta })
        } else if norm_sq > 0.0 {
            let scale = norm_sq.sqrt();
            let normalised = [
                direction[0] / scale,
                direction[1] / scale,
                direction[2] / scale,
                direction[3] / scale,
            ];
            Ok(KernelNormalization::Timelike {
                vector: normalised,
                beta,
            })
        } else {
            Ok(KernelNormalization::Spacelike { direction, beta })
        }
    }

    /// Minimal-norm electric field satisfying the kernel constraint when it
    /// exists.
    pub fn minimal_electric_field(&self) -> Option<MinimalVectorSolution> {
        let det_a = self.a.determinant();
        let omega_a = self.a.mul_vec(&self.omega);
        let numerator = self.s0 * (det_a + dot(&self.omega, &omega_a));
        if !(numerator > 0.0) {
            return None;
        }
        let k = self
            .a
            .adjugate()
            .add(&Matrix3::outer(&self.omega, &self.omega))
            .symmetrize();
        let (values, vectors) = k.symmetric_eigendecomposition();
        let mut index = 0;
        let mut lambda_max = values[0];
        for (i, value) in values.iter().enumerate() {
            if *value > lambda_max {
                lambda_max = *value;
                index = i;
            }
        }
        if lambda_max <= DEFAULT_TOLERANCE {
            return None;
        }
        let magnitude_squared = numerator / lambda_max;
        if !(magnitude_squared > 0.0) {
            return None;
        }
        let direction = vectors.column(index);
        let magnitude = magnitude_squared.sqrt();
        let vector = [
            direction[0] * magnitude,
            direction[1] * magnitude,
            direction[2] * magnitude,
        ];
        Some(MinimalVectorSolution {
            magnitude_squared,
            vector,
        })
    }

    /// Minimal-norm vorticity satisfying the kernel constraint when it exists.
    pub fn minimal_vorticity(&self) -> Option<MinimalVectorSolution> {
        let adj_a = self.a.adjugate();
        let mu = dot(&self.e, &adj_a.mul_vec(&self.e)) - self.s0 * self.a.determinant();
        if mu.abs() <= DEFAULT_TOLERANCE {
            return Some(MinimalVectorSolution {
                magnitude_squared: 0.0,
                vector: [0.0, 0.0, 0.0],
            });
        }
        let k = self
            .a
            .scale(self.s0)
            .sub(&Matrix3::outer(&self.e, &self.e))
            .symmetrize();
        let (values, vectors) = k.symmetric_eigendecomposition();
        if mu > 0.0 {
            let mut index = 0;
            let mut lambda = values[0];
            for (i, value) in values.iter().enumerate() {
                if *value > lambda {
                    lambda = *value;
                    index = i;
                }
            }
            if lambda <= DEFAULT_TOLERANCE {
                return None;
            }
            let magnitude_squared = mu / lambda;
            if !(magnitude_squared > 0.0) {
                return None;
            }
            let direction = vectors.column(index);
            let magnitude = magnitude_squared.sqrt();
            let vector = [
                direction[0] * magnitude,
                direction[1] * magnitude,
                direction[2] * magnitude,
            ];
            Some(MinimalVectorSolution {
                magnitude_squared,
                vector,
            })
        } else {
            let mut index = 0;
            let mut lambda = values[0];
            for (i, value) in values.iter().enumerate() {
                if *value < lambda {
                    lambda = *value;
                    index = i;
                }
            }
            if lambda >= -DEFAULT_TOLERANCE {
                return None;
            }
            let magnitude_squared = mu / lambda;
            if !(magnitude_squared > 0.0) {
                return None;
            }
            let direction = vectors.column(index);
            let magnitude = magnitude_squared.sqrt();
            let vector = [
                direction[0] * magnitude,
                direction[1] * magnitude,
                direction[2] * magnitude,
            ];
            Some(MinimalVectorSolution {
                magnitude_squared,
                vector,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex;

    fn minkowski_norm_sq(v: &Vec4) -> f64 {
        v[0] * v[0] - v[1] * v[1] - v[2] * v[2] - v[3] * v[3]
    }

    fn diag(values: [f64; 3]) -> [[f64; 3]; 3] {
        [
            [values[0], 0.0, 0.0],
            [0.0, values[1], 0.0],
            [0.0, 0.0, values[2]],
        ]
    }

    #[test]
    fn determinant_matches_block_formula() {
        let kernel =
            SpinoTensorKernel::new(1.5, diag([2.0, 1.0, 3.0]), [0.0, 1.5, 0.0], [0.0, 0.0, 1.0])
                .unwrap();

        assert_abs_diff_eq!(kernel.det_d(), 9.0, epsilon = 1e-9);
        assert_abs_diff_eq!(kernel.det_t(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn alpha_beta_match_manual_values() {
        let kernel =
            SpinoTensorKernel::new(1.5, diag([2.0, 1.0, 3.0]), [0.0, 1.5, 0.0], [0.0, 0.0, 1.0])
                .unwrap();

        assert_abs_diff_eq!(kernel.alpha().unwrap(), 1.5, epsilon = 1e-9);
        assert_abs_diff_eq!(kernel.beta().unwrap(), 1.25, epsilon = 1e-9);
    }

    #[test]
    fn pauli_spinor_matches_bloch_coordinates() {
        let theta = 1.234_f64;
        let phi = -0.732_f64;
        let spinor = PauliSpinor::new([
            Complex::new((theta / 2.0).cos(), 0.0),
            Complex::from_polar((theta / 2.0).sin(), phi),
        ])
        .unwrap();

        let bloch = spinor.bloch_current();
        assert_abs_diff_eq!(bloch[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(bloch[1], theta.sin() * phi.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(bloch[2], theta.sin() * phi.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(bloch[3], theta.cos(), epsilon = 1e-9);
    }

    #[test]
    fn tensor_application_matches_block_matrix() {
        let kernel = SpinoTensorKernel::new(
            1.3,
            diag([1.0, 1.5, 2.0]),
            [0.4, -0.2, 0.1],
            [0.1, 0.2, 0.3],
        )
        .unwrap();
        let vector = [1.0, 0.2, -0.4, 0.8];

        let applied = kernel.apply_tensor(&vector);
        let block = kernel.block_tensor();
        let mut manual = [0.0; 4];
        for row in 0..4 {
            let mut acc = 0.0;
            for col in 0..4 {
                acc += block[row][col] * vector[col];
            }
            manual[row] = acc;
        }

        for i in 0..4 {
            assert_abs_diff_eq!(applied[i], manual[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn spino_tensor_vector_respects_definition() {
        let kernel = SpinoTensorKernel::new(
            1.6,
            diag([2.0, 0.5, 3.0]),
            [0.1, 0.3, -0.2],
            [0.0, 0.0, 0.4],
        )
        .unwrap();
        let spinor = PauliSpinor::new([Complex::new(0.9, -0.1), Complex::new(0.3, 0.2)]).unwrap();
        let stv = SpinoTensorVector::new(spinor.clone(), kernel.clone());

        let bloch = stv.bloch_current();
        let induced = stv.induced_vector();
        let expected = kernel.apply_tensor(&bloch);
        for i in 0..4 {
            assert_abs_diff_eq!(induced[i], expected[i], epsilon = 1e-9);
        }

        let rotated = stv.with_phase_shift(0.77);
        let rotated_bloch = rotated.bloch_current();
        let rotated_induced = rotated.induced_vector();
        for i in 0..4 {
            assert_abs_diff_eq!(rotated_bloch[i], bloch[i], epsilon = 1e-9);
            assert_abs_diff_eq!(rotated_induced[i], induced[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn intersection_curve_matches_constraints() {
        let kernel =
            SpinoTensorKernel::new(2.1, diag([2.0, 1.0, 3.0]), [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap();

        let curve = kernel.intersection_curve(1e-9).unwrap();
        let ellipse = match curve {
            IntersectionCurve::Ellipse(ellipse) => ellipse,
            IntersectionCurve::None => panic!("expected ellipse intersection"),
        };

        let sample = ellipse.electric_field(0.3, IntersectionBranch::Positive);
        let d = kernel.d();
        let dd_t = kernel.dd_t();
        let adj_d = d.adjugate();
        let inv_dd_t = dd_t.inverse().unwrap();
        let lhs = dot(&sample, &adj_d.mul_vec(&sample));
        let rhs = kernel.s0() * kernel.det_d();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-6);

        let ellipsoid = dot(&sample, &inv_dd_t.mul_vec(&sample));
        assert_abs_diff_eq!(ellipsoid, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn minimal_electric_field_matches_design_recipe() {
        let kernel =
            SpinoTensorKernel::new(1.5, diag([2.0, 1.0, 3.0]), [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap();

        let solution = kernel.minimal_electric_field().expect("solution exists");
        assert_abs_diff_eq!(solution.magnitude_squared, 2.25, epsilon = 1e-9);
        assert_abs_diff_eq!(solution.vector[0], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(solution.vector[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(solution.vector[1].abs(), 1.5, epsilon = 1e-9);
    }

    #[test]
    fn minimal_vorticity_matches_design_recipe() {
        let kernel = SpinoTensorKernel::new(
            1.5,
            diag([2.0, 1.0, 3.0]),
            [0.0, 2.0_f64.sqrt(), 0.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();

        let solution = kernel.minimal_vorticity().expect("solution exists");
        assert_abs_diff_eq!(solution.magnitude_squared, 2.0 / 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(solution.vector[0], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(solution.vector[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(
            solution.vector[2].abs(),
            (2.0f64 / 3.0).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn classify_timelike_and_normalise() {
        let kernel = SpinoTensorKernel::new(
            2.0 / 3.0,
            diag([2.0, 1.0, 3.0]),
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        .unwrap();

        assert_eq!(kernel.classify(1e-9).unwrap(), CausalClass::Timelike);
        let normalised = kernel.normalise_kernel(1e-9).unwrap();
        match normalised {
            KernelNormalization::Timelike { vector, beta } => {
                assert_abs_diff_eq!(beta, 5.0 / 9.0, epsilon = 1e-9);
                assert_abs_diff_eq!(minkowski_norm_sq(&vector), 1.0, epsilon = 1e-9);
            }
            _ => panic!("expected timelike normalisation"),
        }
    }

    #[test]
    fn classify_lightlike() {
        let e_norm = 1.8f64.sqrt();
        let kernel = SpinoTensorKernel::new(
            1.2,
            diag([2.0, 1.0, 3.0]),
            [0.0, e_norm, 0.0],
            [0.0, 0.0, 1.0],
        )
        .unwrap();

        assert_eq!(kernel.classify(1e-8).unwrap(), CausalClass::Lightlike);
        match kernel.normalise_kernel(1e-8).unwrap() {
            KernelNormalization::Lightlike { beta, .. } => {
                assert_abs_diff_eq!(beta, 1.0, epsilon = 1e-8);
            }
            _ => panic!("expected lightlike kernel"),
        }
    }

    #[test]
    fn detect_non_symmetric_tensor() {
        let err = SpinoTensorKernel::new(
            1.0,
            [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap_err();

        assert!(matches!(err, StvError::NonSymmetric { .. }));
    }

    #[test]
    fn detect_zero_spinor() {
        let err = PauliSpinor::new([Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)]).unwrap_err();
        assert!(matches!(err, StvError::ZeroSpinor));
    }
}
