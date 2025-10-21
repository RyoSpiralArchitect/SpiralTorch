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

/// Alias for a spatial three-vector.
pub type Vec3 = [f64; 3];

/// Alias for a Minkowski four-vector with signature `(+, -, -, -)`.
pub type Vec4 = [f64; 4];

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

    pub fn as_array(&self) -> [[f64; 3]; 3] {
        self.data
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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
}
