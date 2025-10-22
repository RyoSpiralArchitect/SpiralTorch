// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use pyo3::prelude::*;
use rand_distr::{Normal, Distribution};
use crate::backend::faer_dense::Tensor;

#[pyclass(name = "Tensor")]
pub struct PyTensor {
    pub inner: Tensor,
}

#[pymethods]
impl PyTensor {
    /// Create a tensor filled with zeros
    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        Ok(Self { inner: Tensor::zeros(rows, cols)? })
    }

    /// Create a tensor filled with random normal values (mean=0, std=1)
    #[staticmethod]
    pub fn randn(rows: usize, cols: usize) -> PyResult<Self> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| normal.sample(&mut rand::thread_rng()) as f32)
            .collect();
        Ok(Self { inner: Tensor::from_vec(rows, cols, data)? })
    }

    /// Return the shape as a tuple
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Convert to Python list for inspection
    pub fn tolist(&self) -> Vec<Vec<f32>> {
        self.inner.to_vec2d()
    }
}
