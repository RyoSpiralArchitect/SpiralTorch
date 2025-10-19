#![allow(unused)]
#![allow(clippy::too_many_arguments)]

mod devices;
mod utils;
mod tensor;
mod dataset;
mod chrono;
mod atlas;
mod nn;
mod frac;
mod linalg;
mod rl;
mod rec;
mod telemetry;
mod ecosystem;
#[cfg(feature = "golden")]
mod golden;
#[cfg(feature = "collapse")]
mod collapse;

use pyo3::prelude::*;

// ルートモジュール: ここでルート直下に公開したい型/関数を register
#[pymodule]
fn spiraltorch(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 必要に応じてルート直下に型を公開
    chrono::register(py, m)?;
    atlas::register(py, m)?;
    telemetry::register(py, m)?;
    ecosystem::register(py, m)?;
    tensor::register(py, m)?; // Tensor をルート直下に出す場合

    Ok(())
}

// サブモジュール（従来通り個別の #[pymodule] も用意）
#[pymodule]
fn nn(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    nn::register(py, m)
}

#[pymodule]
fn frac(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    frac::register(py, m)
}

#[pymodule]
fn dataset(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    dataset::register(py, m)
}

#[pymodule]
fn linalg(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    linalg::register(py, m)
}

#[pymodule]
fn rl(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    rl::register(py, m)
}

#[pymodule]
fn rec(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    rec::register(py, m)
}

#[pymodule]
fn telemetry(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    telemetry::register(py, m)
}

#[pymodule]
fn ecosystem(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    ecosystem::register(py, m)
}
