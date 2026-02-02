use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Bound;

use st_frac::cosmology::{
    assemble_pzeta as rust_assemble_pzeta, log_lattice_z_points as rust_log_lattice_z_points,
    LogZSeries, SeriesOptions, WeightNormalisation, WindowFunction,
};
use st_frac::fractal_field::FractalFieldGenerator;
use st_frac::mellin::{
    sample_log_uniform_exp_decay, sample_log_uniform_exp_decay_scaled, MellinEvalPlan,
    MellinLogGrid,
};
use st_frac::mellin_types::{ComplexScalar, Scalar};
use st_tensor::fractional::{fracdiff_gl_1d as rust_fracdiff_gl_1d, PadMode};

#[pyfunction]
#[pyo3(signature = (xs, alpha, kernel_len, pad="constant", pad_constant=None))]
fn fracdiff_gl_1d(
    xs: Vec<f32>,
    alpha: f32,
    kernel_len: usize,
    pad: &str,
    pad_constant: Option<f32>,
) -> PyResult<Vec<f32>> {
    let mode = match pad.to_ascii_lowercase().as_str() {
        "constant" => PadMode::Constant(pad_constant.unwrap_or(0.0)),
        "edge" => PadMode::Edge,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pad mode '{other}', expected 'constant' or 'edge'"
            )))
        }
    };
    rust_fracdiff_gl_1d(&xs, alpha, kernel_len, mode)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (alpha, tol=1e-6, max_len=8192))]
fn gl_coeffs_adaptive(alpha: f32, tol: f32, max_len: usize) -> PyResult<Vec<f32>> {
    st_frac::gl_coeffs_adaptive(alpha, tol, max_len)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (log_start, log_step, len))]
fn mellin_exp_decay_samples(
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
) -> PyResult<Vec<ComplexScalar>> {
    sample_log_uniform_exp_decay(log_start, log_step, len).map_err(mellin_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (log_start, log_step, len, rate))]
fn mellin_exp_decay_samples_scaled(
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
    rate: Scalar,
) -> PyResult<Vec<ComplexScalar>> {
    sample_log_uniform_exp_decay_scaled(log_start, log_step, len, rate).map_err(mellin_err_to_py)
}

fn mellin_err_to_py(err: st_frac::mellin_types::MellinError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn cosmology_err_to_py(err: st_frac::cosmology::CosmologyError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn fractal_err_to_py(err: st_frac::fractal_field::FractalFieldError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_window(window: &str) -> PyResult<WindowFunction> {
    match window.to_ascii_lowercase().as_str() {
        "rect" | "rectangular" | "none" => Ok(WindowFunction::Rectangular),
        "hann" => Ok(WindowFunction::Hann),
        other => Err(PyValueError::new_err(format!(
            "unknown window '{other}', expected 'rectangular' or 'hann'"
        ))),
    }
}

fn parse_normalisation(normalisation: &str) -> PyResult<WeightNormalisation> {
    match normalisation.to_ascii_lowercase().as_str() {
        "none" => Ok(WeightNormalisation::None),
        "l1" => Ok(WeightNormalisation::L1),
        "l2" => Ok(WeightNormalisation::L2),
        other => Err(PyValueError::new_err(format!(
            "unknown normalisation '{other}', expected 'none', 'l1', or 'l2'"
        ))),
    }
}

fn window_name(window: WindowFunction) -> &'static str {
    match window {
        WindowFunction::Rectangular => "rectangular",
        WindowFunction::Hann => "hann",
    }
}

fn normalisation_name(norm: WeightNormalisation) -> &'static str {
    match norm {
        WeightNormalisation::None => "none",
        WeightNormalisation::L1 => "l1",
        WeightNormalisation::L2 => "l2",
    }
}

#[pyclass(module = "spiraltorch.frac", name = "FractalFieldGenerator")]
#[derive(Clone, Debug)]
struct PyFractalFieldGenerator {
    inner: FractalFieldGenerator,
}

#[pymethods]
impl PyFractalFieldGenerator {
    #[new]
    #[pyo3(signature = (octaves, lacunarity=2.0, gain=0.5, iterations=16))]
    fn new(octaves: u32, lacunarity: f32, gain: f32, iterations: u32) -> PyResult<Self> {
        let inner =
            FractalFieldGenerator::new(octaves, lacunarity, gain, iterations).map_err(fractal_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn octaves(&self) -> u32 {
        self.inner.octaves()
    }

    #[getter]
    fn lacunarity(&self) -> f32 {
        self.inner.lacunarity()
    }

    #[getter]
    fn gain(&self) -> f32 {
        self.inner.gain()
    }

    #[getter]
    fn iterations(&self) -> u32 {
        self.inner.iterations()
    }

    fn branching_field(&self, log_start: Scalar, log_step: Scalar, len: usize) -> PyResult<Vec<ComplexScalar>> {
        self.inner
            .branching_field(log_start, log_step, len)
            .map_err(fractal_err_to_py)
    }

    fn spawn_grid(&self, log_start: Scalar, log_step: Scalar, len: usize) -> PyResult<PyMellinLogGrid> {
        let grid = self
            .inner
            .spawn_grid(log_start, log_step, len)
            .map_err(fractal_err_to_py)?;
        Ok(PyMellinLogGrid { inner: grid })
    }

    fn weave_with_grid(&self, base: &PyMellinLogGrid) -> PyResult<PyMellinLogGrid> {
        let grid = self
            .inner
            .weave_with_grid(&base.inner)
            .map_err(fractal_err_to_py)?;
        Ok(PyMellinLogGrid { inner: grid })
    }

    fn __repr__(&self) -> String {
        format!(
            "FractalFieldGenerator(octaves={}, lacunarity={:.4}, gain={:.4}, iterations={})",
            self.inner.octaves(),
            self.inner.lacunarity(),
            self.inner.gain(),
            self.inner.iterations()
        )
    }
}

#[pyclass(module = "spiraltorch.frac", name = "LogZSeries")]
#[derive(Clone, Debug)]
struct PyLogZSeries {
    inner: LogZSeries,
}

#[pymethods]
impl PyLogZSeries {
    #[new]
    #[pyo3(signature = (log_start, log_step, samples, window="rectangular", normalisation="l1"))]
    fn new(
        log_start: Scalar,
        log_step: Scalar,
        samples: Vec<Scalar>,
        window: &str,
        normalisation: &str,
    ) -> PyResult<Self> {
        let window = parse_window(window)?;
        let normalisation = parse_normalisation(normalisation)?;
        let inner = LogZSeries::from_samples_with_options(
            log_start,
            log_step,
            samples,
            SeriesOptions {
                window,
                normalisation,
            },
        )
        .map_err(cosmology_err_to_py)?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn log_start(&self) -> Scalar {
        self.inner.log_start()
    }

    #[getter]
    fn log_step(&self) -> Scalar {
        self.inner.log_step()
    }

    #[getter]
    fn samples(&self) -> Vec<Scalar> {
        self.inner.samples().to_vec()
    }

    #[getter]
    fn weights(&self) -> Vec<Scalar> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn window(&self) -> String {
        window_name(self.inner.options().window).to_string()
    }

    #[getter]
    fn normalisation(&self) -> String {
        normalisation_name(self.inner.options().normalisation).to_string()
    }

    fn evaluate_z(&self, z: ComplexScalar) -> PyResult<ComplexScalar> {
        self.inner.evaluate_z(z).map_err(cosmology_err_to_py)
    }

    fn evaluate_many_z(&self, z_values: Vec<ComplexScalar>) -> PyResult<Vec<ComplexScalar>> {
        self.inner
            .evaluate_many_z(&z_values)
            .map_err(cosmology_err_to_py)
    }

    fn ensure_compatible(&self, other: &PyLogZSeries) -> PyResult<()> {
        self.inner
            .ensure_compatible(&other.inner)
            .map_err(cosmology_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "LogZSeries(len={}, log_start={:.4}, log_step={:.4}, window={}, normalisation={})",
            self.inner.len(),
            self.inner.log_start(),
            self.inner.log_step(),
            window_name(self.inner.options().window),
            normalisation_name(self.inner.options().normalisation)
        )
    }
}

#[pyfunction]
#[pyo3(signature = (log_step, s_values))]
fn log_lattice_z_points(log_step: Scalar, s_values: Vec<ComplexScalar>) -> PyResult<Vec<ComplexScalar>> {
    rust_log_lattice_z_points(log_step, &s_values).map_err(cosmology_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (z_points, h_series, epsilon_series, planck_mass))]
fn assemble_pzeta(
    z_points: Vec<ComplexScalar>,
    h_series: &PyLogZSeries,
    epsilon_series: &PyLogZSeries,
    planck_mass: Scalar,
) -> PyResult<Vec<Scalar>> {
    rust_assemble_pzeta(&z_points, &h_series.inner, &epsilon_series.inner, planck_mass)
        .map_err(cosmology_err_to_py)
}

#[pyclass(module = "spiraltorch.frac", name = "MellinLogGrid")]
#[derive(Clone, Debug)]
pub(crate) struct PyMellinLogGrid {
    pub(crate) inner: MellinLogGrid,
}

#[pyclass(module = "spiraltorch.frac", name = "MellinEvalPlan")]
#[derive(Clone, Debug)]
pub(crate) struct PyMellinEvalPlan {
    inner: MellinEvalPlan,
}

#[pymethods]
impl PyMellinEvalPlan {
    #[staticmethod]
    fn mesh(
        log_start: Scalar,
        log_step: Scalar,
        real_values: Vec<Scalar>,
        imag_values: Vec<Scalar>,
    ) -> PyResult<Self> {
        let inner =
            MellinEvalPlan::mesh(log_start, log_step, &real_values, &imag_values).map_err(mellin_err_to_py)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn vertical_line(
        log_start: Scalar,
        log_step: Scalar,
        real: Scalar,
        imag_values: Vec<Scalar>,
    ) -> PyResult<Self> {
        let inner =
            MellinEvalPlan::vertical_line(log_start, log_step, real, &imag_values).map_err(mellin_err_to_py)?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn log_start(&self) -> Scalar {
        self.inner.log_start()
    }

    #[getter]
    fn log_step(&self) -> Scalar {
        self.inner.log_step()
    }

    fn evaluate(&self, grid: &PyMellinLogGrid) -> PyResult<Vec<ComplexScalar>> {
        grid.inner.evaluate_plan(&self.inner).map_err(mellin_err_to_py)
    }

    fn evaluate_magnitude(&self, grid: &PyMellinLogGrid) -> PyResult<Vec<Scalar>> {
        grid.inner
            .evaluate_plan_magnitude(&self.inner)
            .map_err(mellin_err_to_py)
    }

    #[pyo3(signature = (grid, epsilon=0.0))]
    fn evaluate_log_magnitude(&self, grid: &PyMellinLogGrid, epsilon: Scalar) -> PyResult<Vec<Scalar>> {
        grid.inner
            .evaluate_plan_log_magnitude(&self.inner, epsilon)
            .map_err(mellin_err_to_py)
    }

    fn __repr__(&self) -> String {
        let shape = self.inner.shape();
        format!(
            "MellinEvalPlan(shape=({}, {}), log_start={:.4}, log_step={:.4})",
            shape.0,
            shape.1,
            self.inner.log_start(),
            self.inner.log_step()
        )
    }
}

#[pymethods]
impl PyMellinLogGrid {
    #[staticmethod]
    fn exp_decay(log_start: Scalar, log_step: Scalar, len: usize) -> PyResult<Self> {
        let samples =
            sample_log_uniform_exp_decay(log_start, log_step, len).map_err(mellin_err_to_py)?;
        let inner = MellinLogGrid::new(log_start, log_step, samples).map_err(mellin_err_to_py)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn exp_decay_scaled(log_start: Scalar, log_step: Scalar, len: usize, rate: Scalar) -> PyResult<Self> {
        let samples = sample_log_uniform_exp_decay_scaled(log_start, log_step, len, rate)
            .map_err(mellin_err_to_py)?;
        let inner = MellinLogGrid::new(log_start, log_step, samples).map_err(mellin_err_to_py)?;
        Ok(Self { inner })
    }

    #[new]
    fn new(log_start: Scalar, log_step: Scalar, samples: Vec<ComplexScalar>) -> PyResult<Self> {
        let inner = MellinLogGrid::new(log_start, log_step, samples).map_err(mellin_err_to_py)?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn log_start(&self) -> Scalar {
        self.inner.log_start()
    }

    #[getter]
    fn log_step(&self) -> Scalar {
        self.inner.log_step()
    }

    #[getter]
    fn samples(&self) -> Vec<ComplexScalar> {
        self.inner.samples().to_vec()
    }

    #[getter]
    fn weights(&self) -> Vec<Scalar> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn support(&self) -> (Scalar, Scalar) {
        self.inner.support()
    }

    fn weighted_series(&self) -> PyResult<Vec<ComplexScalar>> {
        self.inner.weighted_series().map_err(mellin_err_to_py)
    }

    fn evaluate(&self, s: ComplexScalar) -> PyResult<ComplexScalar> {
        self.inner.evaluate(s).map_err(mellin_err_to_py)
    }

    fn evaluate_many(&self, s_values: Vec<ComplexScalar>) -> PyResult<Vec<ComplexScalar>> {
        self.inner.evaluate_many(&s_values).map_err(mellin_err_to_py)
    }

    fn evaluate_mesh(
        &self,
        real_values: Vec<Scalar>,
        imag_values: Vec<Scalar>,
    ) -> PyResult<Vec<Vec<ComplexScalar>>> {
        if real_values.is_empty() {
            return Ok(Vec::new());
        }
        if imag_values.is_empty() {
            return Ok(vec![Vec::new(); real_values.len()]);
        }
        let flat = self
            .inner
            .evaluate_mesh(&real_values, &imag_values)
            .map_err(mellin_err_to_py)?;
        let cols = imag_values.len();
        Ok(flat.chunks(cols).map(|row| row.to_vec()).collect())
    }

    fn plan_mesh(&self, real_values: Vec<Scalar>, imag_values: Vec<Scalar>) -> PyResult<PyMellinEvalPlan> {
        let inner = self
            .inner
            .plan_mesh(&real_values, &imag_values)
            .map_err(mellin_err_to_py)?;
        Ok(PyMellinEvalPlan { inner })
    }

    fn plan_vertical_line(&self, real: Scalar, imag_values: Vec<Scalar>) -> PyResult<PyMellinEvalPlan> {
        let inner = self
            .inner
            .plan_vertical_line(real, &imag_values)
            .map_err(mellin_err_to_py)?;
        Ok(PyMellinEvalPlan { inner })
    }

    #[pyo3(signature = (real_values, imag_values))]
    fn evaluate_mesh_magnitude_flat(&self, real_values: Vec<Scalar>, imag_values: Vec<Scalar>) -> PyResult<Vec<Scalar>> {
        self.inner
            .evaluate_mesh_magnitude(&real_values, &imag_values)
            .map_err(mellin_err_to_py)
    }

    #[pyo3(signature = (real_values, imag_values, epsilon=0.0))]
    fn evaluate_mesh_log_magnitude_flat(
        &self,
        real_values: Vec<Scalar>,
        imag_values: Vec<Scalar>,
        epsilon: Scalar,
    ) -> PyResult<Vec<Scalar>> {
        self.inner
            .evaluate_mesh_log_magnitude(&real_values, &imag_values, epsilon)
            .map_err(mellin_err_to_py)
    }

    fn evaluate_with_series(
        &self,
        s: ComplexScalar,
        weighted: Vec<ComplexScalar>,
    ) -> PyResult<ComplexScalar> {
        self.inner
            .evaluate_with_series(s, &weighted)
            .map_err(mellin_err_to_py)
    }

    fn evaluate_many_with_series(
        &self,
        s_values: Vec<ComplexScalar>,
        weighted: Vec<ComplexScalar>,
    ) -> PyResult<Vec<ComplexScalar>> {
        self.inner
            .evaluate_many_with_series(&s_values, &weighted)
            .map_err(mellin_err_to_py)
    }

    fn evaluate_vertical_line(
        &self,
        real: Scalar,
        imag_values: Vec<Scalar>,
    ) -> PyResult<Vec<ComplexScalar>> {
        self.inner
            .evaluate_vertical_line(real, &imag_values)
            .map_err(mellin_err_to_py)
    }

    fn hilbert_inner_product(&self, other: &PyMellinLogGrid) -> PyResult<ComplexScalar> {
        self.inner
            .hilbert_inner_product(&other.inner)
            .map_err(mellin_err_to_py)
    }

    fn hilbert_norm(&self) -> PyResult<Scalar> {
        self.inner.hilbert_norm().map_err(mellin_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MellinLogGrid(len={}, log_start={:.4}, log_step={:.4})",
            self.inner.len(),
            self.inner.log_start(),
            self.inner.log_step()
        )
    }
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "frac")?;
    module.add_function(wrap_pyfunction!(gl_coeffs_adaptive, &module)?)?;
    module.add_function(wrap_pyfunction!(fracdiff_gl_1d, &module)?)?;
    module.add_function(wrap_pyfunction!(mellin_exp_decay_samples, &module)?)?;
    module.add_function(wrap_pyfunction!(mellin_exp_decay_samples_scaled, &module)?)?;
    module.add_function(wrap_pyfunction!(log_lattice_z_points, &module)?)?;
    module.add_function(wrap_pyfunction!(assemble_pzeta, &module)?)?;
    module.add_class::<PyMellinEvalPlan>()?;
    module.add_class::<PyMellinLogGrid>()?;
    module.add_class::<PyFractalFieldGenerator>()?;
    module.add_class::<PyLogZSeries>()?;
    module.add("__doc__", "Fractional differencing + Mellin/cosmology tooling")?;
    module.add(
        "__all__",
        vec![
            "assemble_pzeta",
            "log_lattice_z_points",
            "gl_coeffs_adaptive",
            "fracdiff_gl_1d",
            "mellin_exp_decay_samples",
            "mellin_exp_decay_samples_scaled",
            "FractalFieldGenerator",
            "LogZSeries",
            "MellinEvalPlan",
            "MellinLogGrid",
        ],
    )?;
    parent.add_submodule(&module)?;
    Ok(())
}
