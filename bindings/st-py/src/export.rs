use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use spiral_opt::{
    CompressionReport, OptimisationError, QatConfig, QatObserver, QuantizationLeveling,
    QuantizationReport, StructuredPruner, StructuredPruningConfig, StructuredPruningReport,
};

impl From<OptimisationError> for PyErr {
    fn from(value: OptimisationError) -> Self {
        pyo3::exceptions::PyValueError::new_err(value.to_string())
    }
}

#[pyclass(module = "spiraltorch.export")]
pub struct PyQuantizationReport {
    inner: QuantizationReport,
}

#[pymethods]
impl PyQuantizationReport {
    #[getter]
    fn bit_width(&self) -> u8 {
        self.inner.bit_width
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    fn zero_point(&self) -> f32 {
        self.inner.zero_point
    }

    #[getter]
    fn quant_error(&self) -> f32 {
        self.inner.quant_error
    }

    #[getter]
    fn observed_steps(&self) -> u64 {
        self.inner.observed_steps
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("bit_width", self.inner.bit_width)?;
        dict.set_item("observed_min", self.inner.observed_min)?;
        dict.set_item("observed_max", self.inner.observed_max)?;
        dict.set_item("scale", self.inner.scale)?;
        dict.set_item("zero_point", self.inner.zero_point)?;
        dict.set_item("quant_error", self.inner.quant_error)?;
        dict.set_item("observed_steps", self.inner.observed_steps)?;
        Ok(dict)
    }
}

impl From<QuantizationReport> for PyQuantizationReport {
    fn from(inner: QuantizationReport) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch.export")]
pub struct PyStructuredPruningReport {
    inner: StructuredPruningReport,
}

#[pymethods]
impl PyStructuredPruningReport {
    #[getter]
    fn target_sparsity(&self) -> f32 {
        self.inner.target_sparsity
    }

    #[getter]
    fn achieved_sparsity(&self) -> f32 {
        self.inner.achieved_sparsity
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.inner.block_size
    }

    #[getter]
    fn pruned_blocks(&self) -> usize {
        self.inner.pruned_blocks
    }

    #[getter]
    fn kept_blocks(&self) -> usize {
        self.inner.kept_blocks
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("target_sparsity", self.inner.target_sparsity)?;
        dict.set_item("achieved_sparsity", self.inner.achieved_sparsity)?;
        dict.set_item("block_size", self.inner.block_size)?;
        dict.set_item("pruned_blocks", self.inner.pruned_blocks)?;
        dict.set_item("kept_blocks", self.inner.kept_blocks)?;
        dict.set_item("l2_error", self.inner.l2_error)?;
        Ok(dict)
    }
}

impl From<StructuredPruningReport> for PyStructuredPruningReport {
    fn from(inner: StructuredPruningReport) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch.export")]
pub struct PyCompressionReport {
    inner: CompressionReport,
}

#[pymethods]
impl PyCompressionReport {
    #[getter]
    fn original_params(&self) -> usize {
        self.inner.original_params
    }

    #[getter]
    fn remaining_params(&self) -> usize {
        self.inner.remaining_params
    }

    #[getter]
    fn estimated_latency_reduction(&self) -> f32 {
        self.inner.estimated_latency_reduction
    }

    fn has_quantization(&self) -> bool {
        self.inner.quantization.is_some()
    }

    fn has_pruning(&self) -> bool {
        self.inner.pruning.is_some()
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("original_params", self.inner.original_params)?;
        dict.set_item("remaining_params", self.inner.remaining_params)?;
        dict.set_item(
            "estimated_latency_reduction",
            self.inner.estimated_latency_reduction,
        )?;
        if let Some(ref q) = self.inner.quantization {
            dict.set_item(
                "quantization",
                PyQuantizationReport { inner: q.clone() }.as_dict(py)?,
            )?;
        }
        if let Some(ref p) = self.inner.pruning {
            dict.set_item(
                "pruning",
                PyStructuredPruningReport { inner: p.clone() }.as_dict(py)?,
            )?;
        }
        Ok(dict)
    }
}

impl From<CompressionReport> for PyCompressionReport {
    fn from(inner: CompressionReport) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch.export")]
pub struct PyQatObserver {
    inner: QatObserver,
}

#[pymethods]
impl PyQatObserver {
    #[new]
    #[pyo3(signature = (bit_width=8, ema_decay=0.9, clamp_value=Some(6.0), epsilon=1e-6, symmetric=true))]
    fn new(
        bit_width: u8,
        ema_decay: f32,
        clamp_value: Option<f32>,
        epsilon: f32,
        symmetric: bool,
    ) -> PyResult<Self> {
        let config = QatConfig {
            bit_width,
            ema_decay,
            clamp_value,
            epsilon,
        };
        let leveling = if symmetric {
            QuantizationLeveling::Symmetric
        } else {
            QuantizationLeveling::Asymmetric
        };
        Ok(Self {
            inner: QatObserver::new(config, leveling),
        })
    }

    fn observe(&mut self, weights: Vec<f32>) {
        self.inner.observe(&weights);
    }

    fn quantize(&mut self, mut weights: Vec<f32>) -> (Vec<f32>, PyQuantizationReport) {
        let report = self.inner.quantize(&mut weights);
        (weights, PyQuantizationReport::from(report))
    }
}

#[pyfunction]
#[pyo3(signature = (weights, block_size=32, target_sparsity=0.5, min_l2_keep=1e-4))]
fn structured_prune(
    mut weights: Vec<f32>,
    block_size: usize,
    target_sparsity: f32,
    min_l2_keep: f32,
) -> PyResult<(Vec<f32>, PyStructuredPruningReport)> {
    let pruner = StructuredPruner::new();
    let config = StructuredPruningConfig {
        block_size,
        target_sparsity,
        min_l2_keep,
    };
    let report = pruner.apply(&mut weights, config)?;
    Ok((weights, PyStructuredPruningReport::from(report)))
}

#[pyfunction]
#[pyo3(signature = (weights, observer, pruning_config=None, latency_hint=0.25))]
fn compress_weights(
    mut weights: Vec<f32>,
    observer: &PyCell<PyQatObserver>,
    pruning_config: Option<(usize, f32, f32)>,
    latency_hint: f32,
) -> PyResult<(Vec<f32>, PyCompressionReport)> {
    let mut observer_mut = observer.borrow_mut();
    observer_mut.inner.observe(&weights);
    let quant_report = observer_mut.inner.quantize(&mut weights);

    let (remaining_params, pruning_report) = if let Some((block, sparsity, keep)) = pruning_config {
        let pruner = StructuredPruner::new();
        let report = pruner.apply(
            &mut weights,
            StructuredPruningConfig {
                block_size: block,
                target_sparsity: sparsity,
                min_l2_keep: keep,
            },
        )?;
        let remaining = weights.iter().filter(|&&w| w != 0.0).count();
        (remaining, Some(report))
    } else {
        let remaining = weights.iter().filter(|&&w| w != 0.0).count();
        (remaining, None)
    };

    let original_params = weights.len();
    let density = if original_params == 0 {
        0.0
    } else {
        remaining_params as f32 / original_params as f32
    };
    let estimated_reduction = if density >= 1.0 {
        0.0
    } else {
        (1.0 - density).max(latency_hint)
    };

    let compression = CompressionReport::new(
        original_params,
        remaining_params,
        Some(quant_report.clone()),
        pruning_report.clone(),
        estimated_reduction,
    );

    Ok((weights, PyCompressionReport::from(compression)))
}

pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyQatObserver>()?;
    m.add_class::<PyQuantizationReport>()?;
    m.add_class::<PyStructuredPruningReport>()?;
    m.add_class::<PyCompressionReport>()?;
    m.add_function(wrap_pyfunction!(structured_prune, m)?)?;
    m.add_function(wrap_pyfunction!(compress_weights, m)?)?;
    m.add(
        "__doc__",
        "Quantization-aware export and structured pruning helpers",
    )?;
    let _ = py;
    Ok(())
}
