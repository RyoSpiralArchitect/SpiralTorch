use pyo3::prelude::*;
use pyo3::types::PyModule;
#[cfg(feature = "nn")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "nn")]
use crate::pure::PyOpenCartesianTopos;
#[cfg(feature = "nn")]
use crate::tensor::{tensor_err_to_py, PyTensor};

#[cfg(feature = "nn")]
use st_nn::{
    dataset::DataLoaderBatches,
    dataset_from_vec,
    zspace_coherence::{CoherenceDiagnostics, LinguisticChannelReport},
    DataLoader, Dataset, ZSpaceCoherenceSequencer,
};
#[cfg(feature = "nn")]
use st_tensor::OpenCartesianTopos;

#[cfg(feature = "nn")]
fn convert_samples(
    samples: Vec<(PyTensor, PyTensor)>,
) -> Vec<(st_tensor::Tensor, st_tensor::Tensor)> {
    samples
        .into_iter()
        .map(|(input, target)| (input.inner.clone(), target.inner.clone()))
        .collect()
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn")]
pub(crate) struct PyDataset {
    inner: Dataset,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataset {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Dataset::new(),
        }
    }

    #[staticmethod]
    pub fn from_pairs(samples: Vec<(PyTensor, PyTensor)>) -> Self {
        let converted = convert_samples(samples);
        Self {
            inner: Dataset::from_vec(converted),
        }
    }

    pub fn push(&mut self, input: &PyTensor, target: &PyTensor) {
        self.inner.push(input.inner.clone(), target.inner.clone());
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn loader(&self) -> PyDataLoader {
        PyDataLoader::from_loader(self.inner.loader())
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", unsendable)]
pub(crate) struct PyDataLoader {
    inner: DataLoader,
}

#[cfg(feature = "nn")]
impl PyDataLoader {
    fn from_loader(inner: DataLoader) -> Self {
        Self { inner }
    }

    fn iter_inner(&self) -> PyDataLoaderIter {
        PyDataLoaderIter::new(self.inner.clone().into_iter())
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataLoader {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    pub fn prefetch_depth(&self) -> usize {
        self.inner.prefetch_depth()
    }

    pub fn shuffle(&self, seed: u64) -> Self {
        Self::from_loader(self.inner.clone().shuffle(seed))
    }

    pub fn batched(&self, batch_size: usize) -> Self {
        Self::from_loader(self.inner.clone().batched(batch_size))
    }

    pub fn dynamic_batch_by_rows(&self, max_rows: usize) -> Self {
        Self::from_loader(self.inner.clone().dynamic_batch_by_rows(max_rows))
    }

    pub fn prefetch(&self, depth: usize) -> Self {
        Self::from_loader(self.inner.clone().prefetch(depth))
    }

    pub fn iter(&self, py: Python<'_>) -> PyResult<Py<PyDataLoaderIter>> {
        Py::new(py, self.iter_inner())
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyDataLoaderIter>> {
        slf.iter(slf.py())
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", unsendable)]
pub(crate) struct PyDataLoaderIter {
    batches: Option<DataLoaderBatches>,
}

#[cfg(feature = "nn")]
impl PyDataLoaderIter {
    fn new(batches: DataLoaderBatches) -> Self {
        Self {
            batches: Some(batches),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyDataLoaderIter> {
        slf.into()
    }

    fn __next__(&mut self) -> PyResult<Option<(PyTensor, PyTensor)>> {
        let batches = match self.batches.as_mut() {
            Some(iter) => iter,
            None => return Ok(None),
        };
        match batches.next() {
            Some(Ok((input, target))) => Ok(Some((
                PyTensor::from_tensor(input),
                PyTensor::from_tensor(target),
            ))),
            Some(Err(err)) => Err(tensor_err_to_py(err)),
            None => {
                self.batches = None;
                Ok(None)
            }
        }
    }
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (samples))]
fn from_samples(samples: Vec<(PyTensor, PyTensor)>) -> PyDataLoader {
    PyDataLoader::from_loader(dataset_from_vec(convert_samples(samples)))
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceChannelReport")]
pub(crate) struct PyCoherenceChannelReport {
    channel: usize,
    weight: f32,
    backend: String,
    dominant_concept: Option<String>,
    emphasis: f32,
    descriptor: Option<String>,
}

#[cfg(feature = "nn")]
impl PyCoherenceChannelReport {
    fn from_report(report: &LinguisticChannelReport) -> Self {
        Self {
            channel: report.channel(),
            weight: report.weight(),
            backend: report.backend().label().to_string(),
            dominant_concept: report
                .dominant_concept()
                .map(|concept| concept.label().to_string()),
            emphasis: report.emphasis(),
            descriptor: report.descriptor().map(|descriptor| descriptor.to_string()),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceChannelReport {
    #[getter]
    fn channel(&self) -> usize {
        self.channel
    }

    #[getter]
    fn weight(&self) -> f32 {
        self.weight
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }

    #[getter]
    fn dominant_concept(&self) -> Option<&str> {
        self.dominant_concept.as_deref()
    }

    #[getter]
    fn emphasis(&self) -> f32 {
        self.emphasis
    }

    #[getter]
    fn descriptor(&self) -> Option<&str> {
        self.descriptor.as_deref()
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceDiagnostics", unsendable)]
pub(crate) struct PyCoherenceDiagnostics {
    aggregated: PyTensor,
    coherence: Vec<f32>,
    channel_reports: Vec<PyCoherenceChannelReport>,
}

#[cfg(feature = "nn")]
impl PyCoherenceDiagnostics {
    fn from_diagnostics(diagnostics: CoherenceDiagnostics) -> Self {
        let (aggregated, coherence, channel_reports) = diagnostics.into_parts();
        let channel_reports = channel_reports
            .iter()
            .map(PyCoherenceChannelReport::from_report)
            .collect();
        Self {
            aggregated: PyTensor::from_tensor(aggregated),
            coherence,
            channel_reports,
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceDiagnostics {
    #[getter]
    fn aggregated(&self) -> PyTensor {
        self.aggregated.clone()
    }

    #[getter]
    fn coherence(&self) -> Vec<f32> {
        self.coherence.clone()
    }

    #[getter]
    fn channel_reports(&self) -> Vec<PyCoherenceChannelReport> {
        self.channel_reports.clone()
    }
}

#[cfg(feature = "nn")]
#[pyclass(
    module = "spiraltorch.nn",
    name = "ZSpaceCoherenceSequencer",
    unsendable
)]
pub(crate) struct PyZSpaceCoherenceSequencer {
    inner: ZSpaceCoherenceSequencer,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceCoherenceSequencer {
    #[new]
    #[pyo3(signature = (dim, num_heads, curvature, *, topos=None))]
    pub fn new(
        dim: usize,
        num_heads: usize,
        curvature: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<Self> {
        let topos = match topos {
            Some(guard) => guard.inner.clone(),
            None => OpenCartesianTopos::new(curvature, 1e-5, 10.0, 256, 8192)
                .map_err(tensor_err_to_py)?,
        };
        let inner = ZSpaceCoherenceSequencer::new(dim, num_heads, curvature, topos)
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&x.inner).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn forward_with_coherence(&self, x: &PyTensor) -> PyResult<(PyTensor, Vec<f32>)> {
        let (output, coherence) = self
            .inner
            .forward_with_coherence(&x.inner)
            .map_err(tensor_err_to_py)?;
        Ok((PyTensor::from_tensor(output), coherence))
    }

    pub fn project_to_zspace(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let projected = self
            .inner
            .project_to_zspace(&x.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(projected))
    }

    pub fn diagnostics(&self, x: &PyTensor) -> PyResult<PyCoherenceDiagnostics> {
        let diagnostics = self.inner.diagnostics(&x.inner).map_err(tensor_err_to_py)?;
        Ok(PyCoherenceDiagnostics::from_diagnostics(diagnostics))
    }

    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim
    }

    #[getter]
    pub fn num_heads(&self) -> usize {
        self.inner.num_heads
    }

    #[getter]
    pub fn curvature(&self) -> f32 {
        self.inner.curvature
    }

    pub fn maxwell_channels(&self) -> usize {
        self.inner.maxwell_channels()
    }

    pub fn topos(&self) -> PyOpenCartesianTopos {
        PyOpenCartesianTopos::from_topos(self.inner.topos().clone())
    }
}

#[cfg(feature = "nn")]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "nn")?;
    module.add("__doc__", "SpiralTorch neural network primitives")?;
    module.add_class::<PyDataset>()?;
    module.add_class::<PyDataLoader>()?;
    module.add_class::<PyDataLoaderIter>()?;
    module.add_class::<PyCoherenceChannelReport>()?;
    module.add_class::<PyCoherenceDiagnostics>()?;
    module.add_class::<PyZSpaceCoherenceSequencer>()?;
    module.add_function(wrap_pyfunction!(from_samples, &module)?)?;
    module.add(
        "__all__",
        vec![
            "Dataset",
            "DataLoader",
            "DataLoaderIter",
            "CoherenceChannelReport",
            "CoherenceDiagnostics",
            "ZSpaceCoherenceSequencer",
            "from_samples",
        ],
    )?;
    parent.add_submodule(&module)?;
    Ok(())
}

#[cfg(not(feature = "nn"))]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "nn")?;
    module.add("__doc__", "SpiralTorch neural network primitives")?;
    parent.add_submodule(&module)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
