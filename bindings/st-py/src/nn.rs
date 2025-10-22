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
    DataLoader,
    Dataset,
    ZSpaceCoherenceSequencer,
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
#[pyclass(module = "spiraltorch.nn", name = "CoherenceDiagnostics", unsendable)]
pub(crate) struct PyCoherenceDiagnostics {
    channel_weights: Vec<f32>,
    normalized_weights: Vec<f32>,
    normalization: f32,
    fractional_order: f32,
    dominant_channel: Option<usize>,
    mean_coherence: f32,
    z_bias: f32,
    energy_ratio: f32,
    coherence_entropy: f32,
}

#[cfg(feature = "nn")]
impl PyCoherenceDiagnostics {
    fn from_diagnostics(diagnostics: CoherenceDiagnostics) -> Self {
        Self {
            channel_weights: diagnostics.channel_weights().to_vec(),
            normalized_weights: diagnostics.normalized_weights().to_vec(),
            normalization: diagnostics.normalization(),
            fractional_order: diagnostics.fractional_order(),
            dominant_channel: diagnostics.dominant_channel(),
            mean_coherence: diagnostics.mean_coherence(),
            z_bias: diagnostics.z_bias(),
            energy_ratio: diagnostics.energy_ratio(),
            coherence_entropy: diagnostics.coherence_entropy(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceDiagnostics {
    #[getter]
    fn channel_weights(&self) -> Vec<f32> {
        self.channel_weights.clone()
    }

    #[getter]
    fn normalized_weights(&self) -> Vec<f32> {
        self.normalized_weights.clone()
    }

    #[getter]
    fn normalization(&self) -> f32 {
        self.normalization
    }

    #[getter]
    fn fractional_order(&self) -> f32 {
        self.fractional_order
    }

    #[getter]
    fn dominant_channel(&self) -> Option<usize> {
        self.dominant_channel
    }

    #[getter]
    fn mean_coherence(&self) -> f32 {
        self.mean_coherence
    }

    #[getter]
    fn z_bias(&self) -> f32 {
        self.z_bias
    }

    #[getter]
    fn energy_ratio(&self) -> f32 {
        self.energy_ratio
    }

    #[getter]
    fn coherence_entropy(&self) -> f32 {
        self.coherence_entropy
    }

    fn __repr__(&self) -> String {
        format!(
            "CoherenceDiagnostics(mean={:.4}, entropy={:.4}, dominant_channel={:?})",
            self.mean_coherence, self.coherence_entropy, self.dominant_channel
        )
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

    pub fn forward_with_diagnostics(
        &self,
        x: &PyTensor,
    ) -> PyResult<(PyTensor, Vec<f32>, PyCoherenceDiagnostics)> {
        let (output, coherence, diagnostics) = self
            .inner
            .forward_with_diagnostics(&x.inner)
            .map_err(tensor_err_to_py)?;
        Ok((
            PyTensor::from_tensor(output),
            coherence,
            PyCoherenceDiagnostics::from_diagnostics(diagnostics),
        ))
    }

    pub fn project_to_zspace(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let projected = self
            .inner
            .project_to_zspace(&x.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(projected))
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
    module.add_class::<PyCoherenceDiagnostics>()?;
    module.add_class::<PyZSpaceCoherenceSequencer>()?;
    module.add_function(wrap_pyfunction!(from_samples, &module)?)?;
    module.add(
        "__all__",
        vec![
            "Dataset",
            "DataLoader",
            "DataLoaderIter",
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
