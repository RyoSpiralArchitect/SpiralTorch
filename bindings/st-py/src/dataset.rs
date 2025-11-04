#![allow(dead_code)]

#[cfg(feature = "nn")]
use crate::tensor::{tensor_err_to_py, PyTensor};
#[cfg(feature = "nn")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "nn")]
use pyo3::prelude::*;
#[cfg(feature = "nn")]
use pyo3::types::PyList;
#[cfg(feature = "nn")]
use pyo3::PyRefMut;
#[cfg(feature = "nn")]
use st_nn::dataset::{DataLoader as RustDataLoader, DataLoaderBatches, Dataset as RustDataset};

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.dataset", name = "Dataset")]
#[derive(Clone)]
pub(crate) struct PyDataset {
    inner: RustDataset,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataset {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustDataset::new(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (samples=None))]
    pub fn from_samples(
        py: Python<'_>,
        samples: Option<Vec<(Py<PyTensor>, Py<PyTensor>)>>,
    ) -> PyResult<Self> {
        let mut dataset = RustDataset::new();
        if let Some(pairs) = samples {
            for (input, target) in pairs {
                let input_tensor = input.bind(py).borrow().inner.clone();
                let target_tensor = target.bind(py).borrow().inner.clone();
                dataset.push(input_tensor, target_tensor);
            }
        }
        Ok(Self { inner: dataset })
    }

    pub fn push(&mut self, input: &PyTensor, target: &PyTensor) {
        self.inner.push(input.inner.clone(), target.inner.clone());
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn samples(&self) -> PyResult<Vec<(PyTensor, PyTensor)>> {
        let mut out = Vec::with_capacity(self.inner.len());
        for (input, target) in self.inner.iter() {
            out.push((PyTensor::from_tensor(input), PyTensor::from_tensor(target)));
        }
        Ok(out)
    }

    pub fn iter(&self, py: Python<'_>) -> PyResult<Py<PyDataLoaderIter>> {
        PyDataLoader::iterator_py(py, self.inner.loader())
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyDataLoaderIter>> {
        self.iter(py)
    }

    pub fn loader(&self) -> PyDataLoader {
        PyDataLoader::from_loader(self.inner.loader())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.dataset", name = "DataLoader")]
#[derive(Clone)]
pub(crate) struct PyDataLoader {
    inner: RustDataLoader,
}

#[cfg(feature = "nn")]
impl PyDataLoader {
    fn from_loader(loader: RustDataLoader) -> Self {
        Self { inner: loader }
    }

    fn clone_with(&self, loader: RustDataLoader) -> Self {
        Self { inner: loader }
    }

    fn iterator_py(py: Python<'_>, loader: RustDataLoader) -> PyResult<Py<PyDataLoaderIter>> {
        Py::new(py, PyDataLoaderIter::new(loader))
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataLoader {
    #[staticmethod]
    #[pyo3(signature = (samples=None))]
    pub fn from_samples(
        py: Python<'_>,
        samples: Option<Vec<(Py<PyTensor>, Py<PyTensor>)>>,
    ) -> PyResult<Self> {
        let dataset = PyDataset::from_samples(py, samples)?;
        Ok(Self::from_loader(dataset.inner.loader()))
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.len()
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
        self.clone_with(self.inner.clone().shuffle(seed))
    }

    pub fn batched(&self, batch_size: usize) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        Ok(self.clone_with(self.inner.clone().batched(batch_size)))
    }

    pub fn dynamic_batch_by_rows(&self, max_rows: usize) -> PyResult<Self> {
        if max_rows == 0 {
            return Err(PyValueError::new_err("max_rows must be positive"));
        }
        Ok(self.clone_with(self.inner.clone().dynamic_batch_by_rows(max_rows)))
    }

    pub fn prefetch(&self, depth: usize) -> Self {
        self.clone_with(self.inner.clone().prefetch(depth))
    }

    pub fn iter(&self, py: Python<'_>) -> PyResult<Py<PyDataLoaderIter>> {
        Self::iterator_py(py, self.inner.clone())
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyDataLoaderIter>> {
        self.iter(py)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.dataset", name = "DataLoaderIterator")]
pub(crate) struct PyDataLoaderIter {
    inner: Option<DataLoaderBatches>,
}

#[cfg(feature = "nn")]
impl PyDataLoaderIter {
    fn new(loader: RustDataLoader) -> Self {
        Self {
            inner: Some(loader.into_iter()),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataLoaderIter {
    /// Returns an iterator over the dataset.
    /// 
    /// This method implements the Python iterator protocol (__iter__).
    /// The conversion from PyRef to Py<PyDataLoaderIter> is guaranteed to succeed
    /// as it only involves reference counting, not any fallible operations.
    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyDataLoaderIter> {
        slf.into()
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<(PyTensor, PyTensor)>> {
        let Some(iter) = slf.inner.as_mut() else {
            return Ok(None);
        };
        match iter.next() {
            Some(Ok((input, target))) => Ok(Some((
                PyTensor::from_tensor(input),
                PyTensor::from_tensor(target),
            ))),
            Some(Err(err)) => Err(tensor_err_to_py(err)),
            None => {
                slf.inner = None;
                Ok(None)
            }
        }
    }
}

#[cfg(feature = "nn")]
pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let dataset = PyModule::new_bound(py, "dataset")?;
    dataset.add(
        "__doc__",
        "Datasets & loaders backed by the Rust session core.",
    )?;
    dataset.add_class::<PyDataset>()?;
    dataset.add_class::<PyDataLoader>()?;
    dataset.add_class::<PyDataLoaderIter>()?;
    let exports = PyList::new_bound(py, &["Dataset", "DataLoader", "DataLoaderIterator"]);
    dataset.add("__all__", exports)?;
    parent.add_submodule(&dataset)?;
    Ok(())
}

#[cfg(not(feature = "nn"))]
pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let dataset = PyModule::new_bound(py, "dataset")?;
    dataset.add(
        "__doc__",
        "Datasets & loaders (native extension unavailable in this build).",
    )?;
    parent.add_submodule(&dataset)?;
    Ok(())
}
