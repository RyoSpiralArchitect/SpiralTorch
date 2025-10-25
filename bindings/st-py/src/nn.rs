use pyo3::prelude::*;
use pyo3::types::PyModule;
#[cfg(feature = "nn")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "nn")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "nn")]
use crate::pure::PyOpenCartesianTopos;
#[cfg(feature = "nn")]
use crate::tensor::{tensor_err_to_py, PyTensor};

#[cfg(feature = "nn")]
use st_core::theory::zpulse::ZScale;
#[cfg(feature = "nn")]
use st_nn::{
    dataset::DataLoaderBatches,
    dataset_from_vec,
    layers::{NonLiner, NonLinerActivation, NonLinerGeometry, NonLinerHyperbolicConfig},
    zspace_coherence::{
        is_swap_invariant as rust_is_swap_invariant, CoherenceDiagnostics, CoherenceLabel,
        CoherenceObservation, CoherenceSignature, LinguisticChannelReport, PreDiscardPolicy,
        PreDiscardSnapshot, PreDiscardTelemetry,
    },
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
fn parse_non_liner_activation(name: &str) -> PyResult<NonLinerActivation> {
    match name.to_ascii_lowercase().as_str() {
        "tanh" => Ok(NonLinerActivation::Tanh),
        "sigmoid" => Ok(NonLinerActivation::Sigmoid),
        "softsign" => Ok(NonLinerActivation::Softsign),
        other => Err(PyValueError::new_err(format!(
            "unknown activation '{other}', expected 'tanh', 'sigmoid', or 'softsign'"
        ))),
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "NonLiner", unsendable)]
pub(crate) struct PyNonLiner {
    inner: NonLiner,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyNonLiner {
    #[new]
    #[pyo3(signature = (name, features, *, activation="tanh", slope=1.0, gain=1.0, bias=0.0, curvature=None, z_scale=None, retention=0.0))]
    pub fn new(
        name: &str,
        features: usize,
        activation: &str,
        slope: f32,
        gain: f32,
        bias: f32,
        curvature: Option<f32>,
        z_scale: Option<f32>,
        retention: f32,
    ) -> PyResult<Self> {
        let activation = parse_non_liner_activation(activation)?;
        let geometry = if let Some(curvature) = curvature {
            let scale = match z_scale {
                Some(value) => ZScale::new(value)
                    .ok_or_else(|| PyValueError::new_err("z_scale must be positive and finite"))?,
                None => ZScale::ONE,
            };
            let config = NonLinerHyperbolicConfig::new(curvature, scale, retention)
                .map_err(tensor_err_to_py)?;
            NonLinerGeometry::hyperbolic(config)
        } else {
            if z_scale.is_some() {
                return Err(PyValueError::new_err("z_scale requires curvature"));
            }
            if retention != 0.0 {
                return Err(PyValueError::new_err("retention requires curvature"));
            }
            NonLinerGeometry::Euclidean
        };
        let inner =
            NonLiner::with_geometry(name, features, activation, slope, gain, bias, geometry)
                .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    pub fn reset_metrics(&self) {
        self.inner.reset_metrics();
    }

    #[pyo3(signature = (*, curvature=None, z_scale=None, retention=None))]
    pub fn configure_geometry(
        &mut self,
        curvature: Option<f32>,
        z_scale: Option<f32>,
        retention: Option<f32>,
    ) -> PyResult<()> {
        let geometry = if let Some(curvature) = curvature {
            let base = self.inner.geometry();
            let scale = match z_scale {
                Some(value) => ZScale::new(value)
                    .ok_or_else(|| PyValueError::new_err("z_scale must be positive and finite"))?,
                None => base.z_scale().unwrap_or(ZScale::ONE),
            };
            let retention = retention.unwrap_or_else(|| base.retention().unwrap_or(0.0));
            let config = NonLinerHyperbolicConfig::new(curvature, scale, retention)
                .map_err(tensor_err_to_py)?;
            NonLinerGeometry::hyperbolic(config)
        } else {
            if z_scale.is_some() {
                return Err(PyValueError::new_err("z_scale requires curvature"));
            }
            if let Some(retention) = retention {
                if retention != 0.0 {
                    return Err(PyValueError::new_err("retention requires curvature"));
                }
            }
            NonLinerGeometry::Euclidean
        };
        self.inner.set_geometry(geometry);
        Ok(())
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner.zero_accumulators().map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner.apply_step(fallback_lr).map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner.load_state_dict(&map).map_err(tensor_err_to_py)
    }

    #[getter]
    pub fn activation(&self) -> String {
        match self.inner.activation() {
            NonLinerActivation::Tanh => "tanh".to_string(),
            NonLinerActivation::Sigmoid => "sigmoid".to_string(),
            NonLinerActivation::Softsign => "softsign".to_string(),
        }
    }

    #[getter]
    pub fn curvature(&self) -> Option<f32> {
        self.inner.geometry().curvature()
    }

    #[getter]
    pub fn z_scale(&self) -> Option<f32> {
        self.inner.geometry().z_scale().map(|scale| scale.value())
    }

    #[getter]
    pub fn retention(&self) -> Option<f32> {
        self.inner.geometry().retention()
    }

    #[getter]
    pub fn psi_drift(&self) -> Option<f32> {
        self.inner.psi_probe()
    }

    #[getter]
    pub fn last_hyperbolic_radius(&self) -> Option<f32> {
        self.inner.last_hyperbolic_radius()
    }

    #[getter]
    pub fn gain(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.gain().value().clone())
    }

    #[getter]
    pub fn slope(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.slope().value().clone())
    }

    #[getter]
    pub fn bias(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.bias().value().clone())
    }

    pub fn gradients(&self) -> (Option<PyTensor>, Option<PyTensor>, Option<PyTensor>) {
        let gain = self
            .inner
            .gain()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        let slope = self
            .inner
            .slope()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        let bias = self
            .inner
            .bias()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        (gain, slope, bias)
    }
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
#[pyfunction(name = "is_swap_invariant")]
pub(crate) fn py_is_swap_invariant(arrangement: Vec<f32>) -> bool {
    rust_is_swap_invariant(&arrangement)
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
#[pyclass(module = "spiraltorch.nn", name = "CoherenceSignature", unsendable)]
pub(crate) struct PyCoherenceSignature {
    dominant_channel: Option<usize>,
    energy_ratio: f32,
    entropy: f32,
    mean_coherence: f32,
    swap_invariant: bool,
}

#[cfg(feature = "nn")]
impl PyCoherenceSignature {
    fn from_signature(signature: &CoherenceSignature) -> Self {
        Self {
            dominant_channel: signature.dominant_channel(),
            energy_ratio: signature.energy_ratio(),
            entropy: signature.entropy(),
            mean_coherence: signature.mean_coherence(),
            swap_invariant: signature.swap_invariant(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceSignature {
    #[getter]
    fn dominant_channel(&self) -> Option<usize> {
        self.dominant_channel
    }

    #[getter]
    fn energy_ratio(&self) -> f32 {
        self.energy_ratio
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.entropy
    }

    #[getter]
    fn mean_coherence(&self) -> f32 {
        self.mean_coherence
    }

    #[getter]
    fn swap_invariant(&self) -> bool {
        self.swap_invariant
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceObservation", unsendable)]
pub(crate) struct PyCoherenceObservation {
    observation: CoherenceObservation,
    label: CoherenceLabel,
}

#[cfg(feature = "nn")]
impl PyCoherenceObservation {
    fn from_observation(observation: CoherenceObservation) -> Self {
        let label = observation.lift_to_label();
        Self { observation, label }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceObservation {
    #[getter]
    fn is_signature(&self) -> bool {
        matches!(self.observation, CoherenceObservation::Signature(_))
    }

    #[getter]
    fn label(&self) -> String {
        self.label.to_string()
    }

    #[getter]
    fn signature(&self) -> Option<PyCoherenceSignature> {
        match &self.observation {
            CoherenceObservation::Signature(signature) => {
                Some(PyCoherenceSignature::from_signature(signature))
            }
            CoherenceObservation::Undetermined => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.observation {
            CoherenceObservation::Undetermined => format!(
                "CoherenceObservation(label='{}', signature=None)",
                self.label
            ),
            CoherenceObservation::Signature(_) => format!(
                "CoherenceObservation(label='{}', signature=...)",
                self.label
            ),
        }
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceDiagnostics", unsendable)]
pub(crate) struct PyCoherenceDiagnostics {
    aggregated: PyTensor,
    coherence: Vec<f32>,
    channel_reports: Vec<PyCoherenceChannelReport>,
    pre_discard: Option<PyPreDiscardTelemetry>,
    observation: PyCoherenceObservation,
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "PreDiscardTelemetry", unsendable)]
pub(crate) struct PyPreDiscardTelemetry {
    dominance_ratio: f32,
    energy_floor: f32,
    discarded: usize,
    preserved: usize,
    fallback: bool,
    survivor_energy: f32,
    discarded_energy: f32,
    total_energy: f32,
    survivor_energy_ratio: f32,
    discarded_energy_ratio: f32,
    dominant_weight: f32,
}

#[cfg(feature = "nn")]
impl PyPreDiscardTelemetry {
    fn from_telemetry(telemetry: PreDiscardTelemetry) -> Self {
        Self {
            dominance_ratio: telemetry.dominance_ratio(),
            energy_floor: telemetry.energy_floor(),
            discarded: telemetry.discarded(),
            preserved: telemetry.preserved(),
            fallback: telemetry.used_fallback(),
            survivor_energy: telemetry.survivor_energy(),
            discarded_energy: telemetry.discarded_energy(),
            total_energy: telemetry.total_energy(),
            survivor_energy_ratio: telemetry.survivor_energy_ratio(),
            discarded_energy_ratio: telemetry.discarded_energy_ratio(),
            dominant_weight: telemetry.dominant_weight(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyPreDiscardTelemetry {
    #[getter]
    fn dominance_ratio(&self) -> f32 {
        self.dominance_ratio
    }

    #[getter]
    fn energy_floor(&self) -> f32 {
        self.energy_floor
    }

    #[getter]
    fn discarded(&self) -> usize {
        self.discarded
    }

    #[getter]
    fn preserved(&self) -> usize {
        self.preserved
    }

    #[getter]
    fn used_fallback(&self) -> bool {
        self.fallback
    }

    #[getter]
    fn total(&self) -> usize {
        self.discarded + self.preserved
    }

    #[getter]
    fn preserved_ratio(&self) -> f32 {
        if self.discarded + self.preserved == 0 {
            0.0
        } else {
            (self.preserved as f32 / (self.discarded + self.preserved) as f32).clamp(0.0, 1.0)
        }
    }

    #[getter]
    fn discarded_ratio(&self) -> f32 {
        1.0 - self.preserved_ratio()
    }

    #[getter]
    fn survivor_energy(&self) -> f32 {
        self.survivor_energy
    }

    #[getter]
    fn discarded_energy(&self) -> f32 {
        self.discarded_energy
    }

    #[getter]
    fn total_energy(&self) -> f32 {
        self.total_energy
    }

    #[getter]
    fn survivor_energy_ratio(&self) -> f32 {
        self.survivor_energy_ratio
    }

    #[getter]
    fn discarded_energy_ratio(&self) -> f32 {
        self.discarded_energy_ratio
    }

    #[getter]
    fn dominant_weight(&self) -> f32 {
        self.dominant_weight
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "PreDiscardSnapshot", unsendable)]
pub(crate) struct PyPreDiscardSnapshot {
    step: u64,
    telemetry: PyPreDiscardTelemetry,
    survivors: Vec<usize>,
    discarded: Vec<usize>,
    filtered: Vec<f32>,
}

#[cfg(feature = "nn")]
impl PyPreDiscardSnapshot {
    fn from_snapshot(snapshot: PreDiscardSnapshot) -> Self {
        Self {
            step: snapshot.step(),
            telemetry: PyPreDiscardTelemetry::from_telemetry(snapshot.telemetry().clone()),
            survivors: snapshot.survivors().to_vec(),
            discarded: snapshot.discarded().to_vec(),
            filtered: snapshot.filtered().to_vec(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyPreDiscardSnapshot {
    #[getter]
    fn step(&self) -> u64 {
        self.step
    }

    #[getter]
    fn telemetry(&self) -> PyPreDiscardTelemetry {
        self.telemetry.clone()
    }

    #[getter]
    fn survivors(&self) -> Vec<usize> {
        self.survivors.clone()
    }

    #[getter]
    fn discarded(&self) -> Vec<usize> {
        self.discarded.clone()
    }

    #[getter]
    fn filtered(&self) -> Vec<f32> {
        self.filtered.clone()
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "PreDiscardPolicy", unsendable)]
pub(crate) struct PyPreDiscardPolicy {
    dominance_ratio: f32,
    energy_floor: f32,
    min_channels: usize,
}

#[cfg(feature = "nn")]
impl PyPreDiscardPolicy {
    fn from_policy(policy: &PreDiscardPolicy) -> Self {
        Self {
            dominance_ratio: policy.dominance_ratio(),
            energy_floor: policy.energy_floor(),
            min_channels: policy.min_channels(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyPreDiscardPolicy {
    #[new]
    #[pyo3(signature = (dominance_ratio, *, energy_floor=None, min_channels=None))]
    fn new(
        dominance_ratio: f32,
        energy_floor: Option<f32>,
        min_channels: Option<usize>,
    ) -> PyResult<Self> {
        let mut policy = PreDiscardPolicy::new(dominance_ratio).map_err(tensor_err_to_py)?;
        if let Some(floor) = energy_floor {
            policy = policy.with_energy_floor(floor).map_err(tensor_err_to_py)?;
        }
        if let Some(min_channels) = min_channels {
            policy = policy.with_min_channels(min_channels);
        }
        Ok(Self::from_policy(&policy))
    }

    #[getter]
    fn dominance_ratio(&self) -> f32 {
        self.dominance_ratio
    }

    #[getter]
    fn energy_floor(&self) -> f32 {
        self.energy_floor
    }

    #[getter]
    fn min_channels(&self) -> usize {
        self.min_channels
    }
}

#[cfg(feature = "nn")]
impl PyCoherenceDiagnostics {
    fn from_diagnostics(diagnostics: CoherenceDiagnostics) -> Self {
        let observation = PyCoherenceObservation::from_observation(diagnostics.observation());
        let (aggregated, coherence, channel_reports, pre_discard) = diagnostics.into_parts();
        let channel_reports = channel_reports
            .iter()
            .map(PyCoherenceChannelReport::from_report)
            .collect();
        Self {
            aggregated: PyTensor::from_tensor(aggregated),
            coherence,
            channel_reports,
            pre_discard: pre_discard.map(PyPreDiscardTelemetry::from_telemetry),
            observation,
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

    #[getter]
    fn preserved_channels(&self) -> usize {
        self.coherence.iter().filter(|value| **value > 0.0).count()
    }

    #[getter]
    fn discarded_channels(&self) -> usize {
        self.coherence
            .len()
            .saturating_sub(self.preserved_channels())
    }

    #[getter]
    fn pre_discard(&self) -> Option<PyPreDiscardTelemetry> {
        self.pre_discard.clone()
    }

    #[getter]
    fn observation(&self) -> PyCoherenceObservation {
        self.observation.clone()
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

    pub fn diagnostics(&self, x: &PyTensor) -> PyResult<PyCoherenceDiagnostics> {
        let diagnostics = self.inner.diagnostics(&x.inner).map_err(tensor_err_to_py)?;
        Ok(PyCoherenceDiagnostics::from_diagnostics(diagnostics))
    }

    #[pyo3(signature = (dominance_ratio, *, energy_floor=None, min_channels=None))]
    pub fn configure_pre_discard(
        &mut self,
        dominance_ratio: f32,
        energy_floor: Option<f32>,
        min_channels: Option<usize>,
    ) -> PyResult<()> {
        let mut policy = PreDiscardPolicy::new(dominance_ratio).map_err(tensor_err_to_py)?;
        if let Some(floor) = energy_floor {
            policy = policy.with_energy_floor(floor).map_err(tensor_err_to_py)?;
        }
        if let Some(min_channels) = min_channels {
            policy = policy.with_min_channels(min_channels);
        }
        self.inner.enable_pre_discard(policy);
        Ok(())
    }

    pub fn disable_pre_discard(&mut self) {
        self.inner.disable_pre_discard();
    }

    pub fn configure_pre_discard_memory(&mut self, limit: usize) {
        self.inner.configure_pre_discard_memory(limit);
    }

    pub fn clear_pre_discard_snapshots(&self) {
        self.inner.clear_pre_discard_snapshots();
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
    pub fn pre_discard_policy(&self) -> Option<PyPreDiscardPolicy> {
        self.inner
            .pre_discard_policy()
            .map(PyPreDiscardPolicy::from_policy)
    }

    #[getter]
    pub fn pre_discard_snapshots(&self) -> Vec<PyPreDiscardSnapshot> {
        self.inner
            .pre_discard_snapshots()
            .into_iter()
            .map(PyPreDiscardSnapshot::from_snapshot)
            .collect()
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
    module.add_class::<PyNonLiner>()?;
    module.add_class::<PyDataset>()?;
    module.add_class::<PyDataLoader>()?;
    module.add_class::<PyDataLoaderIter>()?;
    module.add_function(wrap_pyfunction!(py_is_swap_invariant, module)?)?;
    module.add_class::<PyCoherenceChannelReport>()?;
    module.add_class::<PyCoherenceSignature>()?;
    module.add_class::<PyCoherenceObservation>()?;
    module.add_class::<PyPreDiscardTelemetry>()?;
    module.add_class::<PyPreDiscardPolicy>()?;
    module.add_class::<PyPreDiscardSnapshot>()?;
    module.add_class::<PyCoherenceDiagnostics>()?;
    module.add_class::<PyZSpaceCoherenceSequencer>()?;
    module.add_function(wrap_pyfunction!(from_samples, &module)?)?;
    module.add(
        "__all__",
        vec![
            "NonLiner",
            "Dataset",
            "DataLoader",
            "DataLoaderIter",
            "CoherenceChannelReport",
            "CoherenceDiagnostics",
            "PreDiscardTelemetry",
            "PreDiscardPolicy",
            "PreDiscardSnapshot",
            "ZSpaceCoherenceSequencer",
            "from_samples",
        ],
    )?;
    parent.add_submodule(&module)?;
    if let Ok(non_liner) = module.getattr("NonLiner") {
        parent.add("NonLiner", non_liner)?;
    }
    if let Ok(sequencer) = module.getattr("ZSpaceCoherenceSequencer") {
        parent.add("ZSpaceCoherenceSequencer", sequencer)?;
    }
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
