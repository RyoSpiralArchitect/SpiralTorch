//! Bounded transport for the Rust-owned QKV corpus schedule and continuation.
use super::qkv::PyQkv;
use super::*;
use st_nn::causal_lm::{CausalLmQkvTrainingConfig, NativeCausalLmQkvTrainer};

#[pyclass(name = "NativeCausalLmQkvTrainer", module = "spiraltorch")]
pub(super) struct PyQkvTrainer {
    inner: NativeCausalLmQkvTrainer,
}

#[pymethods]
impl PyQkvTrainer {
    #[classattr]
    const MAX_TOKENS: usize = NativeCausalLmQkvTrainer::MAX_TOKENS;
    #[classattr]
    const MAX_CHECKPOINT_BYTES: usize = NativeCausalLmQkvTrainer::MAX_CHECKPOINT_BYTES;
    #[classattr]
    const MAX_CHECKPOINT_INTERVAL: u64 = NativeCausalLmQkvTrainer::MAX_CHECKPOINT_INTERVAL;

    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(optimizer, documents, context_length, *, learning_rate, labels=None, batch_windows=None, max_rows=None, epochs=None, weight_decay=None, label_smoothing=None))]
    fn new(
        py: Python<'_>,
        optimizer: &PyQkv,
        documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        batch_windows: Option<&Bound<'_, PyAny>>,
        max_rows: Option<&Bound<'_, PyAny>>,
        epochs: Option<&Bound<'_, PyAny>>,
        weight_decay: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let step = adamw_step(learning_rate, weight_decay)?;
        let objective = loss_objective(label_smoothing, None)?;
        let config = CausalLmQkvTrainingConfig {
            context_length: u32::try_from(dimension(context_length)?).map_err(error)?,
            batch_windows: limit(batch_windows, 1)?,
            max_rows: limit(max_rows, 1)?,
            epochs: u32::try_from(limit(epochs, 1)?).map_err(error)?,
            learning_rate: step.rate(),
            weight_decay: step.weight_decay(),
            label_smoothing: objective.label_smoothing,
        };
        config.validate().map_err(error)?;
        let (docs, labels) = documents::documents(
            documents,
            labels,
            Self::MAX_TOKENS,
            |lengths, has_labels| {
                NativeCausalLmQkvTrainer::document_plan(
                    &optimizer.inner,
                    config,
                    lengths,
                    has_labels,
                )
                .map(|_| ())
                .map_err(error)
            },
        )?;
        py.detach(|| NativeCausalLmQkvTrainer::new(optimizer.inner.clone(), docs, labels, config))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature=(model, documents, payload, expected_id, *, labels=None))]
    fn from_bytes(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        documents: &Bound<'_, PyAny>,
        payload: &Bound<'_, PyBytes>,
        expected_id: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pin = session_input::pin(expected_id)?;
        let raw = payload.as_bytes();
        if raw.len() > Self::MAX_CHECKPOINT_BYTES {
            return Err(error("QKV trainer checkpoint exceeds byte budget"));
        }
        let resume = py
            .detach(|| NativeCausalLmQkvTrainer::prepare_resume(&model.inner, raw, &pin))
            .map_err(error)?;
        let (docs, labels) = documents::documents(
            documents,
            labels,
            Self::MAX_TOKENS,
            |lengths, has_labels| {
                resume
                    .document_plan(lengths, has_labels)
                    .map(|_| ())
                    .map_err(error)
            },
        )?;
        py.detach(|| resume.resume(docs, labels))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[getter]
    fn updates(&self) -> u64 {
        self.inner.updates()
    }
    #[getter]
    fn total_updates(&self) -> u64 {
        self.inner.total_updates()
    }
    #[getter]
    fn active_windows(&self) -> usize {
        self.inner.active_windows()
    }
    #[getter]
    fn data_id(&self) -> &str {
        self.inner.data_id()
    }
    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.config()).map_err(error)
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let report = py.detach(|| -> PyResult<serde_json::Value> {
            let step = self.inner.step().map_err(error)?;
            serde_json::to_value(step).map_err(error)
        })?;
        crate::json::json_to_py(py, &report)
    }

    /// A detached optimizer snapshot for inference/export, never a mutable session owner.
    fn optimizer_snapshot(&self, py: Python<'_>) -> PyQkv {
        PyQkv {
            inner: py.detach(|| self.inner.optimizer().clone()),
        }
    }
    fn save_checkpoint_new(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<String> {
        py.detach(|| self.inner.save_checkpoint_new(path))
            .map_err(error)
    }

    fn advance_to_checkpoint_new(
        &mut self,
        py: Python<'_>,
        stop: &Bound<'_, PyAny>,
        path: std::path::PathBuf,
    ) -> PyResult<(Py<PyAny>, String)> {
        reject_bool(stop)?;
        let stop = stop.extract::<u64>()?;
        let (reports, pin) = py
            .detach(|| self.inner.advance_to_checkpoint_new(stop, path))
            .map_err(error)?;
        let value = serde_json::to_value(reports).map_err(error)?;
        Ok((crate::json::json_to_py(py, &value)?, pin))
    }

    fn checkpoint_with_id<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let (raw, pin) = py.detach(|| self.inner.checkpoint()).map_err(error)?;
        Ok((PyBytes::new(py, &raw), pin))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyQkvTrainer>()
}
