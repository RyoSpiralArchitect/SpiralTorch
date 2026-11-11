//! Bounded transport only; the native QKV optimizer owns all learning semantics.
use super::generation::PyNativeCausalLmGenerator;
use super::*;
use pyo3::types::PyDict;
use st_nn::causal_lm::{
    NativeCausalLmContext, NativeCausalLmProjectionPrefix, NativeCausalLmQkvAdamW,
};
use st_nn::geometry_runtime::AdamWConfig;

#[pyclass(name = "NativeCausalLmQkvAdamW", module = "spiraltorch")]
pub(super) struct PyQkv {
    pub(super) inner: NativeCausalLmQkvAdamW,
}

/// Immutable cached prefix of one exact trained projection snapshot.
#[pyclass(
    name = "NativeCausalLmProjectionPrefix",
    module = "spiraltorch",
    frozen
)]
pub(super) struct PyProjectionPrefix {
    inner: NativeCausalLmProjectionPrefix,
}

#[pymethods]
impl PyProjectionPrefix {
    #[getter]
    fn serving_id(&self) -> &str {
        self.inner.projection_id()
    }
    #[getter]
    fn token_ids(&self) -> Vec<usize> {
        self.inner.token_ids().to_vec()
    }
    #[getter]
    fn cache_values(&self) -> usize {
        self.inner.cache_values()
    }
    #[getter]
    fn max_cache_values(&self) -> usize {
        self.inner.max_cache_values()
    }
    fn logits_f32le<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        bytes(py, self.inner.logits())
    }
    /// Return a new branch; this prefix remains bound to its original weights.
    fn branch(&self, py: Python<'_>, token_id: &Bound<'_, PyAny>) -> PyResult<Self> {
        let id = dimension(token_id)?;
        py.detach(|| self.inner.branch(id))
            .map(|inner| Self { inner })
            .map_err(error)
    }
}

fn scalar(value: Option<&Bound<'_, PyAny>>, default: f32) -> PyResult<f32> {
    match value {
        Some(v) => {
            reject_bool(v)?;
            v.extract()
        }
        None => Ok(default),
    }
}

fn inputs(
    model: &NativeCausalLmQkvAdamW,
    prompts: &Bound<'_, PyAny>,
    gold: &Bound<'_, PyAny>,
    maximum: usize,
) -> PyResult<(Vec<Vec<usize>>, Vec<usize>)> {
    let n = prompts.len()?;
    if n == 0 || n > NativeCausalLmQkvAdamW::MAX_ANSWER_SEQUENCES || gold.len()? != n {
        return Err(error("invalid QKV answer roster"));
    }
    let labels = tokens(gold, n)?;
    let (rows, _) = documents::documents(prompts, None, usize::MAX, |lengths, _| {
        model
            .answer_sequence_input_count(lengths, maximum)
            .map(|_| ())
            .map_err(error)
    })?;
    Ok((rows, labels))
}

fn bytes<'py>(py: Python<'py>, values: &[f32]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, values.len() * 4, |out| {
        for (word, value) in out.chunks_exact_mut(4).zip(values) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    })
}

#[allow(clippy::type_complexity)]
fn causal_documents(
    model: &NativeCausalLmQkvAdamW,
    value: &Bound<'_, PyAny>,
    labels: Option<&Bound<'_, PyAny>>,
    maximum: usize,
) -> PyResult<(Vec<Vec<usize>>, Option<Vec<Vec<i64>>>)> {
    let n = value.len()?;
    if n == 0 || n > NativeCausalLmQkvAdamW::MAX_CAUSAL_DOCUMENTS {
        return Err(error("invalid QKV causal document roster"));
    }
    documents::documents(value, labels, usize::MAX, |lengths, has_labels| {
        model
            .document_input_count(lengths, maximum, has_labels)
            .map(|_| ())
            .map_err(error)
    })
}

#[pymethods]
impl PyQkv {
    #[classattr]
    const MAX_CHECKPOINT_BYTES: usize = NativeCausalLmQkvAdamW::MAX_CHECKPOINT_BYTES;
    #[classattr]
    const MAX_ANSWER_SEQUENCES: usize = NativeCausalLmQkvAdamW::MAX_ANSWER_SEQUENCES;
    #[classattr]
    const MAX_CAUSAL_DOCUMENTS: usize = NativeCausalLmQkvAdamW::MAX_CAUSAL_DOCUMENTS;

    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(model, layer, *, beta1=None, beta2=None, epsilon=None, max_gradient_norm=None))]
    fn new(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        layer: &Bound<'_, PyAny>,
        beta1: Option<&Bound<'_, PyAny>>,
        beta2: Option<&Bound<'_, PyAny>>,
        epsilon: Option<&Bound<'_, PyAny>>,
        max_gradient_norm: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let layer = dimension(layer)?;
        let defaults = AdamWConfig::default();
        let config = AdamWConfig::new(
            scalar(beta1, defaults.beta1())?,
            scalar(beta2, defaults.beta2())?,
            scalar(epsilon, defaults.epsilon())?,
        )
        .map_err(error)?;
        let clip = max_gradient_norm
            .map(|value| {
                st_nn::geometry_runtime::GlobalNormClip::new(scalar(Some(value), 0.)?)
                    .map_err(error)
            })
            .transpose()?;
        py.detach(|| {
            NativeCausalLmQkvAdamW::new_with_gradient_clip(&model.inner, layer, config, clip)
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    #[getter]
    fn steps(&self) -> u32 {
        self.inner.steps()
    }
    #[getter]
    fn layer(&self) -> usize {
        self.inner.layer()
    }
    #[getter]
    fn max_gradient_norm(&self) -> Option<f32> {
        self.inner.max_gradient_norm()
    }
    #[getter]
    fn serving_id(&self, py: Python<'_>) -> String {
        py.detach(|| self.inner.serving_id())
    }

    #[pyo3(signature=(token_documents, context_length, *, microbatch_windows=None, labels=None))]
    fn evaluate_documents(
        &self,
        py: Python<'_>,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        microbatch_windows: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<super::evaluation::PyNativeCausalLmQkvEvaluation> {
        super::evaluation::evaluate_qkv(
            &self.inner,
            py,
            token_documents,
            context_length,
            microbatch_windows,
            labels,
        )
    }

    #[pyo3(signature=(prompt, *, max_cache_values=None))]
    fn prefill(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        max_cache_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyProjectionPrefix> {
        let budget = limit(
            max_cache_values,
            NativeCausalLmContext::DEFAULT_MAX_CACHE_VALUES,
        )?;
        let count = self
            .inner
            .prefill_input_count(prompt.len()?, budget)
            .map_err(error)?;
        let ids = tokens(prompt, count)?;
        py.detach(|| self.inner.prefill(ids, budget))
            .map(|inner| PyProjectionPrefix { inner })
            .map_err(error)
    }

    fn generator(
        &self,
        py: Python<'_>,
        prompt_ids: &Bound<'_, PyAny>,
        config_json: &str,
    ) -> PyResult<PyNativeCausalLmGenerator> {
        let config = st_nn::causal_lm::CausalLmGenerationConfig::from_json(config_json.as_bytes())
            .map_err(error)?;
        let count = self
            .inner
            .generation_input_count(prompt_ids.len()?, &config)
            .map_err(error)?;
        let ids = tokens(prompt_ids, count)?;
        py.detach(|| self.inner.generator(ids, config))
            .map(PyNativeCausalLmGenerator::from_native)
            .map_err(error)
    }

    fn restore_generator(
        &self,
        py: Python<'_>,
        payload: &str,
        expected_id: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeCausalLmGenerator> {
        let pin = session_input::pin(expected_id)?;
        py.detach(|| self.inner.restore_generator(payload.as_bytes(), &pin))
            .map(PyNativeCausalLmGenerator::from_native)
            .map_err(error)
    }

    #[pyo3(signature=(prompts, gold, *, max_rows=None, label_smoothing=None))]
    fn answer_gradients(
        &self,
        py: Python<'_>,
        prompts: &Bound<'_, PyAny>,
        gold: &Bound<'_, PyAny>,
        max_rows: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let cap = limit(max_rows, 1)?;
        let objective = loss_objective(label_smoothing, None)?;
        let (rows, labels) = inputs(&self.inner, prompts, gold, cap)?;
        let (loss, gradient) = py
            .detach(|| {
                let qs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
                self.inner.answer_sequence_gradients_with_label_smoothing(
                    &qs,
                    &labels,
                    cap,
                    objective.label_smoothing,
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        if label_smoothing.is_some() {
            out.set_item("objective_loss", loss)?;
            out.set_item("label_smoothing", objective.label_smoothing)?;
        } else {
            out.set_item("gold_nll", loss)?;
        }
        out.set_item("gradient_shape", gradient.shape())?;
        out.set_item("gradient_f32le", bytes(py, gradient.data())?)?;
        Ok(out.into_any().unbind())
    }

    #[pyo3(signature=(windows, window_shape, *, labels=None, label_smoothing=None))]
    fn loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        window_shape: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let objective = loss_objective(label_smoothing, None)?;
        let shape = shape(window_shape)?;
        let count = self
            .inner
            .loss_input_count(shape, labels.is_some())
            .map_err(error)?;
        let ids = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        let (loss, gradient, target_tokens) = py
            .detach(|| {
                self.inner.causal_loss_gradients(
                    &ids,
                    shape,
                    labels.as_deref(),
                    objective.label_smoothing,
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("objective_loss", loss)?;
        out.set_item("label_smoothing", objective.label_smoothing)?;
        out.set_item("target_tokens", target_tokens)?;
        out.set_item("gradient_shape", gradient.shape())?;
        out.set_item("gradient_f32le", bytes(py, gradient.data())?)?;
        Ok(out.into_any().unbind())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(windows, window_shape, *, learning_rate, labels=None, label_smoothing=None, weight_decay=None))]
    fn step_windows(
        &mut self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        window_shape: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        weight_decay: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let step = adamw_step(learning_rate, weight_decay)?;
        let objective = loss_objective(label_smoothing, None)?;
        let shape = shape(window_shape)?;
        let count = self
            .inner
            .loss_input_count(shape, labels.is_some())
            .map_err(error)?;
        let ids = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        let report = py.detach(|| -> PyResult<serde_json::Value> {
            let report = self
                .inner
                .step_windows(
                    &ids,
                    shape,
                    labels.as_deref(),
                    objective.label_smoothing,
                    step,
                )
                .map_err(error)?;
            serde_json::to_value(report).map_err(error)
        })?;
        crate::json::json_to_py(py, &report)
    }

    #[pyo3(signature=(documents, *, labels=None, max_rows=None, label_smoothing=None))]
    fn document_loss_gradients(
        &self,
        py: Python<'_>,
        documents: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        max_rows: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let cap = limit(max_rows, 1)?;
        let objective = loss_objective(label_smoothing, None)?;
        let (rows, labels) = causal_documents(&self.inner, documents, labels, cap)?;
        let (loss, gradient, target_tokens) = py
            .detach(|| {
                let qs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let ys = labels
                    .as_ref()
                    .map(|rows| rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
                self.inner.document_loss_gradients(
                    &qs,
                    ys.as_deref(),
                    cap,
                    objective.label_smoothing,
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("objective_loss", loss)?;
        out.set_item("label_smoothing", objective.label_smoothing)?;
        out.set_item("target_tokens", target_tokens)?;
        out.set_item("gradient_shape", gradient.shape())?;
        out.set_item("gradient_f32le", bytes(py, gradient.data())?)?;
        Ok(out.into_any().unbind())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(documents, *, learning_rate, labels=None, max_rows=None, label_smoothing=None, weight_decay=None))]
    fn step_documents(
        &mut self,
        py: Python<'_>,
        documents: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        max_rows: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        weight_decay: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let step = adamw_step(learning_rate, weight_decay)?;
        let cap = limit(max_rows, 1)?;
        let objective = loss_objective(label_smoothing, None)?;
        let (rows, labels) = causal_documents(&self.inner, documents, labels, cap)?;
        let report = py.detach(|| -> PyResult<serde_json::Value> {
            let qs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let ys = labels
                .as_ref()
                .map(|rows| rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
            let report = self
                .inner
                .step_documents(&qs, ys.as_deref(), cap, objective.label_smoothing, step)
                .map_err(error)?;
            serde_json::to_value(report).map_err(error)
        })?;
        crate::json::json_to_py(py, &report)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(prompts, gold, *, learning_rate, max_rows=None, weight_decay=None, label_smoothing=None))]
    fn step_answers(
        &mut self,
        py: Python<'_>,
        prompts: &Bound<'_, PyAny>,
        gold: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        max_rows: Option<&Bound<'_, PyAny>>,
        weight_decay: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let step = adamw_step(learning_rate, weight_decay)?;
        let objective = loss_objective(label_smoothing, None)?;
        let smooth = label_smoothing.is_some();
        let cap = limit(max_rows, 1)?;
        let (rows, labels) = inputs(&self.inner, prompts, gold, cap)?;
        let report = py.detach(|| -> PyResult<serde_json::Value> {
            let qs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
            if smooth {
                let v = self
                    .inner
                    .step_answer_sequences_with_label_smoothing(
                        &qs,
                        &labels,
                        cap,
                        objective.label_smoothing,
                        step,
                    )
                    .map_err(error)?;
                serde_json::to_value(v).map_err(error)
            } else {
                let v = self
                    .inner
                    .step_answer_sequences(&qs, &labels, cap, step)
                    .map_err(error)?;
                serde_json::to_value(v).map_err(error)
            }
        })?;
        crate::json::json_to_py(py, &report)
    }

    fn save_checkpoint_new(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<String> {
        py.detach(|| self.inner.save_checkpoint_new(path))
            .map_err(error)
    }

    fn checkpoint_with_id<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let (raw, pin) = py.detach(|| self.inner.checkpoint()).map_err(error)?;
        Ok((PyBytes::new(py, &raw), pin))
    }

    #[staticmethod]
    fn from_bytes(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        payload: &Bound<'_, PyBytes>,
        expected_id: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let pin = session_input::pin(expected_id)?;
        let raw = payload.as_bytes();
        if raw.len() > Self::MAX_CHECKPOINT_BYTES {
            return Err(error("QKV checkpoint exceeds byte budget"));
        }
        py.detach(|| NativeCausalLmQkvAdamW::restore(&model.inner, raw, &pin))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    fn last_logits_f32le<'py>(
        &self,
        py: Python<'py>,
        prompt: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let count = self
            .inner
            .last_logits_input_count(prompt.len()?)
            .map_err(error)?;
        let ids = tokens(prompt, count)?;
        let logits = py.detach(|| self.inner.last_logits(&ids)).map_err(error)?;
        bytes(py, &logits)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyQkv>()?;
    module.add_class::<PyProjectionPrefix>()
}
