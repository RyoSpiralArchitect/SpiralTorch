//! No scoring math here: bounded transport to the shared native evaluator.
use super::{dimension, documents::documents, error, limit, PyNativeCausalLm};
use pyo3::prelude::*;
use st_nn::causal_lm::{CausalLmEvaluation, CausalLmEvaluationConfig};

#[pyclass(name = "NativeCausalLmEvaluation", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmEvaluation {
    inner: CausalLmEvaluation,
}

#[pymethods]
impl PyNativeCausalLmEvaluation {
    #[getter]
    fn nll(&self) -> f64 {
        self.inner.nll
    }
    #[getter]
    fn perplexity(&self) -> Option<f64> {
        self.inner.perplexity
    }
    #[getter]
    fn target_tokens(&self) -> u64 {
        self.inner.target_tokens
    }
    #[getter]
    fn ignored_targets(&self) -> u64 {
        self.inner.ignored_targets
    }
    #[getter]
    fn window_count(&self) -> usize {
        self.inner.window_count
    }
    #[getter]
    fn skipped_windows(&self) -> usize {
        self.inner.skipped_windows
    }
    #[getter]
    fn microbatches(&self) -> usize {
        self.inner.microbatches
    }
    #[getter]
    fn data_id(&self) -> &str {
        &self.inner.data_id
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}

pub(super) fn evaluate(
    model: &PyNativeCausalLm,
    py: Python<'_>,
    token_documents: &Bound<'_, PyAny>,
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
    labels: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNativeCausalLmEvaluation> {
    let config = CausalLmEvaluationConfig {
        context_length: u32::try_from(dimension(context_length)?).map_err(error)?,
        microbatch_windows: limit(microbatch_windows, 1)?,
    };
    config.validate().map_err(error)?;
    let (docs, labels) = documents(token_documents, labels, |lengths, has_labels| {
        model
            .inner
            .evaluation_plan(lengths, config, has_labels)
            .map(|_| ())
            .map_err(error)
    })?;
    py.detach(|| {
        model
            .inner
            .evaluate_documents(&docs, labels.as_deref(), config)
    })
    .map(|inner| PyNativeCausalLmEvaluation { inner })
    .map_err(error)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLmEvaluation>()
}
