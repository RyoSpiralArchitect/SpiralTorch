//! No scoring math here: bounded transport to the shared native evaluator.
use super::{dimension, documents::documents, error, limit, PyNativeCausalLm};
use pyo3::prelude::*;
use st_nn::causal_lm::{
    CausalLmEvaluation, CausalLmEvaluationComparison, CausalLmEvaluationConfig, NativeCausalLm,
    MAX_EVALUATION_VARIANTS,
};

#[pyclass(name = "NativeCausalLmEvaluation", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmEvaluation {
    inner: CausalLmEvaluation,
}

#[pyclass(
    name = "NativeCausalLmEvaluationComparison",
    module = "spiraltorch",
    frozen
)]
pub struct PyNativeCausalLmEvaluationComparison {
    inner: CausalLmEvaluationComparison,
}

#[pymethods]
impl PyNativeCausalLmEvaluationComparison {
    #[getter]
    fn shared_prefix_layers(&self) -> usize {
        self.inner.shared_prefix_layers
    }
    #[getter]
    fn evaluations(&self) -> Vec<PyNativeCausalLmEvaluation> {
        self.inner
            .evaluations
            .iter()
            .cloned()
            .map(|inner| PyNativeCausalLmEvaluation { inner })
            .collect()
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}

#[pymethods]
impl PyNativeCausalLmEvaluation {
    #[getter]
    fn lora_artifact_id(&self) -> Option<&str> {
        self.inner.lora_artifact_id.as_deref()
    }

    #[getter]
    fn readout_accumulator(&self) -> &'static str {
        self.inner.readout_accumulator.as_str()
    }

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

fn policy(
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
) -> PyResult<CausalLmEvaluationConfig> {
    let config = CausalLmEvaluationConfig {
        context_length: u32::try_from(dimension(context_length)?).map_err(error)?,
        microbatch_windows: limit(microbatch_windows, 1)?,
    };
    config.validate().map_err(error)?;
    Ok(config)
}

pub(super) fn evaluate(
    model: &PyNativeCausalLm,
    py: Python<'_>,
    token_documents: &Bound<'_, PyAny>,
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
    labels: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNativeCausalLmEvaluation> {
    let config = policy(context_length, microbatch_windows)?;
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

pub(super) fn compare(
    py: Python<'_>,
    variants: &Bound<'_, PyAny>,
    token_documents: &Bound<'_, PyAny>,
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
    labels: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNativeCausalLmEvaluationComparison> {
    let config = policy(context_length, microbatch_windows)?;
    let count = variants.len()?;
    if !(2..=MAX_EVALUATION_VARIANTS).contains(&count) {
        return Err(error("comparison requires 2 to 16 variants"));
    }
    let owners = (0..count)
        .map(|i| {
            variants
                .get_item(i)?
                .cast_into::<PyNativeCausalLm>()
                .map_err(Into::into)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let models: Vec<_> = owners.iter().map(|obj| &obj.get().inner).collect();
    NativeCausalLm::comparison_prefix_layers(&models).map_err(error)?;
    let (docs, labels) = documents(token_documents, labels, |lengths, has_labels| {
        NativeCausalLm::comparison_plan(&models, lengths, config, has_labels)
            .map(|_| ())
            .map_err(error)
    })?;
    py.detach(|| NativeCausalLm::evaluate_variants(&models, &docs, labels.as_deref(), config))
        .map(|inner| PyNativeCausalLmEvaluationComparison { inner })
        .map_err(error)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLmEvaluation>()?;
    module.add_class::<PyNativeCausalLmEvaluationComparison>()
}
