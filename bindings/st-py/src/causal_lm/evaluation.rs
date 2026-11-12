//! No scoring math here: bounded transport to the shared native evaluator.
use super::{dimension, documents::documents, error, limit, PyNativeCausalLm};
use pyo3::prelude::*;
use st_nn::causal_lm::{
    CausalLmAnswerComparison, CausalLmEvaluation, CausalLmEvaluationComparison,
    CausalLmEvaluationConfig, CausalLmQkvAnswerComparison, CausalLmQkvComparison,
    CausalLmQkvEvaluation, CausalLmResult, NativeCausalLm, NativeCausalLmQkvAdamW,
    MAX_ANSWER_SCORE_CELLS, MAX_EVALUATION_VARIANTS,
};
use st_nn::language_windows::CausalDocumentWindows;

#[pyclass(name = "NativeCausalLmEvaluation", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmEvaluation {
    inner: CausalLmEvaluation,
}

#[pyclass(name = "NativeCausalLmQkvEvaluation", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmQkvEvaluation {
    inner: CausalLmQkvEvaluation,
}

#[pyclass(name = "NativeCausalLmQkvComparison", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmQkvComparison {
    inner: CausalLmQkvComparison,
}

#[pyclass(
    name = "NativeCausalLmQkvAnswerComparison",
    module = "spiraltorch",
    frozen
)]
pub struct PyNativeCausalLmQkvAnswerComparison {
    inner: CausalLmQkvAnswerComparison,
}

#[pymethods]
impl PyNativeCausalLmQkvAnswerComparison {
    #[classattr]
    const MAX_SCORE_CELLS: usize = MAX_ANSWER_SCORE_CELLS;
    fn __len__(&self) -> usize {
        self.inner.comparison.answers().len()
    }
    #[getter]
    fn serving_id(&self) -> &str {
        &self.inner.serving_id
    }
    #[getter]
    fn layer(&self) -> usize {
        self.inner.layer
    }
    #[getter]
    fn steps(&self) -> u32 {
        self.inner.steps
    }
    #[getter]
    fn document_count(&self) -> usize {
        self.inner.comparison.document_count()
    }
    #[getter]
    fn scored_documents(&self) -> usize {
        self.inner.comparison.scored_documents()
    }
    #[getter]
    fn shared_prefix_layers(&self) -> usize {
        self.inner.comparison.comparison().shared_prefix_layers
    }
    #[getter]
    fn evaluations(&self) -> Vec<PyNativeCausalLmEvaluation> {
        wrap_scores(self.inner.comparison.comparison())
    }
    fn answer(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        answer_at(&self.inner.comparison, py, index)
    }
    fn contrast(
        &self,
        py: Python<'_>,
        reference: &Bound<'_, PyAny>,
        candidate: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        answer_contrast(&self.inner.comparison, py, reference, candidate)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| serde_json::to_string(&self.inner))
            .map_err(error)
    }
}

#[pymethods]
impl PyNativeCausalLmQkvComparison {
    #[getter]
    fn serving_id(&self) -> &str {
        &self.inner.serving_id
    }
    #[getter]
    fn layer(&self) -> usize {
        self.inner.layer
    }
    #[getter]
    fn steps(&self) -> u32 {
        self.inner.steps
    }
    #[getter]
    fn shared_prefix_layers(&self) -> usize {
        self.inner.comparison.shared_prefix_layers
    }
    #[getter]
    fn evaluations(&self) -> Vec<PyNativeCausalLmEvaluation> {
        wrap_scores(&self.inner.comparison)
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}

#[pymethods]
impl PyNativeCausalLmQkvEvaluation {
    #[getter]
    fn serving_id(&self) -> &str {
        &self.inner.serving_id
    }
    #[getter]
    fn layer(&self) -> usize {
        self.inner.layer
    }
    #[getter]
    fn steps(&self) -> u32 {
        self.inner.steps
    }
    #[getter]
    fn evaluation(&self) -> PyNativeCausalLmEvaluation {
        PyNativeCausalLmEvaluation {
            inner: self.inner.evaluation.clone(),
        }
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}

#[pyclass(
    name = "NativeCausalLmEvaluationComparison",
    module = "spiraltorch",
    frozen
)]
pub struct PyNativeCausalLmEvaluationComparison {
    inner: CausalLmEvaluationComparison,
}

#[pyclass(
    name = "NativeCausalLmAnswerComparison",
    module = "spiraltorch",
    frozen
)]
pub struct PyNativeCausalLmAnswerComparison {
    inner: CausalLmAnswerComparison,
}

#[pymethods]
impl PyNativeCausalLmAnswerComparison {
    #[classattr]
    const MAX_SCORE_CELLS: usize = MAX_ANSWER_SCORE_CELLS;
    #[classattr]
    const MAX_VARIANTS: usize = MAX_EVALUATION_VARIANTS;
    fn __len__(&self) -> usize {
        self.inner.answers().len()
    }
    #[getter]
    fn document_count(&self) -> usize {
        self.inner.document_count()
    }
    #[getter]
    fn scored_documents(&self) -> usize {
        self.inner.scored_documents()
    }
    #[getter]
    fn shared_prefix_layers(&self) -> usize {
        self.inner.comparison().shared_prefix_layers
    }
    #[getter]
    fn evaluations(&self) -> Vec<PyNativeCausalLmEvaluation> {
        wrap_scores(self.inner.comparison())
    }
    fn answer(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        answer_at(&self.inner, py, index)
    }
    fn contrast(
        &self,
        py: Python<'_>,
        reference: &Bound<'_, PyAny>,
        candidate: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        answer_contrast(&self.inner, py, reference, candidate)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| serde_json::to_string(&self.inner))
            .map_err(error)
    }
}

fn answer_at(
    comparison: &CausalLmAnswerComparison,
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let answer = comparison
        .answers()
        .get(dimension(index)?)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("answer index out of range"))?;
    crate::json::json_to_py(py, &serde_json::to_value(answer).map_err(error)?)
}

fn answer_contrast(
    comparison: &CausalLmAnswerComparison,
    py: Python<'_>,
    reference: &Bound<'_, PyAny>,
    candidate: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (a, b) = (dimension(reference)?, dimension(candidate)?);
    let result = py.detach(|| comparison.contrast(a, b)).map_err(error)?;
    crate::json::json_to_py(py, &serde_json::to_value(result).map_err(error)?)
}

fn wrap_scores(comparison: &CausalLmEvaluationComparison) -> Vec<PyNativeCausalLmEvaluation> {
    comparison
        .evaluations
        .iter()
        .cloned()
        .map(|inner| PyNativeCausalLmEvaluation { inner })
        .collect()
}

#[pymethods]
impl PyNativeCausalLmEvaluationComparison {
    #[getter]
    fn shared_prefix_layers(&self) -> usize {
        self.inner.shared_prefix_layers
    }
    #[getter]
    fn evaluations(&self) -> Vec<PyNativeCausalLmEvaluation> {
        wrap_scores(&self.inner)
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
    let limit = NativeCausalLm::evaluation_token_limit(1).map_err(error)?;
    let (docs, labels) = documents(token_documents, labels, limit, |lengths, has_labels| {
        py.detach(|| {
            model
                .inner
                .evaluation_plan(lengths, config, has_labels)
                .map(|_| ())
        })
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

pub(super) fn evaluate_qkv(
    model: &NativeCausalLmQkvAdamW,
    py: Python<'_>,
    token_documents: &Bound<'_, PyAny>,
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
    labels: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNativeCausalLmQkvEvaluation> {
    let config = policy(context_length, microbatch_windows)?;
    let limit = NativeCausalLm::evaluation_token_limit(1).map_err(error)?;
    let (docs, labels) = documents(token_documents, labels, limit, |lengths, has_labels| {
        py.detach(|| {
            model
                .evaluation_plan(lengths, config, has_labels)
                .map(|_| ())
        })
        .map_err(error)
    })?;
    py.detach(|| model.evaluate_documents(&docs, labels.as_deref(), config))
        .map(|inner| PyNativeCausalLmQkvEvaluation { inner })
        .map_err(error)
}

pub(super) fn compare_qkv(
    model: &NativeCausalLmQkvAdamW,
    source: &PyNativeCausalLm,
    py: Python<'_>,
    token_documents: &Bound<'_, PyAny>,
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
    labels: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNativeCausalLmQkvComparison> {
    let config = policy(context_length, microbatch_windows)?;
    let limit = NativeCausalLm::evaluation_token_limit(2).map_err(error)?;
    let (docs, labels) = documents(token_documents, labels, limit, |lengths, has_labels| {
        py.detach(|| {
            model
                .source_comparison_plan(&source.inner, lengths, config, has_labels)
                .map(|_| ())
        })
        .map_err(error)
    })?;
    py.detach(|| model.evaluate_against_source(&source.inner, &docs, labels.as_deref(), config))
        .map(|inner| PyNativeCausalLmQkvComparison { inner })
        .map_err(error)
}

pub(super) fn compare_qkv_answers(
    model: &NativeCausalLmQkvAdamW,
    source: &PyNativeCausalLm,
    py: Python<'_>,
    token_documents: &Bound<'_, PyAny>,
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
    labels: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNativeCausalLmQkvAnswerComparison> {
    let config = policy(context_length, microbatch_windows)?;
    let limit = NativeCausalLm::evaluation_token_limit(2).map_err(error)?;
    let (docs, labels) = documents(token_documents, labels, limit, |lengths, has_labels| {
        py.detach(|| {
            model
                .source_answer_comparison_plan(&source.inner, lengths, config, has_labels)
                .map(|_| ())
        })
        .map_err(error)
    })?;
    py.detach(|| {
        model.evaluate_against_source_by_answer(&source.inner, &docs, labels.as_deref(), config)
    })
    .map(|inner| PyNativeCausalLmQkvAnswerComparison { inner })
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
    compare_with(
        py,
        variants,
        token_documents,
        labels,
        config,
        NativeCausalLm::comparison_plan,
        NativeCausalLm::evaluate_variants,
    )
    .map(|inner| PyNativeCausalLmEvaluationComparison { inner })
}

pub(super) fn compare_answers(
    py: Python<'_>,
    variants: &Bound<'_, PyAny>,
    token_documents: &Bound<'_, PyAny>,
    context_length: &Bound<'_, PyAny>,
    microbatch_windows: Option<&Bound<'_, PyAny>>,
    labels: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNativeCausalLmAnswerComparison> {
    let config = policy(context_length, microbatch_windows)?;
    compare_with(
        py,
        variants,
        token_documents,
        labels,
        config,
        NativeCausalLm::answer_comparison_plan,
        NativeCausalLm::evaluate_variants_by_answer,
    )
    .map(|inner| PyNativeCausalLmAnswerComparison { inner })
}

type Plan = fn(
    &[&NativeCausalLm],
    &[usize],
    CausalLmEvaluationConfig,
    bool,
) -> CausalLmResult<CausalDocumentWindows>;
type Score<T> = fn(
    &[&NativeCausalLm],
    &[Vec<usize>],
    Option<&[Vec<i64>]>,
    CausalLmEvaluationConfig,
) -> CausalLmResult<T>;

fn compare_with<T: Send>(
    py: Python<'_>,
    variants: &Bound<'_, PyAny>,
    token_documents: &Bound<'_, PyAny>,
    labels: Option<&Bound<'_, PyAny>>,
    config: CausalLmEvaluationConfig,
    plan: Plan,
    score: Score<T>,
) -> PyResult<T> {
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
    let limit = NativeCausalLm::evaluation_token_limit(count).map_err(error)?;
    let (docs, labels) = documents(token_documents, labels, limit, |lengths, has_labels| {
        py.detach(|| plan(&models, lengths, config, has_labels).map(|_| ()))
            .map_err(error)
    })?;
    py.detach(|| score(&models, &docs, labels.as_deref(), config))
        .map_err(error)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLmEvaluation>()?;
    module.add_class::<PyNativeCausalLmQkvEvaluation>()?;
    module.add_class::<PyNativeCausalLmQkvComparison>()?;
    module.add_class::<PyNativeCausalLmQkvAnswerComparison>()?;
    module.add_class::<PyNativeCausalLmEvaluationComparison>()?;
    module.add_class::<PyNativeCausalLmAnswerComparison>()
}
