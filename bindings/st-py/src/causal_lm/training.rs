//! No training semantics here: bounded transport into the Rust corpus trainer.
use super::{
    adamw_step, dimension, documents::documents, error, limit, loss_objective, PyNativeCausalLm,
    PyNativeGeometryAdamW,
};
use pyo3::prelude::*;
use st_nn::causal_lm::{CausalLmTrainingConfig, CausalLmTrainingStep, NativeCausalLmTrainer};
use st_nn::language_windows::MAX_DOCUMENTS;

#[pyclass(name = "NativeCausalLmTrainer", module = "spiraltorch")]
pub struct PyNativeCausalLmTrainer {
    inner: NativeCausalLmTrainer,
}

#[pyclass(name = "NativeCausalLmTrainingStep", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmTrainingStep {
    inner: CausalLmTrainingStep,
}

#[pymethods]
impl PyNativeCausalLmTrainingStep {
    #[getter]
    fn update(&self) -> u64 {
        self.inner.update
    }
    #[getter]
    fn epoch(&self) -> u32 {
        self.inner.epoch
    }
    #[getter]
    fn window_start(&self) -> usize {
        self.inner.window_start
    }
    #[getter]
    fn window_end(&self) -> usize {
        self.inner.window_end
    }
    #[getter]
    fn microbatches(&self) -> u64 {
        self.inner.microbatches
    }
    #[getter]
    fn target_tokens(&self) -> u64 {
        self.inner.target_tokens
    }
    #[getter]
    fn loss(&self) -> f64 {
        self.inner.loss
    }
    #[getter]
    fn repetition_loss(&self) -> Option<f64> {
        self.inner.repetition_loss
    }
    #[getter]
    fn repetition_positions(&self) -> Option<u64> {
        self.inner.repetition_positions
    }
    #[getter]
    fn pre_geometry_artifact_id(&self) -> &str {
        &self.inner.pre_geometry_artifact_id
    }
    #[getter]
    fn post_geometry_artifact_id(&self) -> &str {
        &self.inner.post_geometry_artifact_id
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}

#[pymethods]
impl PyNativeCausalLmTrainer {
    #[classattr]
    const MAX_TOKENS: usize = NativeCausalLmTrainer::MAX_TOKENS;
    #[classattr]
    const MAX_DOCUMENTS: usize = MAX_DOCUMENTS;

    #[new]
    #[pyo3(signature=(model, optimizer, token_documents, context_length, *, learning_rate, microbatch_windows=None, accumulation_steps=None, epochs=None, weight_decay=None, label_smoothing=None, labels=None, repetition=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        optimizer: &PyNativeGeometryAdamW,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        microbatch_windows: Option<&Bound<'_, PyAny>>,
        accumulation_steps: Option<&Bound<'_, PyAny>>,
        epochs: Option<&Bound<'_, PyAny>>,
        weight_decay: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
        repetition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let step = adamw_step(learning_rate, weight_decay)?;
        let objective = loss_objective(label_smoothing, repetition)?;
        let config = CausalLmTrainingConfig {
            context_length: u32::try_from(dimension(context_length)?).map_err(error)?,
            microbatch_windows: limit(microbatch_windows, 1)?,
            accumulation_steps: limit(accumulation_steps, 1)?,
            epochs: u32::try_from(limit(epochs, 1)?).map_err(error)?,
            learning_rate: step.rate(),
            weight_decay: step.weight_decay(),
            label_smoothing: objective.label_smoothing,
            repetition: objective.repetition,
        };
        NativeCausalLmTrainer::validate_owners(&model.inner, &optimizer.inner, config)
            .map_err(error)?;
        let (docs, labels) = documents(token_documents, labels, |lengths, has_labels| {
            NativeCausalLmTrainer::document_plan(&model.inner, config, lengths, has_labels)
                .map(|_| ())
                .map_err(error)
        })?;
        py.detach(|| {
            NativeCausalLmTrainer::new(
                model.inner.clone(),
                optimizer.inner.clone(),
                docs,
                labels,
                config,
            )
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature=(base, token_documents, payload, expected_id, *, labels=None))]
    fn from_json(
        py: Python<'_>,
        base: &PyNativeCausalLm,
        token_documents: &Bound<'_, PyAny>,
        payload: &str,
        expected_id: &str,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let resume = py
            .detach(|| {
                NativeCausalLmTrainer::prepare_restore(
                    base.inner.clone(),
                    payload.as_bytes(),
                    expected_id,
                )
            })
            .map_err(error)?;
        let (docs, labels) = documents(token_documents, labels, |lengths, has_labels| {
            NativeCausalLmTrainer::document_plan(
                resume.model(),
                resume.config(),
                lengths,
                has_labels,
            )
            .map(|_| ())
            .map_err(error)
        })?;
        py.detach(|| resume.resume(docs, labels))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Option<PyNativeCausalLmTrainingStep>> {
        py.detach(|| self.inner.step())
            .map(|v| v.map(|inner| PyNativeCausalLmTrainingStep { inner }))
            .map_err(error)
    }
    #[getter]
    fn model(&self) -> PyNativeCausalLm {
        PyNativeCausalLm {
            inner: self.inner.model().clone(),
        }
    }
    #[getter]
    fn optimizer(&self) -> PyNativeGeometryAdamW {
        PyNativeGeometryAdamW {
            inner: self.inner.optimizer().clone(),
        }
    }
    #[getter]
    fn data_id(&self) -> &str {
        self.inner.data_id()
    }
    #[getter]
    fn completed_updates(&self) -> u64 {
        self.inner.completed_updates()
    }
    #[getter]
    fn total_updates(&self) -> u64 {
        self.inner.total_updates()
    }
    #[getter]
    fn epoch(&self) -> u32 {
        self.inner.epoch()
    }
    #[getter]
    fn next_window(&self) -> usize {
        self.inner.next_window()
    }
    #[getter]
    fn exhausted(&self) -> bool {
        self.inner.exhausted()
    }
    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.config()).map_err(error)
    }
    fn plan_json(&self) -> PyResult<String> {
        self.inner.plan().to_json().map_err(error)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json()).map_err(error)
    }
    #[getter]
    fn checkpoint_id(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.checkpoint_id()).map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLmTrainer>()?;
    module.add_class::<PyNativeCausalLmTrainingStep>()
}
