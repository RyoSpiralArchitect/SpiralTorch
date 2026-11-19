//! Transport for Rust-owned conditional schedules and checkpoint publication.
use super::*;
use pyo3::types::{PyString, PyTuple};
use st_nn::causal_lm::{
    CausalLmConditionalAnswerBatch, NativeCausalLmConditionalAnswerSession,
    NativeCausalLmConditionalServingSnapshot,
};
use std::path::PathBuf;

#[pyclass(
    name = "NativeCausalLmConditionalAnswerSession",
    module = "spiraltorch"
)]
pub(super) struct PyConditionalSession(NativeCausalLmConditionalAnswerSession);

#[pyclass(
    name = "NativeCausalLmConditionalServingSnapshot",
    module = "spiraltorch",
    frozen
)]
struct PyConditionalServing(NativeCausalLmConditionalServingSnapshot);

#[pymethods]
impl PyConditionalServing {
    #[getter]
    fn projection_id(&self) -> &str {
        self.0.projection_id()
    }
    #[getter]
    fn checkpoint_id(&self) -> &str {
        self.0.checkpoint_id()
    }
    #[getter]
    fn position(&self) -> u64 {
        self.0.position()
    }
    #[getter]
    fn total_batches(&self) -> u64 {
        self.0.total_batches()
    }

    #[pyo3(signature=(prompt, *, max_cache_values=None))]
    fn prefill(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        max_cache_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<qkv::PyProjectionPrefix> {
        let cap = limit(
            max_cache_values,
            NativeCausalLmContext::DEFAULT_MAX_CACHE_VALUES,
        )?;
        let n = self
            .0
            .prefill_input_count(prompt.len()?, cap)
            .map_err(error)?;
        let ids = tokens(prompt, n)?;
        py.detach(|| self.0.prefill(ids, cap))
            .map(|inner| qkv::PyProjectionPrefix { inner })
            .map_err(error)
    }
}

fn batches(
    input: &Bound<'_, PyAny>,
    count: usize,
) -> PyResult<Vec<CausalLmConditionalAnswerBatch>> {
    (0..count)
        .map(|i| {
            let batch = input.get_item(i)?;
            if batch.len()? != 2 {
                return Err(error(
                    "expected (indices, answer_pairs) per conditional session batch",
                ));
            }
            let indices = batch.get_item(0)?;
            let n = roster_size(&indices, MAX_OUTPUT_CONTRAST_ROWS)?;
            let labels = pairs(&batch.get_item(1)?, n)?;
            Ok((tokens(&indices, n)?, labels))
        })
        .collect()
}

type ScheduleInputs = (Vec<Vec<usize>>, Vec<CausalLmConditionalAnswerBatch>);

fn read_schedule(
    projection: &NativeCausalLmRankOneProjection,
    pool: &Bound<'_, PyAny>,
    schedule: &Bound<'_, PyAny>,
    policy: CausalLmConditionalAnswerConfig,
    cap: usize,
    epochs: u32,
) -> PyResult<ScheduleInputs> {
    let n = roster_size(
        schedule,
        NativeCausalLmConditionalAnswerSession::MAX_BATCHES,
    )?;
    NativeCausalLmConditionalAnswerSession::scheduled_positions(n, epochs).map_err(error)?;
    let schedule = batches(schedule, n)?;
    let count = roster_size(pool, MAX_CONDITIONAL_BANK_ROWS)?;
    let rows = read_prompts(pool, count, |steps| {
        NativeCausalLmConditionalAnswerSession::admit_schedule_inputs(
            projection, steps, &schedule, policy, cap, epochs,
        )
    })?;
    Ok((rows, schedule))
}

#[pymethods]
impl PyConditionalSession {
    #[classattr]
    const MAX_BATCHES: usize = NativeCausalLmConditionalAnswerSession::MAX_BATCHES;
    #[classattr]
    const MAX_TOTAL_BATCHES: u64 = NativeCausalLmConditionalAnswerSession::MAX_TOTAL_BATCHES;
    #[classattr]
    const MAX_CHECKPOINT_BYTES: usize =
        NativeCausalLmConditionalAnswerSession::MAX_CHECKPOINT_BYTES;
    #[classattr]
    const MAX_CHECKPOINT_INTERVAL: u64 =
        NativeCausalLmConditionalAnswerSession::MAX_CHECKPOINT_INTERVAL;

    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(projection, retention, pool, schedule, *, length, max_trials=None, mass_inward_cosine=None, max_prefix_values=None, epochs=None))]
    fn new(
        py: Python<'_>,
        projection: &PyRankOne,
        retention: &PyConditionalRetention,
        pool: &Bound<'_, PyAny>,
        schedule: &Bound<'_, PyAny>,
        length: &Bound<'_, PyAny>,
        max_trials: Option<&Bound<'_, PyAny>>,
        mass_inward_cosine: Option<&Bound<'_, PyAny>>,
        max_prefix_values: Option<&Bound<'_, PyAny>>,
        epochs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let policy = config(length, max_trials, mass_inward_cosine)?;
        let cap = limit(max_prefix_values, 1 << 20)?;
        retention
            .0
            .verify_checkpoint_owner(&projection.0)
            .map_err(error)?;
        let epochs = u32::try_from(limit(epochs, 1)?).map_err(error)?;
        let (rows, schedule) = read_schedule(&projection.0, pool, schedule, policy, cap, epochs)?;
        py.detach(|| {
            NativeCausalLmConditionalAnswerSession::new_repeated(
                &projection.0,
                &retention.0,
                &rows,
                schedule,
                policy,
                cap,
                epochs,
            )
        })
        .map(Self)
        .map_err(error)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(pool, schedule, *, length, max_trials=None, mass_inward_cosine=None, max_prefix_values=None, epochs=None))]
    fn next_phase(
        &self,
        py: Python<'_>,
        pool: &Bound<'_, PyAny>,
        schedule: &Bound<'_, PyAny>,
        length: &Bound<'_, PyAny>,
        max_trials: Option<&Bound<'_, PyAny>>,
        mass_inward_cosine: Option<&Bound<'_, PyAny>>,
        max_prefix_values: Option<&Bound<'_, PyAny>>,
        epochs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !self.0.is_exhausted() {
            return Err(error(
                "conditional session must be exhausted before starting a new phase",
            ));
        }
        let policy = config(length, max_trials, mass_inward_cosine)?;
        let cap = limit(max_prefix_values, 1 << 20)?;
        let epochs = u32::try_from(limit(epochs, 1)?).map_err(error)?;
        let (rows, schedule) =
            read_schedule(self.0.projection(), pool, schedule, policy, cap, epochs)?;
        py.detach(|| self.0.next_phase(&rows, schedule, policy, cap, epochs))
            .map(Self)
            .map_err(error)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_bytes(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        payload: &Bound<'_, PyBytes>,
        expected_id: &Bound<'_, PyAny>,
        pool: &Bound<'_, PyAny>,
        schedule: &Bound<'_, PyAny>,
        guard_rows: &Bound<'_, PyAny>,
        guard_pairs: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let pin = session_input::pin(expected_id)?;
        let raw = payload.as_bytes();
        let resume = py
            .detach(|| {
                NativeCausalLmConditionalAnswerSession::prepare_resume(&model.inner, raw, &pin)
            })
            .map_err(error)?;
        let (n, batch_count) = resume.schedule_shape();
        let guard_count = resume.guard_plan().roster_len();
        if pool.len()? != n
            || schedule.len()? != batch_count
            || guard_rows.len()? != guard_count
            || guard_pairs.len()? != guard_count
        {
            return Err(error("conditional session resume input shape changed"));
        }
        let pairs = pairs(guard_pairs, guard_count)?;
        resume
            .guard_plan()
            .admit_roster_pairs(&pairs)
            .map_err(error)?;
        let schedule = batches(schedule, batch_count)?;
        let pool = read_prompts(pool, n, |lengths| {
            resume.pool_input_values(lengths, &schedule)
        })?;
        let rows = read_prompts(guard_rows, guard_count, |lengths| {
            resume.guard_plan().roster_input_count(lengths)
        })?;
        py.detach(|| resume.resume(&pool, schedule, &rows, &pairs))
            .map(Self)
            .map_err(error)
    }

    #[staticmethod]
    fn serving_from_checkpoint(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        payload: &Bound<'_, PyBytes>,
        expected_id: &Bound<'_, PyAny>,
    ) -> PyResult<PyConditionalServing> {
        let pin = session_input::pin(expected_id)?;
        let bytes = payload.as_bytes();
        py.detach(|| {
            NativeCausalLmConditionalAnswerSession::serving_from_checkpoint(
                &model.inner,
                bytes,
                &pin,
            )
        })
        .map(PyConditionalServing)
        .map_err(error)
    }

    #[getter]
    fn position(&self) -> u64 {
        self.0.position()
    }
    #[getter]
    fn accepted_updates(&self) -> u64 {
        self.0.accepted_updates()
    }
    #[getter]
    fn total_batches(&self) -> u64 {
        self.0.total_batches()
    }
    #[getter]
    fn epochs(&self) -> u32 {
        self.0.epochs()
    }
    #[getter]
    fn batches_per_epoch(&self) -> usize {
        self.0.batches_per_epoch()
    }
    #[getter]
    fn plan_id(&self) -> &str {
        self.0.plan_id()
    }
    #[getter]
    fn predecessor_checkpoint_id(&self) -> Option<&str> {
        self.0.predecessor_checkpoint_id()
    }
    #[getter]
    fn exhausted(&self) -> bool {
        self.0.is_exhausted()
    }
    fn projection_snapshot(&self) -> PyRankOne {
        PyRankOne(self.0.projection().clone())
    }
    fn step(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut next = self.0.clone();
        let report = py.detach(|| next.step()).map_err(error)?;
        let result = as_py(py, report)?;
        self.0 = next;
        Ok(result)
    }
    fn checkpoint_with_id<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let (raw, pin) = py.detach(|| self.0.checkpoint()).map_err(error)?;
        Ok((PyBytes::new(py, &raw), pin))
    }
    fn save_checkpoint_new(&self, py: Python<'_>, path: PathBuf) -> PyResult<String> {
        py.detach(|| self.0.save_checkpoint_new(path))
            .map_err(error)
    }
    fn advance_to_checkpoint_new<'py>(
        &mut self,
        py: Python<'py>,
        stop: &Bound<'_, PyAny>,
        path: PathBuf,
    ) -> PyResult<Bound<'py, PyTuple>> {
        reject_bool(stop)?;
        let stop = stop.extract::<u64>()?;
        let staged = py
            .detach(|| self.0.prepare_checkpoint_interval(stop, path))
            .map_err(error)?;
        let reports = as_py(py, staged.reports())?.into_bound(py);
        let pin = PyString::new(py, staged.checkpoint_id()).into_any();
        let result = PyTuple::new(py, [reports, pin])?;
        py.detach(|| staged.publish()).map_err(error)?;
        Ok(result)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyConditionalServing>()?;
    module.add_class::<PyConditionalSession>()
}
