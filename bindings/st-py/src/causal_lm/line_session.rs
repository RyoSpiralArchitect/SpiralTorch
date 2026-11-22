use super::*;
use st_nn::causal_lm::NativeCausalLmLineTrainingSession as Session;
use std::path::PathBuf;

#[pyclass(name = "NativeCausalLmLineTrainingSession", module = "spiraltorch")]
pub(super) struct PyLineSession {
    inner: Session,
}

#[pymethods]
impl PyLineSession {
    #[new]
    fn new(py: Python<'_>, origin: &PyNativeCausalLm, plan: &Bound<'_, PyAny>) -> PyResult<Self> {
        let plan = session_input::payload_with_limit(plan, Session::MAX_PLAN_BYTES)?;
        let bytes = plan.as_bytes();
        py.detach(|| Session::new(origin.inner.clone(), bytes))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[staticmethod]
    fn from_json(
        py: Python<'_>,
        origin: &PyNativeCausalLm,
        plan: &Bound<'_, PyAny>,
        payload: &Bound<'_, PyAny>,
        expected_id: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let id = session_input::pin(expected_id)?;
        let p = session_input::payload_with_limit(plan, Session::MAX_PLAN_BYTES)?;
        let raw = session_input::payload_with_limit(payload, Session::MAX_CHECKPOINT_BYTES)?;
        let (p, raw) = (p.as_bytes(), raw.as_bytes());
        py.detach(|| Session::restore(origin.inner.clone(), p, raw, &id))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[getter]
    fn model(&self) -> PyNativeCausalLm {
        PyNativeCausalLm {
            inner: self.inner.model().clone(),
        }
    }
    #[getter]
    fn plan_id(&self) -> &str {
        self.inner.plan_id()
    }
    #[getter]
    fn completed_updates(&self) -> u32 {
        self.inner.completed_updates()
    }
    #[getter]
    fn total_updates(&self) -> u32 {
        self.inner.total_updates()
    }
    #[getter]
    fn exhausted(&self) -> bool {
        self.inner.exhausted()
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        py.detach(|| self.inner.step_adaptation())
            .map_err(error)?
            .map(|r| crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?))
            .transpose()
    }

    fn checkpoint_with_id(&self, py: Python<'_>) -> PyResult<(String, String)> {
        py.detach(|| self.inner.checkpoint_with_id()).map_err(error)
    }

    fn save_checkpoint_new(&self, py: Python<'_>, path: PathBuf) -> PyResult<String> {
        py.detach(|| self.inner.save_checkpoint_new(path))
            .map_err(checkpoint_error)
    }

    fn advance_to_checkpoint_new(
        &mut self,
        py: Python<'_>,
        stop_after: &Bound<'_, PyAny>,
        path: PathBuf,
    ) -> PyResult<(Py<PyAny>, String)> {
        let stop = u32::try_from(dimension(stop_after)?).map_err(error)?;
        let (rs, id) = py
            .detach(|| self.inner.advance_adaptation_to_checkpoint_new(stop, path))
            .map_err(checkpoint_error)?;
        Ok((
            crate::json::json_to_py(py, &serde_json::to_value(rs).map_err(error)?)?,
            id,
        ))
    }
}
