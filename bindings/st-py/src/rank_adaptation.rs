use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use serde::Serialize;
use st_core::runtime::blackcat::bandit::SoftBanditMode;
use st_core::runtime::rank_adaptation::{
    RankAdaptationSelection, RankAdaptationSession, RANK_ADAPTATION_CONTRACT_VERSION,
    RANK_ADAPTATION_SEMANTIC_OWNER,
};

use crate::json::json_to_py;
use crate::planner::{rank_plan_contract_value, PyRankPlan};

type RankPlanMetadata = (
    Option<&'static str>,
    Option<&'static str>,
    Option<&'static str>,
);

fn policy_from_str(policy: &str) -> PyResult<SoftBanditMode> {
    match policy.trim().to_ascii_lowercase().as_str() {
        "ucb" | "upper_confidence_bound" => Ok(SoftBanditMode::UCB),
        "ts" | "thompson_sampling" => Ok(SoftBanditMode::TS),
        _ => Err(PyValueError::new_err(
            "policy must be 'ucb' or 'thompson_sampling'",
        )),
    }
}

fn adaptation_value<T: Serialize>(value: &T) -> PyResult<serde_json::Value> {
    let mut value =
        serde_json::to_value(value).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value
        .as_object_mut()
        .expect("rank-adaptation artifact serializes as an object")
        .insert("execution_client".into(), "python".into());
    Ok(value)
}

fn value_to_py(py: Python<'_>, value: serde_json::Value) -> PyResult<Py<PyAny>> {
    json_to_py(py, &value)
}

#[pyclass(module = "spiraltorch", name = "RankAdaptationSelection")]
pub(crate) struct PyRankAdaptationSelection {
    inner: RankAdaptationSelection,
    metadata: RankPlanMetadata,
}

#[pymethods]
impl PyRankAdaptationSelection {
    #[getter]
    fn selection_id(&self) -> u64 {
        self.inner.receipt().selection_id
    }

    #[getter]
    fn candidate_index(&self) -> usize {
        self.inner.receipt().candidate_index
    }

    #[getter]
    fn spiralk_source_sha256(&self) -> &str {
        &self.inner.receipt().spiralk_source_sha256
    }

    #[getter]
    fn execution_signature(&self) -> &str {
        &self.inner.receipt().execution_signature
    }

    #[getter]
    fn plan(&self) -> PyRankPlan {
        PyRankPlan::from_plan_with_metadata(
            self.inner.plan().clone(),
            self.metadata.0,
            self.metadata.1,
            self.metadata.2,
        )
    }

    fn receipt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut value = adaptation_value(self.inner.receipt())?;
        value["plan"] =
            rank_plan_contract_value(self.inner.plan(), self.metadata.1, self.metadata.2)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        value_to_py(py, value)
    }
}

#[pyclass(module = "spiraltorch", name = "RankAdaptationSession")]
pub(crate) struct PyRankAdaptationSession {
    inner: RankAdaptationSession,
    metadata: RankPlanMetadata,
}

#[pymethods]
impl PyRankAdaptationSession {
    #[new]
    #[pyo3(signature = (base_plan, scripts, *, policy="ucb", seed=0))]
    fn new(
        base_plan: PyRef<'_, PyRankPlan>,
        scripts: Vec<String>,
        policy: &str,
        seed: u64,
    ) -> PyResult<Self> {
        let policy = policy_from_str(policy)?;
        let inner =
            RankAdaptationSession::try_from_spiralk(base_plan.plan(), &scripts, policy, seed)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner,
            metadata: base_plan.metadata(),
        })
    }

    fn choose(&mut self, py: Python<'_>) -> PyResult<Py<PyRankAdaptationSelection>> {
        let mut next = self.inner.clone();
        let inner = next
            .try_choose()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let selection = Py::new(
            py,
            PyRankAdaptationSelection {
                inner,
                metadata: self.metadata,
            },
        )?;
        self.inner = next;
        Ok(selection)
    }

    fn observe(
        &mut self,
        py: Python<'_>,
        selection_id: u64,
        elapsed_ms: f64,
        correctness_passed: bool,
    ) -> PyResult<Py<PyAny>> {
        let mut next = self.inner.clone();
        let receipt = next
            .try_observe(selection_id, elapsed_ms, correctness_passed)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let value = value_to_py(py, adaptation_value(&receipt)?)?;
        self.inner = next;
        Ok(value)
    }

    fn abandon(&mut self, py: Python<'_>, selection_id: u64) -> PyResult<Py<PyAny>> {
        let mut next = self.inner.clone();
        let receipt = next
            .try_abandon(selection_id)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let value = value_to_py(py, adaptation_value(&receipt)?)?;
        self.inner = next;
        Ok(value)
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut value = adaptation_value(&self.inner.snapshot())?;
        let candidates = value
            .get_mut("candidates")
            .and_then(serde_json::Value::as_array_mut)
            .expect("rank-adaptation snapshot contains candidates");
        let plans = self.inner.candidate_plans().collect::<Vec<_>>();
        if candidates.len() != plans.len() {
            return Err(PyRuntimeError::new_err(
                "rank-adaptation candidate snapshot is inconsistent",
            ));
        }
        for (candidate, plan) in candidates.iter_mut().zip(plans) {
            candidate["plan"] = rank_plan_contract_value(plan, self.metadata.1, self.metadata.2)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
        value_to_py(py, value)
    }

    #[getter]
    fn pending_selection_id(&self) -> Option<u64> {
        self.inner.pending_selection_id()
    }
}

pub(crate) fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRankAdaptationSession>()?;
    module.add_class::<PyRankAdaptationSelection>()?;
    module.add(
        "RANK_ADAPTATION_CONTRACT_VERSION",
        RANK_ADAPTATION_CONTRACT_VERSION,
    )?;
    module.add(
        "RANK_ADAPTATION_SEMANTIC_OWNER",
        RANK_ADAPTATION_SEMANTIC_OWNER,
    )?;
    Ok(())
}
