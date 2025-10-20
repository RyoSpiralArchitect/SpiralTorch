use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PySequence};
use spiral_hpo::{
    self as hpo, ExperimentTracker, NoOpTracker, ParamSpec, ParamValue, ResourceConfig,
    SearchError, SearchLoop, SearchLoopState, SearchSpace, Strategy, TrialRecord,
};
use std::sync::Mutex;

fn parse_param_spec(any: &PyAny) -> PyResult<ParamSpec> {
    let dict = any
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("parameter spec must be a mapping"))?;
    let name: String = dict
        .get_item("name")
        .ok_or_else(|| PyValueError::new_err("parameter spec missing 'name'"))?
        .extract()?;
    let kind: String = dict
        .get_item("type")
        .ok_or_else(|| PyValueError::new_err("parameter spec missing 'type'"))?
        .extract()?;
    match kind.to_ascii_lowercase().as_str() {
        "float" => {
            let low: f64 = dict
                .get_item("low")
                .ok_or_else(|| PyValueError::new_err("float spec missing 'low'"))?
                .extract()?;
            let high: f64 = dict
                .get_item("high")
                .ok_or_else(|| PyValueError::new_err("float spec missing 'high'"))?
                .extract()?;
            Ok(ParamSpec::Float { name, low, high })
        }
        "int" => {
            let low: i64 = dict
                .get_item("low")
                .ok_or_else(|| PyValueError::new_err("int spec missing 'low'"))?
                .extract()?;
            let high: i64 = dict
                .get_item("high")
                .ok_or_else(|| PyValueError::new_err("int spec missing 'high'"))?
                .extract()?;
            Ok(ParamSpec::Int { name, low, high })
        }
        "categorical" => {
            let choices: Vec<String> = dict
                .get_item("choices")
                .ok_or_else(|| PyValueError::new_err("categorical spec missing 'choices'"))?
                .extract()?;
            if choices.is_empty() {
                Err(PyValueError::new_err("categorical choices cannot be empty"))
            } else {
                Ok(ParamSpec::Categorical { name, choices })
            }
        }
        other => Err(PyValueError::new_err(format!(
            "unknown parameter type '{other}'"
        ))),
    }
}

fn parse_space(specs: &PyAny) -> PyResult<SearchSpace> {
    if let Ok(dict) = specs.downcast::<PyDict>() {
        // allow mapping -> {name: {...}}
        let mut params = Vec::with_capacity(dict.len());
        for (_, value) in dict.iter() {
            params.push(parse_param_spec(value)?);
        }
        return Ok(SearchSpace::new(params));
    }
    let seq = specs
        .downcast::<PySequence>()
        .map_err(|_| PyValueError::new_err("search space must be a sequence or mapping"))?;
    let mut params = Vec::new();
    for item in seq.iter()? {
        params.push(parse_param_spec(&item?)?);
    }
    Ok(SearchSpace::new(params))
}

fn parse_resource_config(resource: Option<&PyDict>) -> PyResult<ResourceConfig> {
    if let Some(resource) = resource {
        let max_concurrent = resource
            .get_item("max_concurrent")
            .map(|v| v.extract())
            .transpose()?;
        let min_interval_ms = resource
            .get_item("min_interval_ms")
            .map(|v| v.extract())
            .transpose()?;
        Ok(ResourceConfig {
            max_concurrent: max_concurrent.unwrap_or(1),
            min_interval: min_interval_ms,
        })
    } else {
        Ok(ResourceConfig::default())
    }
}

fn parse_strategy(config: &PyDict) -> PyResult<Strategy> {
    let name: String = config
        .get_item("name")
        .ok_or_else(|| PyValueError::new_err("strategy requires 'name'"))?
        .extract()?;
    let seed: u64 = config
        .get_item("seed")
        .map(|v| v.extract())
        .transpose()
        .unwrap_or(Ok(0))?;
    match name.to_ascii_lowercase().as_str() {
        "bayesian" => {
            let exploration: f64 = config
                .get_item("exploration")
                .map(|v| v.extract())
                .transpose()
                .unwrap_or(Ok(0.25))?;
            Ok(Strategy::Bayesian(hpo::strategies::BayesianStrategy::new(
                seed,
                exploration,
            )))
        }
        "population" | "population_based" => {
            let population_size: usize = config
                .get_item("population_size")
                .map(|v| v.extract())
                .transpose()
                .unwrap_or(Ok(16))?;
            let elite_fraction: f64 = config
                .get_item("elite_fraction")
                .map(|v| v.extract())
                .transpose()
                .unwrap_or(Ok(0.25))?;
            let mutation_rate: f64 = config
                .get_item("mutation_rate")
                .map(|v| v.extract())
                .transpose()
                .unwrap_or(Ok(0.3))?;
            Ok(Strategy::Population(
                hpo::strategies::PopulationStrategy::new(
                    seed,
                    population_size,
                    elite_fraction,
                    mutation_rate,
                ),
            ))
        }
        other => Err(PyValueError::new_err(format!("unknown strategy '{other}'"))),
    }
}

fn trial_to_dict<'py>(py: Python<'py>, record: &TrialRecord) -> PyResult<&'py PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("id", record.id)?;
    let params = PyDict::new(py);
    for (key, value) in &record.suggestion {
        match value {
            ParamValue::Float(v) => {
                params.set_item(key, v)?;
            }
            ParamValue::Int(v) => {
                params.set_item(key, v)?;
            }
            ParamValue::Categorical(v) => {
                params.set_item(key, v)?;
            }
        }
    }
    dict.set_item("params", params)?;
    if let Some(metric) = record.metric {
        dict.set_item("metric", metric)?;
    }
    Ok(dict)
}

fn dict_to_state(checkpoint: &str) -> PyResult<SearchLoopState> {
    serde_json::from_str(checkpoint)
        .map_err(|err| PyValueError::new_err(format!("invalid checkpoint: {err}")))
}

fn state_to_json(state: &SearchLoopState) -> PyResult<String> {
    serde_json::to_string_pretty(state)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to serialize checkpoint: {err}")))
}

struct PythonTracker {
    callback: Option<Py<PyAny>>,
}

impl PythonTracker {
    fn new(callback: Option<Py<PyAny>>) -> Self {
        Self { callback }
    }

    fn with_callback(
        &mut self,
        py: Python<'_>,
        method: &str,
        trial: &TrialRecord,
        metric: Option<f64>,
    ) {
        if let Some(ref cb) = self.callback {
            if let Ok(attr) = cb.getattr(py, method) {
                if let Ok(trial_dict) = trial_to_dict(py, trial) {
                    let _ = match metric {
                        Some(metric) => attr.call1((trial_dict, metric)),
                        None => attr.call1((trial_dict,)),
                    };
                }
            }
        }
    }
}

impl ExperimentTracker for PythonTracker {
    fn on_trial_start(&mut self, trial: &TrialRecord) {
        Python::with_gil(|py| self.with_callback(py, "on_trial_start", trial, None));
    }

    fn on_trial_end(&mut self, trial: &TrialRecord, metric: f64) {
        Python::with_gil(|py| self.with_callback(py, "on_trial_end", trial, Some(metric)));
    }

    fn on_checkpoint(&mut self, state: &SearchLoopState) {
        if let Some(ref cb) = self.callback {
            Python::with_gil(|py| {
                if let Ok(attr) = cb.getattr(py, "on_checkpoint") {
                    if let Ok(json) = state_to_json(state) {
                        let _ = attr.call1((json,));
                    }
                }
            });
        }
    }
}

unsafe impl Send for PythonTracker {}

fn tracker_from_py(callback: Option<Py<PyAny>>) -> Box<dyn ExperimentTracker> {
    if callback.is_some() {
        Box::new(PythonTracker::new(callback))
    } else {
        Box::new(NoOpTracker)
    }
}

#[pyclass(name = "SearchLoop", module = "spiraltorch.hpo")]
pub struct PySearchLoop {
    inner: Mutex<SearchLoop>,
}

impl PySearchLoop {
    fn new(inner: SearchLoop) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }
}

#[pymethods]
impl PySearchLoop {
    #[staticmethod]
    #[pyo3(signature = (space, strategy, resource=None, tracker=None))]
    pub fn create(
        space: &PyAny,
        strategy: &PyDict,
        resource: Option<&PyDict>,
        tracker: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            let space = parse_space(space)?;
            let strategy = parse_strategy(strategy)?;
            let resource = parse_resource_config(resource)?;
            let tracker = tracker_from_py(tracker);
            let loop_inner =
                SearchLoop::new(space, strategy, resource, tracker).map_err(PyErr::from)?;
            Ok(PySearchLoop::new(loop_inner))
        })
    }

    #[staticmethod]
    pub fn from_checkpoint(
        space: &PyAny,
        checkpoint: &str,
        tracker: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            let space = parse_space(space)?;
            let state = dict_to_state(checkpoint)?;
            let tracker = tracker_from_py(tracker);
            Ok(PySearchLoop::new(SearchLoop::from_state(
                space, state, tracker,
            )))
        })
    }

    pub fn suggest(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut guard = self.inner.lock().unwrap();
        let record = guard.suggest().map_err(PyErr::from)?;
        Ok(trial_to_dict(py, &record)?.into())
    }

    pub fn observe(&self, trial_id: usize, metric: f64) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard.observe(trial_id, metric).map_err(PyErr::from)
    }

    pub fn checkpoint(&self) -> PyResult<String> {
        let mut guard = self.inner.lock().unwrap();
        let state = guard.checkpoint();
        state_to_json(&state)
    }

    pub fn pending(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let guard = self.inner.lock().unwrap();
        let list = PyList::empty(py);
        for record in guard.pending() {
            list.append(trial_to_dict(py, record)?)?;
        }
        Ok(list.into())
    }

    pub fn completed(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let guard = self.inner.lock().unwrap();
        let list = PyList::empty(py);
        for record in guard.completed() {
            list.append(trial_to_dict(py, record)?)?;
        }
        Ok(list.into())
    }
}

impl From<SearchError> for PyErr {
    fn from(value: SearchError) -> Self {
        match value {
            SearchError::NoAvailableSlot => PyRuntimeError::new_err("no available resource slots"),
            SearchError::UnknownTrial(id) => {
                PyValueError::new_err(format!("unknown trial id {id}"))
            }
            SearchError::EmptySpace => {
                PyValueError::new_err("search space must have at least one parameter")
            }
        }
    }
}

pub fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "hpo")?;
    module.add_class::<PySearchLoop>()?;
    module.add("__doc__", "Hyper-parameter search utilities.")?;
    parent.add_submodule(&module)?;
    Ok(())
}
