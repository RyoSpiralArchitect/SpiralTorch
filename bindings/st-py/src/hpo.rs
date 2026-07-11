#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyAny, PyDict, PyList, PySequence};
use pyo3::IntoPy;
use spiral_hpo::{
    self as hpo, ExperimentTracker, NoOpTracker, Objective, ParamSpec, ParamValue, ResourceConfig,
    SearchError, SearchLoop, SearchLoopState, SearchSpace, Strategy, TrialRecord,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

const MAX_TRACKER_EVENTS_PER_DISPATCH: usize = 1_024;

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            mutex.clear_poison();
            poisoned.into_inner()
        }
    }
}

fn search_error_to_py(err: SearchError) -> PyErr {
    match err {
        SearchError::NoAvailableSlot => PyRuntimeError::new_err("no available resource slots"),
        SearchError::UnknownTrial(id) => PyValueError::new_err(format!("unknown trial id {id}")),
        SearchError::EmptySpace => {
            PyValueError::new_err("search space must have at least one parameter")
        }
        error @ (SearchError::InvalidParameter { .. }
        | SearchError::DuplicateParameter(_)
        | SearchError::InvalidResource(_)
        | SearchError::InvalidStrategy(_)
        | SearchError::NonFiniteMetric { .. }
        | SearchError::InvalidCheckpoint(_)) => PyValueError::new_err(error.to_string()),
        error @ SearchError::TrialIdExhausted => PyRuntimeError::new_err(error.to_string()),
    }
}

fn required_item<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    err: &str,
) -> PyResult<T> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(err.to_string()))?
        .extract()
}

fn parse_param_spec(any: &Bound<'_, PyAny>) -> PyResult<ParamSpec> {
    let dict = any
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("parameter spec must be a mapping"))?;
    let name: String = required_item(dict, "name", "parameter spec missing 'name'")?;
    let kind: String = required_item(dict, "type", "parameter spec missing 'type'")?;
    match kind.to_ascii_lowercase().as_str() {
        "float" => {
            let low: f64 = required_item(dict, "low", "float spec missing 'low'")?;
            let high: f64 = required_item(dict, "high", "float spec missing 'high'")?;
            Ok(ParamSpec::Float { name, low, high })
        }
        "int" => {
            let low: i64 = required_item(dict, "low", "int spec missing 'low'")?;
            let high: i64 = required_item(dict, "high", "int spec missing 'high'")?;
            Ok(ParamSpec::Int { name, low, high })
        }
        "categorical" => {
            let choices: Vec<String> =
                required_item(dict, "choices", "categorical spec missing 'choices'")?;
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

fn parse_space(specs: &Bound<'_, PyAny>) -> PyResult<SearchSpace> {
    if let Ok(dict) = specs.downcast::<PyDict>() {
        // allow mapping -> {name: {...}}
        let mut params = Vec::with_capacity(dict.len());
        for (_, value) in dict.iter() {
            params.push(parse_param_spec(&value)?);
        }
        return Ok(SearchSpace::new(params));
    }
    let seq = specs
        .downcast::<PySequence>()
        .map_err(|_| PyValueError::new_err("search space must be a sequence or mapping"))?;
    let mut params = Vec::new();
    for item in seq.try_iter()? {
        let item = item?;
        params.push(parse_param_spec(&item)?);
    }
    Ok(SearchSpace::new(params))
}

fn parse_resource_config(resource: Option<&Bound<'_, PyDict>>) -> PyResult<ResourceConfig> {
    if let Some(resource) = resource {
        let max_concurrent = resource
            .get_item("max_concurrent")?
            .map(|item| item.extract())
            .transpose()?;
        let min_interval_ms = resource
            .get_item("min_interval_ms")?
            .map(|item| item.extract())
            .transpose()?;
        Ok(ResourceConfig {
            max_concurrent: max_concurrent.unwrap_or(1),
            min_interval: min_interval_ms,
        })
    } else {
        Ok(ResourceConfig::default())
    }
}

fn parse_strategy(config: &Bound<'_, PyDict>) -> PyResult<Strategy> {
    let name: String = config
        .get_item("name")?
        .ok_or_else(|| PyValueError::new_err("strategy requires 'name'"))?
        .extract()?;
    let seed: u64 = config
        .get_item("seed")?
        .map(|value| value.extract())
        .transpose()?
        .unwrap_or(0);
    match name.to_ascii_lowercase().as_str() {
        "bayesian" => {
            let exploration: f64 = config
                .get_item("exploration")?
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or(0.25);
            Ok(Strategy::Bayesian(hpo::strategies::BayesianStrategy::new(
                seed,
                exploration,
            )))
        }
        "population" | "population_based" => {
            let population_size: usize = config
                .get_item("population_size")?
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or(16);
            let elite_fraction: f64 = config
                .get_item("elite_fraction")?
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or(0.25);
            let mutation_rate: f64 = config
                .get_item("mutation_rate")?
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or(0.3);
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

fn trial_to_dict(py: Python<'_>, record: &TrialRecord) -> PyResult<PyObject> {
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
    Ok(dict.into_py(py))
}

fn dict_to_state(checkpoint: &str) -> PyResult<SearchLoopState> {
    serde_json::from_str(checkpoint)
        .map_err(|err| PyValueError::new_err(format!("invalid checkpoint: {err}")))
}

fn state_to_json(state: &SearchLoopState) -> PyResult<String> {
    serde_json::to_string_pretty(state)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to serialize checkpoint: {err}")))
}

#[derive(Clone)]
enum TrackerEvent {
    TrialStart(TrialRecord),
    TrialEnd { trial: TrialRecord, metric: f64 },
    Checkpoint(SearchLoopState),
}

#[derive(Default)]
struct TrackerQueue {
    events: VecDeque<TrackerEvent>,
    dispatching: bool,
}

struct DeferredTracker {
    queue: Arc<Mutex<TrackerQueue>>,
}

impl DeferredTracker {
    fn push(&self, event: TrackerEvent) {
        lock_recover(&self.queue).events.push_back(event);
    }
}

impl ExperimentTracker for DeferredTracker {
    fn on_trial_start(&mut self, trial: &TrialRecord) {
        self.push(TrackerEvent::TrialStart(trial.clone()));
    }

    fn on_trial_end(&mut self, trial: &TrialRecord, metric: f64) {
        self.push(TrackerEvent::TrialEnd {
            trial: trial.clone(),
            metric,
        });
    }

    fn on_checkpoint(&mut self, state: &SearchLoopState) {
        self.push(TrackerEvent::Checkpoint(state.clone()));
    }
}

struct DispatchReset<'a> {
    queue: &'a Mutex<TrackerQueue>,
    armed: bool,
}

impl DispatchReset<'_> {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for DispatchReset<'_> {
    fn drop(&mut self) {
        if self.armed {
            lock_recover(self.queue).dispatching = false;
        }
    }
}

struct PythonTrackerDispatcher {
    callback: Py<PyAny>,
    queue: Arc<Mutex<TrackerQueue>>,
}

impl PythonTrackerDispatcher {
    fn callback_method(&self, py: Python<'_>, method: &str) -> Option<Py<PyAny>> {
        match self.callback.getattr(py, method) {
            Ok(callback) => Some(callback),
            Err(error) => {
                if !error.is_instance_of::<PyAttributeError>(py) {
                    error.write_unraisable(py, Some(self.callback.bind(py)));
                }
                None
            }
        }
    }

    fn report_error(&self, py: Python<'_>, error: PyErr) {
        error.write_unraisable(py, Some(self.callback.bind(py)));
    }

    fn dispatch_trial(
        &self,
        py: Python<'_>,
        method: &str,
        trial: &TrialRecord,
        metric: Option<f64>,
    ) {
        let Some(callback) = self.callback_method(py, method) else {
            return;
        };
        let trial_dict = match trial_to_dict(py, trial) {
            Ok(trial_dict) => trial_dict,
            Err(error) => {
                self.report_error(py, error);
                return;
            }
        };
        let result = match metric {
            Some(metric) => callback.call1(py, (trial_dict, metric)),
            None => callback.call1(py, (trial_dict,)),
        };
        if let Err(error) = result {
            self.report_error(py, error);
        }
    }

    fn dispatch_event(&self, py: Python<'_>, event: TrackerEvent) {
        match event {
            TrackerEvent::TrialStart(trial) => {
                self.dispatch_trial(py, "on_trial_start", &trial, None);
            }
            TrackerEvent::TrialEnd { trial, metric } => {
                self.dispatch_trial(py, "on_trial_end", &trial, Some(metric));
            }
            TrackerEvent::Checkpoint(state) => {
                let Some(callback) = self.callback_method(py, "on_checkpoint") else {
                    return;
                };
                let checkpoint = match state_to_json(&state) {
                    Ok(checkpoint) => checkpoint,
                    Err(error) => {
                        self.report_error(py, error);
                        return;
                    }
                };
                if let Err(error) = callback.call1(py, (checkpoint,)) {
                    self.report_error(py, error);
                }
            }
        }
    }

    fn dispatch(&self, py: Python<'_>) {
        {
            let mut queue = lock_recover(&self.queue);
            if queue.dispatching {
                return;
            }
            queue.dispatching = true;
        }
        let mut reset = DispatchReset {
            queue: &self.queue,
            armed: true,
        };

        for _ in 0..MAX_TRACKER_EVENTS_PER_DISPATCH {
            let event = {
                let mut queue = lock_recover(&self.queue);
                match queue.events.pop_front() {
                    Some(event) => event,
                    None => {
                        queue.dispatching = false;
                        reset.disarm();
                        return;
                    }
                }
            };
            self.dispatch_event(py, event);
        }

        let dropped_events = {
            let mut queue = lock_recover(&self.queue);
            let dropped_events = queue.events.len();
            queue.events.clear();
            queue.dispatching = false;
            reset.disarm();
            dropped_events
        };
        if dropped_events > 0 {
            self.report_error(
                py,
                PyRuntimeError::new_err(format!(
                    "tracker callback event budget exceeded; dropped {dropped_events} queued events"
                )),
            );
        }
    }
}

fn tracker_from_py(
    callback: Option<Py<PyAny>>,
) -> (Box<dyn ExperimentTracker>, Option<PythonTrackerDispatcher>) {
    match callback {
        Some(callback) => {
            let queue = Arc::new(Mutex::new(TrackerQueue::default()));
            (
                Box::new(DeferredTracker {
                    queue: Arc::clone(&queue),
                }),
                Some(PythonTrackerDispatcher { callback, queue }),
            )
        }
        None => (Box::new(NoOpTracker), None),
    }
}

#[pyclass(name = "SearchLoop", module = "spiraltorch.hpo")]
pub struct PySearchLoop {
    inner: Mutex<SearchLoop>,
    tracker_dispatcher: Option<PythonTrackerDispatcher>,
}

impl PySearchLoop {
    fn new(inner: SearchLoop, tracker_dispatcher: Option<PythonTrackerDispatcher>) -> Self {
        Self {
            inner: Mutex::new(inner),
            tracker_dispatcher,
        }
    }

    fn dispatch_tracker_events(&self, py: Python<'_>) {
        if let Some(dispatcher) = &self.tracker_dispatcher {
            dispatcher.dispatch(py);
        }
    }
}

#[pymethods]
impl PySearchLoop {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(dispatcher) = &self.tracker_dispatcher {
            visit.call(&dispatcher.callback)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.tracker_dispatcher = None;
    }

    #[staticmethod]
    #[pyo3(signature = (space, strategy, resource=None, tracker=None, maximize=false))]
    pub fn create(
        space: &Bound<'_, PyAny>,
        strategy: &Bound<'_, PyDict>,
        resource: Option<&Bound<'_, PyDict>>,
        tracker: Option<Py<PyAny>>,
        maximize: bool,
    ) -> PyResult<Self> {
        Python::with_gil(|_py| {
            let space = parse_space(space)?;
            let strategy = parse_strategy(strategy)?;
            let resource = parse_resource_config(resource)?;
            let (tracker, tracker_dispatcher) = tracker_from_py(tracker);
            let objective = Objective::from_maximize(maximize);
            let loop_inner = SearchLoop::new(space, strategy, resource, objective, tracker)
                .map_err(search_error_to_py)?;
            Ok(PySearchLoop::new(loop_inner, tracker_dispatcher))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (space, checkpoint, tracker=None))]
    pub fn from_checkpoint(
        space: &Bound<'_, PyAny>,
        checkpoint: &str,
        tracker: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Python::with_gil(|_py| {
            let space = parse_space(space)?;
            let state = dict_to_state(checkpoint)?;
            let (tracker, tracker_dispatcher) = tracker_from_py(tracker);
            let loop_inner =
                SearchLoop::from_state(space, state, tracker).map_err(search_error_to_py)?;
            Ok(PySearchLoop::new(loop_inner, tracker_dispatcher))
        })
    }

    pub fn suggest(&self, py: Python<'_>) -> PyResult<PyObject> {
        let record = {
            let mut guard = lock_recover(&self.inner);
            guard.suggest().map_err(search_error_to_py)?
        };
        self.dispatch_tracker_events(py);
        trial_to_dict(py, &record)
    }

    pub fn observe(&self, py: Python<'_>, trial_id: usize, metric: f64) -> PyResult<()> {
        {
            let mut guard = lock_recover(&self.inner);
            guard
                .observe(trial_id, metric)
                .map_err(search_error_to_py)?;
        }
        self.dispatch_tracker_events(py);
        Ok(())
    }

    pub fn checkpoint(&self, py: Python<'_>) -> PyResult<String> {
        let state = {
            let mut guard = lock_recover(&self.inner);
            guard.checkpoint()
        };
        self.dispatch_tracker_events(py);
        state_to_json(&state)
    }

    pub fn pending(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let pending = lock_recover(&self.inner).pending().to_vec();
        let list = PyList::empty(py);
        for record in &pending {
            let entry = trial_to_dict(py, record)?;
            list.append(entry.bind(py))?;
        }
        Ok(list.into())
    }

    pub fn completed(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let completed = lock_recover(&self.inner).completed().to_vec();
        let list = PyList::empty(py);
        for record in &completed {
            let entry = trial_to_dict(py, record)?;
            list.append(entry.bind(py))?;
        }
        Ok(list.into())
    }

    pub fn objective(&self) -> PyResult<String> {
        let objective = lock_recover(&self.inner).objective();
        Ok(objective.as_str().to_string())
    }

    pub fn best_trial(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let best = lock_recover(&self.inner).best_trial();
        best.map(|record| trial_to_dict(py, &record)).transpose()
    }

    pub fn summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        let summary = lock_recover(&self.inner).summary();
        let dict = PyDict::new(py);
        dict.set_item("objective", summary.objective.as_str())?;
        dict.set_item("total_trials", summary.total_trials)?;
        dict.set_item("completed_trials", summary.completed_trials)?;
        dict.set_item("pending_trials", summary.pending_trials)?;
        match summary.best_trial {
            Some(best) => {
                let best_obj = trial_to_dict(py, &best)?;
                dict.set_item("best_trial", best_obj)?;
            }
            None => {
                dict.set_item("best_trial", py.None())?;
            }
        }
        Ok(dict.into_py(py))
    }
}

pub fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "hpo")?;
    module.add_class::<PySearchLoop>()?;
    module.add("__doc__", "Hyper-parameter search utilities.")?;
    parent.add_submodule(&module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn test_search_loop() -> SearchLoop {
        SearchLoop::new(
            SearchSpace::new(vec![ParamSpec::Int {
                name: "layers".to_string(),
                low: 1,
                high: 4,
            }]),
            Strategy::Random(hpo::strategies::RandomStrategy::new(7)),
            ResourceConfig::default(),
            Objective::Minimize,
            Box::new(NoOpTracker),
        )
        .unwrap()
    }

    #[test]
    fn poisoned_search_loop_mutex_is_recovered() {
        let search = PySearchLoop::new(test_search_loop(), None);
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _guard = lock_recover(&search.inner);
            panic!("poison search loop");
        }));
        assert!(panic.is_err());

        assert!(lock_recover(&search.inner).pending().is_empty());
    }

    #[test]
    fn deferred_tracker_recovers_a_poisoned_event_queue() {
        let queue = Arc::new(Mutex::new(TrackerQueue::default()));
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _guard = lock_recover(&queue);
            panic!("poison event queue");
        }));
        assert!(panic.is_err());

        let mut tracker = DeferredTracker {
            queue: Arc::clone(&queue),
        };
        tracker.on_trial_start(&TrialRecord {
            id: 0,
            suggestion: Default::default(),
            metric: None,
        });

        assert_eq!(lock_recover(&queue).events.len(), 1);
    }
}
