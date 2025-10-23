use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use pyo3::Bound;

#[cfg(feature = "spiral_rl")]
use crate::tensor::tensor_err_to_py;
#[cfg(feature = "spiral_rl")]
use pyo3::exceptions::{PyDeprecationWarning, PyValueError};
#[cfg(feature = "spiral_rl")]
use st_spiral_rl::{schedules::EpsilonGreedySchedule, DqnAgent, PpoAgent, SacAgent, SpiralRlError};

#[cfg(feature = "spiral_rl")]
fn rl_err_to_py(err: SpiralRlError) -> PyErr {
    match err {
        SpiralRlError::Tensor(err) => tensor_err_to_py(err),
        SpiralRlError::EmptyEpisode
        | SpiralRlError::InvalidStateShape { .. }
        | SpiralRlError::InvalidAction { .. }
        | SpiralRlError::InvalidDiscount { .. }
        | SpiralRlError::InvalidBatch { .. }
        | SpiralRlError::StateDictShape { .. } => PyValueError::new_err(err.to_string()),
    }
}

#[cfg(feature = "spiral_rl")]
#[derive(Clone, Debug)]
struct ReplayConfigData {
    capacity: usize,
    batch_size: usize,
    prioritized: bool,
    alpha: f32,
    beta0: f32,
}

#[cfg(feature = "spiral_rl")]
#[pyclass(module = "spiraltorch.rl", name = "Replay")]
#[derive(Clone)]
pub(crate) struct PyReplayConfig {
    inner: ReplayConfigData,
}

#[cfg(feature = "spiral_rl")]
#[pymethods]
impl PyReplayConfig {
    #[new]
    #[pyo3(signature = (capacity, batch_size, prioritized=false, alpha=0.6, beta0=0.4))]
    pub fn new(
        capacity: usize,
        batch_size: usize,
        prioritized: bool,
        alpha: f32,
        beta0: f32,
    ) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("replay capacity must be positive"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("replay batch_size must be positive"));
        }
        Ok(Self {
            inner: ReplayConfigData {
                capacity,
                batch_size,
                prioritized,
                alpha,
                beta0,
            },
        })
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    #[getter]
    pub fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    #[getter]
    pub fn prioritized(&self) -> bool {
        self.inner.prioritized
    }

    #[getter]
    pub fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    #[getter]
    pub fn beta0(&self) -> f32 {
        self.inner.beta0
    }
}

#[cfg(feature = "spiral_rl")]
#[derive(Clone, Debug)]
struct AgentConfigData {
    algo: String,
    state_dim: usize,
    action_dim: usize,
    discount: f32,
    learning_rate: f32,
    exploration: Option<EpsilonGreedySchedule>,
    optimizer: String,
    clip_grad: Option<f32>,
    replay: Option<ReplayConfigData>,
    target_sync: Option<usize>,
    n_step: Option<usize>,
    seed: Option<u64>,
}

#[cfg(feature = "spiral_rl")]
#[pyclass(module = "spiraltorch.rl", name = "EpsilonGreedy")]
#[derive(Clone)]
pub(crate) struct PyEpsilonGreedy {
    inner: EpsilonGreedySchedule,
}

#[cfg(feature = "spiral_rl")]
impl PyEpsilonGreedy {
    fn from_schedule(schedule: EpsilonGreedySchedule) -> Self {
        Self { inner: schedule }
    }

    fn schedule(&self) -> &EpsilonGreedySchedule {
        &self.inner
    }
}

#[cfg(feature = "spiral_rl")]
#[pymethods]
impl PyEpsilonGreedy {
    #[new]
    #[pyo3(signature = (start, end, decay_steps))]
    pub fn new(start: f32, end: f32, decay_steps: u32) -> Self {
        Self {
            inner: EpsilonGreedySchedule::new(start, end, decay_steps),
        }
    }

    #[getter]
    pub fn start(&self) -> f32 {
        let (start, _, _) = self.inner.parameters();
        start
    }

    #[getter]
    pub fn end(&self) -> f32 {
        let (_, end, _) = self.inner.parameters();
        end
    }

    #[getter]
    pub fn decay_steps(&self) -> u32 {
        let (_, _, steps) = self.inner.parameters();
        steps
    }

    #[getter]
    pub fn step(&self) -> u32 {
        self.inner.step()
    }

    pub fn value(&self) -> f32 {
        self.inner.value()
    }

    pub fn advance(&mut self) -> f32 {
        self.inner.advance()
    }
}

#[cfg(feature = "spiral_rl")]
#[pyclass(module = "spiraltorch.rl", name = "AgentConfig")]
#[derive(Clone)]
pub(crate) struct PyAgentConfig {
    inner: AgentConfigData,
}

#[cfg(feature = "spiral_rl")]
impl PyAgentConfig {
    fn as_data(&self) -> &AgentConfigData {
        &self.inner
    }

    fn into_data(self) -> AgentConfigData {
        self.inner
    }
}

#[cfg(feature = "spiral_rl")]
fn dqn_state_dict(py: Python<'_>, agent: &DqnAgent) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("state_dim", agent.state_dim())?;
    dict.set_item("action_dim", agent.action_dim())?;
    dict.set_item("discount", agent.discount())?;
    dict.set_item("learning_rate", agent.learning_rate())?;
    dict.set_item("epsilon", agent.epsilon())?;
    dict.set_item("table", agent.table().to_vec())?;
    if let Some(schedule) = agent.epsilon_schedule() {
        let schedule_dict = PyDict::new_bound(py);
        let (start, end, steps) = schedule.parameters();
        schedule_dict.set_item("start", start)?;
        schedule_dict.set_item("end", end)?;
        schedule_dict.set_item("steps", steps)?;
        schedule_dict.set_item("step", schedule.step())?;
        dict.set_item("epsilon_schedule", schedule_dict)?;
    } else {
        dict.set_item("epsilon_schedule", py.None())?;
    }
    Ok(dict.into_any().into_py(py))
}

#[cfg(feature = "spiral_rl")]
fn load_dqn_state_dict(agent: &mut DqnAgent, state: &Bound<'_, PyAny>) -> PyResult<()> {
    let dict = state.downcast::<PyDict>()?;

    if let Ok(Some(epsilon)) = dict.get_item("epsilon") {
        let value: f32 = epsilon.extract()?;
        agent.set_epsilon(value);
    }

    if let Ok(Some(table)) = dict.get_item("table") {
        let values: Vec<f32> = table.extract()?;
        agent.set_table(&values).map_err(rl_err_to_py)?;
    }

    if let Ok(Some(schedule_obj)) = dict.get_item("epsilon_schedule") {
        if schedule_obj.is_none() {
            agent.set_epsilon(agent.epsilon());
        } else {
            let schedule_dict = schedule_obj.downcast::<PyDict>()?;
            let start: f32 = schedule_dict
                .get_item("start")?
                .ok_or_else(|| PyValueError::new_err("epsilon_schedule requires 'start'"))?
                .extract()?;
            let end: f32 = schedule_dict
                .get_item("end")?
                .ok_or_else(|| PyValueError::new_err("epsilon_schedule requires 'end'"))?
                .extract()?;
            let steps: u32 = schedule_dict
                .get_item("steps")?
                .ok_or_else(|| PyValueError::new_err("epsilon_schedule requires 'steps'"))?
                .extract()?;
            let progress: u32 = schedule_dict
                .get_item("step")?
                .ok_or_else(|| PyValueError::new_err("epsilon_schedule requires 'step'"))?
                .extract()?;
            let mut schedule = EpsilonGreedySchedule::new(start, end, steps);
            schedule.set_step(progress);
            agent.configure_epsilon_schedule(schedule);
        }
    }

    Ok(())
}

#[cfg(feature = "spiral_rl")]
#[pymethods]
impl PyAgentConfig {
    #[new]
    #[pyo3(signature = (algo, state_dim, action_dim, gamma, lr, exploration=None, optimizer="adam", clip_grad=None, replay=None, target_sync=None, n_step=None, seed=None))]
    pub fn new(
        py: Python<'_>,
        algo: &str,
        state_dim: usize,
        action_dim: usize,
        gamma: f32,
        lr: f32,
        exploration: Option<Py<PyEpsilonGreedy>>,
        optimizer: &str,
        clip_grad: Option<f32>,
        replay: Option<Py<PyReplayConfig>>,
        target_sync: Option<usize>,
        n_step: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&gamma) {
            return Err(PyValueError::new_err("gamma must lie within [0, 1]"));
        }
        if lr <= 0.0 {
            return Err(PyValueError::new_err("learning rate must be positive"));
        }

        let exploration_schedule = exploration
            .as_ref()
            .map(|handle| handle.borrow(py).schedule().clone());
        let replay_cfg = replay
            .as_ref()
            .map(|handle| handle.borrow(py).inner.clone());

        Ok(Self {
            inner: AgentConfigData {
                algo: algo.to_string(),
                state_dim,
                action_dim,
                discount: gamma,
                learning_rate: lr,
                exploration: exploration_schedule,
                optimizer: optimizer.to_string(),
                clip_grad,
                replay: replay_cfg,
                target_sync,
                n_step,
                seed,
            },
        })
    }

    #[getter]
    pub fn algo(&self) -> &str {
        &self.as_data().algo
    }

    #[getter]
    pub fn state_dim(&self) -> usize {
        self.as_data().state_dim
    }

    #[getter]
    pub fn action_dim(&self) -> usize {
        self.as_data().action_dim
    }

    #[getter]
    pub fn gamma(&self) -> f32 {
        self.as_data().discount
    }

    #[getter]
    pub fn lr(&self) -> f32 {
        self.as_data().learning_rate
    }

    #[getter]
    pub fn optimizer(&self) -> &str {
        &self.as_data().optimizer
    }

    #[getter]
    pub fn clip_grad(&self) -> Option<f32> {
        self.as_data().clip_grad
    }

    #[getter]
    pub fn target_sync(&self) -> Option<usize> {
        self.as_data().target_sync
    }

    #[getter]
    pub fn n_step(&self) -> Option<usize> {
        self.as_data().n_step
    }

    #[getter]
    pub fn seed(&self) -> Option<u64> {
        self.as_data().seed
    }

    #[getter]
    pub fn replay<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyReplayConfig>>> {
        if let Some(cfg) = &self.inner.replay {
            Py::new(py, PyReplayConfig { inner: cfg.clone() }).map(Some)
        } else {
            Ok(None)
        }
    }

    #[getter]
    pub fn exploration<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyEpsilonGreedy>>> {
        if let Some(schedule) = &self.inner.exploration {
            Py::new(py, PyEpsilonGreedy::from_schedule(schedule.clone())).map(Some)
        } else {
            Ok(None)
        }
    }
}
#[cfg(feature = "spiral_rl")]
#[pyclass(module = "spiraltorch.spiral_rl", name = "stAgent")]
pub(crate) struct PyDqnAgent {
    inner: DqnAgent,
}

#[cfg(feature = "spiral_rl")]
#[pyclass(module = "spiraltorch.rl", name = "Agent")]
pub(crate) struct PyAgent {
    config: AgentConfigData,
    dqn: DqnAgent,
}

#[cfg(feature = "spiral_rl")]
#[pymethods]
impl PyDqnAgent {
    #[new]
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        discount: f32,
        learning_rate: f32,
    ) -> PyResult<Self> {
        let inner =
            DqnAgent::new(state_dim, action_dim, discount, learning_rate).map_err(rl_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn select_action(&mut self, state: usize) -> usize {
        self.inner.select_action(state)
    }

    pub fn select_actions(&mut self, states: Vec<usize>) -> Vec<usize> {
        self.inner.select_actions(&states)
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f32, next_state: usize) {
        self.inner.update(state, action, reward, next_state);
    }

    #[pyo3(signature = (states, actions, rewards, next_states, dones=None))]
    pub fn update_batch(
        &mut self,
        states: Vec<usize>,
        actions: Vec<usize>,
        rewards: Vec<f32>,
        next_states: Vec<usize>,
        dones: Option<Vec<bool>>,
    ) -> PyResult<()> {
        let done_slice = dones.as_ref().map(|flags| flags.as_slice());
        self.inner
            .update_batch(&states, &actions, &rewards, &next_states, done_slice)
            .map_err(rl_err_to_py)
    }

    #[getter]
    pub fn epsilon(&self) -> f32 {
        self.inner.epsilon()
    }

    #[pyo3(name = "epsilon")]
    pub fn epsilon_method(&self, py: Python<'_>) -> PyResult<f32> {
        let warning_type = py.get_type_bound::<PyDeprecationWarning>();
        let warning_type_any = warning_type.as_any();
        PyErr::warn_bound(
            py,
            &warning_type_any,
            "DqnAgent.epsilon() is deprecated; access the epsilon property instead.",
            1,
        )?;
        Ok(self.inner.epsilon())
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.inner.set_epsilon(epsilon);
    }

    pub fn set_exploration(&mut self, schedule: &PyEpsilonGreedy) {
        self.inner
            .configure_epsilon_schedule(schedule.schedule().clone());
    }

    pub fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        dqn_state_dict(py, &self.inner)
    }

    pub fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        load_dqn_state_dict(&mut self.inner, state)
    }
}

#[cfg(feature = "spiral_rl")]
#[pymethods]
impl PyAgent {
    #[new]
    pub fn new(config: PyAgentConfig) -> PyResult<Self> {
        let data = config.into_data();
        match data.algo.as_str() {
            "dqn" => {
                let mut agent = DqnAgent::new(
                    data.state_dim,
                    data.action_dim,
                    data.discount,
                    data.learning_rate,
                )
                .map_err(rl_err_to_py)?;
                if let Some(schedule) = data.exploration.clone() {
                    agent.configure_epsilon_schedule(schedule);
                }
                Ok(Self {
                    config: data,
                    dqn: agent,
                })
            }
            other => Err(PyValueError::new_err(format!(
                "unsupported algorithm '{other}' (expected 'dqn')"
            ))),
        }
    }

    #[getter]
    pub fn config<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAgentConfig>> {
        Py::new(
            py,
            PyAgentConfig {
                inner: self.config.clone(),
            },
        )
    }

    #[getter]
    pub fn algo(&self) -> &str {
        &self.config.algo
    }

    pub fn select_action(&mut self, state: usize) -> usize {
        self.dqn.select_action(state)
    }

    pub fn select_actions(&mut self, states: Vec<usize>) -> Vec<usize> {
        self.dqn.select_actions(&states)
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f32, next_state: usize) {
        self.dqn.update(state, action, reward, next_state);
    }

    #[pyo3(signature = (states, actions, rewards, next_states, dones=None))]
    pub fn update_batch(
        &mut self,
        states: Vec<usize>,
        actions: Vec<usize>,
        rewards: Vec<f32>,
        next_states: Vec<usize>,
        dones: Option<Vec<bool>>,
    ) -> PyResult<()> {
        let done_slice = dones.as_ref().map(|flags| flags.as_slice());
        self.dqn
            .update_batch(&states, &actions, &rewards, &next_states, done_slice)
            .map_err(rl_err_to_py)
    }

    #[getter]
    pub fn epsilon(&self) -> f32 {
        self.dqn.epsilon()
    }

    #[pyo3(name = "epsilon")]
    pub fn epsilon_method(&self, py: Python<'_>) -> PyResult<f32> {
        let warning_type = py.get_type_bound::<PyDeprecationWarning>();
        let warning_type_any = warning_type.as_any();
        PyErr::warn_bound(
            py,
            &warning_type_any,
            "Agent.epsilon() is deprecated; access the epsilon property instead.",
            1,
        )?;
        Ok(self.dqn.epsilon())
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.dqn.set_epsilon(epsilon);
    }

    pub fn set_exploration(&mut self, schedule: &PyEpsilonGreedy) {
        self.dqn
            .configure_epsilon_schedule(schedule.schedule().clone());
    }

    pub fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        dqn_state_dict(py, &self.dqn)
    }

    pub fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        load_dqn_state_dict(&mut self.dqn, state)
    }
}

#[cfg(feature = "spiral_rl")]
#[pyclass(module = "spiraltorch.spiral_rl", name = "PpoAgent")]
pub(crate) struct PyPpoAgent {
    inner: PpoAgent,
    state_dim: usize,
}

#[cfg(feature = "spiral_rl")]
#[pymethods]
impl PyPpoAgent {
    #[new]
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f32,
        clip_range: f32,
    ) -> PyResult<Self> {
        let inner = PpoAgent::new(state_dim, action_dim, learning_rate, clip_range)
            .map_err(rl_err_to_py)?;
        Ok(Self { inner, state_dim })
    }

    pub fn score_actions(&self, state: Vec<f32>) -> PyResult<Vec<f32>> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        Ok(self.inner.score_actions(&state))
    }

    pub fn value(&self, state: Vec<f32>) -> PyResult<f32> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        Ok(self.inner.value(&state))
    }

    pub fn update(
        &mut self,
        state: Vec<f32>,
        action: usize,
        advantage: f32,
        old_log_prob: f32,
    ) -> PyResult<()> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        self.inner.update(&state, action, advantage, old_log_prob);
        Ok(())
    }
}

#[cfg(feature = "spiral_rl")]
#[pyclass(module = "spiraltorch.spiral_rl", name = "SacAgent")]
pub(crate) struct PySacAgent {
    inner: SacAgent,
    state_dim: usize,
}

#[cfg(feature = "spiral_rl")]
#[pymethods]
impl PySacAgent {
    #[new]
    pub fn new(state_dim: usize, action_dim: usize, temperature: f32) -> PyResult<Self> {
        let inner = SacAgent::new(state_dim, action_dim, temperature).map_err(rl_err_to_py)?;
        Ok(Self { inner, state_dim })
    }

    pub fn sample_action(&self, state: Vec<f32>) -> PyResult<usize> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        Ok(self.inner.sample_action(&state))
    }

    pub fn jitter(&mut self, entropy_target: f32) {
        self.inner.jitter(entropy_target);
    }
}

#[cfg(feature = "spiral_rl")]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "spiral_rl")?;
    module.add("__doc__", "SpiralTorch reinforcement learning agents")?;

    // 1) register public class names first (these names are what Python code expects)
    module.add_class::<PyAgentConfig>()?;
    module.add_class::<PyReplayConfig>()?;
    module.add_class::<PyEpsilonGreedy>()?;
    module.add_class::<PyAgent>()?;
    module.add_class::<PyDqnAgent>()?;
    module.add_class::<PyPpoAgent>()?;
    module.add_class::<PySacAgent>()?;

    // 2) provide aliases / helpers inside the spiral_rl module
    module.add("Agent", module.getattr("Agent")?)?;
    module.add("AgentConfig", module.getattr("AgentConfig")?)?;
    module.add("EpsilonGreedy", module.getattr("EpsilonGreedy")?)?;
    module.add("Replay", module.getattr("Replay")?)?;
    module.add("DqnAgent", module.getattr("stAgent")?)?;
    module.add(
        "__all__",
        vec![
            "Agent",
            "AgentConfig",
            "EpsilonGreedy",
            "Replay",
            "stAgent",
            "DqnAgent",
            "PpoAgent",
            "SacAgent",
        ],
    )?;

    // 3) attach as a submodule of the parent (spiraltorch.spiral_rl)
    parent.add_submodule(&module)?;

    // 4) mirror as st.rl (so users can do `import spiraltorch as st; st.rl...`)
    let module_obj = module.to_object(py);
    parent.add("rl", module_obj.clone_ref(py))?;

    // 5) mirror convenient top-level names under `spiraltorch` for backward compatibility
    parent.add("Agent", module.getattr("Agent")?)?;
    parent.add("AgentConfig", module.getattr("AgentConfig")?)?;
    parent.add("EpsilonGreedy", module.getattr("EpsilonGreedy")?)?;
    parent.add("Replay", module.getattr("Replay")?)?;
    parent.add("stAgent", module.getattr("stAgent")?)?;
    parent.add("DqnAgent", module.getattr("DqnAgent")?)?;
    parent.add("PpoAgent", module.getattr("PpoAgent")?)?;
    parent.add("SacAgent", module.getattr("SacAgent")?)?;

    // 6) (optional but recommended) register legacy top-level module name in sys.modules
    //    so `import spiral_rl` (old code) will find our module object.
    let sys = PyModule::import_bound(py, "sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("spiral_rl", module_obj.clone_ref(py))?;
    modules.set_item("rl", module_obj)?;

    Ok(())
}

#[cfg(not(feature = "spiral_rl"))]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "spiraltorch.spiral_rl")?;
    module.add("__doc__", "SpiralTorch reinforcement learning agents")?;
    parent.add_submodule(&module)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
