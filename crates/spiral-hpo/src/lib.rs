use rand::distributions::{Distribution, StandardNormal};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

pub mod space {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum ParamSpec {
        Float { name: String, low: f64, high: f64 },
        Int { name: String, low: i64, high: i64 },
        Categorical { name: String, choices: Vec<String> },
    }

    impl ParamSpec {
        pub fn name(&self) -> &str {
            match self {
                ParamSpec::Float { name, .. }
                | ParamSpec::Int { name, .. }
                | ParamSpec::Categorical { name, .. } => name,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type", content = "value")]
    pub enum ParamValue {
        Float(f64),
        Int(i64),
        Categorical(String),
    }

    impl ParamValue {
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                ParamValue::Float(v) => Some(*v),
                ParamValue::Int(v) => Some(*v as f64),
                ParamValue::Categorical(_) => None,
            }
        }
    }

    pub type TrialSuggestion = HashMap<String, ParamValue>;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SearchSpace {
        pub params: Vec<ParamSpec>,
    }

    impl SearchSpace {
        pub fn new(params: Vec<ParamSpec>) -> Self {
            Self { params }
        }

        pub fn len(&self) -> usize {
            self.params.len()
        }

        pub fn is_empty(&self) -> bool {
            self.params.is_empty()
        }

        pub fn sample(&self, rng: &mut StdRng) -> TrialSuggestion {
            let mut out = TrialSuggestion::with_capacity(self.params.len());
            for spec in &self.params {
                let value = match spec {
                    ParamSpec::Float { name, low, high } => {
                        let raw = rng.gen::<f64>();
                        let v = low + (high - low) * raw;
                        (name.clone(), ParamValue::Float(v))
                    }
                    ParamSpec::Int { name, low, high } => {
                        let raw = rng.gen::<f64>();
                        let range = (*high - *low + 1) as f64;
                        let v = *low + (raw * range).floor() as i64;
                        (name.clone(), ParamValue::Int(v.clamp(*low, *high)))
                    }
                    ParamSpec::Categorical { name, choices } => {
                        let raw = rng.gen::<f64>();
                        let idx = ((choices.len() as f64) * raw).floor() as usize;
                        let idx = idx.min(choices.len().saturating_sub(1));
                        (name.clone(), ParamValue::Categorical(choices[idx].clone()))
                    }
                };
                out.insert(value.0, value.1);
            }
            out
        }

        pub fn clamp(&self, suggestion: &mut TrialSuggestion) {
            for spec in &self.params {
                if let Some(value) = suggestion.get_mut(spec.name()) {
                    match (spec, value) {
                        (ParamSpec::Float { low, high, .. }, ParamValue::Float(v)) => {
                            *v = v.clamp(*low, *high);
                        }
                        (ParamSpec::Int { low, high, .. }, ParamValue::Int(v)) => {
                            *v = (*v).clamp(*low, *high);
                        }
                        (
                            ParamSpec::Categorical { choices, .. },
                            ParamValue::Categorical(choice),
                        ) => {
                            if !choices.contains(choice) && !choices.is_empty() {
                                *choice = choices[0].clone();
                            }
                        }
                        (ParamSpec::Float { low, high, .. }, ParamValue::Int(v)) => {
                            let new_val = (*v as f64).clamp(*low, *high);
                            *value = ParamValue::Float(new_val);
                        }
                        (ParamSpec::Int { low, high, .. }, ParamValue::Float(v)) => {
                            let iv = (*v).round().clamp(*low as f64, *high as f64) as i64;
                            *value = ParamValue::Int(iv);
                        }
                        _ => {}
                    }
                }
            }
        }

        pub fn draws_per_suggestion(&self) -> usize {
            self.params.len()
        }
    }
}

pub use space::{ParamSpec, ParamValue, SearchSpace, TrialSuggestion};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub suggestion: TrialSuggestion,
    pub metric: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyState {
    Bayesian(super::strategies::BayesianState),
    Population(super::strategies::PopulationState),
}

pub mod strategies {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BayesianState {
        pub seed: u64,
        pub suggestion_count: u64,
        pub observations: Vec<Observation>,
        pub exploration: f64,
    }

    #[derive(Debug, Clone)]
    pub struct BayesianStrategy {
        pub(crate) state: BayesianState,
    }

    impl BayesianStrategy {
        pub fn new(seed: u64, exploration: f64) -> Self {
            Self {
                state: BayesianState {
                    seed,
                    suggestion_count: 0,
                    observations: Vec::new(),
                    exploration,
                },
            }
        }

        fn rng_for(&self, draws: usize) -> StdRng {
            let mut rng = StdRng::seed_from_u64(self.state.seed);
            let skips = self.state.suggestion_count as usize * draws;
            for _ in 0..skips {
                let _: f64 = rng.gen();
            }
            rng
        }

        pub fn suggest(&mut self, space: &SearchSpace) -> TrialSuggestion {
            let draws = space.draws_per_suggestion();
            let mut rng = self.rng_for(draws);
            // advance RNG state for this call so that the checkpointed state remains deterministic
            let mut suggestion = space.sample(&mut rng);
            self.state.suggestion_count += 1;
            if self.state.observations.is_empty() {
                return suggestion;
            }

            // With probability exploration -> random sample, else exploit best observation
            let explore_sample: f64 = rng.gen();
            if explore_sample < self.state.exploration {
                return suggestion;
            }

            let best = self
                .state
                .observations
                .iter()
                .min_by(|a, b| a.metric.total_cmp(&b.metric))
                .map(|obs| obs.suggestion.clone());
            if let Some(best) = best {
                suggestion = best;
                // apply gaussian jitter per float/int param
                for spec in &space.params {
                    match spec {
                        ParamSpec::Float { name, low, high } => {
                            if let Some(ParamValue::Float(v)) = suggestion.get_mut(name) {
                                let base = *v;
                                let sigma = (*high - *low).abs() * 0.1;
                                let noise: f64 = rng.sample(StandardNormal);
                                *v = (base + noise * sigma).clamp(*low, *high);
                            }
                        }
                        ParamSpec::Int { name, low, high } => {
                            if let Some(ParamValue::Int(v)) = suggestion.get_mut(name) {
                                let span = (*high - *low).max(1);
                                let noise: f64 = rng.sample(StandardNormal);
                                let delta = (noise * (span as f64 * 0.25)).round() as i64;
                                *v = (*v + delta).clamp(*low, *high);
                            }
                        }
                        ParamSpec::Categorical { name, choices } => {
                            if let Some(ParamValue::Categorical(value)) = suggestion.get_mut(name) {
                                if !choices.is_empty() {
                                    let idx = rng.gen_range(0..choices.len());
                                    if rng.gen::<f64>() < 0.25 {
                                        *value = choices[idx].clone();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            space.clamp(&mut suggestion);
            suggestion
        }

        pub fn observe(&mut self, observation: Observation) {
            self.state.observations.push(observation);
        }

        pub fn state(&self) -> BayesianState {
            self.state.clone()
        }

        pub fn restore(state: BayesianState) -> Self {
            Self { state }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PopulationState {
        pub seed: u64,
        pub suggestion_count: u64,
        pub population: Vec<Observation>,
        pub population_size: usize,
        pub elite_fraction: f64,
        pub mutation_rate: f64,
    }

    #[derive(Debug, Clone)]
    pub struct PopulationStrategy {
        pub(crate) state: PopulationState,
    }

    impl PopulationStrategy {
        pub fn new(
            seed: u64,
            population_size: usize,
            elite_fraction: f64,
            mutation_rate: f64,
        ) -> Self {
            Self {
                state: PopulationState {
                    seed,
                    suggestion_count: 0,
                    population: Vec::new(),
                    population_size: population_size.max(2),
                    elite_fraction: elite_fraction.clamp(0.05, 0.5),
                    mutation_rate: mutation_rate.clamp(0.01, 1.0),
                },
            }
        }

        fn rng_for(&self, draws: usize) -> StdRng {
            let mut rng = StdRng::seed_from_u64(self.state.seed);
            let skips = self.state.suggestion_count as usize * draws;
            for _ in 0..skips {
                let _: f64 = rng.gen();
            }
            rng
        }

        pub fn suggest(&mut self, space: &SearchSpace) -> TrialSuggestion {
            let draws = space.draws_per_suggestion();
            let mut rng = self.rng_for(draws);
            self.state.suggestion_count += 1;

            if self.state.population.len() < self.state.population_size {
                return space.sample(&mut rng);
            }

            // Select elites
            let elite_count =
                (self.state.population_size as f64 * self.state.elite_fraction).ceil() as usize;
            let elite_count = elite_count.max(1).min(self.state.population.len());
            let mut sorted = self.state.population.clone();
            sorted.sort_by(|a, b| a.metric.total_cmp(&b.metric));
            let elite = &sorted[..elite_count];

            let parent_idx = rng.gen_range(0..elite.len());
            let mut suggestion = elite[parent_idx].suggestion.clone();

            for spec in &space.params {
                if rng.gen::<f64>() > self.state.mutation_rate {
                    continue;
                }
                match spec {
                    ParamSpec::Float { name, low, high } => {
                        if let Some(ParamValue::Float(v)) = suggestion.get_mut(name) {
                            let sigma = (*high - *low).abs() * 0.2;
                            let noise: f64 = rng.sample(StandardNormal);
                            *v = (*v + noise * sigma).clamp(*low, *high);
                        }
                    }
                    ParamSpec::Int { name, low, high } => {
                        if let Some(ParamValue::Int(v)) = suggestion.get_mut(name) {
                            let step = ((*high - *low).max(1) as f64 * 0.2).ceil() as i64;
                            let delta = rng.gen_range(-step..=step);
                            *v = (*v + delta).clamp(*low, *high);
                        }
                    }
                    ParamSpec::Categorical { name, choices } => {
                        if let Some(ParamValue::Categorical(value)) = suggestion.get_mut(name) {
                            if !choices.is_empty() {
                                let idx = rng.gen_range(0..choices.len());
                                *value = choices[idx].clone();
                            }
                        }
                    }
                }
            }

            space.clamp(&mut suggestion);
            suggestion
        }

        pub fn observe(&mut self, observation: Observation) {
            self.state.population.push(observation);
            if self.state.population.len() > self.state.population_size {
                self.state
                    .population
                    .sort_by(|a, b| a.metric.total_cmp(&b.metric));
                self.state.population.truncate(self.state.population_size);
            }
        }

        pub fn state(&self) -> PopulationState {
            self.state.clone()
        }

        pub fn restore(state: PopulationState) -> Self {
            Self { state }
        }
    }
}

use strategies::{BayesianState, BayesianStrategy, PopulationState, PopulationStrategy};

#[derive(Debug, Clone)]
pub enum Strategy {
    Bayesian(BayesianStrategy),
    Population(PopulationStrategy),
}

impl Strategy {
    pub fn name(&self) -> &'static str {
        match self {
            Strategy::Bayesian(_) => "bayesian",
            Strategy::Population(_) => "population",
        }
    }

    pub fn suggest(&mut self, space: &SearchSpace) -> TrialSuggestion {
        match self {
            Strategy::Bayesian(strategy) => strategy.suggest(space),
            Strategy::Population(strategy) => strategy.suggest(space),
        }
    }

    pub fn observe(&mut self, observation: Observation) {
        match self {
            Strategy::Bayesian(strategy) => strategy.observe(observation),
            Strategy::Population(strategy) => strategy.observe(observation),
        }
    }

    pub fn state(&self) -> StrategyState {
        match self {
            Strategy::Bayesian(strategy) => StrategyState::Bayesian(strategy.state()),
            Strategy::Population(strategy) => StrategyState::Population(strategy.state()),
        }
    }

    pub fn restore(state: StrategyState) -> Strategy {
        match state {
            StrategyState::Bayesian(state) => Strategy::Bayesian(BayesianStrategy::restore(state)),
            StrategyState::Population(state) => {
                Strategy::Population(PopulationStrategy::restore(state))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub max_concurrent: usize,
    pub min_interval: Option<u64>,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 1,
            min_interval: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    pub active_slots: usize,
    pub last_scheduled_ms: Option<u128>,
}

#[derive(Debug, Clone)]
pub struct ResourceScheduler {
    config: ResourceConfig,
    state: SchedulerState,
}

impl ResourceScheduler {
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            config,
            state: SchedulerState {
                active_slots: 0,
                last_scheduled_ms: None,
            },
        }
    }

    pub fn from_state(config: ResourceConfig, state: SchedulerState) -> Self {
        Self { config, state }
    }

    pub fn try_reserve(&mut self) -> bool {
        if self.state.active_slots >= self.config.max_concurrent {
            return false;
        }
        if let Some(interval) = self.config.min_interval {
            if let Some(last) = self.state.last_scheduled_ms {
                let now = now_ms();
                if now < last + interval as u128 {
                    return false;
                }
            }
        }
        self.state.active_slots += 1;
        self.state.last_scheduled_ms = Some(now_ms());
        true
    }

    pub fn release(&mut self) {
        if self.state.active_slots > 0 {
            self.state.active_slots -= 1;
        }
    }

    pub fn state(&self) -> SchedulerState {
        self.state.clone()
    }
}

fn now_ms() -> u128 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialRecord {
    pub id: usize,
    pub suggestion: TrialSuggestion,
    pub metric: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchLoopState {
    pub strategy: StrategyState,
    pub completed: Vec<TrialRecord>,
    pub pending: Vec<TrialRecord>,
    pub scheduler: SchedulerState,
    pub next_trial_id: usize,
    pub draws_per_suggestion: usize,
    pub resource: ResourceConfig,
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("all resource slots are busy")]
    NoAvailableSlot,
    #[error("trial id {0} not found")]
    UnknownTrial(usize),
    #[error("search space must define at least one parameter")]
    EmptySpace,
}

pub trait ExperimentTracker: Send {
    fn on_trial_start(&mut self, _trial: &TrialRecord) {}
    fn on_trial_end(&mut self, _trial: &TrialRecord, _metric: f64) {}
    fn on_checkpoint(&mut self, _state: &SearchLoopState) {}
}

pub struct NoOpTracker;
impl ExperimentTracker for NoOpTracker {}

pub struct SearchLoop {
    space: SearchSpace,
    strategy: Strategy,
    scheduler: ResourceScheduler,
    state: SearchLoopState,
    tracker: Box<dyn ExperimentTracker>,
}

impl SearchLoop {
    pub fn new(
        space: SearchSpace,
        strategy: Strategy,
        resource: ResourceConfig,
        tracker: Box<dyn ExperimentTracker>,
    ) -> Result<Self, SearchError> {
        if space.is_empty() {
            return Err(SearchError::EmptySpace);
        }
        let draws = space.draws_per_suggestion();
        Ok(Self {
            scheduler: ResourceScheduler::new(resource.clone()),
            state: SearchLoopState {
                strategy: strategy.state(),
                completed: Vec::new(),
                pending: Vec::new(),
                scheduler: SchedulerState {
                    active_slots: 0,
                    last_scheduled_ms: None,
                },
                next_trial_id: 0,
                draws_per_suggestion: draws,
                resource,
            },
            space,
            strategy,
            tracker,
        })
    }

    pub fn from_state(
        space: SearchSpace,
        state: SearchLoopState,
        tracker: Box<dyn ExperimentTracker>,
    ) -> Self {
        let scheduler =
            ResourceScheduler::from_state(state.resource.clone(), state.scheduler.clone());
        let strategy = Strategy::restore(state.strategy.clone());
        Self {
            strategy,
            scheduler,
            space,
            tracker,
            state,
        }
    }

    pub fn suggest(&mut self) -> Result<TrialRecord, SearchError> {
        if !self.scheduler.try_reserve() {
            return Err(SearchError::NoAvailableSlot);
        }
        let suggestion = self.strategy.suggest(&self.space);
        let trial = TrialRecord {
            id: self.state.next_trial_id,
            suggestion,
            metric: None,
        };
        self.state.next_trial_id += 1;
        self.state.pending.push(trial.clone());
        self.tracker.on_trial_start(&trial);
        self.snapshot_strategy();
        Ok(trial)
    }

    pub fn observe(&mut self, trial_id: usize, metric: f64) -> Result<(), SearchError> {
        if let Some(idx) = self
            .state
            .pending
            .iter()
            .position(|trial| trial.id == trial_id)
        {
            let mut trial = self.state.pending.remove(idx);
            self.scheduler.release();
            trial.metric = Some(metric);
            self.strategy.observe(Observation {
                suggestion: trial.suggestion.clone(),
                metric,
            });
            self.state.completed.push(trial.clone());
            self.tracker.on_trial_end(&trial, metric);
            self.snapshot_strategy();
            Ok(())
        } else {
            Err(SearchError::UnknownTrial(trial_id))
        }
    }

    pub fn checkpoint(&mut self) -> SearchLoopState {
        self.snapshot_strategy();
        let state = self.state.clone();
        self.tracker.on_checkpoint(&state);
        state
    }

    fn snapshot_strategy(&mut self) {
        self.state.strategy = self.strategy.state();
        self.state.scheduler = self.scheduler.state();
    }

    pub fn pending(&self) -> &[TrialRecord] {
        &self.state.pending
    }

    pub fn completed(&self) -> &[TrialRecord] {
        &self.state.completed
    }

    pub fn space(&self) -> &SearchSpace {
        &self.space
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;

    static SPACE: Lazy<SearchSpace> = Lazy::new(|| {
        SearchSpace::new(vec![
            ParamSpec::Float {
                name: "lr".into(),
                low: 1e-4,
                high: 1e-1,
            },
            ParamSpec::Int {
                name: "layers".into(),
                low: 1,
                high: 4,
            },
            ParamSpec::Categorical {
                name: "activation".into(),
                choices: vec!["relu".into(), "gelu".into(), "tanh".into()],
            },
        ])
    });

    fn no_tracker() -> Box<dyn ExperimentTracker> {
        Box::new(NoOpTracker)
    }

    #[test]
    fn deterministic_resume_bayesian() {
        let strategy = Strategy::Bayesian(BayesianStrategy::new(42, 0.3));
        let mut loop_a = SearchLoop::new(
            SPACE.clone(),
            strategy,
            ResourceConfig::default(),
            no_tracker(),
        )
        .unwrap();
        let t1 = loop_a.suggest().unwrap();
        loop_a.observe(t1.id, 0.5).unwrap();
        let checkpoint = loop_a.checkpoint();

        let mut loop_b = SearchLoop::from_state(SPACE.clone(), checkpoint.clone(), no_tracker());
        let next_a = loop_a.suggest().unwrap();
        let next_b = loop_b.suggest().unwrap();
        assert_eq!(next_a.suggestion, next_b.suggestion);

        loop_b.observe(next_b.id, 0.3).unwrap();
        loop_a.observe(next_a.id, 0.3).unwrap();
        let resume_state = loop_b.checkpoint();
        assert_eq!(
            checkpoint.draws_per_suggestion,
            resume_state.draws_per_suggestion
        );
    }

    #[test]
    fn scheduler_respects_concurrency() {
        let strategy = Strategy::Population(PopulationStrategy::new(7, 4, 0.3, 0.5));
        let mut loop_a = SearchLoop::new(
            SPACE.clone(),
            strategy,
            ResourceConfig {
                max_concurrent: 1,
                min_interval: None,
            },
            no_tracker(),
        )
        .unwrap();
        let t1 = loop_a.suggest().unwrap();
        assert!(matches!(
            loop_a.suggest(),
            Err(SearchError::NoAvailableSlot)
        ));
        loop_a.observe(t1.id, 1.0).unwrap();
        assert!(loop_a.suggest().is_ok());
    }
}
