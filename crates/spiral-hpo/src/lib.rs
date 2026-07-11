#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;
use thiserror::Error;

fn rng_for_suggestion(seed: u64, suggestion_count: u64, draws: usize) -> ChaCha12Rng {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    // StdRng in rand 0.8 is ChaCha12, and each skipped f64 consumes two 32-bit words.
    // ChaCha positions wrap after 2^68 words, so masking preserves stream semantics.
    const WORD_POSITION_MASK: u128 = (1_u128 << 68) - 1;
    let skipped_draws = u128::from(suggestion_count).wrapping_mul(draws as u128);
    rng.set_word_pos(skipped_draws.wrapping_mul(2) & WORD_POSITION_MASK);
    rng
}

fn isolate_tracker_callback(callback: impl FnOnce()) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(callback)) {
        // A foreign tracker owns its panic payload, whose destructor can panic as well.
        if let Err(secondary_payload) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
            std::mem::forget(secondary_payload);
        }
    }
}

pub mod analysis;

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

        pub fn validate(&self) -> Result<(), SearchError> {
            if self.params.is_empty() {
                return Err(SearchError::EmptySpace);
            }

            let mut names = HashSet::with_capacity(self.params.len());
            for spec in &self.params {
                let name = spec.name();
                if name.trim().is_empty() {
                    return Err(SearchError::InvalidParameter {
                        name: name.to_string(),
                        reason: "name must not be empty".to_string(),
                    });
                }
                if !names.insert(name.to_string()) {
                    return Err(SearchError::DuplicateParameter(name.to_string()));
                }

                match spec {
                    ParamSpec::Float { low, high, .. } => {
                        if !(low.is_finite() && high.is_finite()) {
                            return Err(SearchError::InvalidParameter {
                                name: name.to_string(),
                                reason: "float bounds must be finite".to_string(),
                            });
                        }
                        if low > high {
                            return Err(SearchError::InvalidParameter {
                                name: name.to_string(),
                                reason: "lower bound must not exceed upper bound".to_string(),
                            });
                        }
                        if !(high - low).is_finite() {
                            return Err(SearchError::InvalidParameter {
                                name: name.to_string(),
                                reason: "float bound span must be finite".to_string(),
                            });
                        }
                    }
                    ParamSpec::Int { low, high, .. } => {
                        if low > high {
                            return Err(SearchError::InvalidParameter {
                                name: name.to_string(),
                                reason: "lower bound must not exceed upper bound".to_string(),
                            });
                        }
                    }
                    ParamSpec::Categorical { choices, .. } => {
                        if choices.is_empty() {
                            return Err(SearchError::InvalidParameter {
                                name: name.to_string(),
                                reason: "categorical choices must not be empty".to_string(),
                            });
                        }
                    }
                }
            }
            Ok(())
        }

        pub fn validate_suggestion(&self, suggestion: &TrialSuggestion) -> Result<(), SearchError> {
            if suggestion.len() != self.params.len() {
                return Err(SearchError::InvalidParameter {
                    name: "<suggestion>".to_string(),
                    reason: format!(
                        "expected {} values, got {}",
                        self.params.len(),
                        suggestion.len()
                    ),
                });
            }

            for spec in &self.params {
                let name = spec.name();
                let value = suggestion
                    .get(name)
                    .ok_or_else(|| SearchError::InvalidParameter {
                        name: name.to_string(),
                        reason: "suggestion is missing this parameter".to_string(),
                    })?;
                let valid = match (spec, value) {
                    (ParamSpec::Float { low, high, .. }, ParamValue::Float(value)) => {
                        value.is_finite() && value >= low && value <= high
                    }
                    (ParamSpec::Int { low, high, .. }, ParamValue::Int(value)) => {
                        value >= low && value <= high
                    }
                    (ParamSpec::Categorical { choices, .. }, ParamValue::Categorical(value)) => {
                        choices.contains(value)
                    }
                    _ => false,
                };
                if !valid {
                    return Err(SearchError::InvalidParameter {
                        name: name.to_string(),
                        reason: "suggested value has the wrong type or is out of bounds"
                            .to_string(),
                    });
                }
            }
            Ok(())
        }

        pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> TrialSuggestion {
            let mut out = TrialSuggestion::with_capacity(self.params.len());
            for spec in &self.params {
                let value = match spec {
                    ParamSpec::Float { name, low, high } => {
                        let raw = rng.gen::<f64>();
                        let v = low + (high - low) * raw;
                        (name.clone(), ParamValue::Float(v))
                    }
                    ParamSpec::Int { name, low, high } => {
                        let value = if low <= high {
                            let raw = rng.gen::<f64>();
                            let span = (*high as i128 - *low as i128 + 1) as u128;
                            let offset = (raw * span as f64).floor() as u128;
                            (*low as i128 + offset.min(span - 1) as i128) as i64
                        } else {
                            *low
                        };
                        (name.clone(), ParamValue::Int(value))
                    }
                    ParamSpec::Categorical { name, choices } => {
                        let raw = rng.gen::<f64>();
                        let idx = ((choices.len() as f64) * raw).floor() as usize;
                        let idx = idx.min(choices.len().saturating_sub(1));
                        let choice = choices.get(idx).cloned().unwrap_or_default();
                        (name.clone(), ParamValue::Categorical(choice))
                    }
                };
                out.insert(value.0, value.1);
            }
            out
        }

        pub fn clamp(&self, suggestion: &mut TrialSuggestion) {
            for spec in &self.params {
                if let Some(value) = suggestion.get_mut(spec.name()) {
                    match spec {
                        ParamSpec::Float { low, high, .. } => match value {
                            ParamValue::Float(v) => {
                                if low.is_finite() && high.is_finite() && low <= high {
                                    *v = v.clamp(*low, *high);
                                }
                            }
                            ParamValue::Int(v)
                                if low.is_finite() && high.is_finite() && low <= high =>
                            {
                                let new_val = (*v as f64).clamp(*low, *high);
                                *value = ParamValue::Float(new_val);
                            }
                            _ => {}
                        },
                        ParamSpec::Int { low, high, .. } => match value {
                            ParamValue::Int(v) => {
                                if low <= high {
                                    *v = (*v).clamp(*low, *high);
                                }
                            }
                            ParamValue::Float(v) if low <= high && v.is_finite() => {
                                let iv = (*v).round().clamp(*low as f64, *high as f64) as i64;
                                *value = ParamValue::Int(iv);
                            }
                            _ => {}
                        },
                        ParamSpec::Categorical { choices, .. } => {
                            if let ParamValue::Categorical(choice) = value {
                                if !choices.contains(choice) && !choices.is_empty() {
                                    *choice = choices[0].clone();
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn draws_per_suggestion(&self) -> usize {
            self.params.len()
        }
    }
}

pub use analysis::TrialSummary;
pub use space::{ParamSpec, ParamValue, SearchSpace, TrialSuggestion};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Objective {
    #[default]
    Minimize,
    Maximize,
}

impl Objective {
    pub fn from_maximize(maximize: bool) -> Self {
        if maximize {
            Objective::Maximize
        } else {
            Objective::Minimize
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Objective::Minimize => "minimize",
            Objective::Maximize => "maximize",
        }
    }

    pub fn ordering(&self, lhs: f64, rhs: f64) -> Ordering {
        match self {
            Objective::Minimize => lhs.total_cmp(&rhs),
            Objective::Maximize => rhs.total_cmp(&lhs),
        }
    }

    pub fn prefers(&self, candidate: f64, incumbent: f64) -> bool {
        match self {
            Objective::Minimize => candidate < incumbent,
            Objective::Maximize => candidate > incumbent,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub suggestion: TrialSuggestion,
    pub metric: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyState {
    Bayesian(crate::strategies::BayesianState),
    Population(crate::strategies::PopulationState),
    Random(crate::strategies::RandomState),
}

pub mod strategies {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RandomState {
        pub seed: u64,
        pub suggestion_count: u64,
    }

    #[derive(Debug, Clone)]
    pub struct RandomStrategy {
        pub(crate) state: RandomState,
    }

    impl RandomStrategy {
        pub fn new(seed: u64) -> Self {
            Self {
                state: RandomState {
                    seed,
                    suggestion_count: 0,
                },
            }
        }

        fn rng_for(&self, draws: usize) -> ChaCha12Rng {
            rng_for_suggestion(self.state.seed, self.state.suggestion_count, draws)
        }

        pub fn suggest(&mut self, space: &SearchSpace) -> TrialSuggestion {
            let draws = space.draws_per_suggestion();
            let mut rng = self.rng_for(draws);
            self.state.suggestion_count = self.state.suggestion_count.saturating_add(1);
            space.sample(&mut rng)
        }

        pub fn observe(&mut self, _observation: Observation) {}

        pub fn state(&self) -> RandomState {
            self.state.clone()
        }

        pub fn restore(state: RandomState) -> Self {
            Self { state }
        }
    }

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

        fn rng_for(&self, draws: usize) -> ChaCha12Rng {
            rng_for_suggestion(self.state.seed, self.state.suggestion_count, draws)
        }

        pub fn suggest(&mut self, space: &SearchSpace, objective: Objective) -> TrialSuggestion {
            let draws = space.draws_per_suggestion();
            let mut rng = self.rng_for(draws);
            // advance RNG state for this call so that the checkpointed state remains deterministic
            let mut suggestion = space.sample(&mut rng);
            self.state.suggestion_count = self.state.suggestion_count.saturating_add(1);
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
                .min_by(|a, b| objective.ordering(a.metric, b.metric))
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
                                let span = high.saturating_sub(*low).max(1);
                                let noise: f64 = rng.sample(StandardNormal);
                                let delta = (noise * (span as f64 * 0.25)).round() as i64;
                                *v = v.saturating_add(delta).clamp(*low, *high);
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

        pub fn observe(&mut self, observation: Observation, objective: Objective) {
            self.state.observations.push(observation);
            self.state
                .observations
                .sort_by(|a, b| objective.ordering(a.metric, b.metric));
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

        fn rng_for(&self, draws: usize) -> ChaCha12Rng {
            rng_for_suggestion(self.state.seed, self.state.suggestion_count, draws)
        }

        pub fn suggest(&mut self, space: &SearchSpace, objective: Objective) -> TrialSuggestion {
            let draws = space.draws_per_suggestion();
            let mut rng = self.rng_for(draws);
            self.state.suggestion_count = self.state.suggestion_count.saturating_add(1);

            if self.state.population.len() < self.state.population_size {
                return space.sample(&mut rng);
            }

            // Select elites
            let elite_count =
                (self.state.population_size as f64 * self.state.elite_fraction).ceil() as usize;
            let elite_count = elite_count.max(1).min(self.state.population.len());
            let mut sorted = self.state.population.clone();
            sorted.sort_by(|a, b| objective.ordering(a.metric, b.metric));
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
                            let step =
                                (high.saturating_sub(*low).max(1) as f64 * 0.2).ceil() as i64;
                            let delta = rng.gen_range(-step..=step);
                            *v = v.saturating_add(delta).clamp(*low, *high);
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

        pub fn observe(&mut self, observation: Observation, objective: Objective) {
            self.state.population.push(observation);
            if self.state.population.len() > self.state.population_size {
                self.state
                    .population
                    .sort_by(|a, b| objective.ordering(a.metric, b.metric));
                self.state.population.truncate(self.state.population_size);
            }
            if self.state.population.len() <= self.state.population_size {
                self.state
                    .population
                    .sort_by(|a, b| objective.ordering(a.metric, b.metric));
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

use strategies::{BayesianStrategy, PopulationStrategy, RandomStrategy};

#[derive(Debug, Clone)]
pub enum Strategy {
    Bayesian(BayesianStrategy),
    Population(PopulationStrategy),
    Random(RandomStrategy),
}

impl Strategy {
    pub fn name(&self) -> &'static str {
        match self {
            Strategy::Bayesian(_) => "bayesian",
            Strategy::Population(_) => "population",
            Strategy::Random(_) => "random",
        }
    }

    pub fn suggestion_count(&self) -> u64 {
        match self {
            Strategy::Bayesian(strategy) => strategy.state.suggestion_count,
            Strategy::Population(strategy) => strategy.state.suggestion_count,
            Strategy::Random(strategy) => strategy.state.suggestion_count,
        }
    }

    pub fn suggest(&mut self, space: &SearchSpace, objective: Objective) -> TrialSuggestion {
        match self {
            Strategy::Bayesian(strategy) => strategy.suggest(space, objective),
            Strategy::Population(strategy) => strategy.suggest(space, objective),
            Strategy::Random(strategy) => strategy.suggest(space),
        }
    }

    pub fn observe(&mut self, observation: Observation, objective: Objective) {
        match self {
            Strategy::Bayesian(strategy) => strategy.observe(observation, objective),
            Strategy::Population(strategy) => strategy.observe(observation, objective),
            Strategy::Random(strategy) => strategy.observe(observation),
        }
    }

    pub fn state(&self) -> StrategyState {
        match self {
            Strategy::Bayesian(strategy) => StrategyState::Bayesian(strategy.state()),
            Strategy::Population(strategy) => StrategyState::Population(strategy.state()),
            Strategy::Random(strategy) => StrategyState::Random(strategy.state()),
        }
    }

    pub fn restore(state: StrategyState) -> Strategy {
        match state {
            StrategyState::Bayesian(state) => Strategy::Bayesian(BayesianStrategy::restore(state)),
            StrategyState::Population(state) => {
                Strategy::Population(PopulationStrategy::restore(state))
            }
            StrategyState::Random(state) => Strategy::Random(RandomStrategy::restore(state)),
        }
    }

    pub fn validate(&self, space: &SearchSpace) -> Result<(), SearchError> {
        match self {
            Strategy::Bayesian(strategy) => {
                let exploration = strategy.state.exploration;
                if !(exploration.is_finite() && (0.0..=1.0).contains(&exploration)) {
                    return Err(SearchError::InvalidStrategy(
                        "bayesian exploration must be finite and within [0, 1]".to_string(),
                    ));
                }
                if strategy.state.observations.len() as u128
                    > u128::from(strategy.state.suggestion_count)
                {
                    return Err(SearchError::InvalidStrategy(
                        "bayesian observation count exceeds suggestion count".to_string(),
                    ));
                }
                for observation in &strategy.state.observations {
                    if !observation.metric.is_finite() {
                        return Err(SearchError::InvalidStrategy(
                            "bayesian observations must have finite metrics".to_string(),
                        ));
                    }
                    space
                        .validate_suggestion(&observation.suggestion)
                        .map_err(|error| SearchError::InvalidStrategy(error.to_string()))?;
                }
            }
            Strategy::Population(strategy) => {
                if strategy.state.population_size < 2 {
                    return Err(SearchError::InvalidStrategy(
                        "population size must be at least 2".to_string(),
                    ));
                }
                if strategy.state.population.len() > strategy.state.population_size {
                    return Err(SearchError::InvalidStrategy(
                        "population history exceeds configured population size".to_string(),
                    ));
                }
                if strategy.state.population.len() as u128
                    > u128::from(strategy.state.suggestion_count)
                {
                    return Err(SearchError::InvalidStrategy(
                        "population history exceeds suggestion count".to_string(),
                    ));
                }
                if !(strategy.state.elite_fraction.is_finite()
                    && (0.0..=1.0).contains(&strategy.state.elite_fraction))
                {
                    return Err(SearchError::InvalidStrategy(
                        "elite fraction must be finite and within [0, 1]".to_string(),
                    ));
                }
                if !(strategy.state.mutation_rate.is_finite()
                    && (0.0..=1.0).contains(&strategy.state.mutation_rate))
                {
                    return Err(SearchError::InvalidStrategy(
                        "mutation rate must be finite and within [0, 1]".to_string(),
                    ));
                }
                for observation in &strategy.state.population {
                    if !observation.metric.is_finite() {
                        return Err(SearchError::InvalidStrategy(
                            "population observations must have finite metrics".to_string(),
                        ));
                    }
                    space
                        .validate_suggestion(&observation.suggestion)
                        .map_err(|error| SearchError::InvalidStrategy(error.to_string()))?;
                }
            }
            Strategy::Random(_) => {}
        }
        Ok(())
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

impl ResourceConfig {
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.max_concurrent == 0 {
            return Err(SearchError::InvalidResource(
                "max_concurrent must be positive".to_string(),
            ));
        }
        Ok(())
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
                if now < last.saturating_add(u128::from(interval)) {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    #[serde(default)]
    pub objective: Objective,
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("all resource slots are busy")]
    NoAvailableSlot,
    #[error("trial id {0} not found")]
    UnknownTrial(usize),
    #[error("search space must define at least one parameter")]
    EmptySpace,
    #[error("invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },
    #[error("parameter '{0}' is defined more than once")]
    DuplicateParameter(String),
    #[error("invalid resource configuration: {0}")]
    InvalidResource(String),
    #[error("invalid strategy configuration: {0}")]
    InvalidStrategy(String),
    #[error("metric for trial {trial_id} must be finite")]
    NonFiniteMetric { trial_id: usize },
    #[error("trial id space is exhausted")]
    TrialIdExhausted,
    #[error("invalid search checkpoint: {0}")]
    InvalidCheckpoint(String),
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
        objective: Objective,
        tracker: Box<dyn ExperimentTracker>,
    ) -> Result<Self, SearchError> {
        space.validate()?;
        strategy.validate(&space)?;
        resource.validate()?;
        if strategy.suggestion_count() != 0 {
            return Err(SearchError::InvalidStrategy(
                "a new search loop requires a fresh strategy".to_string(),
            ));
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
                objective,
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
    ) -> Result<Self, SearchError> {
        space.validate()?;
        state.resource.validate()?;
        if state.draws_per_suggestion != space.draws_per_suggestion() {
            return Err(SearchError::InvalidCheckpoint(format!(
                "draws_per_suggestion mismatch (expected {}, got {})",
                space.draws_per_suggestion(),
                state.draws_per_suggestion
            )));
        }
        if state.scheduler.active_slots != state.pending.len() {
            return Err(SearchError::InvalidCheckpoint(format!(
                "active slot count {} does not match {} pending trials",
                state.scheduler.active_slots,
                state.pending.len()
            )));
        }
        if state.scheduler.active_slots > state.resource.max_concurrent {
            return Err(SearchError::InvalidCheckpoint(format!(
                "active slot count {} exceeds max_concurrent {}",
                state.scheduler.active_slots, state.resource.max_concurrent
            )));
        }
        let record_count = state
            .pending
            .len()
            .checked_add(state.completed.len())
            .ok_or_else(|| {
                SearchError::InvalidCheckpoint("trial record count overflow".to_string())
            })?;
        if record_count != state.next_trial_id {
            return Err(SearchError::InvalidCheckpoint(format!(
                "{} trial records do not match next_trial_id {}",
                record_count, state.next_trial_id
            )));
        }

        let mut trial_ids = HashSet::with_capacity(state.pending.len() + state.completed.len());
        for trial in &state.pending {
            if trial.metric.is_some() {
                return Err(SearchError::InvalidCheckpoint(format!(
                    "pending trial {} already has a metric",
                    trial.id
                )));
            }
            if !trial_ids.insert(trial.id) {
                return Err(SearchError::InvalidCheckpoint(format!(
                    "duplicate trial id {}",
                    trial.id
                )));
            }
            if trial.id >= state.next_trial_id {
                return Err(SearchError::InvalidCheckpoint(format!(
                    "trial id {} is not below next_trial_id {}",
                    trial.id, state.next_trial_id
                )));
            }
            space
                .validate_suggestion(&trial.suggestion)
                .map_err(|error| {
                    SearchError::InvalidCheckpoint(format!(
                        "pending trial {} has an invalid suggestion: {error}",
                        trial.id
                    ))
                })?;
        }
        for trial in &state.completed {
            let Some(metric) = trial.metric else {
                return Err(SearchError::InvalidCheckpoint(format!(
                    "completed trial {} is missing its metric",
                    trial.id
                )));
            };
            if !metric.is_finite() {
                return Err(SearchError::InvalidCheckpoint(format!(
                    "completed trial {} has a non-finite metric",
                    trial.id
                )));
            }
            if !trial_ids.insert(trial.id) {
                return Err(SearchError::InvalidCheckpoint(format!(
                    "duplicate trial id {}",
                    trial.id
                )));
            }
            if trial.id >= state.next_trial_id {
                return Err(SearchError::InvalidCheckpoint(format!(
                    "trial id {} is not below next_trial_id {}",
                    trial.id, state.next_trial_id
                )));
            }
            space
                .validate_suggestion(&trial.suggestion)
                .map_err(|error| {
                    SearchError::InvalidCheckpoint(format!(
                        "completed trial {} has an invalid suggestion: {error}",
                        trial.id
                    ))
                })?;
        }

        let scheduler =
            ResourceScheduler::from_state(state.resource.clone(), state.scheduler.clone());
        let strategy = Strategy::restore(state.strategy.clone());
        strategy.validate(&space)?;
        let expected_suggestion_count = u64::try_from(state.next_trial_id).map_err(|_| {
            SearchError::InvalidCheckpoint(
                "next_trial_id cannot be represented by the strategy state".to_string(),
            )
        })?;
        if strategy.suggestion_count() != expected_suggestion_count {
            return Err(SearchError::InvalidCheckpoint(format!(
                "strategy suggestion count {} does not match next_trial_id {}",
                strategy.suggestion_count(),
                state.next_trial_id
            )));
        }
        Ok(Self {
            strategy,
            scheduler,
            space,
            tracker,
            state,
        })
    }

    pub fn suggest(&mut self) -> Result<TrialRecord, SearchError> {
        let next_trial_id = self
            .state
            .next_trial_id
            .checked_add(1)
            .ok_or(SearchError::TrialIdExhausted)?;
        if !self.scheduler.try_reserve() {
            return Err(SearchError::NoAvailableSlot);
        }
        let suggestion = self.strategy.suggest(&self.space, self.state.objective);
        let trial = TrialRecord {
            id: self.state.next_trial_id,
            suggestion,
            metric: None,
        };
        self.state.next_trial_id = next_trial_id;
        self.state.pending.push(trial.clone());
        self.snapshot_strategy();
        isolate_tracker_callback(|| self.tracker.on_trial_start(&trial));
        Ok(trial)
    }

    pub fn observe(&mut self, trial_id: usize, metric: f64) -> Result<(), SearchError> {
        if !metric.is_finite() {
            return Err(SearchError::NonFiniteMetric { trial_id });
        }
        if let Some(idx) = self
            .state
            .pending
            .iter()
            .position(|trial| trial.id == trial_id)
        {
            let mut trial = self.state.pending.remove(idx);
            self.scheduler.release();
            trial.metric = Some(metric);
            self.strategy.observe(
                Observation {
                    suggestion: trial.suggestion.clone(),
                    metric,
                },
                self.state.objective,
            );
            self.state.completed.push(trial.clone());
            self.snapshot_strategy();
            isolate_tracker_callback(|| self.tracker.on_trial_end(&trial, metric));
            Ok(())
        } else {
            Err(SearchError::UnknownTrial(trial_id))
        }
    }

    pub fn checkpoint(&mut self) -> SearchLoopState {
        self.snapshot_strategy();
        let state = self.state.clone();
        isolate_tracker_callback(|| self.tracker.on_checkpoint(&state));
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

    pub fn objective(&self) -> Objective {
        self.state.objective
    }

    pub fn best_trial(&self) -> Option<TrialRecord> {
        crate::analysis::best_trial(self.state.completed.as_slice(), self.state.objective).cloned()
    }

    pub fn summary(&self) -> crate::analysis::TrialSummary {
        crate::analysis::TrialSummary::from_state(&self.state)
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

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("panic while dropping tracker payload");
        }
    }

    struct HostileTracker;

    impl ExperimentTracker for HostileTracker {
        fn on_trial_start(&mut self, _trial: &TrialRecord) {
            std::panic::panic_any(PanicOnDrop);
        }

        fn on_trial_end(&mut self, _trial: &TrialRecord, _metric: f64) {
            std::panic::panic_any(PanicOnDrop);
        }

        fn on_checkpoint(&mut self, _state: &SearchLoopState) {
            std::panic::panic_any(PanicOnDrop);
        }
    }

    #[test]
    fn seeked_rng_matches_legacy_skip_sequence() {
        let seed = 0x5eed_u64;
        let draws = 3_usize;
        for suggestion_count in [0_u64, 1, 2, 31, 1_000] {
            let mut legacy = StdRng::seed_from_u64(seed);
            for _ in 0..suggestion_count * draws as u64 {
                let _: f64 = legacy.gen();
            }
            let mut seeked = rng_for_suggestion(seed, suggestion_count, draws);

            for _ in 0..8 {
                assert_eq!(legacy.next_u64(), seeked.next_u64());
            }
        }
    }

    #[test]
    fn deterministic_resume_bayesian() {
        let strategy = Strategy::Bayesian(BayesianStrategy::new(42, 0.3));
        let mut loop_a = SearchLoop::new(
            SPACE.clone(),
            strategy,
            ResourceConfig::default(),
            Objective::Minimize,
            no_tracker(),
        )
        .unwrap();
        let t1 = loop_a.suggest().unwrap();
        loop_a.observe(t1.id, 0.5).unwrap();
        let checkpoint = loop_a.checkpoint();

        let mut loop_b =
            SearchLoop::from_state(SPACE.clone(), checkpoint.clone(), no_tracker()).unwrap();
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
    fn deterministic_resume_random() {
        let strategy = Strategy::Random(RandomStrategy::new(1337));
        let mut loop_a = SearchLoop::new(
            SPACE.clone(),
            strategy,
            ResourceConfig::default(),
            Objective::Minimize,
            no_tracker(),
        )
        .unwrap();

        let t1 = loop_a.suggest().unwrap();
        loop_a.observe(t1.id, 1.23).unwrap();
        let checkpoint = loop_a.checkpoint();

        let mut loop_b =
            SearchLoop::from_state(SPACE.clone(), checkpoint.clone(), no_tracker()).unwrap();
        let next_a = loop_a.suggest().unwrap();
        let next_b = loop_b.suggest().unwrap();
        assert_eq!(next_a.suggestion, next_b.suggestion);

        loop_b.observe(next_b.id, 0.7).unwrap();
        loop_a.observe(next_a.id, 0.7).unwrap();
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
            Objective::Minimize,
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

    #[test]
    fn objective_controls_population_ordering() {
        let mut strategy = strategies::PopulationStrategy::new(0, 4, 0.5, 0.0);
        let mut suggestion = TrialSuggestion::new();
        suggestion.insert("param".into(), ParamValue::Float(0.0));
        let obs = |metric: f64| Observation {
            suggestion: suggestion.clone(),
            metric,
        };

        strategy.observe(obs(0.5), Objective::Minimize);
        strategy.observe(obs(0.1), Objective::Minimize);
        assert!(strategy.state.population[0].metric <= strategy.state.population[1].metric);

        strategy.observe(obs(0.9), Objective::Maximize);
        strategy.observe(obs(0.2), Objective::Maximize);
        assert!(strategy.state.population[0].metric >= strategy.state.population[1].metric);
    }

    #[test]
    fn objective_ordering_matches_direction() {
        assert_eq!(Objective::Minimize.ordering(0.1, 0.5), Ordering::Less);
        assert_eq!(Objective::Maximize.ordering(0.1, 0.5), Ordering::Greater);
    }

    #[test]
    fn search_loop_best_trial_respects_objective() {
        let mut loop_min = SearchLoop::new(
            SPACE.clone(),
            Strategy::Bayesian(BayesianStrategy::new(0, 0.3)),
            ResourceConfig::default(),
            Objective::Minimize,
            no_tracker(),
        )
        .unwrap();
        let t1 = loop_min.suggest().unwrap();
        loop_min.observe(t1.id, 0.5).unwrap();
        let t2 = loop_min.suggest().unwrap();
        loop_min.observe(t2.id, 0.2).unwrap();
        let best_min = loop_min.best_trial().unwrap();
        assert_eq!(best_min.id, t2.id);

        let mut loop_max = SearchLoop::new(
            SPACE.clone(),
            Strategy::Bayesian(BayesianStrategy::new(1, 0.3)),
            ResourceConfig::default(),
            Objective::Maximize,
            no_tracker(),
        )
        .unwrap();
        let a = loop_max.suggest().unwrap();
        loop_max.observe(a.id, 0.1).unwrap();
        let b = loop_max.suggest().unwrap();
        loop_max.observe(b.id, 0.9).unwrap();
        let best_max = loop_max.best_trial().unwrap();
        assert_eq!(best_max.id, b.id);
    }

    #[test]
    fn summary_reflects_current_state() {
        let mut loop_min = SearchLoop::new(
            SPACE.clone(),
            Strategy::Bayesian(BayesianStrategy::new(0, 0.3)),
            ResourceConfig::default(),
            Objective::Minimize,
            no_tracker(),
        )
        .unwrap();
        let trial = loop_min.suggest().unwrap();
        let summary = loop_min.summary();
        assert_eq!(summary.pending_trials, 1);
        assert_eq!(summary.completed_trials, 0);
        assert!(!summary.has_best());

        loop_min.observe(trial.id, 0.4).unwrap();
        let summary = loop_min.summary();
        assert_eq!(summary.pending_trials, 0);
        assert_eq!(summary.completed_trials, 1);
        assert!(summary.has_best());
        assert_eq!(summary.best_trial.unwrap().id, trial.id);
    }

    #[test]
    fn invalid_search_spaces_are_rejected_without_sampling() {
        let cases = [
            SearchSpace::new(vec![ParamSpec::Float {
                name: "nan".into(),
                low: f64::NAN,
                high: 1.0,
            }]),
            SearchSpace::new(vec![ParamSpec::Float {
                name: "overflow".into(),
                low: -f64::MAX,
                high: f64::MAX,
            }]),
            SearchSpace::new(vec![ParamSpec::Int {
                name: "reversed".into(),
                low: 3,
                high: 2,
            }]),
            SearchSpace::new(vec![ParamSpec::Categorical {
                name: "empty".into(),
                choices: Vec::new(),
            }]),
            SearchSpace::new(vec![ParamSpec::Int {
                name: "  ".into(),
                low: 0,
                high: 1,
            }]),
            SearchSpace::new(vec![
                ParamSpec::Int {
                    name: "duplicate".into(),
                    low: 0,
                    high: 1,
                },
                ParamSpec::Float {
                    name: "duplicate".into(),
                    low: 0.0,
                    high: 1.0,
                },
            ]),
        ];

        for space in cases {
            assert!(SearchLoop::new(
                space,
                Strategy::Random(RandomStrategy::new(0)),
                ResourceConfig::default(),
                Objective::Minimize,
                no_tracker(),
            )
            .is_err());
        }
    }

    #[test]
    fn full_i64_parameter_range_samples_without_overflow() {
        let space = SearchSpace::new(vec![ParamSpec::Int {
            name: "full".into(),
            low: i64::MIN,
            high: i64::MAX,
        }]);
        space.validate().unwrap();
        let mut rng = StdRng::seed_from_u64(7);

        for _ in 0..32 {
            let sample = space.sample(&mut rng);
            assert!(matches!(sample.get("full"), Some(ParamValue::Int(_))));
        }
    }

    #[test]
    fn non_finite_metric_does_not_consume_pending_trial() {
        let mut search = SearchLoop::new(
            SPACE.clone(),
            Strategy::Random(RandomStrategy::new(9)),
            ResourceConfig::default(),
            Objective::Minimize,
            no_tracker(),
        )
        .unwrap();
        let trial = search.suggest().unwrap();

        assert!(matches!(
            search.observe(trial.id, f64::NAN),
            Err(SearchError::NonFiniteMetric { trial_id }) if trial_id == trial.id
        ));
        assert_eq!(search.pending(), std::slice::from_ref(&trial));
        assert!(search.completed().is_empty());
        assert_eq!(search.scheduler.state().active_slots, 1);

        search.observe(trial.id, 0.5).unwrap();
        assert!(search.pending().is_empty());
        assert_eq!(search.completed().len(), 1);
    }

    #[test]
    fn zero_resource_slots_and_invalid_strategy_are_rejected() {
        assert!(matches!(
            SearchLoop::new(
                SPACE.clone(),
                Strategy::Random(RandomStrategy::new(0)),
                ResourceConfig {
                    max_concurrent: 0,
                    min_interval: None,
                },
                Objective::Minimize,
                no_tracker(),
            ),
            Err(SearchError::InvalidResource(_))
        ));
        assert!(matches!(
            SearchLoop::new(
                SPACE.clone(),
                Strategy::Bayesian(BayesianStrategy::new(0, f64::NAN)),
                ResourceConfig::default(),
                Objective::Minimize,
                no_tracker(),
            ),
            Err(SearchError::InvalidStrategy(_))
        ));
    }

    #[test]
    fn trial_id_exhaustion_does_not_reserve_a_slot() {
        let mut search = SearchLoop::new(
            SPACE.clone(),
            Strategy::Random(RandomStrategy::new(0)),
            ResourceConfig::default(),
            Objective::Minimize,
            no_tracker(),
        )
        .unwrap();
        search.state.next_trial_id = usize::MAX;

        assert!(matches!(
            search.suggest(),
            Err(SearchError::TrialIdExhausted)
        ));
        assert_eq!(search.scheduler.state().active_slots, 0);
        assert!(search.pending().is_empty());
    }

    #[test]
    fn checkpoint_validation_rejects_mismatched_state() {
        let mut search = SearchLoop::new(
            SPACE.clone(),
            Strategy::Random(RandomStrategy::new(0)),
            ResourceConfig::default(),
            Objective::Minimize,
            no_tracker(),
        )
        .unwrap();
        search.suggest().unwrap();
        let state = search.checkpoint();

        let mut wrong_draws = state.clone();
        wrong_draws.draws_per_suggestion += 1;
        assert!(matches!(
            SearchLoop::from_state(SPACE.clone(), wrong_draws, no_tracker()),
            Err(SearchError::InvalidCheckpoint(_))
        ));

        let mut wrong_slots = state.clone();
        wrong_slots.scheduler.active_slots = 0;
        assert!(matches!(
            SearchLoop::from_state(SPACE.clone(), wrong_slots, no_tracker()),
            Err(SearchError::InvalidCheckpoint(_))
        ));

        let mut wrong_trial_count = state.clone();
        wrong_trial_count.next_trial_id += 1;
        assert!(matches!(
            SearchLoop::from_state(SPACE.clone(), wrong_trial_count, no_tracker()),
            Err(SearchError::InvalidCheckpoint(_))
        ));

        let mut wrong_strategy_count = state.clone();
        if let StrategyState::Random(strategy) = &mut wrong_strategy_count.strategy {
            strategy.suggestion_count += 1;
        }
        assert!(matches!(
            SearchLoop::from_state(SPACE.clone(), wrong_strategy_count, no_tracker()),
            Err(SearchError::InvalidCheckpoint(_))
        ));

        let mut wrong_suggestion = state;
        wrong_suggestion.pending[0]
            .suggestion
            .insert("lr".into(), ParamValue::Float(f64::INFINITY));
        assert!(matches!(
            SearchLoop::from_state(SPACE.clone(), wrong_suggestion, no_tracker()),
            Err(SearchError::InvalidCheckpoint(_))
        ));
    }

    #[test]
    fn scheduler_interval_check_saturates_for_untrusted_checkpoint_time() {
        let mut scheduler = ResourceScheduler::from_state(
            ResourceConfig {
                max_concurrent: 1,
                min_interval: Some(u64::MAX),
            },
            SchedulerState {
                active_slots: 0,
                last_scheduled_ms: Some(u128::MAX),
            },
        );

        assert!(!scheduler.try_reserve());
        assert_eq!(scheduler.state().active_slots, 0);
    }

    #[test]
    fn tracker_panics_and_hostile_payloads_do_not_corrupt_search_state() {
        let mut search = SearchLoop::new(
            SPACE.clone(),
            Strategy::Random(RandomStrategy::new(41)),
            ResourceConfig::default(),
            Objective::Minimize,
            Box::new(HostileTracker),
        )
        .unwrap();

        let trial = search.suggest().unwrap();
        assert_eq!(search.pending(), std::slice::from_ref(&trial));
        search.observe(trial.id, 0.25).unwrap();
        assert!(search.pending().is_empty());
        assert_eq!(
            search.completed(),
            &[TrialRecord {
                metric: Some(0.25),
                ..trial
            }]
        );

        let checkpoint = search.checkpoint();
        assert_eq!(checkpoint.scheduler.active_slots, 0);
        assert_eq!(checkpoint.next_trial_id, 1);
        SearchLoop::from_state(SPACE.clone(), checkpoint, no_tracker()).unwrap();
    }
}
