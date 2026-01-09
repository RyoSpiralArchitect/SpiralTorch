use crate::{Objective, SearchLoopState, TrialRecord};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrialSummary {
    pub total_trials: usize,
    pub completed_trials: usize,
    pub pending_trials: usize,
    pub objective: Objective,
    pub best_trial: Option<TrialRecord>,
}

impl TrialSummary {
    pub fn new(completed: &[TrialRecord], pending: usize, objective: Objective) -> Self {
        let best_trial = best_trial(completed, objective).cloned();
        Self {
            total_trials: completed.len() + pending,
            completed_trials: completed.len(),
            pending_trials: pending,
            objective,
            best_trial,
        }
    }

    pub fn from_state(state: &SearchLoopState) -> Self {
        Self::new(&state.completed, state.pending.len(), state.objective)
    }

    pub fn has_best(&self) -> bool {
        self.best_trial.is_some()
    }
}

pub fn best_trial(completed: &[TrialRecord], objective: Objective) -> Option<&TrialRecord> {
    completed
        .iter()
        .filter_map(|record| record.metric.map(|metric| (record, metric)))
        .fold(None, |best, (record, metric)| match best {
            None => Some((record, metric)),
            Some((best_record, best_metric)) => {
                if objective.prefers(metric, best_metric) {
                    Some((record, metric))
                } else {
                    Some((best_record, best_metric))
                }
            }
        })
        .map(|(record, _)| record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::TrialSuggestion;

    fn record(id: usize, metric: Option<f64>) -> TrialRecord {
        TrialRecord {
            id,
            suggestion: TrialSuggestion::new(),
            metric,
        }
    }

    #[test]
    fn best_trial_respects_minimize_objective() {
        let trials = vec![record(0, Some(0.8)), record(1, Some(0.2)), record(2, None)];
        let best = best_trial(&trials, Objective::Minimize).unwrap();
        assert_eq!(best.id, 1);
    }

    #[test]
    fn best_trial_respects_maximize_objective() {
        let trials = vec![
            record(0, Some(0.8)),
            record(1, Some(0.2)),
            record(2, Some(0.95)),
        ];
        let best = best_trial(&trials, Objective::Maximize).unwrap();
        assert_eq!(best.id, 2);
    }

    #[test]
    fn best_trial_handles_empty_metrics() {
        let trials = vec![record(0, None), record(1, None)];
        assert!(best_trial(&trials, Objective::Minimize).is_none());
    }

    #[test]
    fn summary_counts_trials_and_best() {
        let trials = vec![record(0, Some(0.8)), record(1, Some(0.1))];
        let summary = TrialSummary::new(&trials, 2, Objective::Minimize);
        assert_eq!(summary.total_trials, 4);
        assert_eq!(summary.completed_trials, 2);
        assert_eq!(summary.pending_trials, 2);
        assert!(summary.has_best());
        assert_eq!(summary.best_trial.as_ref().unwrap().id, 1);
    }
}
