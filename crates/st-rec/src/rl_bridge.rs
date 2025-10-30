// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::thread_rng;
use rand::SeedableRng;
use spiral_config::determinism;
use st_spiral_rl::{
    EpisodeReport, GeometryFeedback, GeometryFeedbackSignal, PolicyTelemetry, SpiralPolicyGradient,
    SpiralRlError,
};
use st_tensor::{DifferentialResonance, Tensor};

use crate::{Recommendation, SpiralRecError, SpiralRecommender};

/// Errors surfaced by the reinforcement-learning bridge.
#[derive(Debug)]
pub enum RecRlError {
    /// Underlying recommendation harness failed.
    Recommendation(SpiralRecError),
    /// Underlying reinforcement-learning policy failed.
    Reinforcement(SpiralRlError),
    /// Candidate list must match the configured action space.
    CandidateMismatch { expected: usize, provided: usize },
    /// Policy probabilities could not be sampled into an action.
    InvalidProbabilityDistribution { reason: String },
    /// Policy selected an action outside the available candidate list.
    ActionOutOfBounds { actions: usize, selected: usize },
}

impl fmt::Display for RecRlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecRlError::Recommendation(err) => write!(f, "{err}"),
            RecRlError::Reinforcement(err) => write!(f, "{err}"),
            RecRlError::CandidateMismatch { expected, provided } => write!(
                f,
                "candidate list length {provided} does not match policy action space {expected}",
            ),
            RecRlError::InvalidProbabilityDistribution { reason } => {
                write!(f, "invalid policy probability distribution: {reason}")
            }
            RecRlError::ActionOutOfBounds { actions, selected } => write!(
                f,
                "selected action index {selected} is outside candidate list of length {actions}",
            ),
        }
    }
}

impl std::error::Error for RecRlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RecRlError::Recommendation(err) => Some(err),
            RecRlError::Reinforcement(err) => Some(err),
            RecRlError::CandidateMismatch { .. } => None,
            RecRlError::InvalidProbabilityDistribution { .. } => None,
            RecRlError::ActionOutOfBounds { .. } => None,
        }
    }
}

impl From<SpiralRecError> for RecRlError {
    fn from(value: SpiralRecError) -> Self {
        RecRlError::Recommendation(value)
    }
}

impl From<SpiralRlError> for RecRlError {
    fn from(value: SpiralRlError) -> Self {
        RecRlError::Reinforcement(value)
    }
}

impl From<rand::distributions::weighted::WeightedError> for RecRlError {
    fn from(value: rand::distributions::weighted::WeightedError) -> Self {
        RecRlError::InvalidProbabilityDistribution {
            reason: value.to_string(),
        }
    }
}

/// Result type returned by the recommender/policy bridge helpers.
pub type RecRlResult<T> = Result<T, RecRlError>;

/// Decision emitted by the bandit controller.
#[derive(Clone, Debug)]
pub struct RecBanditDecision {
    /// User identifier the policy evaluated.
    pub user: usize,
    /// Candidate recommendations scored by the matrix factorisation model.
    pub candidates: Vec<Recommendation>,
    /// Item selected by the policy.
    pub chosen_item: usize,
    /// Index of the chosen item inside the candidate list.
    pub action_index: usize,
    /// State tensor supplied to the policy (user embedding snapshot).
    pub state: Tensor,
    /// Policy probabilities associated with each candidate action.
    pub probabilities: Vec<f32>,
}

impl RecBanditDecision {
    /// Returns the recommendation selected by the controller.
    pub fn recommendation(&self) -> Recommendation {
        self.candidates[self.action_index].clone()
    }

    /// Decomposes the decision into its state tensor for manual bookkeeping.
    pub fn into_state(self) -> Tensor {
        self.state
    }
}

/// Adapter that routes `SpiralRecommender` embeddings through a policy gradient.
pub struct RecBanditController {
    policy: SpiralPolicyGradient,
    action_dim: usize,
}

impl RecBanditController {
    /// Builds a controller that mirrors the recommender's latent dimensionality.
    pub fn new(
        recommender: &SpiralRecommender,
        action_dim: usize,
        learning_rate: f32,
        discount: f32,
    ) -> RecRlResult<Self> {
        let policy =
            SpiralPolicyGradient::new(recommender.factors(), action_dim, learning_rate, discount)?;
        Ok(Self { policy, action_dim })
    }

    /// Returns the number of actions expected for every decision.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Provides immutable access to the underlying policy.
    pub fn policy(&self) -> &SpiralPolicyGradient {
        &self.policy
    }

    /// Provides mutable access to the underlying policy for advanced tuning.
    pub fn policy_mut(&mut self) -> &mut SpiralPolicyGradient {
        &mut self.policy
    }

    /// Enables the hypergradient tape inside the wrapped policy.
    pub fn enable_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> RecRlResult<()> {
        self.policy
            .enable_hypergrad(curvature, learning_rate)
            .map_err(RecRlError::from)
    }

    /// Attaches the supplied geometry feedback controller.
    pub fn attach_geometry_feedback(&mut self, feedback: GeometryFeedback) {
        self.policy.attach_geometry_feedback(feedback);
    }

    /// Detaches the currently configured geometry feedback controller, if any.
    pub fn detach_geometry_feedback(&mut self) -> Option<GeometryFeedback> {
        self.policy.detach_geometry_feedback()
    }

    /// Returns the last emitted geometry signal, if the controller is active.
    pub fn last_geometry_signal(&self) -> Option<&GeometryFeedbackSignal> {
        self.policy.last_geometry_signal()
    }

    /// Clears the buffered episode transitions without updating the policy.
    pub fn reset_episode(&mut self) {
        self.policy.reset_episode();
    }

    /// Fetches rolling telemetry from the wrapped policy.
    pub fn telemetry(&self) -> PolicyTelemetry {
        self.policy.telemetry()
    }

    /// Builds the recommendation state for the user and greedily selects an item.
    pub fn select_recommendation(
        &self,
        recommender: &SpiralRecommender,
        user: usize,
        candidates: &[usize],
    ) -> RecRlResult<RecBanditDecision> {
        let state = self.load_user_state(recommender, user)?;
        let recs = self.score_candidates(recommender, user, candidates)?;
        self.make_decision_with(user, state, recs, |probs| Ok(Self::greedy_action(probs)))
    }

    /// Pulls the recommender's top-k slate and routes it through the policy.
    pub fn select_top_k(
        &self,
        recommender: &SpiralRecommender,
        user: usize,
        k: usize,
        exclude: Option<&[usize]>,
    ) -> RecRlResult<RecBanditDecision> {
        if k != self.action_dim {
            return Err(RecRlError::CandidateMismatch {
                expected: self.action_dim,
                provided: k,
            });
        }
        let state = self.load_user_state(recommender, user)?;
        let recs = recommender.recommend_top_k(user, k, exclude)?;
        self.make_decision_with(user, state, recs, |probs| Ok(Self::greedy_action(probs)))
    }

    /// Builds the recommendation state and samples an item from the policy distribution.
    pub fn sample_recommendation(
        &self,
        recommender: &SpiralRecommender,
        user: usize,
        candidates: &[usize],
    ) -> RecRlResult<RecBanditDecision> {
        let state = self.load_user_state(recommender, user)?;
        let recs = self.score_candidates(recommender, user, candidates)?;
        self.make_decision_with(user, state, recs, |probs| Self::sample_action(probs))
    }

    /// Samples a top-k slate from the recommender and draws an action from the policy.
    pub fn sample_top_k(
        &self,
        recommender: &SpiralRecommender,
        user: usize,
        k: usize,
        exclude: Option<&[usize]>,
    ) -> RecRlResult<RecBanditDecision> {
        if k != self.action_dim {
            return Err(RecRlError::CandidateMismatch {
                expected: self.action_dim,
                provided: k,
            });
        }
        let state = self.load_user_state(recommender, user)?;
        let recs = recommender.recommend_top_k(user, k, exclude)?;
        self.make_decision_with(user, state, recs, |probs| Self::sample_action(probs))
    }

    fn make_decision_with<F>(
        &self,
        user: usize,
        state: Tensor,
        candidates: Vec<Recommendation>,
        select_action: F,
    ) -> RecRlResult<RecBanditDecision>
    where
        F: FnOnce(&[f32]) -> RecRlResult<usize>,
    {
        if candidates.len() != self.action_dim {
            return Err(RecRlError::CandidateMismatch {
                expected: self.action_dim,
                provided: candidates.len(),
            });
        }

        let probabilities = self.policy.policy(&state)?;
        if probabilities.len() != self.action_dim {
            return Err(RecRlError::CandidateMismatch {
                expected: self.action_dim,
                provided: probabilities.len(),
            });
        }
        let action_index = select_action(&probabilities)?;
        if action_index >= candidates.len() {
            return Err(RecRlError::ActionOutOfBounds {
                actions: candidates.len(),
                selected: action_index,
            });
        }
        let chosen_item = candidates[action_index].item;

        Ok(RecBanditDecision {
            user,
            candidates,
            chosen_item,
            action_index,
            state,
            probabilities,
        })
    }

    fn load_user_state(&self, recommender: &SpiralRecommender, user: usize) -> RecRlResult<Tensor> {
        Ok(recommender.user_embedding(user)?)
    }

    fn score_candidates(
        &self,
        recommender: &SpiralRecommender,
        user: usize,
        candidates: &[usize],
    ) -> RecRlResult<Vec<Recommendation>> {
        if candidates.len() != self.action_dim {
            return Err(RecRlError::CandidateMismatch {
                expected: self.action_dim,
                provided: candidates.len(),
            });
        }
        let mut recs = Vec::with_capacity(candidates.len());
        for &item in candidates {
            let score = recommender.predict(user, item)?;
            recs.push(Recommendation { item, score });
        }
        Ok(recs)
    }

    fn greedy_action(probabilities: &[f32]) -> usize {
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn sample_action(probabilities: &[f32]) -> RecRlResult<usize> {
        let distribution = WeightedIndex::new(probabilities)?;
        if determinism::config().enabled {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            "st-rec/rl_bridge/sample_action".hash(&mut hasher);
            for prob in probabilities {
                prob.to_bits().hash(&mut hasher);
            }
            let seed = determinism::config().seed_for(hasher.finish());
            let mut rng = StdRng::seed_from_u64(seed);
            Ok(distribution.sample(&mut rng))
        } else {
            let mut rng = thread_rng();
            Ok(distribution.sample(&mut rng))
        }
    }

    /// Records the observed reward for the provided transition.
    pub fn observe_reward(&mut self, decision: RecBanditDecision, reward: f32) -> RecRlResult<()> {
        let RecBanditDecision {
            state,
            action_index,
            ..
        } = decision;
        self.record_outcome(state, action_index, reward)
    }

    /// Records a transition and keeps it buffered until the next update step.
    pub fn record_outcome(
        &mut self,
        state: Tensor,
        action_index: usize,
        reward: f32,
    ) -> RecRlResult<()> {
        self.policy
            .record_transition(state, action_index, reward)
            .map_err(RecRlError::from)
    }

    /// Applies a policy update using the buffered recommendation transitions.
    pub fn finish_episode(&mut self) -> RecRlResult<EpisodeReport> {
        self.policy.finish_episode().map_err(RecRlError::from)
    }

    /// Applies a geometry-modulated update using an external resonance snapshot.
    pub fn finish_episode_with_geometry(
        &mut self,
        resonance: &DifferentialResonance,
    ) -> RecRlResult<(EpisodeReport, Option<GeometryFeedbackSignal>)> {
        self.policy
            .finish_episode_with_geometry(resonance)
            .map_err(RecRlError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RatingTriple;

    fn build_basic_recommender() -> SpiralRecommender {
        let mut rec = SpiralRecommender::new(2, 5, 3, 0.05, 0.01, -1.0).unwrap();
        let ratings = vec![
            RatingTriple::new(0, 0, 5.0),
            RatingTriple::new(0, 1, 3.0),
            RatingTriple::new(0, 2, 1.0),
            RatingTriple::new(1, 2, 4.0),
            RatingTriple::new(1, 3, 2.0),
        ];

        for _ in 0..6 {
            rec.train_epoch(&ratings).unwrap();
        }
        rec
    }

    #[test]
    fn bandit_selects_candidate_and_updates() {
        let rec = build_basic_recommender();
        let mut controller = RecBanditController::new(&rec, 3, 0.01, 0.9).unwrap();
        let candidates = vec![0, 1, 2];
        let decision = controller
            .select_recommendation(&rec, 0, &candidates)
            .unwrap();
        assert_eq!(decision.candidates.len(), candidates.len());
        assert_eq!(decision.probabilities.len(), candidates.len());

        controller.observe_reward(decision, 1.0).unwrap();
        let report = controller.finish_episode().unwrap();
        assert_eq!(report.steps, 1);
        assert_eq!(controller.telemetry().buffered_steps, 0);
    }

    #[test]
    fn candidate_mismatch_surfaces_error() {
        let rec = build_basic_recommender();
        let controller = RecBanditController::new(&rec, 3, 0.01, 0.9).unwrap();
        let err = controller
            .select_recommendation(&rec, 0, &[0, 1])
            .unwrap_err();
        match err {
            RecRlError::CandidateMismatch { expected, provided } => {
                assert_eq!(expected, 3);
                assert_eq!(provided, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn sampled_candidate_selection_respects_action_space() {
        let rec = build_basic_recommender();
        let controller = RecBanditController::new(&rec, 3, 0.01, 0.9).unwrap();
        let decision = controller
            .sample_recommendation(&rec, 0, &[0, 1, 2])
            .expect("sampled decision");
        assert_eq!(decision.candidates.len(), 3);
        assert_eq!(decision.probabilities.len(), 3);
        assert!(decision.action_index < 3);
    }

    #[test]
    fn top_k_selection_routes_through_policy() {
        let rec = build_basic_recommender();
        let mut controller = RecBanditController::new(&rec, 3, 0.01, 0.9).unwrap();
        let decision = controller
            .select_top_k(&rec, 0, 3, None)
            .expect("top-k decision");
        assert_eq!(decision.candidates.len(), 3);
        assert!(decision.action_index < 3);

        controller.observe_reward(decision, 0.5).unwrap();
        let report = controller.finish_episode().unwrap();
        assert_eq!(report.steps, 1);
    }

    #[test]
    fn top_k_mismatch_surfaces_error() {
        let rec = build_basic_recommender();
        let controller = RecBanditController::new(&rec, 4, 0.01, 0.9).unwrap();
        let err = controller
            .select_top_k(&rec, 0, 3, None)
            .expect_err("mismatched top-k should error");
        match err {
            RecRlError::CandidateMismatch { expected, provided } => {
                assert_eq!(expected, 4);
                assert_eq!(provided, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn sampled_top_k_routes_through_policy() {
        let rec = build_basic_recommender();
        let controller = RecBanditController::new(&rec, 3, 0.01, 0.9).unwrap();
        let decision = controller
            .sample_top_k(&rec, 0, 3, None)
            .expect("sampled top-k decision");
        assert_eq!(decision.candidates.len(), 3);
        assert!(decision.action_index < 3);
    }

    #[test]
    fn sampled_top_k_mismatch_surfaces_error() {
        let rec = build_basic_recommender();
        let controller = RecBanditController::new(&rec, 4, 0.01, 0.9).unwrap();
        let err = controller
            .sample_top_k(&rec, 0, 3, None)
            .expect_err("mismatched sample top-k should error");
        match err {
            RecRlError::CandidateMismatch { expected, provided } => {
                assert_eq!(expected, 4);
                assert_eq!(provided, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
