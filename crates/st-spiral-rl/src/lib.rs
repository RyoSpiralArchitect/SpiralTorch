// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use st_core::telemetry::hub;
use st_tensor::{AmegaHypergrad, DifferentialResonance, Tensor, TensorError};

pub mod algorithms;
mod geometry;
pub mod schedules;

pub use algorithms::{DqnAgent, PpoAgent, SacAgent};
pub use geometry::{
    GeometryFeedback, GeometryFeedbackConfig, GeometryFeedbackSignal, GeometryTelemetry,
};

/// Reinforcement learning specific error wrapper so callers can surface
/// meaningful diagnostics without inspecting tensor internals.
#[derive(Debug)]
pub enum SpiralRlError {
    /// Wrapped tensor failure bubbling up from the core math routines.
    Tensor(TensorError),
    /// Episode bookkeeping was requested without any recorded transition.
    EmptyEpisode,
    /// Input state does not match the dimensionality expected by the policy.
    InvalidStateShape {
        expected: usize,
        rows: usize,
        cols: usize,
    },
    /// Action index exceeded the configured policy range.
    InvalidAction { action: usize, actions: usize },
    /// Discount factor must stay within the closed interval [0, 1].
    InvalidDiscount { discount: f32 },
    /// Batched updates received tensors with mismatched lengths.
    InvalidBatch {
        expected: usize,
        actions: usize,
        rewards: usize,
        next_states: usize,
        dones: Option<usize>,
    },
    /// State dictionary restore encountered a mismatched tensor length.
    StateDictShape {
        field: &'static str,
        expected: usize,
        received: usize,
    },
}

impl fmt::Display for SpiralRlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpiralRlError::Tensor(err) => write!(f, "{err}"),
            SpiralRlError::EmptyEpisode => write!(
                f,
                "no transitions recorded — call record_transition before finishing the episode",
            ),
            SpiralRlError::InvalidStateShape {
                expected,
                rows,
                cols,
            } => write!(
                f,
                "state tensor must be shaped as (1, {expected}) but received ({rows}, {cols})"
            ),
            SpiralRlError::InvalidAction { action, actions } => {
                write!(
                    f,
                    "action index {action} exceeds configured action count {actions}"
                )
            }
            SpiralRlError::InvalidDiscount { discount } => {
                write!(f, "discount factor must lie in [0, 1]; received {discount}")
            }
            SpiralRlError::InvalidBatch {
                expected,
                actions,
                rewards,
                next_states,
                dones,
            } => write!(
                f,
                "batch inputs must have consistent length {expected}; received actions={actions}, rewards={rewards}, next_states={next_states}, dones={:?}",
                dones
            ),
            SpiralRlError::StateDictShape {
                field,
                expected,
                received,
            } => write!(
                f,
                "state dictionary field '{field}' expected length {expected} but received {received}",
            ),
        }
    }
}

impl std::error::Error for SpiralRlError {}

impl From<TensorError> for SpiralRlError {
    fn from(value: TensorError) -> Self {
        SpiralRlError::Tensor(value)
    }
}

/// Convenient result alias for reinforcement learning helpers.
pub type RlResult<T> = Result<T, SpiralRlError>;

#[derive(Clone, Debug)]
struct Transition {
    state: Tensor,
    action: usize,
    reward: f32,
    probs: Vec<f32>,
}

/// Summary emitted after applying a policy gradient update.
#[derive(Clone, Debug)]
pub struct EpisodeReport {
    /// Total undiscounted reward accumulated over the buffered episode.
    pub total_reward: f32,
    /// Mean discounted return (baseline) used during the most recent update.
    pub mean_return: f32,
    /// Number of transitions folded into the update.
    pub steps: usize,
    /// Flag describing whether a hypergrad tape applied the update.
    pub hypergrad_applied: bool,
}

/// Snapshot of trainer diagnostics for external telemetry collectors.
#[derive(Clone, Debug)]
pub struct PolicyTelemetry {
    /// Number of transitions currently buffered for the active episode.
    pub buffered_steps: usize,
    /// Geometry feedback telemetry, if the controller is attached.
    pub geometry: Option<GeometryTelemetry>,
}

/// Lightweight policy gradient learner that keeps every primitive inside
/// SpiralTorch's Z-space tensor core.
pub struct SpiralPolicyGradient {
    state_dim: usize,
    action_dim: usize,
    learning_rate: f32,
    discount: f32,
    weights: Tensor,
    bias: Vec<f32>,
    episode: Vec<Transition>,
    hypergrad: Option<AmegaHypergrad>,
    geometry_feedback: Option<GeometryFeedback>,
}

impl SpiralPolicyGradient {
    /// Constructs a new policy gradient learner with the provided geometry.
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f32,
        discount: f32,
    ) -> RlResult<Self> {
        if state_dim == 0 || action_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: state_dim,
                cols: action_dim,
            }
            .into());
        }
        if learning_rate <= 0.0 {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            }
            .into());
        }
        if !(0.0..=1.0).contains(&discount) {
            return Err(SpiralRlError::InvalidDiscount { discount });
        }

        let weights = Tensor::from_fn(state_dim, action_dim, |row, col| {
            let seed = (row as f32 + 1.0) * (col as f32 + 1.0);
            (seed.cos() * 0.01).clamp(-0.05, 0.05)
        })?;
        let bias = vec![0.0; action_dim];

        Ok(Self {
            state_dim,
            action_dim,
            learning_rate,
            discount,
            weights,
            bias,
            episode: Vec::new(),
            hypergrad: None,
            geometry_feedback: None,
        })
    }

    /// Enables a hypergradient tape so updates follow the same hyperbolic
    /// curvature employed by the rest of SpiralTorch.
    pub fn enable_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> RlResult<()> {
        let tape = AmegaHypergrad::new(curvature, learning_rate, self.state_dim, self.action_dim)?;
        self.hypergrad = Some(tape);
        Ok(())
    }

    /// Resets the buffered episode without touching the parameters.
    pub fn reset_episode(&mut self) {
        self.episode.clear();
    }

    /// Attaches a geometric observability feedback controller to the policy.
    pub fn attach_geometry_feedback(&mut self, feedback: GeometryFeedback) {
        self.geometry_feedback = Some(feedback);
    }

    /// Detaches the currently configured geometric feedback controller.
    pub fn detach_geometry_feedback(&mut self) -> Option<GeometryFeedback> {
        self.geometry_feedback.take()
    }

    /// Returns the latest feedback signal emitted by the controller, if any.
    pub fn last_geometry_signal(&self) -> Option<&GeometryFeedbackSignal> {
        self.geometry_feedback
            .as_ref()
            .and_then(|feedback| feedback.last_signal())
    }

    /// Returns rolling telemetry so trainers can monitor rank/pressure drift.
    pub fn telemetry(&self) -> PolicyTelemetry {
        let geometry = self
            .geometry_feedback
            .as_ref()
            .map(|feedback| feedback.telemetry().clone());
        PolicyTelemetry {
            buffered_steps: self.episode.len(),
            geometry,
        }
    }

    fn ensure_state_shape(&self, state: &Tensor) -> RlResult<()> {
        let (rows, cols) = state.shape();
        if rows != 1 || cols != self.state_dim {
            return Err(SpiralRlError::InvalidStateShape {
                expected: self.state_dim,
                rows,
                cols,
            });
        }
        Ok(())
    }

    fn logits(&self, state: &Tensor) -> RlResult<Vec<f32>> {
        self.ensure_state_shape(state)?;
        let mut logits = state.matmul(&self.weights)?.data().to_vec();
        for (logit, bias) in logits.iter_mut().zip(self.bias.iter()) {
            *logit += *bias;
        }
        Ok(logits)
    }

    fn softmax(&self, logits: Vec<f32>) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits
            .into_iter()
            .map(|logit| (logit - max).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        if sum <= f32::EPSILON {
            let uniform = 1.0 / self.action_dim as f32;
            return vec![uniform; self.action_dim];
        }
        for value in exps.iter_mut() {
            *value /= sum;
        }
        exps
    }

    /// Returns the policy probabilities for the provided state.
    pub fn policy(&self, state: &Tensor) -> RlResult<Vec<f32>> {
        let logits = self.logits(state)?;
        Ok(self.softmax(logits))
    }

    /// Selects the greedy action for the provided state.
    pub fn select_action(&self, state: &Tensor) -> RlResult<usize> {
        let probs = self.policy(state)?;
        let (action, _) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok(action)
    }

    /// Records a transition (state, action, reward) for the current episode.
    pub fn record_transition(&mut self, state: Tensor, action: usize, reward: f32) -> RlResult<()> {
        if action >= self.action_dim {
            return Err(SpiralRlError::InvalidAction {
                action,
                actions: self.action_dim,
            });
        }
        let probs = self.policy(&state)?;
        self.episode.push(Transition {
            state,
            action,
            reward,
            probs,
        });
        Ok(())
    }

    fn discounted_returns(&self) -> Vec<f32> {
        let mut returns = Vec::with_capacity(self.episode.len());
        let mut running = 0.0f32;
        for step in self.episode.iter().rev() {
            running = step.reward + self.discount * running;
            returns.push(running);
        }
        returns.reverse();
        returns
    }

    /// Applies a policy gradient update using the buffered transitions.
    pub fn finish_episode(&mut self) -> RlResult<EpisodeReport> {
        self.finish_episode_with_rate(self.learning_rate)
    }

    /// Applies an update modulated by a geometric resonance measurement.
    pub fn finish_episode_with_geometry(
        &mut self,
        resonance: &DifferentialResonance,
    ) -> RlResult<(EpisodeReport, Option<GeometryFeedbackSignal>)> {
        let (scale, signal) = if let Some(controller) = self.geometry_feedback.as_mut() {
            #[allow(unused_mut)]
            let mut loop_injected = false;
            let envelopes = hub::drain_loopback_envelopes(8);
            if !envelopes.is_empty() {
                controller.absorb_loopback(&envelopes);
                loop_injected = true;
            }
            #[cfg(feature = "collapse")]
            {
                if let Some(pulse) = hub::get_collapse_pulse() {
                    controller.inject_collapse_bias(pulse.total);
                    if let Some(signal) = pulse.loop_signal {
                        controller.integrate_loop_signal(&signal);
                        loop_injected = true;
                    }
                }
            }
            if !loop_injected {
                if let Some(signal) = hub::get_chrono_loop() {
                    controller.integrate_loop_signal(&signal);
                }
            }
            let signal = controller.process_resonance(resonance);
            let envelope = controller.emit_loopback_envelope(&signal, Some("st-spiral-rl.policy"));
            hub::push_loopback_envelope(envelope);
            (signal.learning_rate_scale.max(f32::EPSILON), Some(signal))
        } else {
            (1.0, None)
        };
        let effective_rate = (self.learning_rate * scale).max(f32::EPSILON);
        let report = self.finish_episode_with_rate(effective_rate)?;
        Ok((report, signal))
    }

    fn finish_episode_with_rate(&mut self, learning_rate: f32) -> RlResult<EpisodeReport> {
        if self.episode.is_empty() {
            return Err(SpiralRlError::EmptyEpisode);
        }
        if !(learning_rate.is_finite()) || learning_rate <= 0.0 {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            }
            .into());
        }
        let returns = self.discounted_returns();
        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;

        let mut weight_grad = vec![0.0f32; self.state_dim * self.action_dim];
        let mut bias_grad = vec![0.0f32; self.action_dim];
        let mut total_reward = 0.0f32;

        for (step, &ret) in self.episode.iter().zip(returns.iter()) {
            total_reward += step.reward;
            let advantage = ret - baseline;
            for action in 0..self.action_dim {
                let prob = step.probs[action];
                let indicator = if action == step.action { 1.0 } else { 0.0 };
                let grad_coeff = (indicator - prob) * advantage * learning_rate;
                bias_grad[action] += grad_coeff;
                for feature in 0..self.state_dim {
                    let idx = feature * self.action_dim + action;
                    weight_grad[idx] += step.state.data()[feature] * grad_coeff;
                }
            }
        }

        for (bias, grad) in self.bias.iter_mut().zip(bias_grad.iter()) {
            *bias += *grad;
        }

        let mut hypergrad_applied = false;
        if let Some(tape) = self.hypergrad.as_mut() {
            let grad_tensor =
                Tensor::from_vec(self.state_dim, self.action_dim, weight_grad.clone())?;
            tape.accumulate_wave(&grad_tensor)?;
            tape.apply(&mut self.weights)?;
            hypergrad_applied = true;
        } else {
            for (weight, grad) in self.weights.data_mut().iter_mut().zip(weight_grad.iter()) {
                *weight += *grad;
            }
        }

        self.episode.clear();

        Ok(EpisodeReport {
            total_reward,
            mean_return: baseline,
            steps: returns.len(),
            hypergrad_applied,
        })
    }

    /// Returns a snapshot of the policy weight matrix.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Returns a copy of the current bias vector.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::telemetry::hub;
    use st_core::theory::observability::{ObservabilityConfig, SlotSymmetry};

    fn resonance_from(values: &[f32]) -> DifferentialResonance {
        let tensor = Tensor::from_vec(1, values.len(), values.to_vec()).unwrap();
        DifferentialResonance {
            homotopy_flow: tensor.clone(),
            functor_linearisation: tensor.clone(),
            recursive_objective: tensor.clone(),
            infinity_projection: tensor.clone(),
            infinity_energy: tensor,
        }
    }

    #[test]
    fn telemetry_surfaces_geometry_feedback() -> Result<(), SpiralRlError> {
        let _ = hub::drain_loopback_envelopes(usize::MAX);
        let mut policy = SpiralPolicyGradient::new(2, 2, 0.01, 0.9)?;
        let snapshot = policy.telemetry();
        assert_eq!(snapshot.buffered_steps, 0);
        assert!(snapshot.geometry.is_none());

        let feedback = GeometryFeedback::new(GeometryFeedbackConfig {
            observability: ObservabilityConfig::new(1, 5, SlotSymmetry::Symmetric),
            ..GeometryFeedbackConfig::default_policy()
        });
        policy.attach_geometry_feedback(feedback);

        let state = Tensor::from_vec(1, 2, vec![0.4, -0.2]).unwrap();
        policy.record_transition(state.clone(), 0, 1.0)?;
        policy.record_transition(state.clone(), 1, 0.5)?;
        let resonance = resonance_from(&[0.6, -0.4, 0.2, -0.1, 0.3]);
        let (_report, signal) = policy.finish_episode_with_geometry(&resonance)?;
        assert!(signal.is_some());

        let telemetry = policy.telemetry();
        assert!(telemetry.geometry.is_some());
        let geo = telemetry.geometry.unwrap();
        assert!(geo.rolling_scale >= 0.0);
        assert!(geo.max_scale <= 3.0);
        assert!(geo.loop_gain >= 0.0);
        assert!(geo.softening_beta >= 0.3);
        let drained = hub::drain_loopback_envelopes(8);
        assert!(!drained.is_empty());
        let broadcast = drained.last().unwrap();
        assert_eq!(broadcast.source.as_deref(), Some("st-spiral-rl.policy"));
        assert!(broadcast.z_signal.is_some());
        Ok(())
    }
}
