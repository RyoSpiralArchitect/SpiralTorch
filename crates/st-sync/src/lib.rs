//! Phase synchronisation utilities for combining multiple refract maps.

#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use anyhow::{anyhow, Result};
use st_softlogic::spiralk::ir::SyncBlock;
use std::collections::HashSet;

/// Metrics describing how well a candidate execution graph satisfies the
/// requested synchronisation constraints.
#[derive(Clone, Debug)]
pub struct PhaseCandidate {
    pub graph_id: String,
    pub phase_delta: f32,
    pub trust: f32,
    pub resonance: f32,
}

/// Result of selecting a candidate after applying the prioritisation rules.
#[derive(Clone, Debug)]
pub struct PhaseSelection {
    pub graph_id: String,
    pub pairs: Vec<String>,
    pub score: f32,
    pub phase_delta: f32,
    pub trust: f32,
    pub resonance: f32,
}

/// Synchronisation engine that merges multiple refract maps subject to phase
/// tolerance constraints.
#[derive(Clone, Copy, Debug)]
pub struct PhaseSync {
    pub default_tolerance: f32,
}

impl PhaseSync {
    #[must_use]
    pub fn new(default_tolerance: f32) -> Self {
        Self { default_tolerance }
    }

    /// Creates a phase synchroniser after validating its default tolerance.
    pub fn try_new(default_tolerance: f32) -> Result<Self> {
        validate_nonnegative_finite("default tolerance", default_tolerance)?;
        Ok(Self { default_tolerance })
    }

    /// Merge the candidate graphs using the heuristics outlined in the design
    /// document.  Candidates within the tolerance window are preferred; if none
    /// fit, the best scoring candidate overall is returned.
    pub fn merge(
        &self,
        sync: &SyncBlock,
        candidates: &[PhaseCandidate],
    ) -> Result<Option<PhaseSelection>> {
        self.validate_inputs(sync, candidates)?;
        if candidates.is_empty() {
            return Ok(None);
        }

        let tolerance = if sync.tolerance > 0.0 {
            sync.tolerance
        } else if self.default_tolerance > 0.0 {
            self.default_tolerance
        } else {
            0.0
        };

        let mut best_index: Option<usize> = None;
        let mut best_within = false;
        let mut best_score = f32::NEG_INFINITY;

        for (index, candidate) in candidates.iter().enumerate() {
            let within = tolerance <= 0.0 || candidate.phase_delta <= tolerance;
            let score = score_candidate(candidate, tolerance);

            let choose = match best_index {
                None => true,
                Some(best_idx) => {
                    if within && !best_within {
                        true
                    } else if within == best_within {
                        if score > best_score + f32::EPSILON {
                            true
                        } else if (score - best_score).abs() <= f32::EPSILON {
                            tie_break(candidate, &candidates[best_idx])
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
            };

            if choose {
                best_index = Some(index);
                best_within = within;
                best_score = score;
            }
        }

        let Some(index) = best_index else {
            return Ok(None);
        };
        let candidate = &candidates[index];
        Ok(Some(PhaseSelection {
            graph_id: candidate.graph_id.clone(),
            pairs: sync.pairs.clone(),
            score: best_score,
            phase_delta: candidate.phase_delta,
            trust: candidate.trust,
            resonance: candidate.resonance,
        }))
    }

    fn validate_inputs(&self, sync: &SyncBlock, candidates: &[PhaseCandidate]) -> Result<()> {
        if sync.name.trim().is_empty() {
            return Err(anyhow!("sync block name must not be empty"));
        }
        if sync.pairs.is_empty() {
            return Err(anyhow!(
                "sync block '{}' must specify at least one pair",
                sync.name
            ));
        }
        let mut pairs = HashSet::with_capacity(sync.pairs.len());
        for (index, pair) in sync.pairs.iter().enumerate() {
            if pair.trim().is_empty() {
                return Err(anyhow!(
                    "sync block '{}' contains an empty pair at index {index}",
                    sync.name
                ));
            }
            if !pairs.insert(pair.as_str()) {
                return Err(anyhow!(
                    "sync block '{}' contains duplicate pair '{pair}'",
                    sync.name
                ));
            }
        }
        validate_nonnegative_finite("sync tolerance", sync.tolerance)?;
        validate_nonnegative_finite("default tolerance", self.default_tolerance)?;

        let mut graph_ids = HashSet::with_capacity(candidates.len());
        for (index, candidate) in candidates.iter().enumerate() {
            if candidate.graph_id.trim().is_empty() {
                return Err(anyhow!("candidate at index {index} has an empty graph id"));
            }
            if !graph_ids.insert(candidate.graph_id.as_str()) {
                return Err(anyhow!(
                    "candidate graph id '{}' is duplicated",
                    candidate.graph_id
                ));
            }
            validate_nonnegative_finite("candidate phase delta", candidate.phase_delta)
                .map_err(|error| anyhow!("candidate '{}': {error}", candidate.graph_id))?;
            validate_finite("candidate trust", candidate.trust)
                .map_err(|error| anyhow!("candidate '{}': {error}", candidate.graph_id))?;
            validate_finite("candidate resonance", candidate.resonance)
                .map_err(|error| anyhow!("candidate '{}': {error}", candidate.graph_id))?;
        }
        Ok(())
    }
}

impl Default for PhaseSync {
    fn default() -> Self {
        Self {
            default_tolerance: 0.05,
        }
    }
}

fn score_candidate(candidate: &PhaseCandidate, tolerance: f32) -> f32 {
    let trust = candidate.trust.clamp(0.0, 1.0);
    let phase_term = if tolerance > 0.0 {
        ((tolerance - candidate.phase_delta) / tolerance).clamp(0.0, 1.0)
    } else {
        1.0 / (1.0 + candidate.phase_delta)
    };
    let resonance_term = candidate.resonance.tanh().max(0.0);
    0.6 * trust + 0.3 * phase_term + 0.1 * resonance_term
}

fn validate_finite(label: &str, value: f32) -> Result<()> {
    if !value.is_finite() {
        return Err(anyhow!("{label} must be finite"));
    }
    Ok(())
}

fn validate_nonnegative_finite(label: &str, value: f32) -> Result<()> {
    validate_finite(label, value)?;
    if value < 0.0 {
        return Err(anyhow!("{label} must not be negative"));
    }
    Ok(())
}

fn tie_break(current: &PhaseCandidate, best: &PhaseCandidate) -> bool {
    if current.trust > best.trust + f32::EPSILON {
        return true;
    }
    if (current.trust - best.trust).abs() <= f32::EPSILON
        && current.phase_delta + f32::EPSILON < best.phase_delta
    {
        return true;
    }
    if (current.trust - best.trust).abs() <= f32::EPSILON
        && (current.phase_delta - best.phase_delta).abs() <= f32::EPSILON
        && current.resonance > best.resonance
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sync_block(tolerance: f32) -> SyncBlock {
        SyncBlock {
            name: "merge".into(),
            pairs: vec!["A".into(), "B".into()],
            tolerance,
        }
    }

    fn candidate(graph_id: &str) -> PhaseCandidate {
        PhaseCandidate {
            graph_id: graph_id.into(),
            phase_delta: 0.03,
            trust: 0.8,
            resonance: 0.4,
        }
    }

    #[test]
    fn prefers_candidates_within_tolerance() {
        let sync_block = SyncBlock {
            name: "merge".into(),
            pairs: vec!["A".into(), "B".into()],
            tolerance: 0.06,
        };

        let candidates = vec![
            PhaseCandidate {
                graph_id: "g1".into(),
                phase_delta: 0.03,
                trust: 0.82,
                resonance: 0.4,
            },
            PhaseCandidate {
                graph_id: "g2".into(),
                phase_delta: 0.08,
                trust: 0.95,
                resonance: 0.2,
            },
        ];

        let sync = PhaseSync::new(0.05);
        let selection = sync.merge(&sync_block, &candidates).unwrap().unwrap();
        assert_eq!(selection.graph_id, "g1");
        assert!((selection.score - score_candidate(&candidates[0], 0.06)).abs() < 1e-3);
    }

    #[test]
    fn falls_back_to_best_score_when_outside_tolerance() {
        let sync_block = SyncBlock {
            name: "merge".into(),
            pairs: vec!["A".into(), "B".into()],
            tolerance: 0.01,
        };

        let candidates = vec![
            PhaseCandidate {
                graph_id: "g1".into(),
                phase_delta: 0.03,
                trust: 0.6,
                resonance: 0.4,
            },
            PhaseCandidate {
                graph_id: "g2".into(),
                phase_delta: 0.04,
                trust: 0.9,
                resonance: 0.1,
            },
        ];

        let sync = PhaseSync::default();
        let selection = sync.merge(&sync_block, &candidates).unwrap().unwrap();
        assert_eq!(selection.graph_id, "g2");
    }

    #[test]
    fn rejects_non_finite_and_negative_tolerances() {
        for tolerance in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01] {
            assert!(PhaseSync::try_new(tolerance).is_err());
            assert!(PhaseSync::default()
                .merge(&sync_block(tolerance), &[candidate("g1")])
                .is_err());
        }

        let sync = sync_block(0.05);
        for default_tolerance in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01] {
            assert!(PhaseSync::new(default_tolerance)
                .merge(&sync, &[candidate("g1")])
                .is_err());
        }
    }

    #[test]
    fn rejects_invalid_candidate_scalars() {
        let sync = sync_block(0.05);
        let cases = [
            PhaseCandidate {
                phase_delta: f32::NAN,
                ..candidate("nan-phase")
            },
            PhaseCandidate {
                phase_delta: -0.01,
                ..candidate("negative-phase")
            },
            PhaseCandidate {
                trust: f32::INFINITY,
                ..candidate("infinite-trust")
            },
            PhaseCandidate {
                resonance: f32::NEG_INFINITY,
                ..candidate("infinite-resonance")
            },
        ];

        for invalid in cases {
            assert!(PhaseSync::default().merge(&sync, &[invalid]).is_err());
        }
    }

    #[test]
    fn rejects_empty_and_duplicate_identities() {
        let mut sync = sync_block(0.05);
        sync.name.clear();
        assert!(PhaseSync::default().merge(&sync, &[]).is_err());

        let mut sync = sync_block(0.05);
        sync.pairs = vec!["A".into(), "A".into()];
        assert!(PhaseSync::default()
            .merge(&sync, &[candidate("g1")])
            .is_err());

        let sync = sync_block(0.05);
        assert!(PhaseSync::default()
            .merge(&sync, &[candidate("g1"), candidate("g1")])
            .is_err());
        assert!(PhaseSync::default()
            .merge(&sync, &[candidate("  ")])
            .is_err());
    }

    #[test]
    fn empty_candidates_return_none_after_configuration_validation() {
        let selection = PhaseSync::try_new(0.0)
            .unwrap()
            .merge(&sync_block(0.0), &[])
            .unwrap();
        assert!(selection.is_none());
    }

    #[test]
    fn finite_extremes_produce_a_bounded_score() {
        let sync = sync_block(f32::MAX);
        let extreme = PhaseCandidate {
            graph_id: "extreme".into(),
            phase_delta: f32::MAX,
            trust: f32::MAX,
            resonance: f32::MAX,
        };

        let selection = PhaseSync::default()
            .merge(&sync, &[extreme])
            .unwrap()
            .unwrap();
        assert!(selection.score.is_finite());
        assert!((0.0..=1.0).contains(&selection.score));
        assert!(selection.phase_delta.is_finite());
        assert!(selection.trust.is_finite());
        assert!(selection.resonance.is_finite());
    }
}
