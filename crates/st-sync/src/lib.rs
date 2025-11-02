//! Phase synchronisation utilities for combining multiple refract maps.

use anyhow::{anyhow, Result};
use st_softlogic::spiralk::ir::SyncBlock;

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
    pub fn new(default_tolerance: f32) -> Self {
        Self { default_tolerance }
    }

    /// Merge the candidate graphs using the heuristics outlined in the design
    /// document.  Candidates within the tolerance window are preferred; if none
    /// fit, the best scoring candidate overall is returned.
    pub fn merge(
        &self,
        sync: &SyncBlock,
        candidates: &[PhaseCandidate],
    ) -> Result<Option<PhaseSelection>> {
        if sync.pairs.is_empty() {
            return Err(anyhow!(
                "sync block '{}' must specify at least one pair",
                sync.name
            ));
        }
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
        ((tolerance - candidate.phase_delta).max(0.0)) / tolerance
    } else {
        1.0 / (1.0 + candidate.phase_delta.abs())
    };
    let resonance_term = candidate.resonance.tanh().max(0.0);
    0.6 * trust + 0.3 * phase_term + 0.1 * resonance_term
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
}
