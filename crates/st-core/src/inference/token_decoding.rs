//! Bounded token selection shared by native clients, without a framework or RNG owner.
//! Raw-logit top-k precedes control; nucleus filtering follows control. Ties use token ID.

use super::generation_control::{
    apply_zspace_generation_control, zspace_entropy, ZSpaceGenerationControlConfig,
    ZSpaceGenerationControlRequest, ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub type DecodingResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const MAX_VOCABULARY: usize = 1_048_576;
pub const MAX_HISTORY: usize = 16_384;
const MAX_NGRAM_COMPARISONS: usize = 16_777_216;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenSelection {
    #[default]
    Greedy,
    Sample,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TokenDistribution {
    Softmax {
        temperature: f64,
    },
    #[serde(rename = "zspace")]
    ZSpace {
        config: ZSpaceGenerationControlConfig,
    },
}

impl Default for TokenDistribution {
    fn default() -> Self {
        Self::Softmax { temperature: 1.0 }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TokenDecodingConfig {
    pub selection: TokenSelection,
    pub distribution: TokenDistribution,
    pub top_k: Option<usize>,
    pub top_p: f64,
}

impl Default for TokenDecodingConfig {
    fn default() -> Self {
        Self {
            selection: TokenSelection::Greedy,
            distribution: TokenDistribution::default(),
            top_k: None,
            top_p: 1.0,
        }
    }
}

impl TokenDecodingConfig {
    /// Admission before a client copies history or computes logits.
    pub fn validate(&self, vocabulary: usize, history_length: usize) -> DecodingResult<()> {
        if vocabulary == 0 || vocabulary > MAX_VOCABULARY || history_length > MAX_HISTORY {
            return Err("token decoding vocabulary or history exceeds budget".into());
        }
        let candidates = self.top_k.unwrap_or(vocabulary);
        if candidates == 0 || candidates > vocabulary {
            return Err("top_k must be in 1..=vocabulary or null".into());
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be finite in (0, 1]".into());
        }
        match &self.distribution {
            TokenDistribution::Softmax { temperature } => {
                if !temperature.is_finite() || *temperature <= 0.0 {
                    return Err("temperature must be finite and positive".into());
                }
            }
            TokenDistribution::ZSpace { config } => {
                config.resolve()?;
                let history = history_length.min(if config.ngram_window == 0 {
                    config.repression_window
                } else {
                    config.ngram_window
                });
                if config.ngram_size > 1
                    && config.ngram_repression_strength > 0.0
                    && history >= config.ngram_size
                {
                    let comparisons = candidates
                        .checked_mul(history - config.ngram_size + 1)
                        .and_then(|n| n.checked_mul(config.ngram_size - 1));
                    if comparisons.is_none_or(|n| n > MAX_NGRAM_COMPARISONS) {
                        return Err("token decoding ngram comparisons exceed budget".into());
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TokenDecision {
    pub token_id: usize,
    /// Probability under the final filtered distribution, not a greedy likelihood.
    pub probability: f64,
    pub entropy: f64,
    pub control_entropy: f64,
    pub effective_temperature: f64,
    pub raw_top_token: usize,
    pub controlled_top_token: usize,
    pub pre_nucleus_candidates: usize,
    pub candidates: usize,
    pub zspace_contract: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct TokenCandidates {
    // Private to preserve the validated, sorted and normalized distribution.
    ids: Vec<usize>,
    probabilities: Vec<f64>,
    report: TokenDecision,
}

impl TokenCandidates {
    pub fn token_ids(&self) -> &[usize] {
        &self.ids
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    pub fn greedy(&self) -> TokenDecision {
        self.decision(0)
    }
    /// Uniform belongs to the caller's sealed RNG policy. Zero-mass tokens are never selected.
    pub fn sample(&self, uniform: f64) -> DecodingResult<TokenDecision> {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err("sampling uniform must be finite in [0, 1)".into());
        }
        let mut cumulative = 0.0;
        for (i, &probability) in self.probabilities.iter().enumerate() {
            cumulative += probability;
            if uniform < cumulative {
                return Ok(self.decision(i));
            }
        }
        Ok(self.decision(self.ids.len() - 1))
    }
    fn decision(&self, index: usize) -> TokenDecision {
        let mut report = self.report.clone();
        report.token_id = self.ids[index];
        report.probability = self.probabilities[index];
        report
    }
}

fn rank(a: usize, b: usize, scores: &[f64]) -> Ordering {
    // Numeric comparison intentionally treats -0 and +0 as equal.
    scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b))
}

pub fn prepare_token_candidates(
    logits: &[f64],
    history: &[usize],
    config: &TokenDecodingConfig,
) -> DecodingResult<TokenCandidates> {
    config.validate(logits.len(), history.len())?;
    if logits.iter().any(|v| !v.is_finite()) || history.iter().any(|&id| id >= logits.len()) {
        return Err("token decoding requires finite logits and in-vocabulary history".into());
    }
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_unstable_by(|&a, &b| rank(a, b, logits));
    ids.truncate(config.top_k.unwrap_or(logits.len()));
    let raw_top_token = ids[0];
    let scores: Vec<_> = ids.iter().map(|&id| logits[id]).collect();
    let (adjusted, probabilities, effective_temperature, zspace_contract) =
        match &config.distribution {
            TokenDistribution::Softmax { temperature } => {
                let maximum = scores[0];
                let mut weights: Vec<_> = scores
                    .iter()
                    .map(|&v| {
                        let delta = v - maximum;
                        let scaled = if delta.is_infinite() && *temperature >= 1.0 {
                            v / temperature - maximum / temperature
                        } else {
                            delta / temperature
                        };
                        scaled.exp()
                    })
                    .collect();
                let total: f64 = weights.iter().sum();
                for value in &mut weights {
                    *value /= total;
                }
                (scores, weights, *temperature, None)
            }
            TokenDistribution::ZSpace { config } => {
                let controlled = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
                    logits: scores,
                    token_ids: ids.iter().map(|&id| id as u64).collect(),
                    recent_tokens: history.iter().map(|&id| id as u64).collect(),
                    config: config.clone(),
                })?;
                (
                    controlled.adjusted_logits,
                    controlled.probabilities,
                    controlled.effective_temperature,
                    Some(ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION),
                )
            }
        };
    if probabilities.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Err("token distribution produced invalid probabilities".into());
    }
    let control_entropy = zspace_entropy(&probabilities)?;
    let pre_nucleus_candidates = ids.len();
    let mut order: Vec<_> = (0..ids.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        adjusted[b]
            .partial_cmp(&adjusted[a])
            .unwrap()
            .then(ids[a].cmp(&ids[b]))
    });
    let controlled_top_token = ids[order[0]];
    let mut kept_ids = Vec::new();
    let mut kept_probs = Vec::new();
    let mut mass = 0.0;
    for i in order {
        if probabilities[i] == 0.0 {
            continue;
        }
        kept_ids.push(ids[i]);
        kept_probs.push(probabilities[i]);
        mass += probabilities[i];
        if config.top_p < 1.0 && mass >= config.top_p {
            break;
        }
    }
    if !mass.is_finite() || mass <= 0.0 {
        return Err("token distribution has no positive finite mass".into());
    }
    for p in &mut kept_probs {
        *p /= mass;
    }
    let report = TokenDecision {
        token_id: kept_ids[0],
        probability: kept_probs[0],
        entropy: zspace_entropy(&kept_probs)?,
        control_entropy,
        effective_temperature,
        raw_top_token,
        controlled_top_token,
        pre_nucleus_candidates,
        candidates: kept_ids.len(),
        zspace_contract,
    };
    Ok(TokenCandidates {
        ids: kept_ids,
        probabilities: kept_probs,
        report,
    })
}

#[cfg(test)]
#[path = "token_decoding_tests.rs"]
mod tests;
