//! Native generation with explicit full-prefix or owned KV execution, never a foreign fallback.
use super::{CausalLmResult, NativeCausalLm, NativeCausalLmContext, NativeCausalLmTrainer};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use st_core::inference::token_decoding::{
    prepare_token_candidates, TokenDecision, TokenDecodingConfig, TokenSelection, MAX_HISTORY,
};

const SCHEMA: &str = "spiraltorch.native_causal_lm_generation.v1";
const RNG: &str = "rand_chacha_0_3_chacha20_seed_u64_top53.v1";
const MAX_CHECKPOINT_BYTES: usize = 1_048_576;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalLmGenerationExecution {
    #[default]
    FullPrefix,
    KvCache {
        max_values: usize,
    },
}

impl CausalLmGenerationExecution {
    fn is_full_prefix(&self) -> bool {
        *self == Self::FullPrefix
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalLmGenerationConfig {
    pub max_new_tokens: usize,
    #[serde(default)]
    pub eos_token_ids: Vec<usize>,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub decoding: TokenDecodingConfig,
    // Preserve byte serialization of the already-issued full-prefix v1 checkpoints.
    #[serde(
        default,
        skip_serializing_if = "CausalLmGenerationExecution::is_full_prefix"
    )]
    pub execution: CausalLmGenerationExecution,
}

impl CausalLmGenerationConfig {
    pub fn from_json(bytes: &[u8]) -> CausalLmResult<Self> {
        if bytes.len() > 65_536 {
            return Err("generation config exceeds byte budget".into());
        }
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Validate the largest prefix actually evaluated, before extracting foreign IDs.
    pub fn validate(&self, model: &NativeCausalLm, prompt_length: usize) -> CausalLmResult<()> {
        if prompt_length == 0 || self.max_new_tokens == 0 || self.max_new_tokens > 1024 {
            return Err("generation requires a nonempty prompt and 1..=1024 new tokens".into());
        }
        let total = prompt_length
            .checked_add(self.max_new_tokens)
            .filter(|&n| n <= MAX_HISTORY)
            .ok_or("generation token count exceeds budget")?;
        match self.execution {
            CausalLmGenerationExecution::FullPrefix => {
                model.input_count([1, total - 1], true)?;
            }
            CausalLmGenerationExecution::KvCache { max_values } => {
                model.context_prefill_count(prompt_length, max_values)?;
                model.cached_input_count(total - 1, max_values)?;
            }
        }
        self.decoding.validate(model.config.vocabulary, total - 1)?;
        if self.eos_token_ids.len() > 256
            || self.eos_token_ids.iter().enumerate().any(|(i, id)| {
                *id >= model.config.vocabulary || self.eos_token_ids[..i].contains(id)
            })
        {
            return Err("EOS IDs must be unique, in vocabulary, and at most 256".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CausalLmGenerationStep {
    pub step: usize,
    #[serde(flatten)]
    pub decision: TokenDecision,
    pub finish_reason: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct NativeCausalLmGenerator {
    model: NativeCausalLm,
    config: CausalLmGenerationConfig,
    prompt_length: usize,
    tokens: Vec<usize>,
    cache: Option<NativeCausalLmContext>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    rng: String,
    base_model_identity: String,
    geometry_artifact_id: Option<String>,
    config: CausalLmGenerationConfig,
    prompt_length: usize,
    tokens: Vec<usize>,
}

fn identity(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

impl NativeCausalLm {
    /// Restore an owner-validated model for inference, WITHOUT verifying corpus identity
    /// or training completion. Use NativeCausalLmTrainer::from_json to resume learning.
    pub fn from_training_checkpoint(
        base: NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<Self> {
        Ok(
            NativeCausalLmTrainer::prepare_restore(base, bytes, expected_id)?
                .model()
                .clone(),
        )
    }
}

impl NativeCausalLmGenerator {
    pub fn new(
        model: NativeCausalLm,
        prompt: Vec<usize>,
        config: CausalLmGenerationConfig,
    ) -> CausalLmResult<Self> {
        config.validate(&model, prompt.len())?;
        if prompt.iter().any(|&id| id >= model.config.vocabulary) {
            return Err("generation prompt contains an out-of-vocabulary token".into());
        }
        Ok(Self {
            model,
            config,
            prompt_length: prompt.len(),
            tokens: prompt,
            cache: None,
        })
    }
    pub fn config(&self) -> &CausalLmGenerationConfig {
        &self.config
    }
    pub fn token_ids(&self) -> &[usize] {
        &self.tokens
    }
    pub fn generated_ids(&self) -> &[usize] {
        &self.tokens[self.prompt_length..]
    }
    pub fn finish_reason(&self) -> Option<&'static str> {
        if self
            .generated_ids()
            .last()
            .is_some_and(|id| self.config.eos_token_ids.contains(id))
        {
            Some("eos")
        } else if self.generated_ids().len() == self.config.max_new_tokens {
            Some("length")
        } else {
            None
        }
    }
    pub fn finished(&self) -> bool {
        self.finish_reason().is_some()
    }

    pub fn step(&mut self) -> CausalLmResult<Option<CausalLmGenerationStep>> {
        if self.finished() {
            return Ok(None);
        }
        let next_cache = match self.config.execution {
            CausalLmGenerationExecution::FullPrefix => None,
            CausalLmGenerationExecution::KvCache { max_values } => {
                Some(if let Some(cache) = &self.cache {
                    if cache.token_ids() != &self.tokens[..self.tokens.len() - 1] {
                        return Err("cached generation cursor differs from token history".into());
                    }
                    cache.preview(*self.tokens.last().unwrap())?
                } else {
                    // Restore follows the original prefill + incremental chronology,
                    // not a numerically different full-prefix re-prefill of all history.
                    let mut cache = NativeCausalLmContext::new(
                        self.model.clone(),
                        self.tokens[..self.prompt_length].to_vec(),
                        max_values,
                    )?;
                    for &token in &self.tokens[self.prompt_length..] {
                        cache = cache.preview(token)?;
                    }
                    cache
                })
            }
        };
        let logits = match &next_cache {
            Some(cache) => cache.logits_tensor().clone(),
            None => self
                .model
                .logits(&self.tokens, [1, self.tokens.len()], true)?,
        };
        let logits: Vec<_> = logits.data().iter().map(|&v| f64::from(v)).collect();
        let candidates = prepare_token_candidates(&logits, &self.tokens, &self.config.decoding)?;
        let decision = match self.config.decoding.selection {
            TokenSelection::Greedy => candidates.greedy(),
            TokenSelection::Sample => {
                // One u64 (two ChaCha words) per accepted sampled token. Reconstructing
                // this position makes errors atomic and avoids serializing opaque RNG state.
                let mut rng = ChaCha20Rng::seed_from_u64(self.config.seed);
                rng.set_word_pos(self.generated_ids().len() as u128 * 2);
                let uniform = (rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
                candidates.sample(uniform)?
            }
        };
        self.tokens.try_reserve(1)?;
        self.tokens.push(decision.token_id);
        self.cache = next_cache;
        Ok(Some(CausalLmGenerationStep {
            step: self.generated_ids().len(),
            decision,
            finish_reason: self.finish_reason(),
        }))
    }

    pub fn to_json(&self) -> CausalLmResult<String> {
        let json = serde_json::to_string(&Checkpoint {
            schema: SCHEMA.into(),
            rng: RNG.into(),
            base_model_identity: self.model.base_id.clone(),
            geometry_artifact_id: self.model.geometry_artifact_id().map(str::to_owned),
            config: self.config.clone(),
            prompt_length: self.prompt_length,
            tokens: self.tokens.clone(),
        })?;
        if json.len() > MAX_CHECKPOINT_BYTES {
            return Err("generation checkpoint exceeds byte budget".into());
        }
        Ok(json)
    }
    pub fn checkpoint_id(&self) -> CausalLmResult<String> {
        Ok(identity(self.to_json()?.as_bytes()))
    }
    pub fn from_json(
        model: NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<Self> {
        if bytes.len() > MAX_CHECKPOINT_BYTES {
            return Err("generation checkpoint exceeds byte budget".into());
        }
        if identity(bytes) != expected_id {
            return Err("generation checkpoint differs from pinned identity".into());
        }
        let saved: Checkpoint = serde_json::from_slice(bytes)?;
        if saved.schema != SCHEMA
            || saved.rng != RNG
            || saved.base_model_identity != model.base_id
            || saved.geometry_artifact_id.as_deref() != model.geometry_artifact_id()
        {
            return Err("generation schema, RNG, base or geometry owner mismatch".into());
        }
        saved.config.validate(&model, saved.prompt_length)?;
        if saved.tokens.len() < saved.prompt_length
            || saved.tokens.len() - saved.prompt_length > saved.config.max_new_tokens
            || saved.tokens.iter().any(|&id| id >= model.config.vocabulary)
        {
            return Err("invalid generation checkpoint token history".into());
        }
        let generated = &saved.tokens[saved.prompt_length..];
        if generated
            .iter()
            .take(generated.len().saturating_sub(1))
            .any(|id| saved.config.eos_token_ids.contains(id))
        {
            return Err("generation checkpoint continued after EOS".into());
        }
        Ok(Self {
            model,
            config: saved.config,
            prompt_length: saved.prompt_length,
            tokens: saved.tokens,
            cache: None,
        })
    }
}
