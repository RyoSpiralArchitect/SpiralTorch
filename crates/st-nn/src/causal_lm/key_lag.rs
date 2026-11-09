use super::*;
use sha2::{Digest, Sha256};

/// Pair a value at j with the projected key at max(0, j-lag), in one block.
/// The structural mask, values and optional score biases remain at j.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalLmKeyLag {
    pub layer: usize,
    pub lag: usize,
}

impl CausalLmKeyLag {
    pub const MODEL_TYPE: &'static str = "spiraltorch_gpt2_key_lag_v1";
    pub(super) fn validate(self, layers: usize, positions: usize) -> CausalLmResult<()> {
        if self.layer >= layers || self.lag == 0 || self.lag >= positions {
            return Err("key lag requires an existing block and 0 < lag < positions".into());
        }
        Ok(())
    }
}

impl NativeCausalLm {
    pub fn key_lag(&self) -> Option<CausalLmKeyLag> {
        self.key_lag
    }
    pub(super) fn key_lag_at(&self, layer: usize) -> usize {
        self.key_lag
            .filter(|x| x.layer == layer)
            .map_or(0, |x| x.lag)
    }

    /// Select the architecture before attaching any adapters. Its base, gradients,
    /// serving prefixes and training checkpoints cannot masquerade as ordinary GPT-2.
    pub fn with_key_lag(&self, policy: CausalLmKeyLag) -> CausalLmResult<Self> {
        policy.validate(self.config.layers, self.config.positions)?;
        if self.key_lag == Some(policy) {
            return Ok(self.clone());
        }
        if self.key_lag.is_some() || self.lora.is_some() || self.geometry.is_some() {
            return Err("key lag must be selected on an unadapted ordinary base".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"spiraltorch.native_causal_lm.gpt2.key_lag.v1\0");
        hash.update(self.base_id.as_bytes());
        hash.update((policy.layer as u64).to_le_bytes());
        hash.update((policy.lag as u64).to_le_bytes());
        let mut m = self.clone();
        m.key_lag = Some(policy);
        m.base_id = format!("sha256:{:x}", hash.finalize());
        Ok(m)
    }
}
