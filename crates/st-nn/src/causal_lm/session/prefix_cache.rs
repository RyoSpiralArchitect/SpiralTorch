//! Caller-owned CPU activations strictly before the earliest attached adapter.
use super::*;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmPrefixCacheAction {
    Hit,
    Inserted,
    CapacityBypass,
    NoFrozenBlock,
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct CausalLmPrefixCacheStats {
    pub max_values: usize,
    pub max_entries: usize,
    pub retained_values: usize,
    pub entries: usize,
    pub hits: u64,
    pub insertions: u64,
    pub capacity_bypasses: u64,
    pub no_frozen_block: u64,
    /// Prefix block invocations avoided beyond the existing within-step sharing.
    pub skipped_block_calls: u64,
    pub last_action: Option<CausalLmPrefixCacheAction>,
}

/// Immutable base/readout binding, bounded f32 payload, no serialization or activation export.
/// Keeps the first entries that fit; a sequential sweep larger than the cache need not thrash.
pub struct NativeCausalLmTrainingPrefixCache {
    base_id: String,
    policy: CausalLmReadoutAccumulator,
    hidden: BTreeMap<[u8; 32], Tensor>,
    stats: CausalLmPrefixCacheStats,
}

impl fmt::Debug for NativeCausalLmTrainingPrefixCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeCausalLmTrainingPrefixCache")
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

pub(super) enum Pending {
    Hit(usize),
    Insert([u8; 32], Tensor),
    Capacity,
    NoDepth,
}

impl NativeCausalLmTrainingPrefixCache {
    pub const MAX_VALUES: usize = 64 * 1024 * 1024;
    pub const MAX_ENTRIES: usize = 16_384;

    pub fn new(
        model: &NativeCausalLm,
        max_values: usize,
        max_entries: usize,
    ) -> CausalLmResult<Self> {
        if !(1..=Self::MAX_VALUES).contains(&max_values)
            || !(1..=Self::MAX_ENTRIES).contains(&max_entries)
        {
            return Err("invalid frozen training prefix cache budget".into());
        }
        Ok(Self {
            base_id: model.base_id.clone(),
            policy: model.readout_accumulator,
            hidden: BTreeMap::new(),
            stats: CausalLmPrefixCacheStats {
                max_values,
                max_entries,
                ..Default::default()
            },
        })
    }

    pub fn stats(&self) -> CausalLmPrefixCacheStats {
        self.stats
    }

    /// Drop all cached hidden states and counters without changing the bound base or budgets.
    pub fn clear(&mut self) {
        self.hidden.clear();
        self.stats = CausalLmPrefixCacheStats {
            max_values: self.stats.max_values,
            max_entries: self.stats.max_entries,
            ..Default::default()
        };
    }

    pub(super) fn admit(&self, model: &NativeCausalLm) -> CausalLmResult<()> {
        if self.base_id != model.base_id || self.policy != model.readout_accumulator {
            return Err(
                "training prefix cache belongs to a different frozen base or readout policy".into(),
            );
        }
        Ok(())
    }

    pub(super) fn resolve(
        &self,
        depth: usize,
        p: &PreparedCausalWindow,
        width: usize,
        compute: impl FnOnce() -> CausalLmResult<Tensor>,
    ) -> CausalLmResult<(Tensor, Pending)> {
        if depth == 0 {
            return Ok((compute()?, Pending::NoDepth));
        }
        let values = p
            .input
            .len()
            .checked_mul(width)
            .ok_or("training prefix cache shape overflow")?;
        if values > self.stats.max_values {
            return Ok((compute()?, Pending::Capacity));
        }
        // Position resets and attention are per row. Flattened IDs alone cannot distinguish batch layouts.
        let mut h = Sha256::new();
        h.update(b"spiraltorch.native_frozen_training_prefix.v1\0");
        for n in [depth, p.shape[0], p.shape[1]] {
            h.update((n as u64).to_le_bytes());
        }
        for &id in &p.input {
            h.update((id as u64).to_le_bytes());
        }
        let key: [u8; 32] = h.finalize().into();
        if let Some(value) = self.hidden.get(&key) {
            return Ok((value.clone(), Pending::Hit(depth)));
        }
        let hidden = compute()?;
        if self.hidden.len() == self.stats.max_entries
            || values > self.stats.max_values - self.stats.retained_values
        {
            return Ok((hidden, Pending::Capacity));
        }
        Ok((hidden.clone(), Pending::Insert(key, hidden)))
    }

    // Commit only after every candidate validates. Failure leaves counters and retained activations untouched.
    pub(super) fn commit(&mut self, pending: Pending) -> CausalLmPrefixCacheAction {
        let s = &mut self.stats;
        let action = match pending {
            Pending::Hit(depth) => {
                s.hits = s.hits.saturating_add(1);
                s.skipped_block_calls = s.skipped_block_calls.saturating_add(depth as u64);
                CausalLmPrefixCacheAction::Hit
            }
            Pending::Insert(k, h) => {
                s.retained_values += h.data().len();
                self.hidden.insert(k, h);
                s.entries = self.hidden.len();
                s.insertions = s.insertions.saturating_add(1);
                CausalLmPrefixCacheAction::Inserted
            }
            Pending::Capacity => {
                s.capacity_bypasses = s.capacity_bypasses.saturating_add(1);
                CausalLmPrefixCacheAction::CapacityBypass
            }
            Pending::NoDepth => {
                s.no_frozen_block = s.no_frozen_block.saturating_add(1);
                CausalLmPrefixCacheAction::NoFrozenBlock
            }
        };
        s.last_action = Some(action);
        action
    }
}
