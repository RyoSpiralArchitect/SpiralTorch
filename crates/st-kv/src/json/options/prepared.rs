// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

use super::JsonSetOptions;
use crate::json::CommandFragment;
use crate::KvResult;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

const PREPARED_CACHE_CAPACITY: usize = 256;

struct PreparedCache {
    capacity: usize,
    entries: HashMap<JsonSetOptions, SharedPreparedJsonSetOptions>,
    insertion_order: VecDeque<JsonSetOptions>,
}

impl PreparedCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
        }
    }

    fn get(&self, options: &JsonSetOptions) -> Option<SharedPreparedJsonSetOptions> {
        self.entries.get(options).map(Arc::clone)
    }

    fn insert(
        &mut self,
        options: JsonSetOptions,
        prepared: SharedPreparedJsonSetOptions,
    ) -> SharedPreparedJsonSetOptions {
        if let Some(existing) = self.get(&options) {
            return existing;
        }

        while self.entries.len() >= self.capacity {
            let Some(oldest) = self.insertion_order.pop_front() else {
                self.entries.clear();
                break;
            };
            self.entries.remove(&oldest);
        }

        self.entries.insert(options, Arc::clone(&prepared));
        self.insertion_order.push_back(options);
        prepared
    }
}

fn cache() -> &'static Mutex<PreparedCache> {
    static CACHE: OnceLock<Mutex<PreparedCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(PreparedCache::new(PREPARED_CACHE_CAPACITY)))
}

fn lock_cache(cache: &Mutex<PreparedCache>) -> MutexGuard<'_, PreparedCache> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Pre-computed Redis `SET` command fragments derived from [`JsonSetOptions`].
#[derive(Debug, Clone)]
pub struct PreparedJsonSetOptions {
    fragments: Box<[CommandFragment]>,
}

/// Shared ownership handle returned by the automated prepared-options cache.
pub type SharedPreparedJsonSetOptions = Arc<PreparedJsonSetOptions>;

impl PreparedJsonSetOptions {
    pub(crate) fn new(fragments: Vec<CommandFragment>) -> Self {
        Self {
            fragments: fragments.into_boxed_slice(),
        }
    }

    /// Prepares a cached fragment sequence from the provided [`JsonSetOptions`].
    pub fn from_options(options: &JsonSetOptions) -> KvResult<Self> {
        let fragments = options.command_fragments()?;
        Ok(Self::new(fragments))
    }

    /// Returns a shared cached configuration for the provided [`JsonSetOptions`].
    ///
    /// The cache retains a bounded set of shared values so repeated calls reuse
    /// their fragments without leaking dynamically generated expiry options for
    /// the lifetime of the process. Concurrent misses are reconciled before
    /// insertion, preserving pointer identity for the cached entry.
    pub fn automated(options: JsonSetOptions) -> KvResult<SharedPreparedJsonSetOptions> {
        {
            let guard = lock_cache(cache());
            if let Some(prepared) = guard.get(&options) {
                return Ok(prepared);
            }
        }

        let prepared = Arc::new(options.prepare()?);
        let mut guard = lock_cache(cache());
        Ok(guard.insert(options, prepared))
    }

    /// Returns the cached fragments without incurring additional validation.
    #[must_use]
    pub fn fragments(&self) -> &[CommandFragment] {
        &self.fragments
    }

    /// Returns whether this prepared configuration would append any fragments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    /// Applies the cached fragments to the provided Redis command.
    pub fn apply(&self, cmd: &mut redis::Cmd) {
        for &fragment in self.fragments.iter() {
            fragment.apply(cmd);
        }
    }
}

/// Wrapper around a cached [`PreparedJsonSetOptions`] that can be cheaply cloned
/// and reapplied without re-validating the originating [`JsonSetOptions`].
#[derive(Debug, Clone)]
pub struct AutomatedJsonSetOptions {
    prepared: SharedPreparedJsonSetOptions,
}

impl AutomatedJsonSetOptions {
    pub(crate) fn new(prepared: SharedPreparedJsonSetOptions) -> Self {
        Self { prepared }
    }

    /// Automates the provided options into a cached prepared sequence.
    pub fn from_options(options: JsonSetOptions) -> KvResult<Self> {
        PreparedJsonSetOptions::automated(options).map(Self::new)
    }

    /// Returns the cached prepared configuration backing this automation.
    pub fn prepared(&self) -> &PreparedJsonSetOptions {
        &self.prepared
    }

    /// Clones the shared ownership handle backing this automation.
    #[must_use]
    pub fn shared_prepared(&self) -> SharedPreparedJsonSetOptions {
        Arc::clone(&self.prepared)
    }

    /// Returns the cached fragments without incurring additional validation.
    #[must_use]
    pub fn fragments(&self) -> &[CommandFragment] {
        self.prepared.fragments()
    }

    /// Returns whether this automated configuration would append any fragments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    /// Applies the cached fragments to the provided Redis command.
    pub fn apply(&self, cmd: &mut redis::Cmd) {
        self.prepared.apply(cmd);
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn bounded_cache_releases_evicted_values() {
        let first_options = JsonSetOptions::new().with_expiry_at_milliseconds(8_765_432_101);
        let first = Arc::new(first_options.prepare().expect("prepare first options"));
        let weak = Arc::downgrade(&first);
        let mut local_cache = PreparedCache::new(1);
        drop(local_cache.insert(first_options, first));

        assert!(weak.upgrade().is_some());

        let second_options = JsonSetOptions::new().with_expiry_at_milliseconds(8_765_432_102);
        let second = Arc::new(second_options.prepare().expect("prepare second options"));
        drop(local_cache.insert(second_options, second));

        assert!(weak.upgrade().is_none());
        assert_eq!(local_cache.entries.len(), 1);
    }

    #[test]
    fn concurrent_cache_misses_converge_on_one_shared_value() {
        const WORKERS: usize = 8;
        let options = JsonSetOptions::new().with_expiry_at_milliseconds(8_765_432_103);
        let barrier = Arc::new(Barrier::new(WORKERS));
        let workers: Vec<_> = (0..WORKERS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    PreparedJsonSetOptions::automated(options).expect("prepare shared options")
                })
            })
            .collect();
        let prepared: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().expect("cache worker"))
            .collect();

        assert!(prepared
            .iter()
            .skip(1)
            .all(|candidate| Arc::ptr_eq(&prepared[0], candidate)));
    }

    #[test]
    fn cache_lock_recovers_after_poisoning() {
        let local_cache = Mutex::new(PreparedCache::new(4));
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _guard = local_cache.lock().expect("local cache lock");
            panic!("poison local cache");
        }));

        assert!(panic.is_err());
        assert!(local_cache.is_poisoned());
        assert!(lock_cache(&local_cache).entries.is_empty());
    }
}
