// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

use super::JsonSetOptions;
use crate::json::CommandFragment;
use crate::KvResult;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn cache() -> &'static Mutex<HashMap<JsonSetOptions, &'static PreparedJsonSetOptions>> {
    static CACHE: OnceLock<Mutex<HashMap<JsonSetOptions, &'static PreparedJsonSetOptions>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Pre-computed Redis `SET` command fragments derived from [`JsonSetOptions`].
#[derive(Debug, Clone)]
pub struct PreparedJsonSetOptions {
    fragments: Box<[CommandFragment]>,
}

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

    /// Returns a cached prepared configuration for the provided [`JsonSetOptions`].
    pub fn automated(options: JsonSetOptions) -> KvResult<&'static Self> {
        let cache = cache();
        let mut guard = cache
            .lock()
            .expect("prepared JSON SET options cache poisoned");

        if let Some(&prepared) = guard.get(&options) {
            return Ok(prepared);
        }

        let prepared = options.prepare()?;
        let leaked: &'static mut PreparedJsonSetOptions = Box::leak(Box::new(prepared));
        let leaked_ref: &'static PreparedJsonSetOptions = leaked;
        guard.insert(options, leaked_ref);

        Ok(leaked_ref)
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

/// Wrapper around a cached [`PreparedJsonSetOptions`] that can be cheaply copied
/// and reapplied without re-validating the originating [`JsonSetOptions`].
#[derive(Debug, Clone, Copy)]
pub struct AutomatedJsonSetOptions {
    prepared: &'static PreparedJsonSetOptions,
}

impl AutomatedJsonSetOptions {
    pub(crate) fn new(prepared: &'static PreparedJsonSetOptions) -> Self {
        Self { prepared }
    }

    /// Automates the provided options into a cached prepared sequence.
    pub fn from_options(options: JsonSetOptions) -> KvResult<Self> {
        PreparedJsonSetOptions::automated(options).map(Self::new)
    }

    /// Returns the cached prepared configuration backing this automation.
    pub fn prepared(self) -> &'static PreparedJsonSetOptions {
        self.prepared
    }

    /// Returns the cached fragments without incurring additional validation.
    #[must_use]
    pub fn fragments(self) -> &'static [CommandFragment] {
        self.prepared.fragments()
    }

    /// Returns whether this automated configuration would append any fragments.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.prepared.is_empty()
    }

    /// Applies the cached fragments to the provided Redis command.
    pub fn apply(self, cmd: &mut redis::Cmd) {
        self.prepared.apply(cmd);
    }
}
