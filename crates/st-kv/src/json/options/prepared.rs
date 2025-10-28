// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

use super::JsonSetOptions;
use crate::json::CommandFragment;
use crate::KvResult;

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
