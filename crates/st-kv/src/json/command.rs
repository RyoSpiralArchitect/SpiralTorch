// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

use redis::Cmd;

/// Represents an atomic fragment to append to a Redis command.
#[cfg(feature = "redis")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandFragment {
    Keyword(&'static str),
    Integer(u64),
}

#[cfg(feature = "redis")]
impl CommandFragment {
    pub(crate) fn apply(self, cmd: &mut Cmd) {
        match self {
            CommandFragment::Keyword(keyword) => {
                cmd.arg(keyword);
            }
            CommandFragment::Integer(value) => {
                cmd.arg(value);
            }
        }
    }
}
