// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "redis")]
mod command;
#[cfg(feature = "redis")]
mod condition;
#[cfg(feature = "redis")]
mod expiry;
#[cfg(feature = "redis")]
pub mod options;

#[cfg(all(test, feature = "redis"))]
mod tests;

#[cfg(feature = "redis")]
pub(crate) use command::CommandFragment;
#[cfg(feature = "redis")]
pub use condition::JsonSetCondition;
#[cfg(feature = "redis")]
pub use expiry::JsonExpiry;
#[cfg(feature = "redis")]
pub use options::{JsonSetOptions, JsonSetOptionsBuilder, PreparedJsonSetOptions};
