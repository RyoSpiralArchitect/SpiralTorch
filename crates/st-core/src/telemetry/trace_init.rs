// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::sync::OnceLock;

use spiral_config::tracing::{self as config_tracing, InitError};
use tracing::warn;

static INIT_GUARD: OnceLock<Result<(), InitError>> = OnceLock::new();

/// Ensures tracing has been initialised for the current process.
pub fn init_tracing() {
    let result = INIT_GUARD.get_or_init(|| match config_tracing::init_tracing() {
        Ok(()) => Ok(()),
        Err(InitError::AlreadyInitialised) => Ok(()),
        Err(err) => Err(err),
    });

    if let Err(err) = result {
        warn!("failed to initialise tracing subscriber: {err}");
    }
}
