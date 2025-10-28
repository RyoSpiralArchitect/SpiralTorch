// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use thiserror::Error;

#[cfg(feature = "redis")]
use redis::Value;

/// Unified error type for key-value helper routines.
#[cfg(feature = "redis")]
#[derive(Error, Debug)]
pub enum KvErr {
    #[error("redis error: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("invalid Redis SET options: {0}")]
    InvalidOptions(&'static str),
    #[error("invalid expiry: {0}")]
    InvalidExpiry(&'static str),
    #[error("unexpected response from {command}: {response:?}")]
    UnexpectedResponse {
        command: &'static str,
        response: Value,
    },
}

#[cfg(not(feature = "redis"))]
#[derive(Error, Debug)]
pub enum KvErr {
    #[error("redis feature not enabled")]
    MissingRedisFeature,
}
