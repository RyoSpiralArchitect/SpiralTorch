// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

use crate::{KvErr, KvResult};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ERR_SYSTEM_TIME_BEFORE_EPOCH: &str = "system time occurs before the Unix epoch";

/// Expiry semantics for Redis `SET`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JsonExpiry {
    /// Expire the key after the provided number of seconds (`EX`).
    Seconds(u64),
    /// Expire the key after the provided number of milliseconds (`PX`).
    Milliseconds(u64),
    /// Expire the key at the given absolute second timestamp (`EXAT`).
    AtSeconds(u64),
    /// Expire the key at the given absolute millisecond timestamp (`PXAT`).
    AtMilliseconds(u64),
}

impl JsonExpiry {
    /// Returns the raw command keyword + value pair for the expiry.
    pub(crate) fn keyword_value(self) -> (&'static str, u64) {
        match self {
            JsonExpiry::Seconds(secs) => ("EX", secs),
            JsonExpiry::Milliseconds(ms) => ("PX", ms),
            JsonExpiry::AtSeconds(secs) => ("EXAT", secs),
            JsonExpiry::AtMilliseconds(ms) => ("PXAT", ms),
        }
    }

    /// Creates an expiry representing seconds.
    #[must_use]
    pub const fn seconds(secs: u64) -> Self {
        Self::Seconds(secs)
    }

    /// Creates an expiry representing milliseconds.
    #[must_use]
    pub const fn milliseconds(ms: u64) -> Self {
        Self::Milliseconds(ms)
    }

    /// Creates an expiry representing an absolute second timestamp since the Unix epoch.
    #[must_use]
    pub const fn at_seconds(secs: u64) -> Self {
        Self::AtSeconds(secs)
    }

    /// Creates an expiry representing an absolute millisecond timestamp since the Unix epoch.
    #[must_use]
    pub const fn at_milliseconds(ms: u64) -> Self {
        Self::AtMilliseconds(ms)
    }

    /// Converts a [`Duration`] into an expiry, favouring millisecond precision.
    ///
    /// Durations that do not cleanly map to milliseconds are rounded up to the
    /// nearest millisecond so that very small values (e.g. microseconds) still
    /// produce a non-zero expiry.
    #[must_use]
    pub fn from_duration(duration: Duration) -> Self {
        if duration.is_zero() {
            return Self::Milliseconds(0);
        }

        let millis = duration.as_millis();

        if millis == 0 {
            return Self::Milliseconds(1);
        }

        if millis > u64::MAX as u128 {
            return Self::Seconds(duration.as_secs());
        }

        let millis_u64 = millis as u64;

        if millis % 1000 == 0 {
            Self::Seconds((millis / 1000) as u64)
        } else {
            Self::Milliseconds(millis_u64)
        }
    }

    /// Converts an absolute [`SystemTime`] deadline into an expiry, preferring
    /// millisecond precision where representable.
    pub fn from_system_time(deadline: SystemTime) -> KvResult<Self> {
        let since_epoch = deadline
            .duration_since(UNIX_EPOCH)
            .map_err(|_| KvErr::InvalidExpiry(ERR_SYSTEM_TIME_BEFORE_EPOCH))?;

        if since_epoch.as_millis() > u64::MAX as u128 {
            Ok(Self::AtSeconds(since_epoch.as_secs()))
        } else {
            Ok(Self::AtMilliseconds(since_epoch.as_millis() as u64))
        }
    }
}

impl From<Duration> for JsonExpiry {
    fn from(duration: Duration) -> Self {
        JsonExpiry::from_duration(duration)
    }
}
