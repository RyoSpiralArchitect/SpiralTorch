// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

use super::{JsonSetOptions, PreparedJsonSetOptions};
use crate::json::{JsonExpiry, JsonSetCondition};
use crate::KvResult;
use std::time::{Duration, SystemTime};

/// Builder for [`JsonSetOptions`] that performs validation on construction.
#[derive(Debug, Default, Clone)]
pub struct JsonSetOptionsBuilder {
    pub(crate) expiry: Option<JsonExpiry>,
    pub(crate) keep_ttl: bool,
    pub(crate) persist: bool,
    pub(crate) condition: JsonSetCondition,
}

impl JsonSetOptionsBuilder {
    /// Applies an explicit expiry.
    #[must_use]
    pub fn expiry<E: Into<JsonExpiry>>(mut self, expiry: E) -> Self {
        self.expiry = Some(expiry.into());
        self.persist = false;
        self
    }

    /// Applies an expiry derived from a [`Duration`].
    #[must_use]
    pub fn duration(mut self, duration: Duration) -> Self {
        self.expiry = Some(JsonExpiry::from_duration(duration));
        self.persist = false;
        self
    }

    /// Applies an absolute second timestamp.
    #[must_use]
    pub fn expiry_at_seconds(self, seconds: u64) -> Self {
        self.expiry(JsonExpiry::at_seconds(seconds))
    }

    /// Applies an absolute millisecond timestamp.
    #[must_use]
    pub fn expiry_at_milliseconds(self, milliseconds: u64) -> Self {
        self.expiry(JsonExpiry::at_milliseconds(milliseconds))
    }

    /// Applies an absolute system time deadline (`PXAT`/`EXAT`).
    pub fn deadline(mut self, deadline: SystemTime) -> KvResult<Self> {
        self.expiry = Some(JsonExpiry::from_system_time(deadline)?);
        self.persist = false;
        Ok(self)
    }

    /// Applies an optional absolute system time deadline.
    pub fn optional_deadline(mut self, deadline: Option<SystemTime>) -> KvResult<Self> {
        self.expiry = match deadline {
            Some(deadline) => {
                let expiry = JsonExpiry::from_system_time(deadline)?;
                self.persist = false;
                Some(expiry)
            }
            None => None,
        };
        Ok(self)
    }

    /// Clears any explicit expiry.
    #[must_use]
    pub fn clear_expiry(mut self) -> Self {
        self.expiry = None;
        self
    }

    /// Retains the existing TTL of the key (`KEEPTTL`).
    #[must_use]
    pub fn keep_ttl(mut self) -> Self {
        self.keep_ttl = true;
        self
    }

    /// Removes any existing expiry (`PERSIST`).
    #[must_use]
    pub fn persist(mut self) -> Self {
        self.persist = true;
        self.expiry = None;
        self
    }

    /// Overrides the write condition.
    #[must_use]
    pub fn condition(mut self, condition: JsonSetCondition) -> Self {
        self.condition = condition;
        self
    }

    /// Applies the `NX` condition.
    #[must_use]
    pub fn nx(self) -> Self {
        self.condition(JsonSetCondition::Nx)
    }

    /// Applies the `XX` condition.
    #[must_use]
    pub fn xx(self) -> Self {
        self.condition(JsonSetCondition::Xx)
    }

    /// Finalises the builder into a validated [`JsonSetOptions`].
    pub fn build(self) -> KvResult<JsonSetOptions> {
        let options = JsonSetOptions {
            expiry: self.expiry,
            keep_ttl: self.keep_ttl,
            persist: self.persist,
            condition: self.condition,
        };

        options.validate()?;
        Ok(options)
    }

    /// Finalises the builder directly into a prepared fragment set.
    pub fn build_prepared(self) -> KvResult<PreparedJsonSetOptions> {
        Ok(self.build()?.prepare()?)
    }
}
