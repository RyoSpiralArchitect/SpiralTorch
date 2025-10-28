// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

use super::CommandFragment;
use crate::{KvErr, KvResult};
use std::time::Duration;

const ERR_EXPLICIT_AND_KEEP_TTL: &str = "cannot set both explicit expiry and KEEPTTL";
const ERR_PERSIST_WITH_EXPIRY: &str = "cannot use PERSIST alongside an explicit expiry";
const ERR_PERSIST_WITH_KEEP_TTL: &str = "cannot combine PERSIST with KEEPTTL";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Controls conditional write behaviour for Redis `SET`.
pub enum JsonSetCondition {
    /// Always perform the write, regardless of key existence.
    Always,
    /// Only write when the key does not yet exist (`NX`).
    Nx,
    /// Only write when the key already exists (`XX`).
    Xx,
}

impl Default for JsonSetCondition {
    fn default() -> Self {
        Self::Always
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Expiry semantics for Redis `SET`.
pub enum JsonExpiry {
    /// Expire the key after the provided number of seconds (`EX`).
    Seconds(u64),
    /// Expire the key after the provided number of milliseconds (`PX`).
    Milliseconds(u64),
}

impl JsonExpiry {
    /// Returns the raw command keyword + value pair for the expiry.
    pub(crate) fn keyword_value(self) -> (&'static str, u64) {
        match self {
            JsonExpiry::Seconds(secs) => ("EX", secs),
            JsonExpiry::Milliseconds(ms) => ("PX", ms),
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
}

impl From<Duration> for JsonExpiry {
    fn from(duration: Duration) -> Self {
        JsonExpiry::from_duration(duration)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Configures Redis `SET` behaviour for JSON helpers.
pub struct JsonSetOptions {
    pub expiry: Option<JsonExpiry>,
    pub keep_ttl: bool,
    pub persist: bool,
    pub condition: JsonSetCondition,
}

impl Default for JsonSetOptions {
    fn default() -> Self {
        Self {
            expiry: None,
            keep_ttl: false,
            persist: false,
            condition: JsonSetCondition::Always,
        }
    }
}

impl JsonSetOptions {
    /// Creates an empty option set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a builder for incremental configuration.
    #[must_use]
    pub fn builder() -> JsonSetOptionsBuilder {
        JsonSetOptionsBuilder::default()
    }

    /// Applies an explicit expiry.
    #[must_use]
    pub fn with_expiry<E: Into<JsonExpiry>>(mut self, expiry: E) -> Self {
        self.expiry = Some(expiry.into());
        self.persist = false;
        self
    }

    /// Applies an expiry in seconds.
    #[must_use]
    pub fn with_expiry_seconds(self, seconds: u64) -> Self {
        self.with_expiry(JsonExpiry::seconds(seconds))
    }

    /// Applies an expiry in milliseconds.
    #[must_use]
    pub fn with_expiry_milliseconds(self, milliseconds: u64) -> Self {
        self.with_expiry(JsonExpiry::milliseconds(milliseconds))
    }

    /// Applies an expiry derived from a [`Duration`].
    #[must_use]
    pub fn with_duration(self, duration: Duration) -> Self {
        self.with_expiry(JsonExpiry::from_duration(duration))
    }

    /// Applies an optional expiry, clearing it when `None` is provided.
    #[must_use]
    pub fn with_optional_expiry(mut self, expiry: Option<JsonExpiry>) -> Self {
        self.expiry = expiry;
        if expiry.is_some() {
            self.persist = false;
        }
        self
    }

    /// Applies an optional expiry derived from a [`Duration`].
    #[must_use]
    pub fn with_optional_duration(mut self, duration: Option<Duration>) -> Self {
        self.expiry = duration.map(JsonExpiry::from_duration);
        if self.expiry.is_some() {
            self.persist = false;
        }
        self
    }

    /// Clears any explicit expiry.
    #[must_use]
    pub fn without_expiry(mut self) -> Self {
        self.expiry = None;
        self
    }

    /// Overrides the write condition.
    #[must_use]
    pub fn with_condition(mut self, condition: JsonSetCondition) -> Self {
        self.condition = condition;
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

    /// Applies the `NX` condition (only write when absent).
    #[must_use]
    pub fn nx(self) -> Self {
        self.with_condition(JsonSetCondition::Nx)
    }

    /// Applies the `XX` condition (only write when present).
    #[must_use]
    pub fn xx(self) -> Self {
        self.with_condition(JsonSetCondition::Xx)
    }

    /// Verifies that the selected options are valid for Redis.
    pub(crate) fn validate(&self) -> KvResult<()> {
        if self.keep_ttl && self.expiry.is_some() {
            return Err(KvErr::InvalidOptions(ERR_EXPLICIT_AND_KEEP_TTL));
        }

        if self.persist && self.expiry.is_some() {
            return Err(KvErr::InvalidOptions(ERR_PERSIST_WITH_EXPIRY));
        }

        if self.persist && self.keep_ttl {
            return Err(KvErr::InvalidOptions(ERR_PERSIST_WITH_KEEP_TTL));
        }

        Ok(())
    }

    pub(crate) fn command_fragments(&self) -> KvResult<Vec<CommandFragment>> {
        self.validate()?;

        let mut fragments = Vec::new();

        if let Some(expiry) = self.expiry {
            let (keyword, value) = expiry.keyword_value();
            fragments.push(CommandFragment::Keyword(keyword));
            fragments.push(CommandFragment::Integer(value));
        }

        if self.persist {
            fragments.push(CommandFragment::Keyword("PERSIST"));
        }

        if self.keep_ttl {
            fragments.push(CommandFragment::Keyword("KEEPTTL"));
        }

        match self.condition {
            JsonSetCondition::Always => {}
            JsonSetCondition::Nx => fragments.push(CommandFragment::Keyword("NX")),
            JsonSetCondition::Xx => fragments.push(CommandFragment::Keyword("XX")),
        }

        Ok(fragments)
    }

    pub(crate) fn apply_to_command(&self, cmd: &mut redis::Cmd) -> KvResult<()> {
        for fragment in self.command_fragments()? {
            fragment.apply(cmd);
        }

        Ok(())
    }
}

/// Builder for [`JsonSetOptions`] that performs validation on construction.
#[derive(Debug, Default, Clone)]
pub struct JsonSetOptionsBuilder {
    expiry: Option<JsonExpiry>,
    keep_ttl: bool,
    persist: bool,
    condition: JsonSetCondition,
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
}
