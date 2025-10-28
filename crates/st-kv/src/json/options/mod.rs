// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "redis")]

mod builder;
mod prepared;

use super::{CommandFragment, JsonExpiry, JsonSetCondition};
use crate::{KvErr, KvResult};
pub use builder::JsonSetOptionsBuilder;
pub use prepared::PreparedJsonSetOptions;
use std::time::{Duration, SystemTime};

const ERR_EXPLICIT_AND_KEEP_TTL: &str = "cannot set both explicit expiry and KEEPTTL";
const ERR_PERSIST_WITH_EXPIRY: &str = "cannot use PERSIST alongside an explicit expiry";
const ERR_PERSIST_WITH_KEEP_TTL: &str = "cannot combine PERSIST with KEEPTTL";

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

    /// Applies an absolute expiry timestamp in seconds (`EXAT`).
    #[must_use]
    pub fn with_expiry_at_seconds(self, seconds: u64) -> Self {
        self.with_expiry(JsonExpiry::at_seconds(seconds))
    }

    /// Applies an absolute expiry timestamp in milliseconds (`PXAT`).
    #[must_use]
    pub fn with_expiry_at_milliseconds(self, milliseconds: u64) -> Self {
        self.with_expiry(JsonExpiry::at_milliseconds(milliseconds))
    }

    /// Applies an expiry derived from a [`Duration`].
    #[must_use]
    pub fn with_duration(self, duration: Duration) -> Self {
        self.with_expiry(JsonExpiry::from_duration(duration))
    }

    /// Applies an expiry derived from a [`SystemTime`] deadline.
    pub fn with_deadline(self, deadline: SystemTime) -> KvResult<Self> {
        let expiry = JsonExpiry::from_system_time(deadline)?;
        Ok(self.with_expiry(expiry))
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

    /// Applies an optional expiry derived from a [`SystemTime`] deadline.
    pub fn with_optional_deadline(mut self, deadline: Option<SystemTime>) -> KvResult<Self> {
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

        let mut fragments = Vec::with_capacity(4);

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

    /// Pre-computes validated command fragments for repeated reuse.
    pub fn prepare(&self) -> KvResult<PreparedJsonSetOptions> {
        PreparedJsonSetOptions::from_options(self)
    }

    pub(crate) fn apply_to_command(&self, cmd: &mut redis::Cmd) -> KvResult<()> {
        for fragment in self.command_fragments()? {
            fragment.apply(cmd);
        }

        Ok(())
    }
}
