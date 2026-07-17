//! Shared route-selection vocabulary.
//!
//! Domain-specific policies own their evidence and scoring formulas. This
//! module only defines the profile names shared by those policies and their
//! clients.

use serde::{Deserialize, Serialize};
use std::str::FromStr;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RouteSelectionProfile {
    #[default]
    Balanced,
    Quality,
    Grounded,
    Efficiency,
    Latency,
}

impl RouteSelectionProfile {
    pub const ALL: [Self; 5] = [
        Self::Balanced,
        Self::Quality,
        Self::Grounded,
        Self::Efficiency,
        Self::Latency,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Quality => "quality",
            Self::Grounded => "grounded",
            Self::Efficiency => "efficiency",
            Self::Latency => "latency",
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("unknown route-policy profile '{profile}'")]
pub struct RouteSelectionProfileError {
    pub profile: String,
}

impl FromStr for RouteSelectionProfile {
    type Err = RouteSelectionProfileError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "balanced" => Ok(Self::Balanced),
            "quality" => Ok(Self::Quality),
            "grounded" => Ok(Self::Grounded),
            "efficiency" => Ok(Self::Efficiency),
            "latency" => Ok(Self::Latency),
            _ => Err(RouteSelectionProfileError {
                profile: value.to_owned(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for RouteSelectionProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let profile = String::deserialize(deserializer)?;
        Self::from_str(&profile).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_ingress_is_normalized_and_closed() {
        assert_eq!(
            RouteSelectionProfile::from_str(" Grounded ").expect("known profile"),
            RouteSelectionProfile::Grounded
        );
        assert_eq!(
            RouteSelectionProfile::from_str("commander")
                .expect_err("unknown profile")
                .to_string(),
            "unknown route-policy profile 'commander'"
        );
    }
}
