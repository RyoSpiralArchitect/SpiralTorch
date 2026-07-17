// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Shared wire encodings for Rust-owned distributed contracts.

pub(crate) mod canonical_u64 {
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub(crate) fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        super::parse_canonical_u64(&encoded).map_err(D::Error::custom)
    }
}

pub(crate) fn parse_canonical_u64(value: &str) -> Result<u64, &'static str> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err("must be a canonical unsigned decimal u64");
    }
    value.parse().map_err(|_| "must fit the complete u64 range")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_u64_accepts_the_complete_range() {
        assert_eq!(parse_canonical_u64("0"), Ok(0));
        assert_eq!(parse_canonical_u64(&u64::MAX.to_string()), Ok(u64::MAX));
    }

    #[test]
    fn canonical_u64_rejects_ambiguous_or_out_of_range_values() {
        for value in ["", "00", "01", "+1", "-1", "1.0", " 1"] {
            assert!(parse_canonical_u64(value).is_err(), "accepted {value:?}");
        }
        assert!(parse_canonical_u64("18446744073709551616").is_err());
    }
}
