// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(test)]

use super::{CommandFragment, JsonExpiry, JsonSetOptions};
use crate::KvErr;
use std::time::Duration;

#[test]
fn json_expiry_from_duration_prefers_seconds_for_whole_seconds() {
    let expiry = JsonExpiry::from_duration(Duration::from_secs(5));
    assert_eq!(expiry, JsonExpiry::Seconds(5));
}

#[test]
fn json_expiry_from_duration_keeps_millisecond_precision() {
    let expiry = JsonExpiry::from_duration(Duration::from_millis(1500));
    assert_eq!(expiry, JsonExpiry::Milliseconds(1500));
}

#[test]
fn json_expiry_from_duration_rounds_up_sub_millisecond_values() {
    let expiry = JsonExpiry::from_duration(Duration::from_micros(500));
    assert_eq!(expiry, JsonExpiry::Milliseconds(1));
}

#[test]
fn json_expiry_from_duration_handles_zero() {
    let expiry = JsonExpiry::from_duration(Duration::ZERO);
    assert_eq!(expiry, JsonExpiry::Milliseconds(0));
}

#[test]
fn json_expiry_from_duration_falls_back_to_seconds_when_overflowing() {
    let duration = Duration::from_secs(u64::MAX);
    let expiry = JsonExpiry::from_duration(duration);
    assert_eq!(expiry, JsonExpiry::Seconds(u64::MAX));
}

#[test]
fn json_set_options_validate_rejects_conflicting_ttl_rules() {
    let options = JsonSetOptions::new().with_expiry_seconds(5).keep_ttl();

    let err = options.validate().expect_err("expected validation failure");
    assert!(matches!(err, KvErr::InvalidOptions(_)));
}

#[test]
fn json_set_options_validate_rejects_persist_with_expiry() {
    let options = JsonSetOptions {
        persist: true,
        expiry: Some(JsonExpiry::seconds(5)),
        ..JsonSetOptions::default()
    };

    let err = options
        .validate()
        .expect_err("expected persist + expiry to fail");
    assert!(matches!(err, KvErr::InvalidOptions(_)));
}

#[test]
fn json_set_options_validate_rejects_persist_with_keep_ttl() {
    let options = JsonSetOptions::new().persist().keep_ttl();

    let err = options
        .validate()
        .expect_err("expected persist + KEEPTTL to fail");
    assert!(matches!(err, KvErr::InvalidOptions(_)));
}

#[test]
fn command_fragments_include_persist_and_condition() {
    let options = JsonSetOptions::new().persist().nx();

    let fragments = options.command_fragments().expect("valid fragments");
    assert_eq!(
        fragments,
        vec![
            CommandFragment::Keyword("PERSIST"),
            CommandFragment::Keyword("NX")
        ]
    );
}

#[test]
fn command_fragments_include_expiry() {
    let options = JsonSetOptions::new().with_expiry_milliseconds(1500).xx();

    let fragments = options.command_fragments().expect("valid fragments");
    assert_eq!(
        fragments,
        vec![
            CommandFragment::Keyword("PX"),
            CommandFragment::Integer(1500),
            CommandFragment::Keyword("XX"),
        ]
    );
}

#[test]
fn builder_allows_valid_configuration() {
    let options = JsonSetOptions::builder()
        .duration(Duration::from_secs(10))
        .nx()
        .build()
        .expect("builder should succeed");

    let fragments = options.command_fragments().expect("valid fragments");
    assert_eq!(
        fragments,
        vec![
            CommandFragment::Keyword("EX"),
            CommandFragment::Integer(10),
            CommandFragment::Keyword("NX"),
        ]
    );
}

#[test]
fn builder_rejects_invalid_configuration() {
    let err = JsonSetOptions::builder()
        .expiry(JsonExpiry::seconds(5))
        .keep_ttl()
        .build()
        .expect_err("expected builder validation failure");

    assert!(matches!(err, KvErr::InvalidOptions(_)));
}

#[test]
fn builder_supports_persist() {
    let options = JsonSetOptions::builder()
        .persist()
        .xx()
        .build()
        .expect("persist should be valid");

    let fragments = options.command_fragments().expect("valid fragments");
    assert_eq!(
        fragments,
        vec![
            CommandFragment::Keyword("PERSIST"),
            CommandFragment::Keyword("XX")
        ]
    );
}
