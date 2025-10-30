// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(test)]

use super::{
    AutomatedJsonSetOptions, CommandFragment, JsonExpiry, JsonSetOptions, PreparedJsonSetOptions,
};
use crate::KvErr;
use std::time::{Duration, UNIX_EPOCH};

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
fn json_expiry_from_system_time_prefers_milliseconds() {
    let deadline = UNIX_EPOCH + Duration::from_millis(1_733_000_000_500);
    let expiry = JsonExpiry::from_system_time(deadline).expect("system time should be valid");
    assert_eq!(expiry, JsonExpiry::AtMilliseconds(1_733_000_000_500));
}

#[test]
fn json_expiry_from_system_time_overflow_falls_back_to_seconds() {
    let large_secs = (i64::MAX as u64) - 1;
    let deadline = UNIX_EPOCH + Duration::from_secs(large_secs);
    let expiry = JsonExpiry::from_system_time(deadline).expect("system time should be valid");
    assert_eq!(expiry, JsonExpiry::AtSeconds(large_secs));
}

#[test]
fn json_expiry_from_system_time_rejects_pre_epoch() {
    let deadline = UNIX_EPOCH - Duration::from_secs(1);
    let err = JsonExpiry::from_system_time(deadline).expect_err("pre-epoch time should fail");
    assert!(matches!(err, KvErr::InvalidExpiry(_)));
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
fn json_set_options_support_deadlines() {
    let deadline = UNIX_EPOCH + Duration::from_millis(1_733_888_000_250);
    let options = JsonSetOptions::new()
        .with_deadline(deadline)
        .expect("deadline should be valid")
        .nx();

    let millis = deadline.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;

    let fragments = options.command_fragments().expect("valid fragments");
    assert_eq!(
        fragments,
        vec![
            CommandFragment::Keyword("PXAT"),
            CommandFragment::Integer(millis),
            CommandFragment::Keyword("NX"),
        ]
    );
}

#[test]
fn json_set_options_optional_deadline_clears_when_none() {
    let options = JsonSetOptions::new()
        .with_expiry_seconds(5)
        .with_optional_deadline(None)
        .expect("optional deadline should accept None");

    assert!(options.expiry.is_none());
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
fn command_fragments_include_absolute_expiry() {
    let options = JsonSetOptions::new()
        .with_expiry_at_seconds(1_733_000_000)
        .nx();

    let fragments = options.command_fragments().expect("valid fragments");
    assert_eq!(
        fragments,
        vec![
            CommandFragment::Keyword("EXAT"),
            CommandFragment::Integer(1_733_000_000),
            CommandFragment::Keyword("NX"),
        ]
    );
}

#[test]
fn prepared_json_set_options_cache_fragments() {
    let options = JsonSetOptions::new().with_expiry_milliseconds(1500).xx();
    let prepared = options.prepare().expect("preparation should succeed");

    assert_eq!(
        prepared.fragments(),
        &[
            CommandFragment::Keyword("PX"),
            CommandFragment::Integer(1500),
            CommandFragment::Keyword("XX"),
        ]
    );
    assert!(!prepared.is_empty());
}

#[test]
fn prepared_json_set_options_apply_appends_fragments() {
    let options = JsonSetOptions::new().with_expiry_at_milliseconds(1_733_000_000_500);
    let prepared =
        PreparedJsonSetOptions::from_options(&options).expect("preparation should succeed");

    let mut cmd = redis::cmd("SET");
    prepared.apply(&mut cmd);

    let packed = cmd.get_packed_command();
    let as_string = String::from_utf8(packed).expect("command should be valid utf8");
    assert!(as_string.contains("PXAT"));
    assert!(as_string.contains("1733000000500"));
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

#[test]
fn builder_supports_absolute_expiry() {
    let options = JsonSetOptions::builder()
        .expiry_at_milliseconds(1_733_000_000_500)
        .xx()
        .build()
        .expect("absolute expiry should be valid");

    let fragments = options.command_fragments().expect("valid fragments");
    assert_eq!(
        fragments,
        vec![
            CommandFragment::Keyword("PXAT"),
            CommandFragment::Integer(1_733_000_000_500),
            CommandFragment::Keyword("XX"),
        ]
    );
}

#[test]
fn builder_supports_deadline_and_preparation() {
    let deadline = UNIX_EPOCH + Duration::from_secs(1_733_000_000);
    let builder = JsonSetOptions::builder()
        .deadline(deadline)
        .expect("deadline should be valid")
        .nx();

    let prepared = builder
        .build_prepared()
        .expect("prepared fragments should build");

    let millis = deadline.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;

    assert_eq!(
        prepared.fragments(),
        &[
            CommandFragment::Keyword("PXAT"),
            CommandFragment::Integer(millis),
            CommandFragment::Keyword("NX"),
        ]
    );
}

#[test]
fn builder_optional_deadline_handles_none() {
    let builder = JsonSetOptions::builder()
        .optional_deadline(None)
        .expect("optional deadline should allow None")
        .keep_ttl();

    let options = builder.build().expect("builder should succeed");
    assert!(options.expiry.is_none());
    assert!(options.keep_ttl);
}

#[test]
fn automated_prepared_options_cache_reuses_instances() {
    let options = JsonSetOptions::new().nx();

    let first = PreparedJsonSetOptions::automated(options).expect("automation should prepare");
    let second = PreparedJsonSetOptions::automated(options).expect("automation should reuse");

    assert!(std::ptr::eq(first, second));
    assert_eq!(first.fragments(), &[CommandFragment::Keyword("NX")]);
}

#[test]
fn automated_prepared_options_respects_validation() {
    let options = JsonSetOptions::new().with_expiry_seconds(5).keep_ttl();

    let err =
        PreparedJsonSetOptions::automated(options).expect_err("invalid options should be rejected");
    assert!(matches!(err, KvErr::InvalidOptions(_)));
}

#[test]
fn automated_json_set_options_wrap_cached_prepared_instances() {
    let options = JsonSetOptions::new().xx();
    let automated = options
        .automated_owned()
        .expect("automated options should prepare");
    let prepared = PreparedJsonSetOptions::automated(options).expect("prepared cache should exist");

    assert!(std::ptr::eq(automated.prepared(), prepared));
    assert_eq!(automated.fragments(), &[CommandFragment::Keyword("XX")]);
    assert!(!automated.is_empty());
}

#[test]
fn automated_json_set_options_apply_delegates_to_fragments() {
    let options = JsonSetOptions::new().persist();
    let automated = AutomatedJsonSetOptions::from_options(options)
        .expect("automation should succeed for persist");

    let mut cmd = redis::cmd("SET");
    automated.apply(&mut cmd);

    let packed = cmd.get_packed_command();
    let as_string = String::from_utf8(packed).expect("command should be valid utf8");
    assert!(as_string.contains("PERSIST"));
}

#[test]
fn automated_owned_respects_validation_rules() {
    let invalid = JsonSetOptions::new().with_expiry_seconds(30).keep_ttl();
    let err = invalid
        .automated_owned()
        .expect_err("invalid automated options should error");
    assert!(matches!(err, KvErr::InvalidOptions(_)));
}
