// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use thiserror::Error;

/// Result alias specialised for key-value helper routines.
pub type KvResult<T> = std::result::Result<T, KvErr>;

#[derive(Error, Debug)]
pub enum KvErr {
    #[error("redis error: {0}")]
    Redis(String),
    #[error("serialization error: {0}")]
    Serialization(String),
}

#[cfg(feature = "redis")]
use redis::Commands;

#[cfg(feature = "redis")]
use serde::{de::DeserializeOwned, Serialize};

#[cfg(feature = "redis")]
fn redis_connection(url: &str) -> KvResult<redis::Connection> {
    let client = redis::Client::open(url)?;
    client.get_connection().map_err(Into::into)
}

#[cfg(feature = "redis")]
fn with_connection<T, F>(url: &str, f: F) -> KvResult<T>
where
    F: FnOnce(&mut redis::Connection) -> redis::RedisResult<T>,
{
    let mut conn = redis_connection(url)?;
    f(&mut conn).map_err(Into::into)
}

#[cfg(feature = "redis")]
impl From<redis::RedisError> for KvErr {
    fn from(err: redis::RedisError) -> Self {
        KvErr::Redis(err.to_string())
    }
}

#[cfg(feature = "redis")]
impl From<serde_json::Error> for KvErr {
    fn from(err: serde_json::Error) -> Self {
        KvErr::Serialization(err.to_string())
    }
}

/// Fetch the raw string value for a given key.
#[cfg(feature = "redis")]
pub fn redis_get_raw(url: &str, key: &str) -> KvResult<Option<String>> {
    with_connection(url, |conn| conn.get::<_, Option<String>>(key))
}

/// Store a raw string value at the given key.
#[cfg(feature = "redis")]
pub fn redis_set_raw(url: &str, key: &str, value: &str) -> KvResult<()> {
    with_connection(url, |conn| conn.set(key, value))
}

/// Delete the specified key, returning `true` when a value was removed.
#[cfg(feature = "redis")]
pub fn redis_del(url: &str, key: &str) -> KvResult<bool> {
    let deleted: usize = with_connection(url, |conn| conn.del(key))?;
    Ok(deleted > 0)
}

/// Determine whether a key currently exists.
#[cfg(feature = "redis")]
pub fn redis_exists(url: &str, key: &str) -> KvResult<bool> {
    with_connection(url, |conn| conn.exists(key))
}

/// Fetch values from a list within the provided range.
#[cfg(feature = "redis")]
pub fn redis_lrange(url: &str, key: &str, start: isize, stop: isize) -> KvResult<Vec<String>> {
    with_connection(url, |conn| conn.lrange(key, start, stop))
}

/// Push new values onto the head of a list.
#[cfg(feature = "redis")]
pub fn redis_lpush(url: &str, key: &str, values: &[String]) -> KvResult<usize> {
    with_connection(url, |conn| conn.lpush(key, values))
}

/// Push new values onto the tail of a list.
#[cfg(feature = "redis")]
pub fn redis_rpush(url: &str, key: &str, values: &[String]) -> KvResult<usize> {
    with_connection(url, |conn| conn.rpush(key, values))
}

/// Pop a value from the head of a list.
#[cfg(feature = "redis")]
pub fn redis_lpop(url: &str, key: &str) -> KvResult<Option<String>> {
    with_connection(url, |conn| conn.lpop::<_, Option<String>>(key, None))
}

/// Pop a value from the tail of a list.
#[cfg(feature = "redis")]
pub fn redis_rpop(url: &str, key: &str) -> KvResult<Option<String>> {
    with_connection(url, |conn| conn.rpop::<_, Option<String>>(key, None))
}

/// Retrieve the raw string value of a field inside a hash.
#[cfg(feature = "redis")]
pub fn redis_hget_raw(url: &str, key: &str, field: &str) -> KvResult<Option<String>> {
    with_connection(url, |conn| conn.hget::<_, _, Option<String>>(key, field))
}

/// Set a raw string value for the specified hash field.
#[cfg(feature = "redis")]
pub fn redis_hset_raw(url: &str, key: &str, field: &str, value: &str) -> KvResult<bool> {
    with_connection(url, |conn| conn.hset(key, field, value))
}

/// Delete a field from a hash, returning `true` if it existed.
#[cfg(feature = "redis")]
pub fn redis_hdel(url: &str, key: &str, field: &str) -> KvResult<bool> {
    let removed: usize = with_connection(url, |conn| conn.hdel(key, field))?;
    Ok(removed > 0)
}

/// Increment an integer value stored at a key by the provided amount.
#[cfg(feature = "redis")]
pub fn redis_incr_by(url: &str, key: &str, amount: i64) -> KvResult<i64> {
    with_connection(url, |conn| conn.incr(key, amount))
}

/// Decrement an integer value stored at a key by the provided amount.
#[cfg(feature = "redis")]
pub fn redis_decr_by(url: &str, key: &str, amount: i64) -> KvResult<i64> {
    with_connection(url, |conn| conn.decr(key, amount))
}

/// Apply an expiry (TTL) in seconds to a key.
#[cfg(feature = "redis")]
pub fn redis_expire(url: &str, key: &str, seconds: usize) -> KvResult<bool> {
    with_connection(url, |conn| conn.expire(key, seconds))
}

/// Retrieve keys by glob-style pattern.
#[cfg(feature = "redis")]
pub fn redis_keys(url: &str, pattern: &str) -> KvResult<Vec<String>> {
    with_connection(url, |conn| conn.keys(pattern))
}

/// Fetch a JSON-encoded value and deserialize it into the requested type.
#[cfg(feature = "redis")]
pub fn redis_get_json<T>(url: &str, key: &str) -> KvResult<Option<T>>
where
    T: DeserializeOwned,
{
    match redis_get_raw(url, key)? {
        Some(raw) => Ok(Some(serde_json::from_str(&raw)?)),
        None => Ok(None),
    }
}

/// Serialize the provided value to JSON and store it at the given key.
#[cfg(feature = "redis")]
pub fn redis_set_json<T>(url: &str, key: &str, value: &T) -> KvResult<()>
where
    T: Serialize,
{
    let payload = serde_json::to_string(value)?;
    redis_set_raw(url, key, &payload)
}

/// Fetch a JSON-encoded hash field and deserialize it into the requested type.
#[cfg(feature = "redis")]
pub fn redis_hget_json<T>(url: &str, key: &str, field: &str) -> KvResult<Option<T>>
where
    T: DeserializeOwned,
{
    match redis_hget_raw(url, key, field)? {
        Some(raw) => Ok(Some(serde_json::from_str(&raw)?)),
        None => Ok(None),
    }
}

/// Serialize the provided value to JSON and store it under the specified hash field.
#[cfg(feature = "redis")]
pub fn redis_hset_json<T>(url: &str, key: &str, field: &str, value: &T) -> KvResult<bool>
where
    T: Serialize,
{
    let payload = serde_json::to_string(value)?;
    redis_hset_raw(url, key, field, &payload)
}
