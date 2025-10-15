// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "redis")]
use redis::Commands;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KvErr {
    #[error("redis error: {0}")]
    Redis(String),
}

#[cfg(feature = "redis")]
pub fn redis_get_raw(url: &str, key: &str) -> Result<Option<String>, KvErr> {
    let client = redis::Client::open(url).map_err(|e| KvErr::Redis(e.to_string()))?;
    let mut conn = client
        .get_connection()
        .map_err(|e| KvErr::Redis(e.to_string()))?;
    let s: Option<String> = conn.get(key).map_err(|e| KvErr::Redis(e.to_string()))?;
    Ok(s)
}
#[cfg(feature = "redis")]
pub fn redis_lrange(url: &str, key: &str, start: isize, stop: isize) -> Result<Vec<String>, KvErr> {
    let client = redis::Client::open(url).map_err(|e| KvErr::Redis(e.to_string()))?;
    let mut conn = client
        .get_connection()
        .map_err(|e| KvErr::Redis(e.to_string()))?;
    let list: Vec<String> = conn
        .lrange(key, start, stop)
        .map_err(|e| KvErr::Redis(e.to_string()))?;
    Ok(list)
}
