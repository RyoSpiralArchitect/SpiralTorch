// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

mod error;
pub use error::KvErr;

/// Result alias specialised for key-value helper routines.
pub type KvResult<T> = std::result::Result<T, KvErr>;

#[cfg(feature = "redis")]
mod json;
#[cfg(feature = "redis")]
pub use json::{JsonExpiry, JsonSetCondition, JsonSetOptions, JsonSetOptionsBuilder};

#[cfg(feature = "redis")]
use redis::{Commands, FromRedisValue, Pipeline, ToRedisArgs, Value};
#[cfg(feature = "redis")]
use serde::de::DeserializeOwned;
#[cfg(feature = "redis")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "redis")]
use std::collections::HashMap;

#[cfg(feature = "redis")]
pub fn redis_get_raw(url: &str, key: &str) -> KvResult<Option<String>> {
    with_redis(url, |kv| kv.get(key))
}

#[cfg(feature = "redis")]
pub fn redis_lrange(url: &str, key: &str, start: isize, stop: isize) -> KvResult<Vec<String>> {
    with_redis(url, |kv| kv.lrange(key, start, stop))
}

#[cfg(feature = "redis")]
pub fn redis_set_raw<V: ToRedisArgs>(url: &str, key: &str, value: V) -> KvResult<()> {
    with_redis(url, |kv| kv.set(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_set_ex<V: ToRedisArgs>(
    url: &str,
    key: &str,
    value: V,
    seconds: usize,
) -> KvResult<()> {
    with_redis(url, |kv| kv.set_ex(key, value, seconds))
}

#[cfg(feature = "redis")]
pub fn redis_set_nx<V: ToRedisArgs>(url: &str, key: &str, value: V) -> KvResult<bool> {
    with_redis(url, |kv| kv.set_nx(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_getset<T, V>(url: &str, key: &str, value: V) -> KvResult<Option<T>>
where
    T: FromRedisValue,
    V: ToRedisArgs,
{
    with_redis(url, |kv| kv.getset(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_del(url: &str, key: &str) -> KvResult<usize> {
    with_redis(url, |kv| kv.del(key))
}

#[cfg(feature = "redis")]
pub fn redis_exists(url: &str, key: &str) -> KvResult<bool> {
    with_redis(url, |kv| kv.exists(key))
}

#[cfg(feature = "redis")]
pub fn redis_expire(url: &str, key: &str, seconds: usize) -> KvResult<bool> {
    with_redis(url, |kv| kv.expire(key, seconds))
}

#[cfg(feature = "redis")]
pub fn redis_ttl(url: &str, key: &str) -> KvResult<isize> {
    with_redis(url, |kv| kv.ttl(key))
}

#[cfg(feature = "redis")]
pub fn redis_incr_by<N>(url: &str, key: &str, amount: N) -> KvResult<N>
where
    N: ToRedisArgs + FromRedisValue,
{
    with_redis(url, |kv| kv.incr_by(key, amount))
}

#[cfg(feature = "redis")]
pub fn redis_lpush<V: ToRedisArgs>(url: &str, key: &str, values: V) -> KvResult<usize> {
    with_redis(url, |kv| kv.lpush(key, values))
}

#[cfg(feature = "redis")]
pub fn redis_rpush<V: ToRedisArgs>(url: &str, key: &str, values: V) -> KvResult<usize> {
    with_redis(url, |kv| kv.rpush(key, values))
}

#[cfg(feature = "redis")]
pub fn redis_lpush_json<T: Serialize>(url: &str, key: &str, value: &T) -> KvResult<usize> {
    with_redis(url, |kv| kv.lpush_json(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_rpush_json<T: Serialize>(url: &str, key: &str, value: &T) -> KvResult<usize> {
    with_redis(url, |kv| kv.rpush_json(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_ltrim(url: &str, key: &str, start: isize, stop: isize) -> KvResult<()> {
    with_redis(url, |kv| kv.ltrim(key, start, stop))
}

#[cfg(feature = "redis")]
pub fn redis_lrange_json<T: DeserializeOwned>(
    url: &str,
    key: &str,
    start: isize,
    stop: isize,
) -> KvResult<Vec<T>> {
    with_redis(url, |kv| kv.lrange_json(key, start, stop))
}

#[cfg(feature = "redis")]
pub fn redis_set_json<T: Serialize>(url: &str, key: &str, value: &T) -> KvResult<()> {
    with_redis(url, |kv| kv.set_json(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_get_json<T: DeserializeOwned>(url: &str, key: &str) -> KvResult<Option<T>> {
    with_redis(url, |kv| kv.get_json(key))
}

#[cfg(feature = "redis")]
pub fn redis_set_json_ex<T: Serialize>(
    url: &str,
    key: &str,
    value: &T,
    seconds: usize,
) -> KvResult<()> {
    with_redis(url, |kv| kv.set_json_ex(key, value, seconds))
}

#[cfg(feature = "redis")]
pub fn redis_set_json_nx<T: Serialize>(url: &str, key: &str, value: &T) -> KvResult<bool> {
    with_redis(url, |kv| kv.set_json_nx(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_set_json_with_options<T: Serialize>(
    url: &str,
    key: &str,
    value: &T,
    options: &JsonSetOptions,
) -> KvResult<bool> {
    with_redis(url, |kv| kv.set_json_with_options(key, value, options))
}

#[cfg(feature = "redis")]
pub fn redis_get_or_set_json<T, F>(url: &str, key: &str, default: F) -> KvResult<T>
where
    T: Serialize + DeserializeOwned,
    F: FnOnce() -> T,
{
    with_redis(url, |kv| kv.get_or_set_json(key, default))
}

#[cfg(feature = "redis")]
pub fn redis_get_or_set_json_with_options<T, F>(
    url: &str,
    key: &str,
    default: F,
    options: &JsonSetOptions,
) -> KvResult<T>
where
    T: Serialize + DeserializeOwned,
    F: FnOnce() -> T,
{
    with_redis(url, |kv| {
        kv.get_or_set_json_with_options(key, default, options)
    })
}

#[cfg(feature = "redis")]
pub fn redis_hset<F, V>(url: &str, key: &str, field: F, value: V) -> KvResult<bool>
where
    F: ToRedisArgs,
    V: ToRedisArgs,
{
    with_redis(url, |kv| kv.hset(key, field, value))
}

#[cfg(feature = "redis")]
pub fn redis_hget<F, T>(url: &str, key: &str, field: F) -> KvResult<Option<T>>
where
    F: ToRedisArgs,
    T: FromRedisValue,
{
    with_redis(url, |kv| kv.hget(key, field))
}

#[cfg(feature = "redis")]
pub fn redis_hdel<F: ToRedisArgs>(url: &str, key: &str, field: F) -> KvResult<usize> {
    with_redis(url, |kv| kv.hdel(key, field))
}

#[cfg(feature = "redis")]
pub fn redis_hgetall<T: FromRedisValue>(url: &str, key: &str) -> KvResult<HashMap<String, T>> {
    with_redis(url, |kv| kv.hgetall(key))
}

#[cfg(feature = "redis")]
pub fn redis_pipeline<R, F>(url: &str, build: F) -> KvResult<R>
where
    R: FromRedisValue,
    F: FnOnce(Pipeline) -> Pipeline,
{
    with_redis(url, |kv| kv.pipeline(build))
}

#[cfg(feature = "redis")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub algo_topk: u8,
    pub ctile: u32,
    pub mode_midk: u8,
    pub mode_bottomk: u8,
    pub tile_cols: u32,
    pub radix: u32,
    pub segments: u32,
}

#[cfg(feature = "redis")]
pub fn redis_set_choice(url: &str, key: &str, choice: &Choice) -> KvResult<()> {
    redis_set_json(url, key, choice)
}

#[cfg(feature = "redis")]
pub fn redis_set_choice_ex(url: &str, key: &str, choice: &Choice, seconds: usize) -> KvResult<()> {
    redis_set_json_ex(url, key, choice, seconds)
}

#[cfg(feature = "redis")]
pub fn redis_set_choice_nx(url: &str, key: &str, choice: &Choice) -> KvResult<bool> {
    redis_set_json_nx(url, key, choice)
}

#[cfg(feature = "redis")]
pub fn redis_set_choice_with_options(
    url: &str,
    key: &str,
    choice: &Choice,
    options: &JsonSetOptions,
) -> KvResult<bool> {
    redis_set_json_with_options(url, key, choice, options)
}

#[cfg(feature = "redis")]
pub fn redis_get_choice(url: &str, key: &str) -> KvResult<Option<Choice>> {
    redis_get_json(url, key)
}

#[cfg(feature = "redis")]
pub fn redis_get_or_set_choice<F>(url: &str, key: &str, default: F) -> KvResult<Choice>
where
    F: FnOnce() -> Choice,
{
    redis_get_or_set_json(url, key, default)
}

#[cfg(feature = "redis")]
pub fn redis_get_or_set_choice_with_options<F>(
    url: &str,
    key: &str,
    default: F,
    options: &JsonSetOptions,
) -> KvResult<Choice>
where
    F: FnOnce() -> Choice,
{
    redis_get_or_set_json_with_options(url, key, default, options)
}

#[cfg(feature = "redis")]
pub fn redis_push_choice(
    url: &str,
    key: &str,
    choice: &Choice,
    max_len: Option<usize>,
) -> KvResult<usize> {
    with_redis(url, |kv| {
        let len = kv.lpush_json(key, choice)?;
        if let Some(max) = max_len {
            if max == 0 {
                kv.del(key)?;
            } else {
                let end = max.saturating_sub(1) as isize;
                kv.ltrim(key, 0, end)?;
            }
        }
        Ok(len)
    })
}

#[cfg(feature = "redis")]
pub fn redis_lrange_choice(
    url: &str,
    key: &str,
    start: isize,
    stop: isize,
) -> KvResult<Vec<Choice>> {
    redis_lrange_json(url, key, start, stop)
}

#[cfg(feature = "redis")]
pub fn with_redis<F, R>(url: &str, f: F) -> KvResult<R>
where
    F: FnOnce(&mut RedisKv) -> KvResult<R>,
{
    let mut kv = RedisKv::connect(url)?;
    f(&mut kv)
}

#[cfg(feature = "redis")]
pub struct RedisKv {
    conn: redis::Connection,
}

#[cfg(feature = "redis")]
impl RedisKv {
    pub fn connect(url: &str) -> KvResult<Self> {
        let client = redis::Client::open(url)?;
        Self::from_client(client)
    }

    pub fn from_client(client: redis::Client) -> KvResult<Self> {
        let conn = client.get_connection()?;
        Ok(Self { conn })
    }

    pub fn into_inner(self) -> redis::Connection {
        self.conn
    }

    pub fn get<T: FromRedisValue>(&mut self, key: &str) -> KvResult<Option<T>> {
        Ok(self.conn.get(key)?)
    }

    pub fn lrange<T: FromRedisValue>(
        &mut self,
        key: &str,
        start: isize,
        stop: isize,
    ) -> KvResult<Vec<T>> {
        Ok(self.conn.lrange(key, start, stop)?)
    }

    pub fn lpush<V: ToRedisArgs>(&mut self, key: &str, values: V) -> KvResult<usize> {
        Ok(self.conn.lpush(key, values)?)
    }

    pub fn rpush<V: ToRedisArgs>(&mut self, key: &str, values: V) -> KvResult<usize> {
        Ok(self.conn.rpush(key, values)?)
    }

    pub fn ltrim(&mut self, key: &str, start: isize, stop: isize) -> KvResult<()> {
        Ok(self.conn.ltrim(key, start, stop)?)
    }

    pub fn lpush_json<T>(&mut self, key: &str, value: &T) -> KvResult<usize>
    where
        T: Serialize,
    {
        let payload = serde_json::to_string(value)?;
        self.lpush(key, payload)
    }

    pub fn rpush_json<T>(&mut self, key: &str, value: &T) -> KvResult<usize>
    where
        T: Serialize,
    {
        let payload = serde_json::to_string(value)?;
        self.rpush(key, payload)
    }

    pub fn lrange_json<T>(&mut self, key: &str, start: isize, stop: isize) -> KvResult<Vec<T>>
    where
        T: DeserializeOwned,
    {
        let entries: Vec<String> = self.lrange(key, start, stop)?;
        entries
            .into_iter()
            .map(|raw| serde_json::from_str(&raw).map_err(KvErr::from))
            .collect()
    }

    pub fn set<V: ToRedisArgs>(&mut self, key: &str, value: V) -> KvResult<()> {
        Ok(self.conn.set(key, value)?)
    }

    pub fn set_ex<V: ToRedisArgs>(&mut self, key: &str, value: V, seconds: usize) -> KvResult<()> {
        Ok(self.conn.set_ex(key, value, seconds)?)
    }

    pub fn set_nx<V: ToRedisArgs>(&mut self, key: &str, value: V) -> KvResult<bool> {
        Ok(self.conn.set_nx(key, value)?)
    }

    pub fn getset<T, V>(&mut self, key: &str, value: V) -> KvResult<Option<T>>
    where
        T: FromRedisValue,
        V: ToRedisArgs,
    {
        Ok(self.conn.getset(key, value)?)
    }

    pub fn del(&mut self, key: &str) -> KvResult<usize> {
        Ok(self.conn.del(key)?)
    }

    pub fn exists(&mut self, key: &str) -> KvResult<bool> {
        Ok(self.conn.exists(key)?)
    }

    pub fn expire(&mut self, key: &str, seconds: usize) -> KvResult<bool> {
        Ok(self.conn.expire(key, seconds)?)
    }

    pub fn ttl(&mut self, key: &str) -> KvResult<isize> {
        Ok(self.conn.ttl(key)?)
    }

    pub fn incr_by<N>(&mut self, key: &str, amount: N) -> KvResult<N>
    where
        N: ToRedisArgs + FromRedisValue,
    {
        Ok(self.conn.incr(key, amount)?)
    }

    pub fn set_json<T: Serialize>(&mut self, key: &str, value: &T) -> KvResult<()> {
        let payload = serde_json::to_string(value)?;
        self.set(key, payload)
    }

    pub fn set_json_with_options<T>(
        &mut self,
        key: &str,
        value: &T,
        options: &JsonSetOptions,
    ) -> KvResult<bool>
    where
        T: Serialize,
    {
        let payload = serde_json::to_string(value)?;
        self.set_json_payload_with_options(key, payload, options)
    }

    pub fn get_json<T>(&mut self, key: &str) -> KvResult<Option<T>>
    where
        T: DeserializeOwned,
    {
        match self.get::<String>(key)? {
            Some(raw) => Ok(Some(serde_json::from_str(&raw)?)),
            None => Ok(None),
        }
    }

    pub fn set_json_ex<T>(&mut self, key: &str, value: &T, seconds: usize) -> KvResult<()>
    where
        T: Serialize,
    {
        let payload = serde_json::to_string(value)?;
        self.set_ex(key, payload, seconds)
    }

    pub fn set_json_nx<T>(&mut self, key: &str, value: &T) -> KvResult<bool>
    where
        T: Serialize,
    {
        let payload = serde_json::to_string(value)?;
        self.set_nx(key, payload)
    }

    fn set_json_payload_with_options(
        &mut self,
        key: &str,
        payload: String,
        options: &JsonSetOptions,
    ) -> KvResult<bool> {
        let mut cmd = redis::cmd("SET");
        cmd.arg(key).arg(payload);

        options.apply_to_command(&mut cmd)?;

        let response: Value = cmd.query(&mut self.conn)?;
        match response {
            Value::Nil => Ok(false),
            Value::Okay | Value::Status(_) | Value::Data(_) => Ok(true),
            other => Err(KvErr::UnexpectedResponse {
                command: "SET",
                response: other,
            }),
        }
    }

    pub fn get_or_set_json<T, F>(&mut self, key: &str, default: F) -> KvResult<T>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce() -> T,
    {
        if let Some(existing) = self.get_json(key)? {
            return Ok(existing);
        }

        let value = default();
        self.set_json(key, &value)?;
        Ok(value)
    }

    pub fn get_or_set_json_with_options<T, F>(
        &mut self,
        key: &str,
        default: F,
        options: &JsonSetOptions,
    ) -> KvResult<T>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce() -> T,
    {
        if let Some(existing) = self.get_json(key)? {
            return Ok(existing);
        }

        let value = default();
        let inserted = self.set_json_with_options(key, &value, options)?;

        if inserted {
            return Ok(value);
        }

        if let Some(existing) = self.get_json(key)? {
            Ok(existing)
        } else {
            Ok(value)
        }
    }

    pub fn hset<F, V>(&mut self, key: &str, field: F, value: V) -> KvResult<bool>
    where
        F: ToRedisArgs,
        V: ToRedisArgs,
    {
        let updated: usize = self.conn.hset(key, field, value)?;
        Ok(updated > 0)
    }

    pub fn hget<F, T>(&mut self, key: &str, field: F) -> KvResult<Option<T>>
    where
        F: ToRedisArgs,
        T: FromRedisValue,
    {
        Ok(self.conn.hget(key, field)?)
    }

    pub fn hdel<F>(&mut self, key: &str, field: F) -> KvResult<usize>
    where
        F: ToRedisArgs,
    {
        Ok(self.conn.hdel(key, field)?)
    }

    pub fn hgetall<T>(&mut self, key: &str) -> KvResult<HashMap<String, T>>
    where
        T: FromRedisValue,
    {
        Ok(self.conn.hgetall(key)?)
    }

    pub fn execute_pipeline<R>(&mut self, pipeline: Pipeline) -> KvResult<R>
    where
        R: FromRedisValue,
    {
        let result = pipeline.query(&mut self.conn)?;
        Ok(result)
    }

    pub fn pipeline<R, F>(&mut self, build: F) -> KvResult<R>
    where
        R: FromRedisValue,
        F: FnOnce(Pipeline) -> Pipeline,
    {
        let pipeline = build(redis::pipe());
        self.execute_pipeline(pipeline)
    }
}
