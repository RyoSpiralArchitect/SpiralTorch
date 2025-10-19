// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "redis")]
use redis::{Commands, FromRedisValue, ToRedisArgs};
#[cfg(feature = "redis")]
use serde::de::DeserializeOwned;
#[cfg(feature = "redis")]
use serde::Serialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KvErr {
    #[error("redis error: {0}")]
    Redis(String),
    #[cfg(feature = "redis")]
    #[error("serde error: {0}")]
    Serde(String),
}

#[cfg(feature = "redis")]
pub fn redis_get_raw(url: &str, key: &str) -> Result<Option<String>, KvErr> {
    with_redis(url, |kv| kv.get(key))
}
#[cfg(feature = "redis")]
pub fn redis_lrange(url: &str, key: &str, start: isize, stop: isize) -> Result<Vec<String>, KvErr> {
    with_redis(url, |kv| kv.lrange(key, start, stop))
}

#[cfg(feature = "redis")]
pub fn redis_set_raw<V: ToRedisArgs>(url: &str, key: &str, value: V) -> Result<(), KvErr> {
    with_redis(url, |kv| kv.set(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_del(url: &str, key: &str) -> Result<usize, KvErr> {
    with_redis(url, |kv| kv.del(key))
}

#[cfg(feature = "redis")]
pub fn redis_exists(url: &str, key: &str) -> Result<bool, KvErr> {
    with_redis(url, |kv| kv.exists(key))
}

#[cfg(feature = "redis")]
pub fn redis_lpush<V: ToRedisArgs>(url: &str, key: &str, values: V) -> Result<usize, KvErr> {
    with_redis(url, |kv| kv.lpush(key, values))
}

#[cfg(feature = "redis")]
pub fn redis_rpush<V: ToRedisArgs>(url: &str, key: &str, values: V) -> Result<usize, KvErr> {
    with_redis(url, |kv| kv.rpush(key, values))
}

#[cfg(feature = "redis")]
pub fn redis_ltrim(url: &str, key: &str, start: isize, stop: isize) -> Result<(), KvErr> {
    with_redis(url, |kv| kv.ltrim(key, start, stop))
}

#[cfg(feature = "redis")]
pub fn redis_set_json<T: Serialize>(url: &str, key: &str, value: &T) -> Result<(), KvErr> {
    with_redis(url, |kv| kv.set_json(key, value))
}

#[cfg(feature = "redis")]
pub fn redis_get_json<T: DeserializeOwned>(url: &str, key: &str) -> Result<Option<T>, KvErr> {
    with_redis(url, |kv| kv.get_json(key))
}

#[cfg(feature = "redis")]
pub fn with_redis<F, R>(url: &str, f: F) -> Result<R, KvErr>
where
    F: FnOnce(&mut RedisKv) -> Result<R, KvErr>,
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
    pub fn connect(url: &str) -> Result<Self, KvErr> {
        let client = redis::Client::open(url).map_err(|e| KvErr::Redis(e.to_string()))?;
        Self::from_client(client)
    }

    pub fn from_client(client: redis::Client) -> Result<Self, KvErr> {
        let conn = client
            .get_connection()
            .map_err(|e| KvErr::Redis(e.to_string()))?;
        Ok(Self { conn })
    }

    pub fn get<T>(&mut self, key: &str) -> Result<Option<T>, KvErr>
    where
        T: FromRedisValue,
    {
        let value: Option<T> = self
            .conn
            .get(key)
            .map_err(|e| KvErr::Redis(e.to_string()))?;
        Ok(value)
    }

    pub fn set<V>(&mut self, key: &str, value: V) -> Result<(), KvErr>
    where
        V: ToRedisArgs,
    {
        self.conn
            .set::<_, _, ()>(key, value)
            .map_err(|e| KvErr::Redis(e.to_string()))?;
        Ok(())
    }

    pub fn del(&mut self, key: &str) -> Result<usize, KvErr> {
        self.conn.del(key).map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn exists(&mut self, key: &str) -> Result<bool, KvErr> {
        self.conn
            .exists(key)
            .map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn lrange<T>(&mut self, key: &str, start: isize, stop: isize) -> Result<Vec<T>, KvErr>
    where
        T: FromRedisValue,
    {
        let list: Vec<T> = self
            .conn
            .lrange(key, start, stop)
            .map_err(|e| KvErr::Redis(e.to_string()))?;
        Ok(list)
    }

    pub fn lpush<V>(&mut self, key: &str, values: V) -> Result<usize, KvErr>
    where
        V: ToRedisArgs,
    {
        self.conn
            .lpush(key, values)
            .map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn rpush<V>(&mut self, key: &str, values: V) -> Result<usize, KvErr>
    where
        V: ToRedisArgs,
    {
        self.conn
            .rpush(key, values)
            .map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn ltrim(&mut self, key: &str, start: isize, stop: isize) -> Result<(), KvErr> {
        self.conn
            .ltrim(key, start, stop)
            .map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn expire(&mut self, key: &str, seconds: usize) -> Result<bool, KvErr> {
        self.conn
            .expire(key, seconds)
            .map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn ttl(&mut self, key: &str) -> Result<isize, KvErr> {
        self.conn.ttl(key).map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn incr_by<N>(&mut self, key: &str, amount: N) -> Result<N, KvErr>
    where
        N: ToRedisArgs + FromRedisValue,
    {
        self.conn
            .incr(key, amount)
            .map_err(|e| KvErr::Redis(e.to_string()))
    }

    pub fn set_json<T>(&mut self, key: &str, value: &T) -> Result<(), KvErr>
    where
        T: Serialize,
    {
        let payload = serde_json::to_string(value).map_err(|e| KvErr::Serde(e.to_string()))?;
        self.set(key, payload)
    }

    pub fn get_json<T>(&mut self, key: &str) -> Result<Option<T>, KvErr>
    where
        T: DeserializeOwned,
    {
        match self.get::<String>(key)? {
            Some(raw) => serde_json::from_str(&raw)
                .map(Some)
                .map_err(|e| KvErr::Serde(e.to_string())),
            None => Ok(None),
        }
    }
}
