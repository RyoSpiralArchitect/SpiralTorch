// crates/st-kv/src/lib.rs  (v1.9.0)
#[cfg(feature="redis")]
use redis::Commands;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KvErr {
    #[error("redis error: {0}")]
    Redis(String),
}

#[cfg(feature="redis")]
pub fn redis_get_raw(url:&str, key:&str) -> Result<Option<String>, KvErr> {
    let client = redis::Client::open(url).map_err(|e|KvErr::Redis(e.to_string()))?;
    let mut conn = client.get_connection().map_err(|e|KvErr::Redis(e.to_string()))?;
    let s: Option<String> = conn.get(key).map_err(|e|KvErr::Redis(e.to_string()))?;
    Ok(s)
}
