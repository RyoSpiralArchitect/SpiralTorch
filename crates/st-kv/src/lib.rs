#[cfg(feature="redis")]
use redis::Commands;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default)]
pub struct Choice { pub use_2ce: bool, pub wg:u32, pub kl:u32, pub ch:u32 }

#[derive(Error, Debug)]
pub enum KvErr {
    #[error("redis error: {0}")]
    Redis(String),
    #[error("json error")]
    Json,
}

#[cfg(feature="redis")]
pub fn redis_get_choice(url:&str, key:&str) -> Result<Option<Choice>, KvErr> {
    let client = redis::Client::open(url).map_err(|e|KvErr::Redis(e.to_string()))?;
    let mut conn = client.get_connection().map_err(|e|KvErr::Redis(e.to_string()))?;
    let s: Option<String> = conn.get(key).map_err(|e|KvErr::Redis(e.to_string()))?;
    if let Some(js) = s {
        #[derive(serde::Deserialize)]
        struct V{ use_2ce:bool, wg:u32, kl:u32, ch:u32 }
        let v: V = serde_json::from_str(&js).map_err(|_|KvErr::Json)?;
        Ok(Some(Choice{ use_2ce:v.use_2ce, wg:v.wg, kl:v.kl, ch:v.ch }))
    } else { Ok(None) }
}

#[cfg(not(feature="redis"))]
pub fn redis_get_choice(_url:&str, _key:&str) -> Result<Option<Choice>, ()> { Ok(None) }
