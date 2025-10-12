//! Pull soft-rule medians from Redis and inject as low-weight soft(...).
//! Key space (example): spiral:soft:v1:bucket:{rowsLg2}:{colsLg2}:{kLg2}
//! Value: JSON like {"mk":2,"weight":0.1} meaning prefer warp keep-k with small weight.

#[cfg(feature="kv-redis")]
use redis::Commands;

pub struct SoftInject { pub line: String, pub weight: f32 }

fn lg2(x:usize)->u32 { (std::mem::size_of::<usize>() as u32 * 8 - ((x.max(1)-1) as u32).leading_zeros()) }

pub fn fetch_soft_from_redis(rows:usize, cols:usize, k:usize) -> Option<SoftInject> {
    let url = std::env::var("REDIS_URL").ok()?;
    #[cfg(feature="kv-redis")]
    {
        let client = redis::Client::open(url).ok()?;
        let mut conn = client.get_connection().ok()?;
        let key = format!("spiral:soft:v1:bucket:{}:{}:{}", lg2(rows), lg2(cols), lg2(k));
        let s: Option<String> = conn.get(key).ok()?;
        if let Some(js) = s {
            let v: serde_json::Value = serde_json::from_str(&js).ok()?;
            if let Some(mk) = v.get("mk").and_then(|x|x.as_u64()) {
                let w = v.get("weight").and_then(|x|x.as_f64()).unwrap_or(0.1);
                return Some(SoftInject{ line: format!("soft(mk, {}, {}, 1)", mk), weight: w as f32 });
            }
        }
    }
    None
}
