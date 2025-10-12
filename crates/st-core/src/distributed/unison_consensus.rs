//! Unison consensus via KV and local observations.

use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeurChoice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub prob: f32,   // optional: probability/confidence
}

pub fn combine_trimmed_mean(samples:&[HeurChoice]) -> Option<HeurChoice> {
    if samples.is_empty(){ return None }
    // crude: pick majority vote on booleans; mean on numerics (ignoring prob for now)
    let u2 = samples.iter().filter(|s|s.use_2ce).count() * 2 >= samples.len();
    let wg = (samples.iter().map(|s|s.wg as f32).sum::<f32>() / samples.len() as f32).round() as u32;
    let kl = (samples.iter().map(|s|s.kl as f32).sum::<f32>() / samples.len() as f32).round() as u32;
    let ch = (samples.iter().map(|s|s.ch as f32).sum::<f32>() / samples.len() as f32).round() as u32;
    let prob = samples.iter().map(|s|s.prob).sum::<f32>() / samples.len() as f32;
    Some(HeurChoice{ use_2ce:u2, wg, kl, ch, prob })
}
