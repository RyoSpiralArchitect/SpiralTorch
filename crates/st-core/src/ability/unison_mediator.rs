// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Unison mediator: read Redis bucket, compute median Choice, inject low-weight soft rules.
use crate::backend::soft_logic::SoftRule;
use crate::backend::wgpu_heuristics::{
    SOFT_NAME_CH, SOFT_NAME_KL, SOFT_NAME_USE2CE, SOFT_NAME_WG,
};

#[cfg(feature="kv-redis")]
pub fn soft_from_redis(rows:u32, cols:u32, k:u32, subgroup:bool) -> Vec<SoftRule> {
    let mut out = Vec::new();
    let url = match std::env::var("REDIS_URL"){ Ok(u)=>u, Err(_)=> return out };
    let lg2c = (32 - (cols.max(1)-1).leading_zeros()) as u32;
    let lg2k = (32 - (k.max(1)-1).leading_zeros()) as u32;
    let key = format!("spiral:heur:v1:sg:{}:c:{}:k:{}", if subgroup{1}else{0}, lg2c, lg2k);
    if let Ok(Some(v)) = st_kv::redis_get_choice(&url, &key) {
        // low-weight nudges
        out.push(SoftRule{ name: SOFT_NAME_USE2CE, weight: 0.05, score: if v.use_2ce { 1.0 } else { -1.0 } });
        out.push(SoftRule{ name: SOFT_NAME_WG,     weight: 0.05, score: v.wg as f32 });
        out.push(SoftRule{ name: SOFT_NAME_KL,     weight: 0.05, score: v.kl as f32 });
        out.push(SoftRule{ name: SOFT_NAME_CH,     weight: 0.05, score: v.ch as f32 });
    }
    out
}
#[cfg(not(feature="kv-redis"))]
pub fn soft_from_redis(_rows:u32,_cols:u32,_k:u32,_subgroup:bool)->Vec<()> { Vec::new() }
