// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Unison mediator: read Redis bucket, compute median Choice, inject low-weight soft rules.
use crate::backend::wgpu_heuristics::Choice;
#[cfg(feature="logic")]
use st_logic::{SoftRule, Field, Value};

#[cfg(feature="kv-redis")]
pub fn soft_from_redis(rows:u32, cols:u32, k:u32, subgroup:bool) -> Vec<SoftRule> {
    let mut out = Vec::new();
    let url = match std::env::var("REDIS_URL"){ Ok(u)=>u, Err(_)=> return out };
    let lg2c = (32 - (cols.max(1)-1).leading_zeros()) as u32;
    let lg2k = (32 - (k.max(1)-1).leading_zeros()) as u32;
    let key = format!("spiral:heur:v1:sg:{}:c:{}:k:{}", if subgroup{1}else{0}, lg2c, lg2k);
    if let Ok(Some(v)) = st_kv::redis_get_choice(&url, &key) {
        // low-weight nudges
        out.push(SoftRule{ field: Field::Use2ce, value: Value::B(v.use_2ce), weight: 0.05 });
        out.push(SoftRule{ field: Field::Wg,     value: Value::U(v.wg),      weight: 0.05 });
        out.push(SoftRule{ field: Field::Kl,     value: Value::U(v.kl),      weight: 0.05 });
        out.push(SoftRule{ field: Field::Ch,     value: Value::U(v.ch),      weight: 0.05 });
    }
    out
}
#[cfg(not(feature="kv-redis"))]
pub fn soft_from_redis(_rows:u32,_cols:u32,_k:u32,_subgroup:bool)->Vec<()> { Vec::new() }
