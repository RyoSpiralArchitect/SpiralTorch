// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[derive(Clone, Debug)]
pub struct TopKShard<T> {
    pub vals: Vec<T>,
    pub idxs: Vec<i32>,
}

pub fn merge_two_shards_f32(a: &TopKShard<f32>, b: &TopKShard<f32>, k: usize) -> TopKShard<f32> {
    // naive merge for demo (stable)
    let mut pairs: Vec<(f32, i32)> = a
        .vals
        .iter()
        .cloned()
        .zip(a.idxs.iter().cloned())
        .chain(b.vals.iter().cloned().zip(b.idxs.iter().cloned()))
        .collect();
    pairs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    let mut out_vals = Vec::with_capacity(k);
    let mut out_idx = Vec::with_capacity(k);
    for (v, i) in pairs.into_iter().take(k) {
        out_vals.push(v);
        out_idx.push(i);
    }
    TopKShard {
        vals: out_vals,
        idxs: out_idx,
    }
}
