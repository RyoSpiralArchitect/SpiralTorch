// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct RopeKey {
    pub theta: f64,
    pub t: usize,
    pub dtype: &'static str,
    pub device: &'static str,
}

impl PartialEq for RopeKey {
    fn eq(&self, other: &Self) -> bool {
        (self.theta - other.theta).abs() < 1e-9
            && self.t == other.t
            && self.dtype == other.dtype
            && self.device == other.device
    }
}

impl Eq for RopeKey {}

impl Hash for RopeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.theta.to_bits().hash(state);
        self.t.hash(state);
        self.dtype.hash(state);
        self.device.hash(state);
    }
}

struct RopeEntry {
    key: RopeKey,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl RopeEntry {
    fn new(key: RopeKey) -> Self {
        let mut cos = vec![0.0; key.t];
        let mut sin = vec![0.0; key.t];
        let mut phase: f64 = 0.0;
        for idx in 0..key.t {
            cos[idx] = phase.cos();
            sin[idx] = phase.sin();
            phase = (phase + key.theta) % TAU;
        }
        Self { key, cos, sin }
    }
}

/// Minimal rotating positional embedding cache with LRU eviction.
pub struct RopeLRU {
    cap: usize,
    entries: VecDeque<RopeEntry>,
}

impl RopeLRU {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            entries: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: RopeKey) -> (&[f64], &[f64]) {
        if let Some(pos) = self.entries.iter().position(|entry| entry.key == key) {
            if pos != 0 {
                if let Some(entry) = self.entries.remove(pos) {
                    self.entries.push_front(entry);
                }
            }
            let entry = self.entries.front().expect("entries non-empty");
            return (entry.cos.as_slice(), entry.sin.as_slice());
        }

        self.entries.push_front(RopeEntry::new(key));
        if self.entries.len() > self.cap {
            self.entries.pop_back();
        }
        let entry = self
            .entries
            .front()
            .expect("entries non-empty after insert");
        (entry.cos.as_slice(), entry.sin.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    fn make_key(theta: f64, t: usize) -> RopeKey {
        RopeKey {
            theta,
            t,
            dtype: "f16",
            device: "wgpu",
        }
    }

    #[test]
    fn retrieves_cached_entries_without_reallocation() {
        let mut lru = RopeLRU::new(4);
        let key = make_key(0.42, 8);
        let (cos_ptr_first, sin_ptr_first) = {
            let (cos, sin) = lru.get(key.clone());
            (cos.as_ptr(), sin.as_ptr())
        };
        let (cos_ptr_second, sin_ptr_second) = {
            let (cos, sin) = lru.get(key);
            (cos.as_ptr(), sin.as_ptr())
        };
        assert!(ptr::eq(cos_ptr_first, cos_ptr_second));
        assert!(ptr::eq(sin_ptr_first, sin_ptr_second));
    }

    #[test]
    fn zero_capacity_promotes_to_minimum_one() {
        let mut lru = RopeLRU::new(0);
        let (cos, sin) = lru.get(make_key(0.1, 3));
        assert_eq!(cos.len(), 3);
        assert_eq!(sin.len(), 3);
    }
}
