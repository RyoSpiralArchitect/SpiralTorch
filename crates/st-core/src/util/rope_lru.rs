// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::hash::{Hash, Hasher};

/// Exact cache identity, including signed zero and NaN payload bits.
/// Nearby angles must not reuse each other's phase sequence. Bitwise identity
/// defines Eq/Hash even for non-finite keys; it is not a finite-value validator.
#[derive(Clone, Debug)]
pub struct RopeKey {
    pub theta: f64,
    pub t: usize,
    pub dtype: &'static str,
    pub device: &'static str,
}

impl PartialEq for RopeKey {
    fn eq(&self, other: &Self) -> bool {
        self.theta.to_bits() == other.theta.to_bits()
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

    #[test]
    fn equality_and_hash_share_exact_bit_identity() {
        use std::collections::{hash_map::DefaultHasher, HashSet};
        let hash = |key: &RopeKey| {
            let mut h = DefaultHasher::new();
            key.hash(&mut h);
            h.finish()
        };
        let patterns = [
            0u64,
            1,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7ff8_0000_0000_0001,
            0x3ff0_0000_0000_0000,
        ];
        let mut keys = HashSet::new();
        for bits in patterns {
            let key = make_key(f64::from_bits(bits), 8);
            assert_eq!(key, key.clone());
            assert_eq!(hash(&key), hash(&key.clone()));
            assert!(keys.insert(key.clone()));
            assert!(!keys.insert(key));
        }
        assert_eq!(keys.len(), patterns.len());
        let a = make_key(1.0, 8);
        let b = make_key(1.0 + 5e-10, 8);
        let c = make_key(1.0 + 1.25e-9, 8);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn nearby_angle_requests_do_not_depend_on_cache_history() {
        let mut populated = RopeLRU::new(2);
        let first = populated.get(make_key(1.0, 32)).0.to_vec();
        let (cos, sin) = populated.get(make_key(1.0 + 5e-10, 32));
        let actual = (cos.to_vec(), sin.to_vec());
        let mut fresh = RopeLRU::new(2);
        let (cos, sin) = fresh.get(make_key(1.0 + 5e-10, 32));
        assert_eq!(actual, (cos.to_vec(), sin.to_vec()));
        assert_ne!(first, actual.0);
    }

    #[test]
    fn hits_promote_and_eviction_respects_all_key_fields() {
        let a = make_key(0.25, 8);
        let mut b = a.clone();
        b.dtype = "f32";
        let mut c = a.clone();
        c.device = "cpu";
        let mut d = a.clone();
        d.t = 9;
        let mut cache = RopeLRU::new(3);
        let pointer = cache.get(a.clone()).0.as_ptr();
        cache.get(b.clone());
        cache.get(c.clone());
        assert_eq!(cache.get(a.clone()).0.as_ptr(), pointer);
        cache.get(d.clone());
        assert_eq!(
            cache
                .entries
                .iter()
                .map(|e| e.key.clone())
                .collect::<Vec<_>>(),
            vec![d, a, c]
        );
        assert!(!cache.entries.iter().any(|e| e.key == b));
    }

    #[test]
    fn finite_phases_keep_the_existing_recurrence_and_empty_sequence() {
        let mut cache = RopeLRU::new(2);
        for theta in [0.0, -0.0, -0.25, 0.42, TAU, 1e100] {
            let mut phase: f64 = 0.0;
            let (cos, sin) = cache.get(make_key(theta, 64));
            for (&c, &s) in cos.iter().zip(sin) {
                // libm calls and compile-time folding may differ by one ULP;
                // exact bit identity is required of keys/hits, not libm.
                assert!((c - phase.cos()).abs() <= 4.0 * f64::EPSILON);
                assert!((s - phase.sin()).abs() <= 4.0 * f64::EPSILON);
                phase = (phase + theta) % TAU;
            }
        }
        let (cos, sin) = cache.get(make_key(0.25, 0));
        assert!(cos.is_empty() && sin.is_empty());
    }
}
