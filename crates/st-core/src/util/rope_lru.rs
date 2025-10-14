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

/// Minimal rotating positional embedding cache with LRU eviction.
pub struct RopeLRU {
    cap: usize,
    keys: VecDeque<RopeKey>,
    cos: Vec<Vec<f64>>,
    sin: Vec<Vec<f64>>,
}

impl RopeLRU {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            keys: VecDeque::new(),
            cos: Vec::new(),
            sin: Vec::new(),
        }
    }

    pub fn get(&mut self, key: RopeKey) -> (&[f64], &[f64]) {
        if let Some(pos) = self.keys.iter().position(|k| k == &key) {
            self.keys.rotate_left(pos);
            self.cos.rotate_left(pos);
            self.sin.rotate_left(pos);
            return (&self.cos[0], &self.sin[0]);
        }

        let mut cos = vec![0.0; key.t];
        let mut sin = vec![0.0; key.t];
        let mut phase: f64 = 0.0;
        for idx in 0..key.t {
            cos[idx] = phase.cos();
            sin[idx] = phase.sin();
            phase = (phase + key.theta) % TAU;
        }

        self.keys.push_front(key);
        self.cos.insert(0, cos);
        self.sin.insert(0, sin);

        while self.keys.len() > self.cap {
            self.keys.pop_back();
            self.cos.pop();
            self.sin.pop();
        }

        (&self.cos[0], &self.sin[0])
    }
}
