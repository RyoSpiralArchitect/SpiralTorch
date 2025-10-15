// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Fractal relation streaming utilities that pair well with async runtimes.
//!
//! The goal of this module is to expose a tiny "Tokio-uring style" scheduler
//! that keeps the memory footprint predictable while continuously folding new
//! relation patches into a coherent Z-space gradient.  The implementation is
//! runtime agnostic; the async helpers simply forward to the synchronous core
//! so that any executor (Tokio, Tokio-uring, async-std, or a manual loop) can
//! drive the queue without extra dependencies.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use super::{PureResult, Tensor, TensorError};

/// Describes a coherent relation patch sampled from a fractal walk.
#[derive(Clone, Debug)]
pub struct FractalPatch {
    relation: Tensor,
    coherence: f32,
    tension: f32,
    depth: u32,
}

impl FractalPatch {
    /// Construct a new relation patch.
    pub fn new(relation: Tensor, coherence: f32, tension: f32, depth: u32) -> PureResult<Self> {
        if coherence <= 0.0 {
            return Err(TensorError::NonPositiveCoherence { coherence });
        }
        if tension <= 0.0 {
            return Err(TensorError::NonPositiveTension { tension });
        }
        Ok(Self {
            relation,
            coherence,
            tension,
            depth,
        })
    }

    /// Returns the original relation tensor.
    pub fn relation(&self) -> &Tensor {
        &self.relation
    }

    /// Coherence weight used when aggregating patches.
    pub fn coherence(&self) -> f32 {
        self.coherence
    }

    /// Tension weight used when attenuating patches.
    pub fn tension(&self) -> f32 {
        self.tension
    }

    /// Depth in the fractal lattice.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Internal helper that normalises the relation energy so that combining
    /// multiple patches only depends on the coherence/tension weights.
    pub fn normalized_relation(&self) -> PureResult<Tensor> {
        let norm = self.relation.squared_l2_norm().sqrt();
        if norm == 0.0 {
            let (rows, cols) = self.relation.shape();
            Tensor::zeros(rows, cols)
        } else {
            self.relation.scale(1.0 / norm)
        }
    }

    /// Weight applied by the streaming scheduler.
    pub fn weight(&self) -> f32 {
        self.coherence / (1.0 + self.tension)
    }
}

#[derive(Debug)]
struct Inner {
    queue: VecDeque<FractalPatch>,
    capacity: usize,
    total_weight: f32,
}

/// Lightweight queue that mimics a Tokio-uring style submission/completion
/// model where we aggressively recycle memory instead of cloning tensors.
#[derive(Clone, Debug)]
pub struct UringFractalScheduler {
    inner: Arc<Mutex<Inner>>,
}

impl UringFractalScheduler {
    /// Create a scheduler with the requested capacity. Capacity must be non-zero
    /// to avoid silently discarding every patch.
    pub fn new(capacity: usize) -> PureResult<Self> {
        if capacity == 0 {
            return Err(TensorError::EmptyInput("fractal scheduler capacity"));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                queue: VecDeque::with_capacity(capacity),
                capacity,
                total_weight: 0.0,
            })),
        })
    }

    /// Push a new relation patch into the queue, recycling the oldest item when
    /// the capacity is exceeded.
    pub fn push(&self, patch: FractalPatch) -> PureResult<()> {
        let mut inner = self.inner.lock().expect("scheduler mutex poisoned");
        while inner.queue.len() >= inner.capacity {
            if let Some(old) = inner.queue.pop_front() {
                inner.total_weight -= old.weight();
            }
        }
        inner.total_weight += patch.weight();
        inner.queue.push_back(patch);
        Ok(())
    }

    /// Async-friendly wrapper over [`push`].
    pub async fn push_async(&self, patch: FractalPatch) -> PureResult<()> {
        self.push(patch)
    }

    /// Pop the oldest relation patch from the queue.
    pub fn pop(&self) -> Option<FractalPatch> {
        let mut inner = self.inner.lock().expect("scheduler mutex poisoned");
        let patch = inner.queue.pop_front();
        if let Some(ref p) = patch {
            inner.total_weight -= p.weight();
        }
        patch
    }

    /// Async-friendly wrapper over [`pop`].
    pub async fn pop_async(&self) -> Option<FractalPatch> {
        self.pop()
    }

    /// Number of queued relation patches.
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().expect("scheduler mutex poisoned");
        inner.queue.len()
    }

    /// Returns true when no patches are staged.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sum of the coherence weights currently staged.
    pub fn total_weight(&self) -> f32 {
        let inner = self.inner.lock().expect("scheduler mutex poisoned");
        inner.total_weight
    }

    /// Fold the queue into a single relation tensor where coherence acts as the
    /// combining factor and tension keeps the stream smooth.
    pub fn fold_coherence(&self) -> PureResult<Tensor> {
        let inner = self.inner.lock().expect("scheduler mutex poisoned");
        if inner.queue.is_empty() {
            return Err(TensorError::EmptyInput("fractal scheduler queue"));
        }

        let mut accumulator: Option<Tensor> = None;
        let mut total_weight = 0.0f32;
        for patch in inner.queue.iter() {
            let weight = patch.weight();
            if weight <= 0.0 {
                continue;
            }
            total_weight += weight;
            let normalized = patch.normalized_relation()?;
            if let Some(acc) = accumulator.as_mut() {
                acc.add_scaled(&normalized, weight)?;
            } else {
                accumulator = Some(normalized.scale(weight)?);
            }
        }

        let mut acc = accumulator.ok_or(TensorError::EmptyInput("fractal scheduler queue"))?;
        if total_weight > 0.0 {
            acc = acc.scale(1.0 / total_weight)?;
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(1, values.len(), values.to_vec()).unwrap()
    }

    #[test]
    fn rejects_invalid_patch() {
        let relation = tensor(&[1.0, 0.0]);
        assert!(matches!(
            FractalPatch::new(relation.clone(), 0.0, 1.0, 0),
            Err(TensorError::NonPositiveCoherence { .. })
        ));
        assert!(matches!(
            FractalPatch::new(relation, 1.0, 0.0, 0),
            Err(TensorError::NonPositiveTension { .. })
        ));
    }

    #[test]
    fn scheduler_enforces_capacity() {
        let scheduler = UringFractalScheduler::new(2).unwrap();
        let p1 = FractalPatch::new(tensor(&[1.0, 0.0]), 1.0, 1.0, 0).unwrap();
        let p2 = FractalPatch::new(tensor(&[0.0, 1.0]), 1.0, 1.0, 1).unwrap();
        let p3 = FractalPatch::new(tensor(&[1.0, 1.0]), 1.0, 1.0, 2).unwrap();

        scheduler.push(p1).unwrap();
        scheduler.push(p2).unwrap();
        scheduler.push(p3).unwrap();

        assert_eq!(scheduler.len(), 2);
        let mut remaining = vec![scheduler.pop().unwrap(), scheduler.pop().unwrap()];
        remaining.sort_by_key(|p| p.depth());
        assert_eq!(remaining[0].relation().data(), &[0.0, 1.0]);
        assert_eq!(remaining[1].relation().data(), &[1.0, 1.0]);
    }

    #[test]
    fn coherence_fold_blends_relations() {
        let scheduler = UringFractalScheduler::new(4).unwrap();
        let p1 = FractalPatch::new(tensor(&[1.0, 0.0]), 1.0, 0.5, 0).unwrap();
        let p2 = FractalPatch::new(tensor(&[0.0, 1.0]), 2.0, 1.0, 1).unwrap();
        scheduler.push(p1).unwrap();
        scheduler.push(p2).unwrap();

        let blended = scheduler.fold_coherence().unwrap();
        let values = blended.data();
        assert_eq!(values.len(), 2);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
        assert!(values[1] > values[0]);
    }
}
