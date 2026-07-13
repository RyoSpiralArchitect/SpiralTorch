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

fn checked_f64_to_f32(label: &'static str, value: f64) -> PureResult<f32> {
    if !value.is_finite() || value.abs() > f32::MAX as f64 {
        let value = if value.is_nan() {
            f32::NAN
        } else if value.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value as f32)
}

fn checked_weight_sum(label: &'static str, weights: impl Iterator<Item = f32>) -> PureResult<f32> {
    checked_f64_to_f32(label, weights.map(|weight| weight as f64).sum())
}

/// Describes a coherent relation patch sampled from a fractal walk.
#[derive(Clone, Debug)]
pub struct FractalPatch {
    relation: Tensor,
    coherence: f32,
    tension: f32,
    weight: f32,
    depth: u32,
}

impl FractalPatch {
    /// Construct a new relation patch.
    pub fn new(relation: Tensor, coherence: f32, tension: f32, depth: u32) -> PureResult<Self> {
        if !coherence.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "fractal_patch_coherence",
                value: coherence,
            });
        }
        if coherence <= 0.0 {
            return Err(TensorError::NonPositiveCoherence { coherence });
        }
        if !tension.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "fractal_patch_tension",
                value: tension,
            });
        }
        if tension <= 0.0 {
            return Err(TensorError::NonPositiveTension { tension });
        }
        for &value in relation.data() {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "fractal_patch_relation",
                    value,
                });
            }
        }
        let weight = checked_f64_to_f32(
            "fractal_patch_weight",
            coherence as f64 / (1.0 + tension as f64),
        )?;
        if weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight { weight });
        }
        Ok(Self {
            relation,
            coherence,
            tension,
            weight,
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
        let squared_norm = self
            .relation
            .data()
            .iter()
            .map(|&value| {
                let value = value as f64;
                value * value
            })
            .sum::<f64>();
        let mut normalized = self.relation.clone();
        if squared_norm == 0.0 {
            normalized.data_mut().fill(0.0);
            return Ok(normalized);
        }
        let norm = squared_norm.sqrt();
        for value in normalized.data_mut() {
            *value = checked_f64_to_f32("fractal_patch_normalized_relation", *value as f64 / norm)?;
        }
        Ok(normalized)
    }

    /// Weight applied by the streaming scheduler.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

#[derive(Debug)]
struct Inner {
    queue: VecDeque<FractalPatch>,
    capacity: usize,
    total_weight: f32,
    relation_shape: Option<(usize, usize)>,
}

/// Atomic view of the patches currently staged by a [`UringFractalScheduler`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractalSchedulerSnapshot {
    capacity: usize,
    queued: usize,
    total_weight: f32,
    relation_shape: Option<(usize, usize)>,
    oldest_depth: Option<u32>,
    newest_depth: Option<u32>,
}

impl FractalSchedulerSnapshot {
    /// Maximum number of patches retained by the scheduler.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of patches represented by this snapshot.
    pub fn queued(&self) -> usize {
        self.queued
    }

    /// Sum of the coherence/tension-derived patch weights.
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Common relation shape, or `None` when the queue is empty.
    pub fn relation_shape(&self) -> Option<(usize, usize)> {
        self.relation_shape
    }

    /// Depth of the oldest queued patch.
    pub fn oldest_depth(&self) -> Option<u32> {
        self.oldest_depth
    }

    /// Depth of the newest queued patch.
    pub fn newest_depth(&self) -> Option<u32> {
        self.newest_depth
    }

    /// Returns true when no patches were staged at snapshot time.
    pub fn is_empty(&self) -> bool {
        self.queued == 0
    }
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
                relation_shape: None,
            })),
        })
    }

    fn lock_inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Push a new relation patch into the queue, recycling the oldest item when
    /// the capacity is exceeded.
    pub fn push(&self, patch: FractalPatch) -> PureResult<()> {
        let mut inner = self.lock_inner();
        let evict_count = inner.queue.len().saturating_sub(inner.capacity - 1);
        if let Some(retained) = inner.queue.get(evict_count) {
            let expected = retained.relation().shape();
            let actual = patch.relation().shape();
            if expected != actual {
                return Err(TensorError::ShapeMismatch {
                    left: expected,
                    right: actual,
                });
            }
        }
        let total_weight = checked_weight_sum(
            "fractal_scheduler_total_weight",
            inner
                .queue
                .iter()
                .skip(evict_count)
                .map(FractalPatch::weight)
                .chain(core::iter::once(patch.weight())),
        )?;
        for _ in 0..evict_count {
            inner.queue.pop_front();
        }
        let relation_shape = patch.relation().shape();
        inner.queue.push_back(patch);
        inner.total_weight = total_weight;
        inner.relation_shape = Some(relation_shape);
        Ok(())
    }

    /// Async-friendly wrapper over [`Self::push`].
    pub async fn push_async(&self, patch: FractalPatch) -> PureResult<()> {
        self.push(patch)
    }

    /// Pop the oldest relation patch from the queue.
    pub fn pop(&self) -> Option<FractalPatch> {
        let mut inner = self.lock_inner();
        let patch = inner.queue.pop_front();
        if patch.is_some() {
            inner.total_weight = inner
                .queue
                .iter()
                .map(|patch| patch.weight() as f64)
                .sum::<f64>() as f32;
            inner.relation_shape = inner.queue.front().map(|patch| patch.relation().shape());
        }
        patch
    }

    /// Async-friendly wrapper over [`Self::pop`].
    pub async fn pop_async(&self) -> Option<FractalPatch> {
        self.pop()
    }

    /// Captures queue diagnostics under one lock acquisition.
    pub fn snapshot(&self) -> FractalSchedulerSnapshot {
        let inner = self.lock_inner();
        FractalSchedulerSnapshot {
            capacity: inner.capacity,
            queued: inner.queue.len(),
            total_weight: inner.total_weight,
            relation_shape: inner.relation_shape,
            oldest_depth: inner.queue.front().map(FractalPatch::depth),
            newest_depth: inner.queue.back().map(FractalPatch::depth),
        }
    }

    /// Number of queued relation patches.
    pub fn len(&self) -> usize {
        let inner = self.lock_inner();
        inner.queue.len()
    }

    /// Returns true when no patches are staged.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum number of relation patches retained by the scheduler.
    pub fn capacity(&self) -> usize {
        self.snapshot().capacity()
    }

    /// Common relation shape currently staged by the scheduler.
    pub fn relation_shape(&self) -> Option<(usize, usize)> {
        self.snapshot().relation_shape()
    }

    /// Sum of the coherence/tension-derived weights currently staged.
    pub fn total_weight(&self) -> f32 {
        self.snapshot().total_weight()
    }

    /// Fold the queue into a single relation tensor where coherence acts as the
    /// combining factor and tension keeps the stream smooth.
    pub fn fold_coherence(&self) -> PureResult<Tensor> {
        let patches = {
            let inner = self.lock_inner();
            if inner.queue.is_empty() {
                return Err(TensorError::EmptyInput("fractal scheduler queue"));
            }
            inner.queue.iter().cloned().collect::<Vec<_>>()
        };
        let shape = patches[0].relation().shape();
        let total_weight = checked_weight_sum(
            "fractal_scheduler_fold_weight",
            patches.iter().map(FractalPatch::weight),
        )? as f64;
        let mut accumulator = Tensor::zeros(shape.0, shape.1)?;
        let mut normalised_total = 0.0f64;
        for patch in &patches {
            let normalized = patch.normalized_relation()?;
            let weight = checked_f64_to_f32(
                "fractal_scheduler_normalised_weight",
                patch.weight() as f64 / total_weight,
            )?;
            normalised_total += weight as f64;
            accumulator.add_scaled(&normalized, weight)?;
        }
        if !normalised_total.is_finite() || normalised_total <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "fractal_scheduler_normalised_weight_sum",
                value: normalised_total as f32,
            });
        }
        let correction = checked_f64_to_f32(
            "fractal_scheduler_weight_normalisation",
            1.0 / normalised_total,
        )?;
        accumulator.scale(correction)
    }

    /// Fold the queued relations directly into the provided tensor buffer.
    pub fn fold_coherence_into(&self, target: &mut Tensor) -> PureResult<()> {
        let folded = self.fold_coherence()?;
        if target.shape() != folded.shape() {
            return Err(TensorError::ShapeMismatch {
                left: target.shape(),
                right: folded.shape(),
            });
        }
        target.data_mut().copy_from_slice(folded.data());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn unwrap_ok<T, E: core::fmt::Debug>(result: Result<T, E>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => panic!("expected Ok(..), got Err({error:?})"),
        }
    }

    #[track_caller]
    fn unwrap_err<T, E: core::fmt::Debug>(result: Result<T, E>) -> E {
        match result {
            Ok(_) => panic!("expected Err(..), got Ok(..)"),
            Err(error) => error,
        }
    }

    #[track_caller]
    fn unwrap_some<T>(option: Option<T>) -> T {
        match option {
            Some(value) => value,
            None => panic!("expected Some(..), got None"),
        }
    }

    fn tensor(values: &[f32]) -> Tensor {
        unwrap_ok(Tensor::from_vec(1, values.len(), values.to_vec()))
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
    fn rejects_non_finite_patch_fields_and_weight_underflow() {
        let relation = tensor(&[1.0, 0.0]);
        let error = unwrap_err(FractalPatch::new(relation.clone(), f32::NAN, 1.0, 0));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "fractal_patch_coherence",
                ..
            }
        ));
        let error = unwrap_err(FractalPatch::new(relation.clone(), 1.0, f32::INFINITY, 0));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "fractal_patch_tension",
                ..
            }
        ));
        let non_finite = unwrap_ok(Tensor::from_vec(1, 2, vec![1.0, f32::NAN]));
        let error = unwrap_err(FractalPatch::new(non_finite, 1.0, 1.0, 0));
        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "fractal_patch_relation",
                ..
            }
        ));
        let error = unwrap_err(FractalPatch::new(relation, f32::from_bits(1), f32::MAX, 0));
        assert!(matches!(error, TensorError::NonPositiveWeight { .. }));
    }

    #[test]
    fn relation_normalization_handles_extreme_finite_scales() {
        let large = unwrap_ok(FractalPatch::new(
            tensor(&[f32::MAX, f32::MAX]),
            1.0,
            1.0,
            0,
        ));
        let large_normalized = unwrap_ok(large.normalized_relation());
        let large_norm = large_normalized
            .data()
            .iter()
            .map(|&value| (value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((large_norm - 1.0).abs() < 1e-6);
        assert!(large_normalized.data().iter().all(|value| *value > 0.0));

        let tiny = unwrap_ok(FractalPatch::new(
            tensor(&[f32::from_bits(1), 0.0]),
            1.0,
            1.0,
            0,
        ));
        assert_eq!(unwrap_ok(tiny.normalized_relation()).data(), &[1.0, 0.0]);
    }

    #[test]
    fn scheduler_enforces_capacity() {
        let scheduler = unwrap_ok(UringFractalScheduler::new(2));
        let p1 = unwrap_ok(FractalPatch::new(tensor(&[1.0, 0.0]), 1.0, 1.0, 0));
        let p2 = unwrap_ok(FractalPatch::new(tensor(&[0.0, 1.0]), 1.0, 1.0, 1));
        let p3 = unwrap_ok(FractalPatch::new(tensor(&[1.0, 1.0]), 1.0, 1.0, 2));

        unwrap_ok(scheduler.push(p1));
        unwrap_ok(scheduler.push(p2));
        unwrap_ok(scheduler.push(p3));

        assert_eq!(scheduler.len(), 2);
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.capacity(), 2);
        assert_eq!(snapshot.queued(), 2);
        assert_eq!(snapshot.relation_shape(), Some((1, 2)));
        assert_eq!(snapshot.oldest_depth(), Some(1));
        assert_eq!(snapshot.newest_depth(), Some(2));
        assert!(!snapshot.is_empty());
        let mut remaining = [unwrap_some(scheduler.pop()), unwrap_some(scheduler.pop())];
        remaining.sort_by_key(|p| p.depth());
        assert_eq!(remaining[0].relation().data(), &[0.0, 1.0]);
        assert_eq!(remaining[1].relation().data(), &[1.0, 1.0]);
    }

    #[test]
    fn scheduler_rejects_shape_mismatch_without_eviction() {
        let scheduler = unwrap_ok(UringFractalScheduler::new(2));
        let first = unwrap_ok(FractalPatch::new(tensor(&[1.0, 0.0]), 2.0, 1.0, 0));
        unwrap_ok(scheduler.push(first));
        let second = unwrap_ok(FractalPatch::new(tensor(&[0.0, 1.0]), 1.0, 1.0, 1));
        unwrap_ok(scheduler.push(second));
        let total_before = scheduler.total_weight();
        let mismatched = unwrap_ok(FractalPatch::new(
            unwrap_ok(Tensor::from_vec(2, 1, vec![0.0, 1.0])),
            1.0,
            1.0,
            2,
        ));

        let error = unwrap_err(scheduler.push(mismatched));

        assert!(matches!(error, TensorError::ShapeMismatch { .. }));
        assert_eq!(scheduler.len(), 2);
        assert_eq!(scheduler.relation_shape(), Some((1, 2)));
        assert_eq!(scheduler.total_weight(), total_before);
        assert_eq!(unwrap_some(scheduler.pop()).depth(), 0);
        assert_eq!(unwrap_some(scheduler.pop()).depth(), 1);
    }

    #[test]
    fn single_capacity_scheduler_can_replace_relation_shape() {
        let scheduler = unwrap_ok(UringFractalScheduler::new(1));
        unwrap_ok(scheduler.push(unwrap_ok(FractalPatch::new(
            tensor(&[1.0, 0.0]),
            1.0,
            1.0,
            0,
        ))));
        unwrap_ok(scheduler.push(unwrap_ok(FractalPatch::new(
            unwrap_ok(Tensor::from_vec(2, 1, vec![0.0, 1.0])),
            1.0,
            1.0,
            1,
        ))));

        assert_eq!(scheduler.capacity(), 1);
        assert_eq!(scheduler.len(), 1);
        assert_eq!(scheduler.relation_shape(), Some((2, 1)));
        assert_eq!(unwrap_some(scheduler.pop()).depth(), 1);
        let snapshot = scheduler.snapshot();
        assert!(snapshot.is_empty());
        assert_eq!(snapshot.relation_shape(), None);
        assert_eq!(snapshot.total_weight(), 0.0);
        assert_eq!(snapshot.oldest_depth(), None);
        assert_eq!(snapshot.newest_depth(), None);
    }

    #[test]
    fn scheduler_weight_overflow_is_transactional() {
        let scheduler = unwrap_ok(UringFractalScheduler::new(2));
        let huge = || {
            unwrap_ok(FractalPatch::new(
                tensor(&[1.0]),
                f32::MAX,
                f32::MIN_POSITIVE,
                0,
            ))
        };
        unwrap_ok(scheduler.push(huge()));

        let error = unwrap_err(scheduler.push(huge()));

        assert!(matches!(
            error,
            TensorError::NonFiniteValue {
                label: "fractal_scheduler_total_weight",
                ..
            }
        ));
        assert_eq!(scheduler.len(), 1);
        assert_eq!(scheduler.total_weight(), f32::MAX);
    }

    #[test]
    fn coherence_fold_blends_relations() {
        let scheduler = unwrap_ok(UringFractalScheduler::new(4));
        let p1 = unwrap_ok(FractalPatch::new(tensor(&[1.0, 0.0]), 1.0, 0.5, 0));
        let p2 = unwrap_ok(FractalPatch::new(tensor(&[0.0, 1.0]), 2.0, 1.0, 1));
        unwrap_ok(scheduler.push(p1));
        unwrap_ok(scheduler.push(p2));

        let blended = unwrap_ok(scheduler.fold_coherence());
        let values = blended.data();
        assert_eq!(values.len(), 2);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
        assert!(values[1] > values[0]);
    }

    #[test]
    fn coherence_fold_avoids_intermediate_weight_overflow() {
        let scheduler = unwrap_ok(UringFractalScheduler::new(2));
        let coherence = f32::MAX * 0.5;
        unwrap_ok(scheduler.push(unwrap_ok(FractalPatch::new(
            tensor(&[1.0, 0.0]),
            coherence,
            1.0,
            0,
        ))));
        unwrap_ok(scheduler.push(unwrap_ok(FractalPatch::new(
            tensor(&[0.0, 1.0]),
            coherence,
            1.0,
            1,
        ))));

        let folded = unwrap_ok(scheduler.fold_coherence());

        assert!(folded.data().iter().all(|value| value.is_finite()));
        assert!((folded.data()[0] - 0.5).abs() < 1e-6);
        assert!((folded.data()[1] - 0.5).abs() < 1e-6);
    }
}
