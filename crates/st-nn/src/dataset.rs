// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::{PureResult, Tensor};
use futures::stream::{self, Stream};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use std::sync::Arc;

type Sample = (Tensor, Tensor);

/// Lightweight in-memory dataset that keeps input/target tensors paired together.
#[derive(Clone, Debug, Default)]
pub struct Dataset {
    samples: Vec<Sample>,
}

impl Dataset {
    /// Creates an empty dataset.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Builds a dataset from an iterator of `(input, target)` pairs.
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Sample>,
    {
        Self {
            samples: iter.into_iter().collect(),
        }
    }

    /// Builds a dataset from an owning vector.
    pub fn from_vec(samples: Vec<Sample>) -> Self {
        Self { samples }
    }

    /// Appends a new sample to the dataset.
    pub fn push(&mut self, input: Tensor, target: Tensor) {
        self.samples.push((input, target));
    }

    /// Returns the number of samples stored in the dataset.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when no samples are registered.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns an owning iterator that yields cloned samples.
    pub fn iter(&self) -> impl Iterator<Item = Sample> + '_ {
        self.samples.iter().cloned()
    }

    /// Returns an asynchronous stream of cloned samples.
    pub fn stream(&self) -> impl Stream<Item = Sample> + '_ {
        stream::iter(self.iter())
    }

    /// Consumes the dataset and turns it into a streaming [`DataLoader`].
    pub fn into_loader(self) -> DataLoader {
        DataLoader::new(self.samples.into())
    }

    /// Creates a streaming [`DataLoader`] that borrows the dataset by cloning the
    /// underlying tensors. This keeps API parity with the old iterator surface
    /// while enabling builders like `.shuffle().batched().prefetch()`.
    pub fn loader(&self) -> DataLoader {
        DataLoader::new(self.samples.clone().into())
    }
}

/// Entry point mirroring the ergonomic `st_nn::dataset::from_vec([...])` helper
/// exposed to Python callers. The returned [`DataLoader`] can be further
/// refined using the builder-style adapters defined below.
pub fn from_vec(samples: Vec<Sample>) -> DataLoader {
    DataLoader::new(samples.into())
}

fn default_order(len: usize) -> Arc<Vec<usize>> {
    Arc::new((0..len).collect())
}

fn stack_batch(batch: &[Sample]) -> PureResult<(Tensor, Tensor)> {
    let (inputs, targets): (Vec<_>, Vec<_>) = batch.iter().cloned().unzip();
    let input = Tensor::cat_rows(&inputs)?;
    let target = Tensor::cat_rows(&targets)?;
    Ok((input, target))
}

#[derive(Clone, Copy, Debug)]
enum BatchStrategy {
    Fixed { batch_size: usize },
    DynamicRows { max_rows: usize },
}

impl BatchStrategy {
    fn fixed(batch_size: usize) -> Self {
        Self::Fixed {
            batch_size: batch_size.max(1),
        }
    }

    fn dynamic_rows(max_rows: usize) -> Self {
        Self::DynamicRows {
            max_rows: max_rows.max(1),
        }
    }

    fn batch_size_hint(&self) -> usize {
        match *self {
            Self::Fixed { batch_size } => batch_size,
            Self::DynamicRows { max_rows } => max_rows,
        }
    }
}

#[derive(Clone)]
struct BatchState {
    samples: Arc<[Sample]>,
    order: Arc<Vec<usize>>,
    strategy: BatchStrategy,
    position: usize,
}

impl BatchState {
    fn new(samples: Arc<[Sample]>, order: Arc<Vec<usize>>, strategy: BatchStrategy) -> Self {
        Self {
            samples,
            order,
            strategy,
            position: 0,
        }
    }

    fn next_indices(&mut self) -> Option<Vec<usize>> {
        if self.position >= self.order.len() {
            return None;
        }
        match self.strategy {
            BatchStrategy::Fixed { batch_size } => {
                let start = self.position;
                let end = (self.position + batch_size).min(self.order.len());
                self.position = end;
                if start == end {
                    None
                } else {
                    Some(self.order[start..end].to_vec())
                }
            }
            BatchStrategy::DynamicRows { max_rows } => {
                let mut rows_acc = 0usize;
                let mut indices = Vec::new();
                while self.position < self.order.len() {
                    let idx = self.order[self.position];
                    let sample_rows = self.samples[idx].0.shape().0.max(1);
                    let would_exceed = !indices.is_empty() && rows_acc + sample_rows > max_rows;
                    if would_exceed && rows_acc > 0 {
                        break;
                    }
                    indices.push(idx);
                    rows_acc += sample_rows;
                    self.position += 1;
                    if rows_acc >= max_rows {
                        break;
                    }
                }
                if indices.is_empty() {
                    // No rows accumulated due to `max_rows == 0` (guarded) or empty order.
                    // Include at least the next sample to make forward progress.
                    if self.position < self.order.len() {
                        let idx = self.order[self.position];
                        self.position += 1;
                        Some(vec![idx])
                    } else {
                        None
                    }
                } else {
                    Some(indices)
                }
            }
        }
    }

    fn next_batch(&mut self) -> Option<PureResult<(Tensor, Tensor)>> {
        let indices = self.next_indices()?;
        let mut batch = Vec::with_capacity(indices.len());
        for idx in indices {
            batch.push(self.samples[idx].clone());
        }
        Some(stack_batch(&batch))
    }
}

#[derive(Clone)]
struct ImmediateBatches {
    state: BatchState,
}

impl ImmediateBatches {
    fn new(state: BatchState) -> Self {
        Self { state }
    }
}

impl Iterator for ImmediateBatches {
    type Item = PureResult<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.state.next_batch()
    }
}

// Prefetch のスレッド化を取りやめ、同期イテレータとして実装
struct PrefetchBatches {
    state: BatchState,
}

impl PrefetchBatches {
    fn spawn(state: BatchState, _depth: usize) -> Self {
        Self { state }
    }
}

impl Iterator for PrefetchBatches {
    type Item = PureResult<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.state.next_batch()
    }
}

enum DataLoaderBackend {
    Immediate(ImmediateBatches),
    Prefetch(PrefetchBatches),
}

/// Iterator over mini-batches produced by a [`DataLoader`].
pub struct DataLoaderBatches {
    backend: DataLoaderBackend,
}

impl DataLoaderBatches {
    fn immediate(samples: Arc<[Sample]>, order: Arc<Vec<usize>>, strategy: BatchStrategy) -> Self {
        Self {
            backend: DataLoaderBackend::Immediate(ImmediateBatches::new(BatchState::new(
                samples, order, strategy,
            ))),
        }
    }

    fn prefetch(
        samples: Arc<[Sample]>,
        order: Arc<Vec<usize>>,
        strategy: BatchStrategy,
        depth: usize,
    ) -> Self {
        Self {
            backend: DataLoaderBackend::Prefetch(PrefetchBatches::spawn(
                BatchState::new(samples, order, strategy),
                depth,
            )),
        }
    }
}

impl Iterator for DataLoaderBatches {
    type Item = PureResult<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.backend {
            DataLoaderBackend::Immediate(iter) => iter.next(),
            DataLoaderBackend::Prefetch(iter) => iter.next(),
        }
    }
}

/// Builder-style streaming loader that supports deterministic shuffling, fixed
/// batch sizes, and background prefetch for feeding training loops.
#[derive(Clone)]
pub struct DataLoader {
    samples: Arc<[Sample]>,
    order: Arc<Vec<usize>>,
    strategy: BatchStrategy,
    prefetch: usize,
}

impl DataLoader {
    fn new(samples: Arc<[Sample]>) -> Self {
        let len = samples.len();
        Self {
            samples,
            order: default_order(len),
            strategy: BatchStrategy::fixed(1),
            prefetch: 0,
        }
    }

    /// Returns the number of individual samples referenced by the loader.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when the underlying dataset holds no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.strategy.batch_size_hint()
    }

    /// Returns the configured prefetch depth.
    pub fn prefetch_depth(&self) -> usize {
        self.prefetch
    }

    /// Creates a new loader with the same dataset but a deterministically
    /// shuffled visitation order using the provided seed.
    pub fn shuffle(mut self, seed: u64) -> Self {
        let mut indices: Vec<usize> = (0..self.samples.len()).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        self.order = Arc::new(indices);
        self
    }

    /// Updates the loader to emit batches of `batch_size` samples.
    pub fn batched(mut self, batch_size: usize) -> Self {
        self.strategy = BatchStrategy::fixed(batch_size);
        self
    }

    /// Dynamically groups samples so each batch covers at most `max_rows` input rows.
    ///
    /// The loader continues to guarantee forward progress even if a single
    /// sample exceeds `max_rows` by emitting that sample as a standalone batch.
    pub fn dynamic_batch_by_rows(mut self, max_rows: usize) -> Self {
        self.strategy = BatchStrategy::dynamic_rows(max_rows);
        self
    }

    /// Enables background prefetching with the given channel depth.
    pub fn prefetch(mut self, depth: usize) -> Self {
        self.prefetch = depth;
        self
    }

    /// Creates a new iterator over the configured batches.
    pub fn iter(&self) -> DataLoaderBatches {
        self.clone().into_iter()
    }
}

impl IntoIterator for DataLoader {
    type Item = PureResult<(Tensor, Tensor)>;
    type IntoIter = DataLoaderBatches;

    fn into_iter(self) -> Self::IntoIter {
        if self.prefetch == 0 {
            DataLoaderBatches::immediate(self.samples, self.order, self.strategy)
        } else {
            DataLoaderBatches::prefetch(self.samples, self.order, self.strategy, self.prefetch)
        }
    }
}

/// Extension trait that provides a `.batched(batch_size)` adapter mirroring the
/// ergonomics of Python data loaders. This remains for backwards compatibility
/// with the original iterator surface.
pub trait BatchIter: Sized {
    fn batched(self, batch_size: usize) -> BatchIterator<Self>;
}

impl<I> BatchIter for I
where
    I: Iterator<Item = Sample>,
{
    fn batched(self, batch_size: usize) -> BatchIterator<Self> {
        BatchIterator {
            iter: self,
            batch_size: batch_size.max(1),
        }
    }
}

/// Iterator that stacks samples into fixed-size batches.
pub struct BatchIterator<I> {
    iter: I,
    batch_size: usize,
}

impl<I> Iterator for BatchIterator<I>
where
    I: Iterator<Item = Sample>,
{
    type Item = PureResult<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            match self.iter.next() {
                Some((input, target)) => {
                    inputs.push(input);
                    targets.push(target);
                }
                None => break,
            }
        }
        if inputs.is_empty() {
            return None;
        }
        let input = match Tensor::cat_rows(&inputs) {
            Ok(tensor) => tensor,
            Err(err) => return Some(Err(err)),
        };
        let target = match Tensor::cat_rows(&targets) {
            Ok(tensor) => tensor,
            Err(err) => return Some(Err(err)),
        };
        Some(Ok((input, target)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_batches_rows() {
        let samples = (0..6).map(|i| {
            let input = Tensor::from_vec(1, 2, vec![i as f32, (i + 1) as f32]).unwrap();
            let target = Tensor::from_vec(1, 1, vec![i as f32 * 2.0]).unwrap();
            (input, target)
        });
        let dataset = Dataset::from_iter(samples);
        let mut batches = dataset.iter().batched(3);
        let first = batches.next().unwrap().unwrap();
        assert_eq!(first.0.shape(), (3, 2));
        assert_eq!(first.1.shape(), (3, 1));
        let second = batches.next().unwrap().unwrap();
        assert_eq!(second.0.shape(), (3, 2));
        assert_eq!(second.1.shape(), (3, 1));
        assert!(batches.next().is_none());
    }

    #[test]
    fn dataset_stream_yields_all_samples() {
        let mut dataset = Dataset::new();
        dataset.push(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::zeros(1, 1).unwrap(),
        );
        dataset.push(
            Tensor::from_vec(1, 1, vec![2.0]).unwrap(),
            Tensor::zeros(1, 1).unwrap(),
        );
        let collected: Vec<_> = futures::executor::block_on_stream(dataset.stream()).collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn dataloader_shuffles_deterministically() {
        let samples: Vec<(Tensor, Tensor)> = (0..4)
            .map(|i| {
                let input = Tensor::from_vec(1, 1, vec![i as f32]).unwrap();
                let target = Tensor::from_vec(1, 1, vec![i as f32]).unwrap();
                (input, target)
            })
            .collect();
        let shuffled = from_vec(samples)
            .shuffle(42)
            .batched(2)
            .iter()
            .map(|batch| batch.unwrap().0.data()[0])
            .collect::<Vec<_>>();
        let shuffled_again = from_vec(
            (0..4)
                .map(|i| {
                    let input = Tensor::from_vec(1, 1, vec![i as f32]).unwrap();
                    let target = Tensor::from_vec(1, 1, vec![i as f32]).unwrap();
                    (input, target)
                })
                .collect(),
        )
        .shuffle(42)
        .batched(2)
        .iter()
        .map(|batch| batch.unwrap().0.data()[0])
        .collect::<Vec<_>>();
        assert_eq!(shuffled, shuffled_again);
    }

    #[test]
    fn dataloader_prefetch_matches_eager() {
        let samples: Vec<(Tensor, Tensor)> = (0..8)
            .map(|i| {
                let input = Tensor::from_vec(1, 1, vec![i as f32]).unwrap();
                let target = Tensor::from_vec(1, 1, vec![i as f32]).unwrap();
                (input, target)
            })
            .collect();
        let eager: Vec<_> = from_vec(samples.clone())
            .batched(4)
            .iter()
            .map(|batch| batch.unwrap().0)
            .collect();
        let prefetched: Vec<_> = from_vec(samples)
            .batched(4)
            .prefetch(2)
            .iter()
            .map(|batch| batch.unwrap().0)
            .collect();
        assert_eq!(eager.len(), prefetched.len());
        for (lhs, rhs) in eager.iter().zip(prefetched.iter()) {
            assert_eq!(lhs.data(), rhs.data());
        }
    }

    #[test]
    fn dataloader_dynamic_rows_groups_by_capacity() {
        let samples: Vec<(Tensor, Tensor)> = vec![
            (
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                Tensor::zeros(1, 1).unwrap(),
            ),
            (
                Tensor::from_vec(2, 1, vec![2.0, 3.0]).unwrap(),
                Tensor::zeros(2, 1).unwrap(),
            ),
            (
                Tensor::from_vec(1, 1, vec![4.0]).unwrap(),
                Tensor::zeros(1, 1).unwrap(),
            ),
            (
                Tensor::from_vec(4, 1, vec![5.0, 6.0, 7.0, 8.0]).unwrap(),
                Tensor::zeros(4, 1).unwrap(),
            ),
        ];
        let mut batches = from_vec(samples).dynamic_batch_by_rows(3).iter();
        let first = batches.next().unwrap().unwrap();
        assert_eq!(first.0.shape(), (3, 1));
        assert_eq!(first.1.shape(), (3, 1));
        let second = batches.next().unwrap().unwrap();
        assert_eq!(second.0.shape(), (1, 1));
        assert_eq!(second.1.shape(), (1, 1));
        let third = batches.next().unwrap().unwrap();
        assert_eq!(third.0.shape(), (4, 1));
        assert_eq!(third.1.shape(), (4, 1));
        assert!(batches.next().is_none());
    }

    #[test]
    fn dataloader_dynamic_rows_prefetch_matches_immediate() {
        let samples: Vec<(Tensor, Tensor)> = (0..6)
            .map(|i| {
                let rows = (i % 3) + 1;
                let input = Tensor::from_vec(rows, 1, vec![i as f32; rows]).unwrap();
                let target = Tensor::zeros(rows, 1).unwrap();
                (input, target)
            })
            .collect();
        let eager: Vec<_> = from_vec(samples.clone())
            .dynamic_batch_by_rows(4)
            .iter()
            .map(|batch| batch.unwrap().0.shape().0)
            .collect();
        let prefetched: Vec<_> = from_vec(samples)
            .dynamic_batch_by_rows(4)
            .prefetch(2)
            .iter()
            .map(|batch| batch.unwrap().0.shape().0)
            .collect();
        assert_eq!(eager, prefetched);
    }
}
