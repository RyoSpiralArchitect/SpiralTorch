use crate::{PureResult, Tensor};
use futures::stream::{self, Stream};

/// Lightweight in-memory dataset that keeps input/target tensors paired together.
#[derive(Clone, Debug, Default)]
pub struct Dataset {
    samples: Vec<(Tensor, Tensor)>,
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
        I: IntoIterator<Item = (Tensor, Tensor)>,
    {
        Self {
            samples: iter.into_iter().collect(),
        }
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
    pub fn iter(&self) -> impl Iterator<Item = (Tensor, Tensor)> + '_ {
        self.samples.iter().cloned()
    }

    /// Returns an asynchronous stream of cloned samples.
    pub fn stream(&self) -> impl Stream<Item = (Tensor, Tensor)> + '_ {
        stream::iter(self.iter())
    }
}

/// Extension trait that provides a `.batched(batch_size)` adapter mirroring the
/// ergonomics of Python data loaders.
pub trait BatchIter: Sized {
    fn batched(self, batch_size: usize) -> BatchIterator<Self>;
}

impl<I> BatchIter for I
where
    I: Iterator<Item = (Tensor, Tensor)>,
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
    I: Iterator<Item = (Tensor, Tensor)>,
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
}
