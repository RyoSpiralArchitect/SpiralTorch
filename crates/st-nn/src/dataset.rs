use crate::{PureResult, Tensor, TensorError};
use futures::stream::{self, Stream};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;

type Sample = (Tensor, Tensor);

/// Vocabulary size for tokenizerless next-byte language-model helpers.
pub const BYTE_LM_VOCAB: usize = 256;

/// Row accounting for tokenizerless byte-LM samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteLmSampleStats {
    pub samples: usize,
    pub total_rows: usize,
    pub active_rows: usize,
}

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

fn byte_lm_one_hot_inputs(bytes: &[u8]) -> PureResult<Tensor> {
    let mut data = vec![0.0f32; bytes.len() * BYTE_LM_VOCAB];
    for (row, byte) in bytes.iter().copied().enumerate() {
        data[row * BYTE_LM_VOCAB + byte as usize] = 1.0;
    }
    Tensor::from_vec(bytes.len(), BYTE_LM_VOCAB, data)
}

fn byte_lm_sparse_targets(bytes: &[u8]) -> PureResult<Tensor> {
    Tensor::from_vec(
        bytes.len(),
        1,
        bytes.iter().copied().map(|byte| byte as f32).collect(),
    )
}

/// Converts text into tokenizerless sliding-window next-byte samples.
///
/// Each sample stores `context` one-hot byte rows and sparse `(context, 1)`
/// next-byte class-id targets. UTF-8 bytes are preserved directly, so there is
/// no unknown-token path.
pub fn byte_lm_windows(text: &str, context: usize) -> PureResult<Vec<(Tensor, Tensor)>> {
    if context == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: context,
            cols: BYTE_LM_VOCAB,
        });
    }
    let bytes = text.as_bytes();
    if bytes.len() <= context {
        return Err(TensorError::EmptyInput("byte_lm_windows"));
    }
    let mut samples = Vec::with_capacity(bytes.len() - context);
    for start in 0..(bytes.len() - context) {
        let input = &bytes[start..start + context];
        let target = &bytes[start + 1..start + context + 1];
        samples.push((
            byte_lm_one_hot_inputs(input)?,
            byte_lm_sparse_targets(target)?,
        ));
    }
    Ok(samples)
}

/// Converts multiple documents into tokenizerless next-byte samples.
///
/// Documents shorter than or equal to `context` bytes are skipped so mixed
/// corpora can include short titles, blank lines, or metadata rows. An error is
/// returned only when no document yields any trainable window.
pub fn byte_lm_corpus_windows(texts: &[&str], context: usize) -> PureResult<Vec<(Tensor, Tensor)>> {
    if context == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: context,
            cols: BYTE_LM_VOCAB,
        });
    }
    let mut samples = Vec::new();
    for text in texts {
        if text.as_bytes().len() <= context {
            continue;
        }
        samples.extend(byte_lm_windows(text, context)?);
    }
    if samples.is_empty() {
        return Err(TensorError::EmptyInput("byte_lm_corpus_windows"));
    }
    Ok(samples)
}

fn padded_byte_lm_sample(
    text: &str,
    pad_rows: usize,
    ignore_index: i32,
) -> PureResult<(Tensor, Tensor)> {
    if pad_rows == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: pad_rows,
            cols: BYTE_LM_VOCAB,
        });
    }
    let bytes = text.as_bytes();
    let mut input = vec![0.0f32; pad_rows * BYTE_LM_VOCAB];
    let mut target = vec![ignore_index as f32; pad_rows];
    for row in 0..pad_rows {
        let Some(&byte) = bytes.get(row) else {
            break;
        };
        let Some(&next_byte) = bytes.get(row + 1) else {
            break;
        };
        input[row * BYTE_LM_VOCAB + byte as usize] = 1.0;
        target[row] = next_byte as f32;
    }
    Ok((
        Tensor::from_vec(pad_rows, BYTE_LM_VOCAB, input)?,
        Tensor::from_vec(pad_rows, 1, target)?,
    ))
}

/// Converts variable-length texts into fixed-row tokenizerless byte-LM samples.
///
/// Rows without a next-byte target are assigned `ignore_index`, matching
/// `SoftmaxCrossEntropy::with_ignore_index...` padding semantics.
pub fn padded_byte_lm_samples(
    texts: &[&str],
    pad_rows: usize,
    ignore_index: i32,
) -> PureResult<Vec<(Tensor, Tensor)>> {
    texts
        .iter()
        .map(|text| padded_byte_lm_sample(text, pad_rows, ignore_index))
        .collect()
}

/// Counts rows for tokenizerless byte-LM samples, optionally respecting padding.
pub fn byte_lm_sample_stats(
    samples: &[(Tensor, Tensor)],
    ignore_index: Option<i32>,
) -> ByteLmSampleStats {
    let total_rows = samples.iter().map(|(input, _)| input.shape().0).sum();
    let active_rows = samples
        .iter()
        .map(|(_, target)| match ignore_index {
            Some(index) => target
                .data()
                .iter()
                .filter(|&&value| value != index as f32)
                .count(),
            None => target.shape().0,
        })
        .sum();
    ByteLmSampleStats {
        samples: samples.len(),
        total_rows,
        active_rows,
    }
}

fn is_ignored_target(value: f32, ignore_index: Option<i32>) -> PureResult<bool> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "byte_lm_target",
            value,
        });
    }
    let rounded = value.round();
    Ok(ignore_index
        .is_some_and(|index| (value - rounded).abs() <= f32::EPSILON && rounded == index as f32))
}

fn validate_byte_lm_target(value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue {
            label: "byte_lm_target",
            value,
        });
    }
    let rounded = value.round();
    if (value - rounded).abs() > f32::EPSILON || rounded < 0.0 || rounded >= BYTE_LM_VOCAB as f32 {
        return Err(TensorError::InvalidClassIndex {
            index: value,
            classes: BYTE_LM_VOCAB,
        });
    }
    Ok(())
}

fn validate_byte_lm_input_row(row: &[f32], allow_empty: bool) -> PureResult<()> {
    let mut ones = 0usize;
    let mut non_zero = 0usize;
    for &value in row {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "byte_lm_input",
                value,
            });
        }
        if value == 1.0 {
            ones += 1;
        } else if value != 0.0 {
            non_zero += 1;
        }
    }
    if non_zero == 0 && (ones == 1 || (allow_empty && ones == 0)) {
        return Ok(());
    }
    Err(TensorError::IoError {
        message: format!(
            "byte_lm input row must be one-hot{}: ones={ones} non_zero={non_zero}",
            if allow_empty { " or empty" } else { "" }
        ),
    })
}

/// Validates tokenizerless byte-LM samples before training.
///
/// This is stricter than [`byte_lm_sample_stats`]: it rejects empty sample sets,
/// non-`256` input columns, target shape mismatches, non-finite values,
/// out-of-range sparse byte ids, malformed one-hot rows, and all-padding
/// batches. It returns the same row accounting on success so callers can log
/// the accepted training surface.
pub fn validate_byte_lm_samples(
    samples: &[(Tensor, Tensor)],
    ignore_index: Option<i32>,
) -> PureResult<ByteLmSampleStats> {
    if samples.is_empty() {
        return Err(TensorError::EmptyInput("validate_byte_lm_samples"));
    }

    let mut total_rows = 0usize;
    let mut active_rows = 0usize;
    for (input, target) in samples {
        let (input_rows, input_cols) = input.shape();
        if input_cols != BYTE_LM_VOCAB {
            return Err(TensorError::ShapeMismatch {
                left: (input_rows, BYTE_LM_VOCAB),
                right: input.shape(),
            });
        }
        if target.shape() != (input_rows, 1) {
            return Err(TensorError::ShapeMismatch {
                left: (input_rows, 1),
                right: target.shape(),
            });
        }

        total_rows += input_rows;
        for row in 0..input_rows {
            let target_value = target.data()[row];
            let ignored = is_ignored_target(target_value, ignore_index)?;
            if !ignored {
                validate_byte_lm_target(target_value)?;
                active_rows += 1;
            }
            let start = row * input_cols;
            let end = start + input_cols;
            validate_byte_lm_input_row(&input.data()[start..end], ignored)?;
        }
    }

    if active_rows == 0 {
        return Err(TensorError::EmptyInput(
            "validate_byte_lm_samples_active_rows",
        ));
    }

    Ok(ByteLmSampleStats {
        samples: samples.len(),
        total_rows,
        active_rows,
    })
}

/// Deterministically interleaves target fine-tuning samples with source replay samples.
///
/// After every `target_per_replay` target samples, one replay sample is inserted.
/// Replay samples cycle when the target set is longer than the replay set. This
/// gives FT loops a reproducible way to preserve source behavior while still
/// visiting target samples in their original order.
pub fn interleave_replay_samples(
    target_samples: &[Sample],
    replay_samples: &[Sample],
    target_per_replay: usize,
) -> PureResult<Vec<Sample>> {
    if target_per_replay == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: target_per_replay,
            cols: 1,
        });
    }
    if target_samples.is_empty() {
        return Err(TensorError::EmptyInput("interleave_replay_samples_target"));
    }
    if replay_samples.is_empty() {
        return Err(TensorError::EmptyInput("interleave_replay_samples_replay"));
    }

    let replay_slots = target_samples.len() / target_per_replay;
    let mut mixed = Vec::with_capacity(target_samples.len() + replay_slots);
    let mut replay_idx = 0usize;
    for (idx, sample) in target_samples.iter().cloned().enumerate() {
        mixed.push(sample);
        if (idx + 1) % target_per_replay == 0 {
            mixed.push(replay_samples[replay_idx % replay_samples.len()].clone());
            replay_idx += 1;
        }
    }
    Ok(mixed)
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

fn chunk_indices(order: &[usize], batch_size: usize) -> impl Iterator<Item = &[usize]> {
    order.chunks(batch_size.max(1))
}

#[derive(Clone)]
struct ImmediateBatches {
    samples: Arc<[Sample]>,
    order: Arc<Vec<usize>>,
    batch_size: usize,
    position: usize,
}

impl ImmediateBatches {
    fn new(samples: Arc<[Sample]>, order: Arc<Vec<usize>>, batch_size: usize) -> Self {
        Self {
            samples,
            order,
            batch_size: batch_size.max(1),
            position: 0,
        }
    }
}

impl Iterator for ImmediateBatches {
    type Item = PureResult<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.order.len() {
            return None;
        }
        let start = self.position;
        let end = (self.position + self.batch_size).min(self.order.len());
        self.position = end;
        let indices = &self.order[start..end];
        if indices.is_empty() {
            return None;
        }
        let mut batch = Vec::with_capacity(indices.len());
        for &idx in indices {
            batch.push(self.samples[idx].clone());
        }
        Some(stack_batch(&batch))
    }
}

struct PrefetchBatches {
    rx: Receiver<Option<PureResult<(Tensor, Tensor)>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PrefetchBatches {
    fn spawn(
        samples: Arc<[Sample]>,
        order: Arc<Vec<usize>>,
        batch_size: usize,
        depth: usize,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel(depth.max(1));
        let handle = thread::spawn(move || {
            for indices in chunk_indices(&order, batch_size) {
                let mut batch = Vec::with_capacity(indices.len());
                for &idx in indices {
                    batch.push(samples[idx].clone());
                }
                if tx.send(Some(stack_batch(&batch))).is_err() {
                    return;
                }
            }
            let _ = tx.send(None);
        });
        Self {
            rx,
            handle: Some(handle),
        }
    }
}

impl Iterator for PrefetchBatches {
    type Item = PureResult<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.recv().ok()? {
            Some(batch) => Some(batch),
            None => None,
        }
    }
}

impl Drop for PrefetchBatches {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
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
    fn immediate(samples: Arc<[Sample]>, order: Arc<Vec<usize>>, batch_size: usize) -> Self {
        Self {
            backend: DataLoaderBackend::Immediate(ImmediateBatches::new(
                samples, order, batch_size,
            )),
        }
    }

    fn prefetch(
        samples: Arc<[Sample]>,
        order: Arc<Vec<usize>>,
        batch_size: usize,
        depth: usize,
    ) -> Self {
        Self {
            backend: DataLoaderBackend::Prefetch(PrefetchBatches::spawn(
                samples, order, batch_size, depth,
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
    batch_size: usize,
    prefetch: usize,
}

impl DataLoader {
    fn new(samples: Arc<[Sample]>) -> Self {
        let len = samples.len();
        Self {
            samples,
            order: default_order(len),
            batch_size: 1,
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
        self.batch_size
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
        self.batch_size = batch_size.max(1);
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
            DataLoaderBatches::immediate(self.samples, self.order, self.batch_size)
        } else {
            DataLoaderBatches::prefetch(self.samples, self.order, self.batch_size, self.prefetch)
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
    fn dataloader_batches_multirow_sequence_samples() {
        let samples = vec![
            (
                Tensor::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
                Tensor::from_vec(2, 1, vec![1.0, 2.0]).unwrap(),
            ),
            (
                Tensor::from_vec(2, 3, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
                Tensor::from_vec(2, 1, vec![2.0, 0.0]).unwrap(),
            ),
        ];
        let mut batches = from_vec(samples).batched(2).iter();
        let (input, target) = batches.next().unwrap().unwrap();
        assert_eq!(input.shape(), (4, 3));
        assert_eq!(target.shape(), (4, 1));
        assert_eq!(target.data(), &[1.0, 2.0, 2.0, 0.0]);
        assert!(batches.next().is_none());
    }

    #[test]
    fn byte_lm_windows_preserve_next_byte_targets() {
        let samples = byte_lm_windows("abcd", 2).unwrap();
        assert_eq!(samples.len(), 2);
        let (input, target) = &samples[0];
        assert_eq!(input.shape(), (2, BYTE_LM_VOCAB));
        assert_eq!(target.shape(), (2, 1));
        assert_eq!(target.data(), &[b'b' as f32, b'c' as f32]);
        assert_eq!(input.data()[b'a' as usize], 1.0);
        assert_eq!(input.data()[BYTE_LM_VOCAB + b'b' as usize], 1.0);

        let stats = byte_lm_sample_stats(&samples, None);
        assert_eq!(
            stats,
            ByteLmSampleStats {
                samples: 2,
                total_rows: 4,
                active_rows: 4,
            }
        );
    }

    #[test]
    fn byte_lm_corpus_windows_skip_short_documents() {
        let samples = byte_lm_corpus_windows(&["abcd", "xy", "", "wxyz"], 2).unwrap();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].1.data(), &[b'b' as f32, b'c' as f32]);
        assert_eq!(samples[2].1.data(), &[b'x' as f32, b'y' as f32]);

        let stats = byte_lm_sample_stats(&samples, None);
        assert_eq!(
            stats,
            ByteLmSampleStats {
                samples: 4,
                total_rows: 8,
                active_rows: 8,
            }
        );
    }

    #[test]
    fn padded_byte_lm_samples_mask_missing_targets() {
        let samples = padded_byte_lm_samples(&["ab", ""], 3, -1).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].0.shape(), (3, BYTE_LM_VOCAB));
        assert_eq!(samples[0].1.data(), &[b'b' as f32, -1.0, -1.0]);
        assert_eq!(samples[1].1.data(), &[-1.0, -1.0, -1.0]);

        let stats = byte_lm_sample_stats(&samples, Some(-1));
        assert_eq!(
            stats,
            ByteLmSampleStats {
                samples: 2,
                total_rows: 6,
                active_rows: 1,
            }
        );
    }

    #[test]
    fn validate_byte_lm_samples_accepts_windowed_and_padded_rows() {
        let samples = byte_lm_windows("abcd", 2).unwrap();
        let stats = validate_byte_lm_samples(&samples, None).unwrap();
        assert_eq!(
            stats,
            ByteLmSampleStats {
                samples: 2,
                total_rows: 4,
                active_rows: 4,
            }
        );

        let padded = padded_byte_lm_samples(&["ab", ""], 3, -1).unwrap();
        let stats = validate_byte_lm_samples(&padded, Some(-1)).unwrap();
        assert_eq!(
            stats,
            ByteLmSampleStats {
                samples: 2,
                total_rows: 6,
                active_rows: 1,
            }
        );
    }

    #[test]
    fn validate_byte_lm_samples_rejects_malformed_training_rows() {
        assert!(matches!(
            validate_byte_lm_samples(&[], None),
            Err(TensorError::EmptyInput("validate_byte_lm_samples"))
        ));

        let bad_cols = vec![(
            Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )];
        assert!(matches!(
            validate_byte_lm_samples(&bad_cols, None),
            Err(TensorError::ShapeMismatch {
                left: (1, BYTE_LM_VOCAB),
                right: (1, 2)
            })
        ));

        let mut bad_input = vec![0.0; BYTE_LM_VOCAB];
        bad_input[0] = 0.5;
        bad_input[1] = 0.5;
        let bad_input = vec![(
            Tensor::from_vec(1, BYTE_LM_VOCAB, bad_input).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )];
        assert!(matches!(
            validate_byte_lm_samples(&bad_input, None),
            Err(TensorError::IoError { .. })
        ));

        let mut input = vec![0.0; BYTE_LM_VOCAB];
        input[0] = 1.0;
        let bad_target = vec![(
            Tensor::from_vec(1, BYTE_LM_VOCAB, input).unwrap(),
            Tensor::from_vec(1, 1, vec![BYTE_LM_VOCAB as f32]).unwrap(),
        )];
        assert!(matches!(
            validate_byte_lm_samples(&bad_target, None),
            Err(TensorError::InvalidClassIndex {
                index,
                classes: BYTE_LM_VOCAB
            }) if index == BYTE_LM_VOCAB as f32
        ));

        let all_padding = padded_byte_lm_samples(&[""], 2, -1).unwrap();
        assert!(matches!(
            validate_byte_lm_samples(&all_padding, Some(-1)),
            Err(TensorError::EmptyInput(
                "validate_byte_lm_samples_active_rows"
            ))
        ));
    }

    #[test]
    fn byte_lm_helpers_reject_empty_shapes() {
        assert!(matches!(
            byte_lm_windows("abc", 0),
            Err(TensorError::InvalidDimensions {
                rows: 0,
                cols: BYTE_LM_VOCAB
            })
        ));
        assert!(matches!(
            padded_byte_lm_samples(&["abc"], 0, -1),
            Err(TensorError::InvalidDimensions {
                rows: 0,
                cols: BYTE_LM_VOCAB
            })
        ));
        assert!(matches!(
            byte_lm_windows("ab", 2),
            Err(TensorError::EmptyInput("byte_lm_windows"))
        ));
        assert!(matches!(
            byte_lm_corpus_windows(&["ab", ""], 2),
            Err(TensorError::EmptyInput("byte_lm_corpus_windows"))
        ));
        let sample = vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )];
        assert!(matches!(
            interleave_replay_samples(&sample, &sample, 0),
            Err(TensorError::InvalidDimensions { rows: 0, cols: 1 })
        ));
        assert!(matches!(
            interleave_replay_samples(&[], &sample, 1),
            Err(TensorError::EmptyInput("interleave_replay_samples_target"))
        ));
        assert!(matches!(
            interleave_replay_samples(&sample, &[], 1),
            Err(TensorError::EmptyInput("interleave_replay_samples_replay"))
        ));
    }

    #[test]
    fn interleave_replay_samples_inserts_cycled_source_replay() {
        let make = |value: f32| {
            (
                Tensor::from_vec(1, 1, vec![value]).unwrap(),
                Tensor::from_vec(1, 1, vec![value]).unwrap(),
            )
        };
        let target = vec![make(0.0), make(1.0), make(2.0), make(3.0), make(4.0)];
        let replay = vec![make(100.0), make(200.0)];
        let mixed = interleave_replay_samples(&target, &replay, 2).unwrap();
        let order: Vec<f32> = mixed.iter().map(|(input, _)| input.data()[0]).collect();
        assert_eq!(order, vec![0.0, 1.0, 100.0, 2.0, 3.0, 200.0, 4.0]);
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
}
