use core::ops::Range;

#[derive(Debug, Clone, Copy, Default)]
pub struct BlockRecord {
    pub start: usize,
    pub len: usize,
    pub norm_sq: f32,
}

impl BlockRecord {
    #[inline]
    pub fn new(start: usize, len: usize, norm_sq: f32) -> Self {
        Self { start, len, norm_sq }
    }

    #[inline]
    pub fn range(&self) -> Range<usize> {
        self.start..self.start + self.len
    }
}

#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 256;

#[inline]
fn block_len(total_elems: usize, block_size: usize, block_index: usize, block_count: usize) -> usize {
    if block_index + 1 == block_count {
        total_elems - block_index * block_size
    } else {
        block_size
    }
}

#[cfg(feature = "parallel")]
fn fill_record(record: &mut BlockRecord, block_index: usize, block_size: usize, weights: &[f32], block_count: usize) {
    let start = block_index * block_size;
    let len = block_len(weights.len(), block_size, block_index, block_count);
    let slice = &weights[start..start + len];
    record.start = start;
    record.len = len;
    record.norm_sq = sum_squares(slice);
}

#[inline]
fn sum_squares(values: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;

    let mut chunks = values.chunks_exact(4);
    for lanes in &mut chunks {
        acc0 = lanes[0].mul_add(lanes[0], acc0);
        acc1 = lanes[1].mul_add(lanes[1], acc1);
        acc0 = lanes[2].mul_add(lanes[2], acc0);
        acc1 = lanes[3].mul_add(lanes[3], acc1);
    }

    let mut acc = acc0 + acc1;
    for &value in chunks.remainder() {
        acc = value.mul_add(value, acc);
    }

    acc
}

pub(crate) fn compute_block_norms(
    weights: &[f32],
    block_size: usize,
    block_count: usize,
    block_norms: &mut Vec<BlockRecord>,
) {
    if block_count == 0 {
        block_norms.clear();
        return;
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        if block_count >= PARALLEL_THRESHOLD {
            block_norms.resize(block_count, BlockRecord::default());
            block_norms
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, record)| fill_record(record, index, block_size, weights, block_count));
            return;
        }
    }

    block_norms.clear();
    block_norms.reserve(block_count);

    for block_index in 0..block_count {
        let start = block_index * block_size;
        let len = block_len(weights.len(), block_size, block_index, block_count);
        let slice = &weights[start..start + len];
        let norm_sq = sum_squares(slice);
        block_norms.push(BlockRecord::new(start, len, norm_sq));
    }
}

#[inline]
pub(crate) fn zero_block(weights: &mut [f32], block: &BlockRecord) {
    weights[block.range()].fill(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_squares_matches_naive() {
        let data: Vec<f32> = (0..101).map(|v| (v as f32 - 3.2) * 0.5).collect();
        let naive = data.iter().map(|v| v * v).sum::<f32>();
        assert!((sum_squares(&data) - naive).abs() < 1e-6);
    }

    #[test]
    fn compute_block_norms_covers_tail() {
        let data: Vec<f32> = (1..=15).map(|v| v as f32).collect();
        let mut blocks = Vec::new();
        compute_block_norms(&data, 4, (data.len() + 3) / 4, &mut blocks);
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[3].len, 3);
        let total = blocks.iter().map(|b| b.norm_sq).sum::<f32>();
        let naive = data.iter().map(|v| v * v).sum::<f32>();
        assert!((total - naive).abs() < 1e-5);
    }
}
