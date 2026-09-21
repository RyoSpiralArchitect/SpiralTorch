//! Checked logical views, independent of storage, device, and execution.

use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum NdLayoutError {
    #[error("shape, stride, or offset exceeds the address space")]
    Overflow,
    #[error("axes must be a permutation of all dimensions")]
    InvalidPermutation,
    #[error("slice is outside the selected axis")]
    InvalidSlice,
    #[error("reshape requires contiguous storage and the same element count")]
    InvalidReshape,
    #[error("dimensions cannot be broadcast to the requested shape")]
    InvalidBroadcast,
}

/// Element strides, not byte strides. Views never allocate or move tensor data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NdLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    len: usize,
}

/// Read-only last-axis rows with unit column stride. Leading axes flatten in
/// logical order: address = offset + row * row_stride + column. Row stride may
/// be zero for broadcast rows. This does not grant mutable/nonoverlapping access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowMajorRows {
    rows: usize,
    cols: usize,
    row_stride: usize,
    offset: usize,
}

impl RowMajorRows {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl NdLayout {
    /// A scalar has shape `[]` and one element; zero-sized axes are allowed.
    pub fn contiguous(shape: &[usize]) -> Result<Self, NdLayoutError> {
        let mut strides = vec![0; shape.len()];
        let mut len = 1usize;
        for axis in (0..shape.len()).rev() {
            strides[axis] = len;
            len = len
                .checked_mul(shape[axis])
                .ok_or(NdLayoutError::Overflow)?;
        }
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            offset: 0,
            len,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_contiguous(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for axis in (0..self.rank()).rev() {
            if self.shape[axis] > 1 && self.strides[axis] != expected {
                return false;
            }
            expected *= self.shape[axis];
        }
        true
    }

    /// Describe nonempty last-axis rows without packing. Return None for a
    /// scalar, strided columns, or non-affine flattened leading axes. Singleton
    /// strides are unobservable; offsets and broadcast rows are preserved.
    pub fn row_major_rows(&self) -> Option<RowMajorRows> {
        let cols = *self.shape.last()?;
        if self.is_empty() || (cols > 1 && *self.strides.last()? != 1) {
            return None;
        }
        let mut rows = 1usize;
        let mut row_stride: Option<usize> = None;
        for axis in (0..self.rank() - 1).rev() {
            let size = self.shape[axis];
            if size > 1 {
                if let Some(stride) = row_stride {
                    if self.strides[axis] != stride.checked_mul(rows)? {
                        return None;
                    }
                } else {
                    row_stride = Some(self.strides[axis]);
                }
            }
            rows = rows.checked_mul(size)?;
        }
        let row_stride = row_stride.unwrap_or(cols);
        self.offset
            .checked_add((rows - 1).checked_mul(row_stride)?)?
            .checked_add(cols)?;
        Some(RowMajorRows {
            rows,
            cols,
            row_stride,
            offset: self.offset,
        })
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self, NdLayoutError> {
        let mut seen = vec![false; self.rank()];
        if axes.len() != self.rank() {
            return Err(NdLayoutError::InvalidPermutation);
        }
        for &axis in axes {
            if axis >= self.rank() || seen[axis] {
                return Err(NdLayoutError::InvalidPermutation);
            }
            seen[axis] = true;
        }
        Ok(Self {
            shape: axes.iter().map(|&axis| self.shape[axis]).collect(),
            strides: axes.iter().map(|&axis| self.strides[axis]).collect(),
            offset: self.offset,
            len: self.len,
        })
    }

    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self, NdLayoutError> {
        let size = *self.shape.get(axis).ok_or(NdLayoutError::InvalidSlice)?;
        if start > size || length > size - start {
            return Err(NdLayoutError::InvalidSlice);
        }
        let mut next = self.clone();
        next.shape[axis] = length;
        next.len = Self::contiguous(&next.shape)?.len;
        // An empty view has no addressable elements, including an end slice.
        if !next.is_empty() {
            next.offset = self
                .offset
                .checked_add(
                    start
                        .checked_mul(self.strides[axis])
                        .ok_or(NdLayoutError::Overflow)?,
                )
                .ok_or(NdLayoutError::Overflow)?;
        }
        Ok(next)
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self, NdLayoutError> {
        let mut next = Self::contiguous(shape)?;
        if !self.is_contiguous() || next.len != self.len {
            return Err(NdLayoutError::InvalidReshape);
        }
        next.offset = self.offset;
        Ok(next)
    }

    /// Read-only expansion. Expanded dimensions have zero element stride.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, NdLayoutError> {
        if shape.len() < self.rank() {
            return Err(NdLayoutError::InvalidBroadcast);
        }
        let mut next = Self::contiguous(shape)?;
        next.strides.fill(0);
        next.offset = self.offset;
        let leading = shape.len() - self.rank();
        for axis in 0..self.rank() {
            if self.shape[axis] == shape[leading + axis] {
                next.strides[leading + axis] = self.strides[axis];
            } else if self.shape[axis] != 1 {
                return Err(NdLayoutError::InvalidBroadcast);
            }
        }
        Ok(next)
    }

    /// Exclusive upper storage bound, including offsets and gaps, not logical length.
    pub fn required_storage_len(&self) -> Result<usize, NdLayoutError> {
        if self.is_empty() {
            return Ok(0);
        }
        self.shape
            .iter()
            .zip(&self.strides)
            .try_fold(self.offset, |end, (&n, &s)| {
                end.checked_add((n - 1).checked_mul(s).ok_or(NdLayoutError::Overflow)?)
                    .ok_or(NdLayoutError::Overflow)
            })?
            .checked_add(1)
            .ok_or(NdLayoutError::Overflow)
    }

    /// Resolve a row-major logical index into the original storage.
    pub fn storage_index(&self, mut logical: usize) -> Option<usize> {
        if logical >= self.len {
            return None;
        }
        let mut address = self.offset;
        for axis in (0..self.rank()).rev() {
            address = address
                .checked_add((logical % self.shape[axis]).checked_mul(self.strides[axis])?)?;
            logical /= self.shape[axis];
        }
        Some(address)
    }
}

/// Right-aligned broadcasting, including scalar and zero-sized axes.
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, NdLayoutError> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = vec![1; rank];
    for (axis, out) in shape.iter_mut().enumerate() {
        let a = axis.checked_sub(rank - lhs.len()).map_or(1, |i| lhs[i]);
        let b = axis.checked_sub(rank - rhs.len()).map_or(1, |i| rhs[i]);
        *out = if a == b || b == 1 {
            a
        } else if a == 1 {
            b
        } else {
            return Err(NdLayoutError::InvalidBroadcast);
        };
    }
    NdLayout::contiguous(&shape)?;
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_rows(layout: &NdLayout) {
        let actual = layout.row_major_rows();
        let expected = layout
            .shape()
            .last()
            .copied()
            .filter(|_| !layout.is_empty())
            .and_then(|cols| {
                let rows = layout.len() / cols;
                let stride = if rows > 1 {
                    layout.storage_index(cols)?.checked_sub(layout.offset())?
                } else {
                    cols
                };
                (0..layout.len())
                    .all(|i| {
                        layout.storage_index(i)
                            == Some(layout.offset() + (i / cols) * stride + i % cols)
                    })
                    .then_some((rows, cols, stride, layout.offset()))
            });
        assert_eq!(
            actual.map(|r| (r.rows(), r.cols(), r.row_stride(), r.offset())),
            expected,
            "{layout:?}"
        );
    }

    #[test]
    fn row_addressing_matches_every_element_of_permuted_and_sliced_small_layouts() {
        for a in 0..=3 {
            for b in 0..=3 {
                for c in 0..=3 {
                    let base = NdLayout::contiguous(&[a, b, c]).unwrap();
                    for axes in [
                        [0, 1, 2],
                        [0, 2, 1],
                        [1, 0, 2],
                        [1, 2, 0],
                        [2, 0, 1],
                        [2, 1, 0],
                    ] {
                        let permuted = base.permute(&axes).unwrap();
                        check_rows(&permuted);
                        for axis in 0..3 {
                            for start in 0..=permuted.shape()[axis] {
                                for length in 0..=permuted.shape()[axis] - start {
                                    check_rows(&permuted.narrow(axis, start, length).unwrap());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn row_addressing_handles_broadcast_singletons_offsets_and_overflow() {
        for shape in [&[3][..], &[1, 3], &[2, 1, 3], &[1, 2, 3], &[1, 1, 1]] {
            let base = NdLayout::contiguous(shape).unwrap();
            check_rows(&base);
            for target in [&[2, 2, 3][..], &[4, 2, 1, 3], &[2, 3, 1], &[2, 2, 2]] {
                if let Ok(view) = base.broadcast_to(target) {
                    check_rows(&view);
                }
            }
        }
        assert!(NdLayout::contiguous(&[])
            .unwrap()
            .row_major_rows()
            .is_none());
        let positions = NdLayout::contiguous(&[65, 64, 4])
            .unwrap()
            .narrow(2, 0, 3)
            .unwrap();
        assert_eq!(positions.row_major_rows().unwrap().row_stride(), 4);
        let offset = positions.narrow(0, 1, 64).unwrap();
        assert_eq!(offset.row_major_rows().unwrap().offset(), 256);
        assert!(positions
            .narrow(1, 1, 63)
            .unwrap()
            .row_major_rows()
            .is_none());
        let overflowing = NdLayout {
            shape: vec![2, 3],
            strides: vec![usize::MAX, 1],
            offset: 1,
            len: 6,
        };
        assert!(overflowing.row_major_rows().is_none());
    }

    #[test]
    fn broadcasting_uses_zero_strides_and_preserves_storage_bounds() {
        let base = NdLayout::contiguous(&[2, 1, 3]).unwrap();
        let view = base
            .narrow(0, 1, 1)
            .unwrap()
            .broadcast_to(&[4, 5, 3])
            .unwrap();
        assert_eq!(view.strides(), &[0, 0, 1]);
        assert_eq!(view.required_storage_len().unwrap(), 6);
        for i in 0..view.len() {
            assert_eq!(view.storage_index(i), Some(3 + i % 3));
        }
        assert!(!view.is_contiguous());
        assert!(view.reshape(&[60]).is_err());
        assert_eq!(broadcast_shape(&[2, 1, 3], &[5, 1]).unwrap(), [2, 5, 3]);
        assert_eq!(broadcast_shape(&[0, 3], &[1, 3]).unwrap(), [0, 3]);
        assert_eq!(broadcast_shape(&[], &[0, 3]).unwrap(), [0, 3]);
        assert!(broadcast_shape(&[2, 3], &[4, 3]).is_err());
        assert!(base.broadcast_to(&[1, 3]).is_err());
        let empty = base.broadcast_to(&[2, 0, 3]).unwrap();
        assert_eq!(empty.required_storage_len().unwrap(), 0);
        assert_eq!(empty.storage_index(0), None);
    }

    #[test]
    fn scalar_empty_and_overflow() {
        let scalar = NdLayout::contiguous(&[]).unwrap();
        assert_eq!(scalar.len(), 1);
        assert_eq!(scalar.storage_index(0), Some(0));
        assert_eq!(scalar.storage_index(1), None);
        let empty = NdLayout::contiguous(&[2, 0, 3]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.storage_index(0), None);
        assert_eq!(
            NdLayout::contiguous(&[usize::MAX, 2]),
            Err(NdLayoutError::Overflow)
        );
    }

    #[test]
    fn views_preserve_storage_coordinates() {
        let base = NdLayout::contiguous(&[2, 3, 4]).unwrap();
        assert_eq!(base.strides(), &[12, 4, 1]);
        let view = base.permute(&[1, 0, 2]).unwrap().narrow(0, 1, 2).unwrap();
        assert_eq!(view.shape(), &[2, 2, 4]);
        assert_eq!(view.offset(), 4);
        assert_eq!(
            (0..16)
                .map(|i| view.storage_index(i).unwrap())
                .collect::<Vec<_>>(),
            vec![4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23]
        );
        assert!(view.reshape(&[4, 4]).is_err());
        let slice = base.narrow(0, 1, 1).unwrap().reshape(&[3, 4]).unwrap();
        assert_eq!(slice.storage_index(0), Some(12));
        assert_eq!(slice.storage_index(11), Some(23));
    }

    #[test]
    fn invalid_views_and_singleton_axes() {
        let base = NdLayout::contiguous(&[2, 1, 3]).unwrap();
        assert!(base.permute(&[1, 0, 2]).unwrap().is_contiguous());
        for axes in [&[0, 0, 2][..], &[0, 1][..], &[0, 1, 3][..]] {
            assert!(base.permute(axes).is_err());
        }
        assert!(base.narrow(3, 0, 1).is_err());
        assert!(base.narrow(0, 1, 2).is_err());
        assert!(base.narrow(0, usize::MAX, 1).is_err());
        assert!(base.reshape(&[5]).is_err());
        assert!(base.narrow(0, 2, 0).unwrap().is_empty());
    }
}
