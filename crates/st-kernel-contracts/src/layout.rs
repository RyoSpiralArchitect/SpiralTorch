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
}

/// Element strides, not byte strides. Views never allocate or move tensor data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NdLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    len: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

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
