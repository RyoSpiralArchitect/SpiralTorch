// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::slice;

const ALIGNMENT: usize = 16;

#[derive(Debug)]
pub struct AlignedVec {
    ptr: NonNull<f32>,
    len: usize,
    cap: usize,
}

// SAFETY: `AlignedVec` owns its allocation and does not permit interior mutation; moving it across
// threads is safe, and sharing `&AlignedVec` is safe because it only yields immutable access to the
// underlying `f32` data.
unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

impl AlignedVec {
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: Self::dangling_aligned(),
                len: 0,
                cap: 0,
            };
        }
        let ptr = Self::allocate(capacity);
        Self {
            ptr,
            len: 0,
            cap: capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, value: f32) {
        let new_len = match self.len.checked_add(1) {
            Some(new_len) => new_len,
            None => panic!("AlignedVec length overflow (len={}, additional=1)", self.len),
        };
        if self.len == self.cap {
            self.grow(1);
        }
        // SAFETY: `self.len < self.cap` after `grow`, and elements within `0..self.len` are
        // initialized; writing to `self.len` stays within allocation bounds.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len = new_len;
    }

    pub fn extend_from_slice(&mut self, slice: &[f32]) {
        if slice.is_empty() {
            return;
        }
        let new_len = match self.len.checked_add(slice.len()) {
            Some(new_len) => new_len,
            None => panic!(
                "AlignedVec length overflow (len={}, additional={})",
                self.len,
                slice.len()
            ),
        };
        if new_len > self.cap {
            self.grow(slice.len());
        }
        // SAFETY: destination has at least `slice.len()` available after `grow`, and the source
        // and destination do not overlap.
        unsafe { ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.as_ptr().add(self.len), slice.len()) };
        self.len = new_len;
    }

    pub fn resize(&mut self, len: usize, value: f32) {
        if len <= self.len {
            self.len = len;
            return;
        }
        let additional = len - self.len;
        if self.len + additional > self.cap {
            self.grow(additional);
        }
        for _ in 0..additional {
            // SAFETY: `self.len < self.cap` after `grow`, so writing at `self.len` stays in-bounds.
            unsafe { self.ptr.as_ptr().add(self.len).write(value) };
            self.len += 1;
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `self.ptr` points to an allocation for at least `self.len` elements (or is
        // dangling for `len == 0`), and elements within `0..self.len` are initialized.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: same as `as_slice`, and we hold `&mut self` so the returned slice is unique.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn allocate(capacity: usize) -> NonNull<f32> {
        if capacity == 0 {
            return Self::dangling_aligned();
        }
        let layout = Self::layout(capacity);
        // SAFETY: `layout` is derived from a valid `Layout`, and the allocator contract is upheld
        // by deallocating via `Self::deallocate` with the same `layout`.
        let ptr = unsafe { alloc(layout) } as *mut f32;
        match NonNull::new(ptr) {
            Some(ptr) => ptr,
            None => handle_alloc_error(layout),
        }
    }

    fn layout(capacity: usize) -> Layout {
        let size = match capacity.checked_mul(mem::size_of::<f32>()) {
            Some(size) => size,
            None => panic!("AlignedVec allocation overflow (capacity={capacity})"),
        };
        match Layout::from_size_align(size, ALIGNMENT) {
            Ok(layout) => layout,
            Err(_) => panic!("AlignedVec invalid layout (size={size}, align={ALIGNMENT})"),
        }
    }

    fn dangling_aligned() -> NonNull<f32> {
        // SAFETY: `ALIGNMENT` is non-zero and a multiple of `align_of::<f32>()`, so this is a
        // well-aligned non-null dangling pointer suitable for `len == 0` slices.
        unsafe { NonNull::new_unchecked(ALIGNMENT as *mut f32) }
    }

    fn deallocate(ptr: NonNull<f32>, capacity: usize) {
        if capacity == 0 {
            return;
        }
        let layout = Self::layout(capacity);
        // SAFETY: `ptr` was allocated by `Self::allocate(capacity)` with the same `layout` and is
        // deallocated exactly once (on `Drop` or after `grow`).
        unsafe { dealloc(ptr.as_ptr() as *mut u8, layout) };
    }

    fn grow(&mut self, additional: usize) {
        let required = match self.len.checked_add(additional) {
            Some(required) => required,
            None => panic!(
                "AlignedVec capacity overflow (len={}, additional={})",
                self.len, additional
            ),
        };
        let doubled = match self.cap.checked_mul(2) {
            Some(doubled) => doubled,
            None => required,
        };
        let new_cap = doubled.max(required).max(1);
        let new_ptr = Self::allocate(new_cap);
        if self.len > 0 {
            // SAFETY: both source and destination are valid for `self.len` elements, and they do
            // not overlap because `new_ptr` comes from a fresh allocation.
            unsafe { ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len) };
        }
        let old_ptr = self.ptr;
        let old_cap = self.cap;
        self.ptr = new_ptr;
        self.cap = new_cap;
        Self::deallocate(old_ptr, old_cap);
    }
}

impl Default for AlignedVec {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Clone for AlignedVec {
    fn clone(&self) -> Self {
        let mut cloned = AlignedVec::with_capacity(self.len);
        cloned.extend_from_slice(self.as_slice());
        cloned
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        Self::deallocate(self.ptr, self.cap);
    }
}

impl Deref for AlignedVec {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl From<Vec<f32>> for AlignedVec {
    fn from(source: Vec<f32>) -> Self {
        aligned_from_slice(&source)
    }
}

pub fn aligned_zeroed(len: usize) -> AlignedVec {
    let mut data = AlignedVec::with_capacity(len);
    data.resize(len, 0.0);
    data
}

pub fn aligned_from_vec(source: Vec<f32>) -> AlignedVec {
    AlignedVec::from(source)
}

pub fn aligned_from_slice(slice: &[f32]) -> AlignedVec {
    let mut data = AlignedVec::with_capacity(slice.len());
    data.extend_from_slice(slice);
    data
}

pub fn aligned_with_capacity(len: usize) -> AlignedVec {
    AlignedVec::with_capacity(len)
}

pub fn is_ptr_aligned(ptr: *const f32, alignment: usize) -> bool {
    assert!(alignment != 0, "alignment must be non-zero");
    (ptr as usize).is_multiple_of(alignment)
}

#[cfg(test)]
mod tests {
    use super::{
        aligned_from_slice, aligned_with_capacity, aligned_zeroed, is_ptr_aligned, AlignedVec,
        ALIGNMENT,
    };

    #[test]
    fn aligned_vec_is_16b_aligned() {
        let data = aligned_with_capacity(1);
        assert!(is_ptr_aligned(data.as_slice().as_ptr(), ALIGNMENT));
    }

    #[test]
    fn aligned_vec_empty_is_16b_aligned() {
        let data = aligned_with_capacity(0);
        assert!(is_ptr_aligned(data.as_slice().as_ptr(), ALIGNMENT));
    }

    #[test]
    fn aligned_vec_grows_and_preserves_contents() {
        let mut data = AlignedVec::with_capacity(1);
        for idx in 0..100 {
            data.push(idx as f32);
        }
        assert_eq!(data.len(), 100);
        assert!(is_ptr_aligned(data.as_slice().as_ptr(), ALIGNMENT));
        for idx in 0..100 {
            assert_eq!(data.as_slice()[idx], idx as f32);
        }
    }

    #[test]
    fn aligned_from_slice_matches_input() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let data = aligned_from_slice(&input);
        assert_eq!(data.as_slice(), &input);
        assert!(is_ptr_aligned(data.as_slice().as_ptr(), ALIGNMENT));
    }

    #[test]
    fn aligned_zeroed_is_zero() {
        let data = aligned_zeroed(8);
        assert_eq!(data.len(), 8);
        assert!(data.as_slice().iter().all(|&value| value == 0.0));
        assert!(is_ptr_aligned(data.as_slice().as_ptr(), ALIGNMENT));
    }

    #[test]
    fn aligned_vec_clone_is_deep_copy() {
        let data = aligned_from_slice(&[1.0, 2.0, 3.0]);
        let mut cloned = data.clone();
        cloned.as_mut_slice()[0] = 10.0;
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(cloned.as_slice(), &[10.0, 2.0, 3.0]);
    }

    #[test]
    fn aligned_vec_resize_fills_values() {
        let mut data = aligned_with_capacity(0);
        data.resize(4, 1.25);
        assert_eq!(data.as_slice(), &[1.25, 1.25, 1.25, 1.25]);
    }

    #[test]
    fn aligned_vec_extend_appends_values() {
        let mut data = AlignedVec::with_capacity(1);
        data.push(1.0);
        data.extend_from_slice(&[2.0, 3.0]);
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0]);
        assert!(is_ptr_aligned(data.as_slice().as_ptr(), ALIGNMENT));
    }

    #[test]
    fn aligned_vec_resize_shrinks_len() {
        let mut data = aligned_from_slice(&[1.0, 2.0, 3.0]);
        data.resize(1, 0.0);
        assert_eq!(data.as_slice(), &[1.0]);
    }
}
