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

impl AlignedVec {
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                cap: 0,
            };
        }
        let ptr = unsafe { Self::allocate(capacity) };
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
        if self.len == self.cap {
            self.grow(1);
        }
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }

    pub fn extend_from_slice(&mut self, slice: &[f32]) {
        if slice.is_empty() {
            return;
        }
        if self.len + slice.len() > self.cap {
            self.grow(slice.len());
        }
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.as_ptr().add(self.len), slice.len());
        }
        self.len += slice.len();
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
            unsafe {
                self.ptr.as_ptr().add(self.len).write(value);
            }
            self.len += 1;
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    unsafe fn allocate(capacity: usize) -> NonNull<f32> {
        let layout = Self::layout(capacity);
        let ptr = alloc(layout);
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        NonNull::new_unchecked(ptr as *mut f32)
    }

    fn layout(capacity: usize) -> Layout {
        Layout::from_size_align(capacity * mem::size_of::<f32>(), ALIGNMENT).unwrap()
    }

    fn grow(&mut self, additional: usize) {
        let new_cap = (self.cap.max(1) * 2).max(self.len + additional);
        let new_ptr = unsafe { Self::allocate(new_cap) };
        if self.len > 0 {
            unsafe {
                ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
        }
        if self.cap != 0 {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.cap));
            }
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
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
        if self.cap != 0 {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.cap));
            }
        }
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
    (ptr as usize) % alignment == 0
}
