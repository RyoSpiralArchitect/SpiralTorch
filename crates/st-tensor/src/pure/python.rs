// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{AmegaHypergrad, LanguageWaveEncoder, OpenCartesianTopos, Tensor, TensorError};
use core::mem;
use core::ptr;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};

thread_local! {
    static ERROR_STORAGE: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn store_error_message(message: impl AsRef<str>) -> i32 {
    let message = CString::new(message.as_ref()).unwrap_or_else(|_| c"tensor error".to_owned());
    ERROR_STORAGE.with(|slot| {
        *slot.borrow_mut() = Some(message);
    });
    -1
}

fn store_error(err: TensorError) -> i32 {
    store_error_message(err.to_string())
}

fn clear_error() {
    ERROR_STORAGE.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

#[inline]
fn catch_unwind_or<T>(fallback: T, panic_message: &'static str, f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(_) => {
            store_error_message(panic_message);
            fallback
        }
    }
}

#[inline]
fn is_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(mem::align_of::<T>())
}

#[inline]
fn require_aligned<T>(ptr: *const T, label: &'static str) -> bool {
    if is_aligned(ptr) {
        true
    } else {
        store_error_message(format!("{label} is misaligned"));
        false
    }
}

fn data_length_mismatch(expected: usize, got: usize) -> TensorError {
    TensorError::DataLength { expected, got }
}

fn checked_element_count(
    rows: usize,
    cols: usize,
    label: &'static str,
) -> Result<usize, TensorError> {
    rows.checked_mul(cols).ok_or_else(|| {
        TensorError::Generic(format!(
            "{label} element count overflows usize ({rows} x {cols})"
        ))
    })
}

#[no_mangle]
pub extern "C" fn st_pure_last_error() -> *const c_char {
    catch_unwind_or(
        ptr::null(),
        "panic in st_pure_last_error",
        || {
            ERROR_STORAGE.with(|slot| match slot.borrow().as_ref() {
                Some(msg) => msg.as_ptr(),
                None => ptr::null(),
            })
        },
    )
}

#[no_mangle]
pub extern "C" fn st_pure_clear_last_error() {
    catch_unwind_or((), "panic in st_pure_clear_last_error", || {
        clear_error();
    });
}

#[no_mangle]
pub extern "C" fn st_pure_topos_new(
    curvature: f32,
    tolerance: f32,
    saturation: f32,
    max_depth: usize,
    max_volume: usize,
) -> *mut OpenCartesianTopos {
    catch_unwind_or(ptr::null_mut(), "panic in st_pure_topos_new", || {
        match OpenCartesianTopos::new(curvature, tolerance, saturation, max_depth, max_volume) {
            Ok(topos) => Box::into_raw(Box::new(topos)),
            Err(err) => {
                store_error(err);
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
/// # Safety
/// - If `ptr` is non-null it must be a valid pointer obtained from `st_pure_topos_new`.
/// - The pointer must be freed at most once.
pub unsafe extern "C" fn st_pure_topos_free(ptr: *mut OpenCartesianTopos) {
    catch_unwind_or((), "panic in st_pure_topos_free", || {
        if ptr.is_null() {
            return;
        }
        if !require_aligned(ptr, "topos handle") {
            return;
        }
        // Safety: the pointer must originate from `st_pure_topos_new` and be freed exactly once.
        unsafe { drop(Box::from_raw(ptr)) }
    });
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_new(
    curvature: f32,
    learning_rate: f32,
    rows: usize,
    cols: usize,
) -> *mut AmegaHypergrad {
    catch_unwind_or(ptr::null_mut(), "panic in st_pure_hypergrad_new", || {
        match AmegaHypergrad::new(curvature, learning_rate, rows, cols) {
            Ok(hypergrad) => Box::into_raw(Box::new(hypergrad)),
            Err(err) => {
                store_error(err);
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
/// # Safety
/// - If `topos` is non-null it must be a valid pointer obtained from `st_pure_topos_new` and not
///   yet freed.
pub unsafe extern "C" fn st_pure_hypergrad_with_topos(
    curvature: f32,
    learning_rate: f32,
    rows: usize,
    cols: usize,
    topos: *const OpenCartesianTopos,
) -> *mut AmegaHypergrad {
    catch_unwind_or(ptr::null_mut(), "panic in st_pure_hypergrad_with_topos", || {
        if topos.is_null() {
            store_error(TensorError::EmptyInput("topos handle"));
            return ptr::null_mut();
        }
        if !require_aligned(topos, "topos handle") {
            return ptr::null_mut();
        }
        // Safety: caller must pass a valid topos pointer originating from `st_pure_topos_new`.
        let guard = unsafe { &*topos };
        match AmegaHypergrad::with_topos(curvature, learning_rate, rows, cols, guard.clone()) {
            Ok(hypergrad) => Box::into_raw(Box::new(hypergrad)),
            Err(err) => {
                store_error(err);
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
/// # Safety
/// - If `ptr` is non-null it must be a valid pointer obtained from `st_pure_hypergrad_new` or
///   `st_pure_hypergrad_with_topos`.
/// - The pointer must be freed at most once.
pub unsafe extern "C" fn st_pure_hypergrad_free(ptr: *mut AmegaHypergrad) {
    catch_unwind_or((), "panic in st_pure_hypergrad_free", || {
        if ptr.is_null() {
            return;
        }
        if !require_aligned(ptr, "hypergrad handle") {
            return;
        }
        // Safety: the pointer must originate from `st_pure_hypergrad_new`/`st_pure_hypergrad_with_topos`
        // and be freed exactly once.
        unsafe { drop(Box::from_raw(ptr)) }
    });
}

#[no_mangle]
pub extern "C" fn st_pure_encoder_new(
    curvature: f32,
    temperature: f32,
) -> *mut LanguageWaveEncoder {
    catch_unwind_or(ptr::null_mut(), "panic in st_pure_encoder_new", || {
        match LanguageWaveEncoder::new(curvature, temperature) {
            Ok(encoder) => Box::into_raw(Box::new(encoder)),
            Err(err) => {
                store_error(err);
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
/// # Safety
/// - If `ptr` is non-null it must be a valid pointer obtained from `st_pure_encoder_new`.
/// - The pointer must be freed at most once.
pub unsafe extern "C" fn st_pure_encoder_free(ptr: *mut LanguageWaveEncoder) {
    catch_unwind_or((), "panic in st_pure_encoder_free", || {
        if ptr.is_null() {
            return;
        }
        if !require_aligned(ptr, "encoder handle") {
            return;
        }
        // Safety: the pointer must originate from `st_pure_encoder_new` and be freed exactly once.
        unsafe { drop(Box::from_raw(ptr)) }
    });
}

#[no_mangle]
/// # Safety
/// - `hypergrad` must point to a valid `AmegaHypergrad` created by this library.
/// - `rows_out` and `cols_out` must be valid writable pointers.
pub unsafe extern "C" fn st_pure_hypergrad_shape(
    hypergrad: *const AmegaHypergrad,
    rows_out: *mut usize,
    cols_out: *mut usize,
) -> i32 {
    catch_unwind_or(-1, "panic in st_pure_hypergrad_shape", || {
        if hypergrad.is_null() {
            return store_error(TensorError::EmptyInput("hypergrad handle"));
        }
        if !require_aligned(hypergrad, "hypergrad handle") {
            return -1;
        }
        if rows_out.is_null() {
            return store_error(TensorError::EmptyInput("rows pointer"));
        }
        if cols_out.is_null() {
            return store_error(TensorError::EmptyInput("cols pointer"));
        }
        if !require_aligned(rows_out, "rows pointer") {
            return -1;
        }
        if !require_aligned(cols_out, "cols pointer") {
            return -1;
        }
        // Safety: caller guarantees `hypergrad` is a valid pointer to an `AmegaHypergrad`.
        let hypergrad = unsafe { &*hypergrad };
        let (rows, cols) = hypergrad.shape();
        unsafe {
            *rows_out = rows;
            *cols_out = cols;
        }
        0
    })
}

unsafe fn tensor_from_pointer(
    rows: usize,
    cols: usize,
    ptr: *const f32,
    len: usize,
    label: &'static str,
) -> Result<Tensor, TensorError> {
    if ptr.is_null() {
        return Err(TensorError::EmptyInput(label));
    }
    if !is_aligned(ptr) {
        return Err(TensorError::InvalidValue {
            label: "tensor pointer is misaligned",
        });
    }
    let expected = checked_element_count(rows, cols, label)?;
    if len != expected {
        return Err(data_length_mismatch(expected, len));
    }
    // Safety: the caller guarantees `ptr` is valid for reads of `len` f32 values.
    let slice = unsafe { slice::from_raw_parts(ptr, len) };
    Tensor::from_vec(rows, cols, slice.to_vec())
}

#[no_mangle]
/// # Safety
/// - `hypergrad` must point to a valid mutable `AmegaHypergrad` created by this library.
/// - `prediction`/`target` must point to arrays of `prediction_len`/`target_len` `f32` values.
pub unsafe extern "C" fn st_pure_hypergrad_accumulate_pair(
    hypergrad: *mut AmegaHypergrad,
    prediction: *const f32,
    prediction_len: usize,
    target: *const f32,
    target_len: usize,
) -> i32 {
    catch_unwind_or(-1, "panic in st_pure_hypergrad_accumulate_pair", || {
        if hypergrad.is_null() {
            return store_error(TensorError::EmptyInput("hypergrad handle"));
        }
        if !require_aligned(hypergrad, "hypergrad handle") {
            return -1;
        }
        if prediction.is_null() {
            return store_error(TensorError::EmptyInput("prediction"));
        }
        if !require_aligned(prediction, "prediction") {
            return -1;
        }
        if target.is_null() {
            return store_error(TensorError::EmptyInput("target"));
        }
        if !require_aligned(target, "target") {
            return -1;
        }
        // Safety: caller guarantees `hypergrad` points to a valid, mutable `AmegaHypergrad`.
        let hypergrad = unsafe { &mut *hypergrad };
        let (rows, cols) = hypergrad.shape();
        let prediction_tensor = match unsafe {
            tensor_from_pointer(rows, cols, prediction, prediction_len, "prediction")
        } {
            Ok(tensor) => tensor,
            Err(err) => return store_error(err),
        };
        let target_tensor =
            match unsafe { tensor_from_pointer(rows, cols, target, target_len, "target") } {
                Ok(tensor) => tensor,
                Err(err) => return store_error(err),
            };
        match hypergrad.accumulate_pair(&prediction_tensor, &target_tensor) {
            Ok(_) => 0,
            Err(err) => store_error(err),
        }
    })
}

#[no_mangle]
/// # Safety
/// - `hypergrad` must point to a valid mutable `AmegaHypergrad` created by this library.
/// - `weights` must point to an array of `weights_len` `f32` values.
pub unsafe extern "C" fn st_pure_hypergrad_apply(
    hypergrad: *mut AmegaHypergrad,
    weights: *mut f32,
    weights_len: usize,
) -> i32 {
    catch_unwind_or(-1, "panic in st_pure_hypergrad_apply", || {
        if hypergrad.is_null() {
            return store_error(TensorError::EmptyInput("hypergrad handle"));
        }
        if !require_aligned(hypergrad, "hypergrad handle") {
            return -1;
        }
        if weights.is_null() {
            return store_error(TensorError::EmptyInput("weights"));
        }
        if !require_aligned(weights, "weights") {
            return -1;
        }
        // Safety: caller guarantees `hypergrad` points to a valid, mutable `AmegaHypergrad`.
        let hypergrad = unsafe { &mut *hypergrad };
        let (rows, cols) = hypergrad.shape();
        let expected = match checked_element_count(rows, cols, "weights") {
            Ok(expected) => expected,
            Err(err) => return store_error(err),
        };
        if weights_len != expected {
            return store_error(data_length_mismatch(expected, weights_len));
        }
        // Safety: caller guarantees `weights` is valid for reads of `expected` f32 values.
        let slice = unsafe { slice::from_raw_parts(weights, expected) };
        let mut tensor = match Tensor::from_vec(rows, cols, slice.to_vec()) {
            Ok(tensor) => tensor,
            Err(err) => return store_error(err),
        };
        match hypergrad.apply(&mut tensor) {
            Ok(_) => {
                let data = tensor.data();
                // Safety: caller guarantees `weights` is valid for writes of `expected` f32 values.
                let dest = unsafe { slice::from_raw_parts_mut(weights, expected) };
                dest.copy_from_slice(data);
                0
            }
            Err(err) => store_error(err),
        }
    })
}

#[no_mangle]
/// # Safety
/// - `hypergrad` must point to a valid `AmegaHypergrad` created by this library.
/// - If `out` is non-null it must be valid for writes of `out_len` `f32` values.
pub unsafe extern "C" fn st_pure_hypergrad_gradient(
    hypergrad: *const AmegaHypergrad,
    out: *mut f32,
    out_len: usize,
) -> usize {
    catch_unwind_or(0, "panic in st_pure_hypergrad_gradient", || {
        if hypergrad.is_null() {
            store_error(TensorError::EmptyInput("hypergrad handle"));
            return 0;
        }
        if !require_aligned(hypergrad, "hypergrad handle") {
            return 0;
        }
        // Safety: caller guarantees `hypergrad` points to a valid `AmegaHypergrad`.
        let hypergrad = unsafe { &*hypergrad };
        let gradient = hypergrad.gradient();
        if out.is_null() {
            return gradient.len();
        }
        if !require_aligned(out, "out") {
            return 0;
        }
        if out_len < gradient.len() {
            store_error(data_length_mismatch(gradient.len(), out_len));
            return 0;
        }
        // Safety: caller guarantees `out` is valid for writes of `gradient.len()` f32 values.
        let dest = unsafe { slice::from_raw_parts_mut(out, gradient.len()) };
        dest.copy_from_slice(gradient);
        gradient.len()
    })
}

#[no_mangle]
/// # Safety
/// - `hypergrad` must point to a valid mutable `AmegaHypergrad` created by this library.
/// - `encoder` must point to a valid `LanguageWaveEncoder` created by this library.
/// - `text` must point to a NUL-terminated C string.
pub unsafe extern "C" fn st_pure_hypergrad_absorb_text(
    hypergrad: *mut AmegaHypergrad,
    encoder: *const LanguageWaveEncoder,
    text: *const c_char,
) -> i32 {
    catch_unwind_or(-1, "panic in st_pure_hypergrad_absorb_text", || {
        if hypergrad.is_null() {
            return store_error(TensorError::EmptyInput("hypergrad handle"));
        }
        if !require_aligned(hypergrad, "hypergrad handle") {
            return -1;
        }
        if encoder.is_null() {
            return store_error(TensorError::EmptyInput("encoder handle"));
        }
        if !require_aligned(encoder, "encoder handle") {
            return -1;
        }
        if text.is_null() {
            return store_error(TensorError::EmptyInput("text pointer"));
        }
        // Safety: caller guarantees `hypergrad` points to a valid, mutable `AmegaHypergrad`.
        let hypergrad = unsafe { &mut *hypergrad };
        // Safety: caller guarantees `encoder` points to a valid `LanguageWaveEncoder`.
        let encoder = unsafe { &*encoder };
        // Safety: caller guarantees `text` points to a NUL-terminated C string.
        let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        match hypergrad.absorb_text(encoder, &text) {
            Ok(_) => 0,
            Err(err) => store_error(err),
        }
    })
}

#[no_mangle]
/// # Safety
/// - `encoder` must point to a valid `LanguageWaveEncoder` created by this library.
/// - `text` must point to a NUL-terminated C string.
/// - If `out` is non-null it must be valid for writes of `out_len` `f32` values.
pub unsafe extern "C" fn st_pure_encoder_encode_z_space(
    encoder: *const LanguageWaveEncoder,
    text: *const c_char,
    out: *mut f32,
    out_len: usize,
) -> usize {
    catch_unwind_or(0, "panic in st_pure_encoder_encode_z_space", || {
        if encoder.is_null() {
            store_error(TensorError::EmptyInput("encoder handle"));
            return 0;
        }
        if !require_aligned(encoder, "encoder handle") {
            return 0;
        }
        if text.is_null() {
            store_error(TensorError::EmptyInput("text pointer"));
            return 0;
        }
        // Safety: caller guarantees `encoder` points to a valid `LanguageWaveEncoder`.
        let encoder = unsafe { &*encoder };
        // Safety: caller guarantees `text` points to a NUL-terminated C string.
        let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        let tensor = match encoder.encode_z_space(&text) {
            Ok(tensor) => tensor,
            Err(err) => {
                store_error(err);
                return 0;
            }
        };
        let data = tensor.data();
        if out.is_null() {
            return data.len();
        }
        if !require_aligned(out, "out") {
            return 0;
        }
        if out_len < data.len() {
            store_error(data_length_mismatch(data.len(), out_len));
            return 0;
        }
        // Safety: caller guarantees `out` is valid for writes of `data.len()` f32 values.
        let dest = unsafe { slice::from_raw_parts_mut(out, data.len()) };
        dest.copy_from_slice(data);
        data.len()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn last_error_string() -> Option<String> {
        let ptr = st_pure_last_error();
        if ptr.is_null() {
            None
        } else {
            // Safety: `st_pure_last_error` returns a stable pointer to an internal `CString` while
            // the error storage remains unchanged.
            Some(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    fn last_error_string_lossy() -> String {
        match last_error_string() {
            Some(msg) => msg,
            None => "<no error>".to_string(),
        }
    }

    #[test]
    fn ffi_hypergrad_round_trip() {
        st_pure_clear_last_error();
        let curvature = -1.0;
        let encoder = st_pure_encoder_new(curvature, 0.5);
        if encoder.is_null() {
            panic!("failed to create encoder: {}", last_error_string_lossy());
        }
        let text = c"spiral torch pure";
        let required =
            unsafe { st_pure_encoder_encode_z_space(encoder, text.as_ptr(), ptr::null_mut(), 0) };
        assert!(required > 0);
        let hypergrad = st_pure_hypergrad_new(curvature, 0.1, 1, required);
        if hypergrad.is_null() {
            panic!("failed to create hypergrad: {}", last_error_string_lossy());
        }
        let status = unsafe { st_pure_hypergrad_absorb_text(hypergrad, encoder, text.as_ptr()) };
        assert_eq!(status, 0);

        let mut weights = vec![0.0f32; required];
        let apply_status =
            unsafe { st_pure_hypergrad_apply(hypergrad, weights.as_mut_ptr(), weights.len()) };
        assert_eq!(apply_status, 0);

        let mut gradient = vec![0.0f32; required];
        let copied =
            unsafe { st_pure_hypergrad_gradient(hypergrad, gradient.as_mut_ptr(), gradient.len()) };
        assert_eq!(copied, gradient.len());

        unsafe {
            st_pure_encoder_free(encoder);
            st_pure_hypergrad_free(hypergrad);
        }
    }

    #[test]
    fn ffi_encoder_len_probe() {
        st_pure_clear_last_error();
        let encoder = st_pure_encoder_new(-1.0, 0.3);
        if encoder.is_null() {
            panic!("failed to create encoder: {}", last_error_string_lossy());
        }
        let text = c"hyperbolic ffi";
        let required =
            unsafe { st_pure_encoder_encode_z_space(encoder, text.as_ptr(), ptr::null_mut(), 0) };
        assert!(required > 0);
        let mut buffer = vec![0.0f32; required];
        let copied = unsafe {
            st_pure_encoder_encode_z_space(
                encoder,
                text.as_ptr(),
                buffer.as_mut_ptr(),
                buffer.len(),
            )
        };
        assert_eq!(copied, required);
        unsafe { st_pure_encoder_free(encoder) };
    }

    #[test]
    fn ffi_apply_rejects_length_mismatch() {
        st_pure_clear_last_error();
        let hypergrad = st_pure_hypergrad_new(-1.0, 0.1, 1, 4);
        if hypergrad.is_null() {
            panic!("failed to create hypergrad: {}", last_error_string_lossy());
        }
        let mut weights = vec![0.0f32; 3];
        let status =
            unsafe { st_pure_hypergrad_apply(hypergrad, weights.as_mut_ptr(), weights.len()) };
        assert_eq!(status, -1);
        assert!(!st_pure_last_error().is_null());
        unsafe { st_pure_hypergrad_free(hypergrad) };
    }

    #[test]
    fn ffi_gradient_rejects_short_buffer() {
        st_pure_clear_last_error();
        let hypergrad = st_pure_hypergrad_new(-1.0, 0.1, 1, 4);
        if hypergrad.is_null() {
            panic!("failed to create hypergrad: {}", last_error_string_lossy());
        }
        let mut buffer = vec![0.0f32; 3];
        let written =
            unsafe { st_pure_hypergrad_gradient(hypergrad, buffer.as_mut_ptr(), buffer.len()) };
        assert_eq!(written, 0);
        assert!(!st_pure_last_error().is_null());
        unsafe { st_pure_hypergrad_free(hypergrad) };
    }

    #[test]
    fn ffi_last_error_is_thread_local() {
        st_pure_clear_last_error();
        let hypergrad = st_pure_hypergrad_new(1.0, 0.1, 1, 1);
        assert!(hypergrad.is_null());
        assert!(!st_pure_last_error().is_null());

        let handle = thread::spawn(|| {
            assert!(st_pure_last_error().is_null());
        });
        match handle.join() {
            Ok(()) => {}
            Err(_) => panic!("thread panicked"),
        }
    }

    #[test]
    fn ffi_rejects_misaligned_handles() {
        st_pure_clear_last_error();
        let mut storage = vec![0u8; mem::size_of::<AmegaHypergrad>() + 1];
        let misaligned = unsafe { storage.as_mut_ptr().add(1) as *const AmegaHypergrad };
        let mut rows = 0usize;
        let mut cols = 0usize;
        let status =
            unsafe { st_pure_hypergrad_shape(misaligned, &mut rows as *mut usize, &mut cols as *mut usize) };
        assert_eq!(status, -1);
        assert!(!st_pure_last_error().is_null());
    }

    #[test]
    fn ffi_rejects_misaligned_buffers() {
        st_pure_clear_last_error();
        let hypergrad = st_pure_hypergrad_new(-1.0, 0.1, 1, 4);
        if hypergrad.is_null() {
            panic!("failed to create hypergrad: {}", last_error_string_lossy());
        }
        let mut buffer = vec![0u8; 4 * mem::size_of::<f32>() + 1];
        let misaligned = unsafe { buffer.as_mut_ptr().add(1) as *mut f32 };
        let status = unsafe { st_pure_hypergrad_apply(hypergrad, misaligned, 4) };
        assert_eq!(status, -1);
        assert!(!st_pure_last_error().is_null());
        unsafe { st_pure_hypergrad_free(hypergrad) };
    }
}
