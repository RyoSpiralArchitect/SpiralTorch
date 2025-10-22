// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal C-ABI shims that surface a stable subset of SpiralTorch tensor
//! primitives. These functions are consumed by the Julia and Go bindings and
//! can be used by other foreign-language integrations that require a stable
//! binary interface.

use once_cell::sync::Lazy;
use st_core::ecosystem::foreign::{ForeignLanguage, ForeignRegistry, ForeignRuntimeDescriptor};
use st_tensor::{PureResult, Tensor};
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::slice;
use std::sync::Mutex;
use std::time::Duration;

type FfiResult<T> = Result<T, ()>;

static LAST_ERROR: Lazy<Mutex<Option<CString>>> = Lazy::new(|| Mutex::new(None));

fn set_last_error(message: impl Into<String>) {
    let mut slot = LAST_ERROR.lock().expect("last error mutex poisoned");
    let owned = message.into();
    *slot = Some(
        CString::new(owned.clone())
            .unwrap_or_else(|_| CString::new("<error message contained null byte>").unwrap()),
    );
}

fn clear_last_error() {
    let mut slot = LAST_ERROR.lock().expect("last error mutex poisoned");
    *slot = None;
}

fn ok<T>(value: T) -> FfiResult<T> {
    clear_last_error();
    Ok(value)
}

fn err<T>(message: impl Into<String>) -> FfiResult<T> {
    set_last_error(message);
    Err(())
}

fn tensor_from_result(result: PureResult<Tensor>) -> *mut Tensor {
    match result {
        Ok(tensor) => {
            clear_last_error();
            Box::into_raw(Box::new(tensor))
        }
        Err(err) => {
            set_last_error(err.to_string());
            ptr::null_mut()
        }
    }
}

fn tensor_from_result_with_message(result: PureResult<Tensor>, context: &str) -> *mut Tensor {
    match result {
        Ok(tensor) => {
            clear_last_error();
            Box::into_raw(Box::new(tensor))
        }
        Err(err) => {
            set_last_error(format!("{context}: {err}"));
            ptr::null_mut()
        }
    }
}

fn tensor_from_data(rows: usize, cols: usize, data: &[f32]) -> PureResult<Tensor> {
    Tensor::from_vec(rows, cols, data.to_vec())
}

fn require_non_null<T>(ptr: *const T, label: &str) -> FfiResult<*const T> {
    if ptr.is_null() {
        return err(format!("{label} pointer was null"));
    }
    ok(ptr)
}

fn require_non_null_mut<T>(ptr: *mut T, label: &str) -> FfiResult<*mut T> {
    if ptr.is_null() {
        return err(format!("{label} pointer was null"));
    }
    clear_last_error();
    Ok(ptr)
}

fn parse_c_string(ptr: *const c_char, label: &str) -> FfiResult<String> {
    if ptr.is_null() {
        return err(format!("{label} pointer was null"));
    }
    // SAFETY: pointer checked for null above.
    let raw = unsafe { CStr::from_ptr(ptr) };
    raw.to_str().map(|s| s.to_string()).map_err(|_| {
        set_last_error(format!("{label} contained invalid UTF-8"));
        ()
    })
}

fn parse_capabilities(ptr: *const c_char) -> FfiResult<Vec<String>> {
    if ptr.is_null() {
        return ok(Vec::new());
    }
    let raw = parse_c_string(ptr, "capabilities")?;
    let caps = raw
        .split(',')
        .map(|cap| cap.trim())
        .filter(|cap| !cap.is_empty())
        .map(|cap| cap.to_string())
        .collect();
    ok(caps)
}

/// Write the SpiralTorch semantic version into the provided buffer.
///
/// The function returns the number of bytes required to represent the string
/// (not counting the trailing null terminator). If the provided buffer has
/// enough capacity (`capacity >= len + 1`), the string is copied and a null
/// terminator is appended.
#[no_mangle]
pub extern "C" fn spiraltorch_version(buffer: *mut c_char, capacity: usize) -> usize {
    let version = env!("CARGO_PKG_VERSION");
    let bytes = version.as_bytes();
    if capacity > 0 && !buffer.is_null() {
        // Reserve space for the trailing null terminator.
        let max_copy = capacity.saturating_sub(1);
        let to_copy = bytes.len().min(max_copy);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buffer, to_copy);
            // Null terminate the buffer even if truncation occurred.
            *buffer.add(to_copy) = 0;
        }
    }
    bytes.len()
}

/// Returns the length of the last error message (in bytes, excluding the
/// trailing null terminator).
#[no_mangle]
pub extern "C" fn spiraltorch_last_error_length() -> usize {
    let slot = LAST_ERROR.lock().expect("last error mutex poisoned");
    slot.as_ref().map(|msg| msg.as_bytes().len()).unwrap_or(0)
}

/// Copies the last error message into the provided buffer and returns the
/// number of bytes copied (excluding the null terminator). If no error is
/// present the function returns `0` and the buffer is left untouched.
#[no_mangle]
pub extern "C" fn spiraltorch_last_error_message(buffer: *mut c_char, capacity: usize) -> usize {
    if buffer.is_null() || capacity == 0 {
        return 0;
    }
    let slot = LAST_ERROR.lock().expect("last error mutex poisoned");
    if let Some(message) = slot.as_ref() {
        let bytes = message.as_bytes();
        let max_copy = capacity.saturating_sub(1);
        let to_copy = bytes.len().min(max_copy);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buffer, to_copy);
            *buffer.add(to_copy) = 0;
        }
        to_copy
    } else {
        0
    }
}

/// Clears the last error so subsequent calls observe an empty state.
#[no_mangle]
pub extern "C" fn spiraltorch_clear_last_error() {
    clear_last_error();
}

/// Constructs a tensor filled with zeros. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_zeros(rows: usize, cols: usize) -> *mut Tensor {
    tensor_from_result(Tensor::zeros(rows, cols))
}

/// Constructs a tensor from a dense row-major buffer. Returns `NULL` on failure.
#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_from_dense(
    rows: usize,
    cols: usize,
    data: *const f32,
    len: usize,
) -> *mut Tensor {
    if data.is_null() {
        set_last_error("tensor_from_dense received null data pointer");
        return ptr::null_mut();
    }
    let required = rows.saturating_mul(cols);
    if required != len {
        set_last_error(format!(
            "tensor_from_dense expected {required} elements but received {len}"
        ));
        return ptr::null_mut();
    }
    let slice = slice::from_raw_parts(data, len);
    tensor_from_result_with_message(tensor_from_data(rows, cols, slice), "tensor_from_dense")
}

/// Releases a tensor previously allocated by this library.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_free(handle: *mut Tensor) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle));
    }
}

fn with_tensor<'a, T>(
    handle: *const Tensor,
    f: impl FnOnce(&'a Tensor) -> FfiResult<T>,
) -> FfiResult<T> {
    let handle = require_non_null(handle, "tensor handle")?;
    // SAFETY: we validated that the pointer is not null above.
    let tensor = unsafe { &*handle };
    f(tensor)
}

/// Retrieves the tensor shape and writes it into the provided output pointers.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_shape(
    handle: *const Tensor,
    rows_out: *mut usize,
    cols_out: *mut usize,
) -> bool {
    let result = with_tensor(handle, |tensor| {
        let rows_ptr = require_non_null_mut(rows_out, "rows_out")?;
        let cols_ptr = require_non_null_mut(cols_out, "cols_out")?;
        let (rows, cols) = tensor.shape();
        unsafe {
            *rows_ptr = rows;
            *cols_ptr = cols;
        }
        ok(())
    });
    result.is_ok()
}

/// Returns the number of elements stored in the tensor. On failure the
/// function returns `0` and records an error.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_elements(handle: *const Tensor) -> usize {
    match with_tensor(handle, |tensor| ok(tensor.data().len())) {
        Ok(len) => len,
        Err(_) => 0,
    }
}

/// Copies the tensor data into the provided buffer. Returns `true` on success.
#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_copy_data(
    handle: *const Tensor,
    out: *mut f32,
    len: usize,
) -> bool {
    let result = with_tensor(handle, |tensor| {
        let data = tensor.data();
        if data.len() != len {
            return err(format!(
                "tensor_copy_data expected {len} elements but tensor stores {}",
                data.len()
            ));
        }
        let out_ptr = require_non_null_mut(out, "out")?;
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), out_ptr, len);
        }
        ok(())
    });
    result.is_ok()
}

fn as_tensor<'a>(handle: *const Tensor, label: &str) -> Option<&'a Tensor> {
    let handle = match require_non_null(handle, label) {
        Ok(handle) => handle,
        Err(_) => return None,
    };
    // SAFETY: we validated the pointer is not null above.
    Some(unsafe { &*handle })
}

fn tensor_binary_op(
    lhs: *const Tensor,
    rhs: *const Tensor,
    context: &str,
    op: impl FnOnce(&Tensor, &Tensor) -> PureResult<Tensor>,
) -> *mut Tensor {
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    tensor_from_result_with_message(op(left, right), context)
}

fn tensor_unary_op(
    handle: *const Tensor,
    context: &str,
    op: impl FnOnce(&Tensor) -> PureResult<Tensor>,
) -> *mut Tensor {
    let tensor = match as_tensor(handle, "tensor handle") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    tensor_from_result_with_message(op(tensor), context)
}

/// Element-wise tensor addition. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_add(lhs: *const Tensor, rhs: *const Tensor) -> *mut Tensor {
    tensor_binary_op(lhs, rhs, "tensor_add", Tensor::add)
}

/// Element-wise tensor subtraction. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_sub(lhs: *const Tensor, rhs: *const Tensor) -> *mut Tensor {
    tensor_binary_op(lhs, rhs, "tensor_sub", Tensor::sub)
}

/// Returns a new tensor with every element scaled by `value`.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_scale(handle: *const Tensor, value: f32) -> *mut Tensor {
    tensor_unary_op(handle, "tensor_scale", |tensor| tensor.scale(value))
}

/// Returns a new tensor that is the transpose of the provided tensor.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_transpose(handle: *const Tensor) -> *mut Tensor {
    tensor_unary_op(handle, "tensor_transpose", |tensor| Ok(tensor.transpose()))
}

/// Returns a reshaped view of the tensor with the requested dimensions.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_reshape(
    handle: *const Tensor,
    rows: usize,
    cols: usize,
) -> *mut Tensor {
    let tensor = match as_tensor(handle, "tensor handle") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    tensor_from_result_with_message(tensor.reshape(rows, cols), "tensor_reshape")
}

/// Matrix multiplication (`lhs @ rhs`). Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_matmul(lhs: *const Tensor, rhs: *const Tensor) -> *mut Tensor {
    tensor_binary_op(lhs, rhs, "tensor_matmul", Tensor::matmul)
}

/// Element-wise tensor multiplication. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_hadamard(
    lhs: *const Tensor,
    rhs: *const Tensor,
) -> *mut Tensor {
    tensor_binary_op(lhs, rhs, "tensor_hadamard", Tensor::hadamard)
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_foreign_register(
    language: *const c_char,
    runtime_id: *const c_char,
    version: *const c_char,
    capabilities: *const c_char,
) -> bool {
    let language_str = match parse_c_string(language, "language") {
        Ok(value) => value,
        Err(_) => return false,
    };
    let runtime = match parse_c_string(runtime_id, "runtime_id") {
        Ok(value) => value,
        Err(_) => return false,
    };
    let version = match parse_c_string(version, "version") {
        Ok(value) => value,
        Err(_) => return false,
    };
    let capabilities = match parse_capabilities(capabilities) {
        Ok(value) => value,
        Err(_) => return false,
    };

    let language = ForeignLanguage::parse(&language_str);
    let descriptor =
        ForeignRuntimeDescriptor::new(runtime.clone(), language, version, capabilities);
    ForeignRegistry::global().register_runtime(descriptor);
    clear_last_error();
    true
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_foreign_heartbeat(runtime_id: *const c_char) -> bool {
    let runtime = match parse_c_string(runtime_id, "runtime_id") {
        Ok(value) => value,
        Err(_) => return false,
    };
    if ForeignRegistry::global().record_heartbeat(&runtime) {
        clear_last_error();
        true
    } else {
        set_last_error(format!("runtime '{runtime}' is not registered"));
        false
    }
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_foreign_record_latency(
    runtime_id: *const c_char,
    operation: *const c_char,
    latency_ns: u64,
    ok_flag: u8,
) -> bool {
    let runtime = match parse_c_string(runtime_id, "runtime_id") {
        Ok(value) => value,
        Err(_) => return false,
    };
    let operation = match parse_c_string(operation, "operation") {
        Ok(value) => value,
        Err(_) => return false,
    };
    let ok = ok_flag != 0;
    let duration = Duration::from_nanos(latency_ns);
    if ForeignRegistry::global().record_latency(&runtime, &operation, duration, ok) {
        clear_last_error();
        true
    } else {
        set_last_error(format!("runtime '{runtime}' is not registered"));
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libc::c_char;
    use std::ffi::{CStr, CString};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn version_roundtrip() {
        let len = spiraltorch_version(ptr::null_mut(), 0);
        assert!(len > 0);
        let mut buffer = vec![0 as c_char; len + 1];
        let written = spiraltorch_version(buffer.as_mut_ptr(), buffer.len());
        assert_eq!(written, len);
        let as_str = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str().unwrap();
        assert_eq!(as_str, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn tensor_lifecycle() {
        let rows = 2;
        let cols = 2;
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let handle =
            unsafe { spiraltorch_tensor_from_dense(rows, cols, data.as_ptr(), data.len()) };
        assert!(!handle.is_null(), "tensor handle should be valid");

        let mut rows_out = 0usize;
        let mut cols_out = 0usize;
        let ok = spiraltorch_tensor_shape(handle, &mut rows_out, &mut cols_out);
        assert!(ok);
        assert_eq!(rows_out, rows);
        assert_eq!(cols_out, cols);

        let elements = spiraltorch_tensor_elements(handle);
        assert_eq!(elements, data.len());

        let mut copy = vec![0.0_f32; elements];
        let copied = unsafe { spiraltorch_tensor_copy_data(handle, copy.as_mut_ptr(), copy.len()) };
        assert!(copied);
        assert_eq!(copy, data);

        spiraltorch_tensor_free(handle);
    }

    #[test]
    fn reports_errors() {
        unsafe {
            let handle = spiraltorch_tensor_from_dense(2, 3, ptr::null(), 0);
            assert!(handle.is_null());
        }
        let len = spiraltorch_last_error_length();
        assert!(len > 0);
        let mut buffer = vec![0i8; len + 1];
        let written = spiraltorch_last_error_message(buffer.as_mut_ptr(), buffer.len());
        assert_eq!(written, len.min(buffer.len() - 1));
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) };
        assert!(message.to_string_lossy().contains("null"));
    }

    #[test]
    fn tensor_add_and_sub() {
        let a_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0_f32, 6.0, 7.0, 8.0];
        let a = unsafe { spiraltorch_tensor_from_dense(2, 2, a_data.as_ptr(), a_data.len()) };
        let b = unsafe { spiraltorch_tensor_from_dense(2, 2, b_data.as_ptr(), b_data.len()) };
        assert!(!a.is_null());
        assert!(!b.is_null());

        let sum = spiraltorch_tensor_add(a, b);
        assert!(!sum.is_null());
        let mut buffer = vec![0.0_f32; 4];
        let ok = unsafe { spiraltorch_tensor_copy_data(sum, buffer.as_mut_ptr(), buffer.len()) };
        assert!(ok);
        assert_eq!(buffer, vec![6.0, 8.0, 10.0, 12.0]);

        let diff = spiraltorch_tensor_sub(sum, a);
        assert!(!diff.is_null());
        let mut diff_buffer = vec![0.0_f32; 4];
        let ok = unsafe {
            spiraltorch_tensor_copy_data(diff, diff_buffer.as_mut_ptr(), diff_buffer.len())
        };
        assert!(ok);
        assert_eq!(diff_buffer, b_data);

        spiraltorch_tensor_free(diff);
        spiraltorch_tensor_free(sum);
        spiraltorch_tensor_free(b);
        spiraltorch_tensor_free(a);
    }

    #[test]
    fn tensor_scale_and_matmul() {
        let left_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right_data = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let left =
            unsafe { spiraltorch_tensor_from_dense(2, 3, left_data.as_ptr(), left_data.len()) };
        let right =
            unsafe { spiraltorch_tensor_from_dense(3, 2, right_data.as_ptr(), right_data.len()) };
        assert!(!left.is_null());
        assert!(!right.is_null());

        let scaled = spiraltorch_tensor_scale(left, 0.5);
        assert!(!scaled.is_null());
        let mut buffer = vec![0.0_f32; 6];
        let ok = unsafe { spiraltorch_tensor_copy_data(scaled, buffer.as_mut_ptr(), buffer.len()) };
        assert!(ok);
        assert_eq!(buffer, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);

        let product = spiraltorch_tensor_matmul(left, right);
        assert!(!product.is_null());
        let mut product_buffer = vec![0.0_f32; 4];
        let ok = unsafe {
            spiraltorch_tensor_copy_data(product, product_buffer.as_mut_ptr(), product_buffer.len())
        };
        assert!(ok);
        assert_eq!(product_buffer, vec![58.0, 64.0, 139.0, 154.0]);

        spiraltorch_tensor_free(product);
        spiraltorch_tensor_free(scaled);
        spiraltorch_tensor_free(right);
        spiraltorch_tensor_free(left);
    }

    #[test]
    fn tensor_transpose_reshape_and_hadamard() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let handle = unsafe { spiraltorch_tensor_from_dense(2, 3, data.as_ptr(), data.len()) };
        assert!(!handle.is_null());

        let transposed = spiraltorch_tensor_transpose(handle);
        assert!(!transposed.is_null());
        let mut transposed_shape = (0usize, 0usize);
        let ok =
            spiraltorch_tensor_shape(transposed, &mut transposed_shape.0, &mut transposed_shape.1);
        assert!(ok);
        assert_eq!(transposed_shape, (3, 2));
        let mut buffer = vec![0.0_f32; 6];
        let copied =
            unsafe { spiraltorch_tensor_copy_data(transposed, buffer.as_mut_ptr(), buffer.len()) };
        assert!(copied);
        assert_eq!(buffer, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let reshaped = spiraltorch_tensor_reshape(handle, 3, 2);
        assert!(!reshaped.is_null());
        let mut reshaped_buffer = vec![0.0_f32; 6];
        let reshaped_ok = unsafe {
            spiraltorch_tensor_copy_data(
                reshaped,
                reshaped_buffer.as_mut_ptr(),
                reshaped_buffer.len(),
            )
        };
        assert!(reshaped_ok);
        assert_eq!(reshaped_buffer, data);

        let other_data = vec![1.0_f32, 0.5, 1.5, 2.0, 2.5, 3.0];
        let other =
            unsafe { spiraltorch_tensor_from_dense(2, 3, other_data.as_ptr(), other_data.len()) };
        assert!(!other.is_null());
        let hadamard = spiraltorch_tensor_hadamard(handle, other);
        assert!(!hadamard.is_null());
        let mut hadamard_buffer = vec![0.0_f32; 6];
        let hadamard_ok = unsafe {
            spiraltorch_tensor_copy_data(
                hadamard,
                hadamard_buffer.as_mut_ptr(),
                hadamard_buffer.len(),
            )
        };
        assert!(hadamard_ok);
        assert_eq!(hadamard_buffer, vec![1.0, 1.0, 4.5, 8.0, 12.5, 18.0]);

        spiraltorch_tensor_free(hadamard);
        spiraltorch_tensor_free(other);
        spiraltorch_tensor_free(reshaped);
        spiraltorch_tensor_free(transposed);
        spiraltorch_tensor_free(handle);
    }

    #[test]
    fn foreign_registry_roundtrip() {
        let language = CString::new("go").unwrap();
        let runtime_id = format!(
            "ffi-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let runtime = CString::new(runtime_id.clone()).unwrap();
        let version = CString::new("1.0.0").unwrap();
        let capabilities = CString::new("tensor.add").unwrap();
        let operation = CString::new("tensor.add").unwrap();
        unsafe {
            assert!(spiraltorch_foreign_register(
                language.as_ptr(),
                runtime.as_ptr(),
                version.as_ptr(),
                capabilities.as_ptr()
            ));
            assert!(spiraltorch_foreign_record_latency(
                runtime.as_ptr(),
                operation.as_ptr(),
                1_000_000,
                1
            ));
        }
        let snapshot = ForeignRegistry::global().snapshot();
        let status = snapshot
            .iter()
            .find(|entry| entry.descriptor.id == runtime_id)
            .expect("runtime should be registered");
        assert_eq!(status.descriptor.language, ForeignLanguage::Go);
        assert!(status
            .kernels
            .iter()
            .any(|kernel| kernel.operation == "tensor.add"));
    }
}
