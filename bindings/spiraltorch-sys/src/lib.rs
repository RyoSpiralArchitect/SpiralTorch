// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal C-ABI shims that surface a stable subset of SpiralTorch tensor
//! primitives. These functions are consumed by the Julia and Go bindings and
//! can be used by other foreign-language integrations that require a stable
//! binary interface.

use st_core::runtime::golden::{
    GoldenRuntime, GoldenRuntimeConfig, GoldenTaskError, GoldenTensorError,
};
use st_tensor::{
    dlpack::{self, DLManagedTensor},
    MatmulBackend, PureResult, SoftmaxBackend, Tensor,
};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::slice;

type FfiResult<T> = Result<T, ()>;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: impl Into<String>) {
    let owned = message.into();
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(
            CString::new(owned.clone())
                .unwrap_or_else(|_| CString::new("<error message contained null byte>").unwrap()),
        );
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
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

#[repr(C)]
pub struct RuntimeHandle(GoldenRuntime);

fn runtime_from_ptr<'a>(handle: *const RuntimeHandle, label: &str) -> Option<&'a GoldenRuntime> {
    let handle = match require_non_null(handle, label) {
        Ok(handle) => handle,
        Err(_) => return None,
    };
    // SAFETY: pointer validated above.
    Some(unsafe { &(*handle).0 })
}

#[repr(C)]
pub struct RoundtableSummary {
    pub above: usize,
    pub here: usize,
    pub beneath: usize,
    pub energy_above: f32,
    pub energy_here: f32,
    pub energy_beneath: f32,
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
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|msg| msg.as_bytes().len())
            .unwrap_or(0)
    })
}

/// Copies the last error message into the provided buffer and returns the
/// number of bytes copied (excluding the null terminator). If no error is
/// present the function returns `0` and the buffer is left untouched.
#[no_mangle]
pub extern "C" fn spiraltorch_last_error_message(buffer: *mut c_char, capacity: usize) -> usize {
    if buffer.is_null() || capacity == 0 {
        return 0;
    }
    LAST_ERROR.with(|slot| {
        if let Some(message) = slot.borrow().as_ref() {
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
    })
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

/// Copies a single row into the provided buffer. Returns `true` on success.
#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_copy_row(
    handle: *const Tensor,
    index: usize,
    out: *mut f32,
    len: usize,
) -> bool {
    let result = with_tensor(handle, |tensor| {
        let (rows, cols) = tensor.shape();
        if index >= rows {
            return err(format!(
                "tensor_copy_row index {index} out of range for {rows} rows"
            ));
        }
        if cols != len {
            return err(format!(
                "tensor_copy_row expected buffer length {cols} but received {len}"
            ));
        }
        if cols == 0 {
            return ok(());
        }
        let out_ptr = require_non_null_mut(out, "out")?;
        let data = tensor.data();
        let offset = index * cols;
        unsafe {
            ptr::copy_nonoverlapping(data[offset..].as_ptr(), out_ptr, cols);
        }
        ok(())
    });
    result.is_ok()
}

/// Copies a single column into the provided buffer. Returns `true` on success.
#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_copy_column(
    handle: *const Tensor,
    index: usize,
    out: *mut f32,
    len: usize,
) -> bool {
    let result = with_tensor(handle, |tensor| {
        let (rows, cols) = tensor.shape();
        if index >= cols {
            return err(format!(
                "tensor_copy_column index {index} out of range for {cols} columns"
            ));
        }
        if rows != len {
            return err(format!(
                "tensor_copy_column expected buffer length {rows} but received {len}"
            ));
        }
        if rows == 0 {
            return ok(());
        }
        let out_ptr = require_non_null_mut(out, "out")?;
        let data = tensor.data();
        // SAFETY: we validated that `out_ptr` is non-null and that `len == rows`.
        let out_slice = unsafe { slice::from_raw_parts_mut(out_ptr, rows) };
        for r in 0..rows {
            out_slice[r] = data[r * cols + index];
        }
        ok(())
    });
    result.is_ok()
}

/// Exports the tensor as a managed DLPack tensor. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_to_dlpack(handle: *const Tensor) -> *mut DLManagedTensor {
    let tensor = match as_tensor(handle, "tensor handle") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };

    match tensor.to_dlpack() {
        Ok(ptr) => {
            clear_last_error();
            ptr
        }
        Err(err) => {
            set_last_error(format!("tensor_to_dlpack: {err}"));
            ptr::null_mut()
        }
    }
}

/// Constructs a tensor from a managed DLPack tensor. Returns `NULL` on failure.
///
/// # Safety
/// The caller must ensure `managed` points to a valid `DLManagedTensor`.
#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_from_dlpack(
    managed: *mut DLManagedTensor,
) -> *mut Tensor {
    if managed.is_null() {
        set_last_error("tensor_from_dlpack received null managed tensor pointer");
        return ptr::null_mut();
    }

    match Tensor::from_dlpack(managed) {
        Ok(tensor) => {
            clear_last_error();
            Box::into_raw(Box::new(tensor))
        }
        Err(err) => {
            set_last_error(format!("tensor_from_dlpack: {err}"));
            ptr::null_mut()
        }
    }
}

/// Drops the exported state allocated when creating a DLPack tensor.
///
/// # Safety
/// The caller must ensure `managed` either originates from
/// `spiraltorch_tensor_to_dlpack` or has been obtained from another SpiralTorch
/// API that documents compatibility with this function.
#[no_mangle]
pub unsafe extern "C" fn spiraltorch_dlpack_drop_exported_state(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // SAFETY: Caller guarantees pointer validity per the function contract.
    unsafe {
        dlpack::drop_exported_state(managed);
    }
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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpiraltorchMatmulBackend {
    Auto = 0,
    CpuFaer = 1,
    CpuSimd = 2,
    CpuNaive = 3,
    GpuWgpu = 4,
    GpuHip = 5,
}

fn map_matmul_backend(backend: SpiraltorchMatmulBackend) -> Result<MatmulBackend, &'static str> {
    match backend {
        SpiraltorchMatmulBackend::Auto => Ok(MatmulBackend::Auto),
        SpiraltorchMatmulBackend::CpuFaer => Ok(MatmulBackend::CpuFaer),
        SpiraltorchMatmulBackend::CpuSimd => Ok(MatmulBackend::CpuSimd),
        SpiraltorchMatmulBackend::CpuNaive => Ok(MatmulBackend::CpuNaive),
        SpiraltorchMatmulBackend::GpuWgpu => {
            #[cfg(feature = "wgpu")]
            {
                Ok(MatmulBackend::GpuWgpu)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                Err("wgpu backend support is not compiled in")
            }
        }
        SpiraltorchMatmulBackend::GpuHip => {
            #[cfg(feature = "hip")]
            {
                Ok(MatmulBackend::GpuHip)
            }
            #[cfg(not(feature = "hip"))]
            {
                Err("hip backend support is not compiled in")
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpiraltorchSoftmaxBackend {
    Auto = 0,
    Cpu = 1,
    GpuWgpu = 2,
}

fn map_softmax_backend(backend: SpiraltorchSoftmaxBackend) -> Result<SoftmaxBackend, &'static str> {
    match backend {
        SpiraltorchSoftmaxBackend::Auto => Ok(SoftmaxBackend::Auto),
        SpiraltorchSoftmaxBackend::Cpu => Ok(SoftmaxBackend::Cpu),
        SpiraltorchSoftmaxBackend::GpuWgpu => {
            #[cfg(feature = "wgpu")]
            {
                Ok(SoftmaxBackend::GpuWgpu)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                Err("wgpu backend support is not compiled in")
            }
        }
    }
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

unsafe fn copy_bias_buffer(
    bias: *const f32,
    len: usize,
    expected: usize,
    context: &str,
) -> FfiResult<Vec<f32>> {
    if bias.is_null() {
        return err(format!("{context} received null bias pointer"));
    }
    if len != expected {
        return err(format!(
            "{context} expected bias with {expected} elements but received {len}"
        ));
    }
    let slice = slice::from_raw_parts(bias, len);
    ok(slice.to_vec())
}

fn runtime_spawn(
    runtime: &GoldenRuntime,
    context: &str,
    task: impl FnOnce() -> PureResult<Tensor> + Send + 'static,
) -> *mut Tensor {
    match runtime.spawn_blocking(task) {
        Ok(handle) => match handle.join() {
            Ok(tensor) => tensor_from_result_with_message(Ok(tensor), context),
            Err(GoldenTaskError::Task(err)) => tensor_from_result_with_message(Err(err), context),
            Err(GoldenTaskError::Runtime(err)) => {
                set_last_error(format!("{context}: {err}"));
                ptr::null_mut()
            }
            Err(GoldenTaskError::Panic) => {
                set_last_error(format!("{context} task panicked"));
                ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(format!("{context}: {err}"));
            ptr::null_mut()
        }
    }
}

fn runtime_tensor_generate(
    runtime: *const RuntimeHandle,
    context: &str,
    task: impl FnOnce(&GoldenRuntime) -> Result<Tensor, GoldenTensorError>,
) -> *mut Tensor {
    let runtime = match runtime_from_ptr(runtime, "runtime handle") {
        Some(runtime) => runtime,
        None => return ptr::null_mut(),
    };

    match task(runtime) {
        Ok(tensor) => {
            clear_last_error();
            Box::into_raw(Box::new(tensor))
        }
        Err(GoldenTensorError::Runtime(err)) => {
            set_last_error(format!("{context}: {err}"));
            ptr::null_mut()
        }
        Err(GoldenTensorError::Tensor(err)) => {
            set_last_error(format!("{context}: {err}"));
            ptr::null_mut()
        }
    }
}

fn runtime_tensor_binary_op<F>(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    context: &str,
    op: F,
) -> *mut Tensor
where
    F: FnOnce(&Tensor, &Tensor) -> PureResult<Tensor> + Send + 'static,
{
    let runtime = match runtime_from_ptr(runtime, "runtime handle") {
        Some(runtime) => runtime,
        None => return ptr::null_mut(),
    };
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    runtime_spawn(runtime, context, move || op(&left, &right))
}

fn runtime_tensor_unary_op<F>(
    runtime: *const RuntimeHandle,
    handle: *const Tensor,
    context: &str,
    op: F,
) -> *mut Tensor
where
    F: FnOnce(&Tensor) -> PureResult<Tensor> + Send + 'static,
{
    let runtime = match runtime_from_ptr(runtime, "runtime handle") {
        Some(runtime) => runtime,
        None => return ptr::null_mut(),
    };
    let tensor = match as_tensor(handle, "tensor handle") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    runtime_spawn(runtime, context, move || op(&tensor))
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

fn optional_seed(seed: u64, has_seed: bool) -> Option<u64> {
    if has_seed {
        Some(seed)
    } else {
        None
    }
}

/// Samples tensor elements from a uniform distribution in `[min, max)`.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_random_uniform(
    rows: usize,
    cols: usize,
    min: f32,
    max: f32,
    seed: u64,
    has_seed: bool,
) -> *mut Tensor {
    tensor_from_result_with_message(
        Tensor::random_uniform(rows, cols, min, max, optional_seed(seed, has_seed)),
        "tensor_random_uniform",
    )
}

/// Samples tensor elements from a normal distribution with the provided mean and standard deviation.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_random_normal(
    rows: usize,
    cols: usize,
    mean: f32,
    std: f32,
    seed: u64,
    has_seed: bool,
) -> *mut Tensor {
    tensor_from_result_with_message(
        Tensor::random_normal(rows, cols, mean, std, optional_seed(seed, has_seed)),
        "tensor_random_normal",
    )
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

/// Matrix multiplication with an explicit backend override. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_matmul_with_backend(
    lhs: *const Tensor,
    rhs: *const Tensor,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    tensor_binary_op(lhs, rhs, "tensor_matmul_with_backend", move |lhs, rhs| {
        lhs.matmul_with_backend(rhs, backend)
    })
}

unsafe fn tensor_matmul_bias_relu_core(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: MatmulBackend,
    context: &str,
) -> *mut Tensor {
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let (_, cols) = right.shape();
    let bias = match copy_bias_buffer(bias, bias_len, cols, context) {
        Ok(bias) => bias,
        Err(_) => return ptr::null_mut(),
    };
    tensor_from_result_with_message(
        left.matmul_bias_relu_with_backend(right, &bias, backend),
        context,
    )
}

unsafe fn tensor_matmul_bias_add_relu_core(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    residual: *const Tensor,
    backend: MatmulBackend,
    context: &str,
) -> *mut Tensor {
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let residual = match as_tensor(residual, "residual tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let (_, cols) = right.shape();
    let bias = match copy_bias_buffer(bias, bias_len, cols, context) {
        Ok(bias) => bias,
        Err(_) => return ptr::null_mut(),
    };
    tensor_from_result_with_message(
        left.matmul_bias_add_relu_with_backend(right, &bias, residual, backend),
        context,
    )
}

unsafe fn tensor_matmul_bias_gelu_core(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: MatmulBackend,
    context: &str,
) -> *mut Tensor {
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor,
        None => return ptr::null_mut(),
    };
    let (_, cols) = right.shape();
    let bias = match copy_bias_buffer(bias, bias_len, cols, context) {
        Ok(bias) => bias,
        Err(_) => return ptr::null_mut(),
    };
    tensor_from_result_with_message(
        left.matmul_bias_gelu_with_backend(right, &bias, backend),
        context,
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_matmul_bias_relu(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
) -> *mut Tensor {
    tensor_matmul_bias_relu_core(
        lhs,
        rhs,
        bias,
        bias_len,
        MatmulBackend::Auto,
        "tensor_matmul_bias_relu",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_matmul_bias_relu_with_backend(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    tensor_matmul_bias_relu_core(
        lhs,
        rhs,
        bias,
        bias_len,
        backend,
        "tensor_matmul_bias_relu_with_backend",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_matmul_bias_add_relu(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    residual: *const Tensor,
) -> *mut Tensor {
    tensor_matmul_bias_add_relu_core(
        lhs,
        rhs,
        bias,
        bias_len,
        residual,
        MatmulBackend::Auto,
        "tensor_matmul_bias_add_relu",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_matmul_bias_add_relu_with_backend(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    residual: *const Tensor,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    tensor_matmul_bias_add_relu_core(
        lhs,
        rhs,
        bias,
        bias_len,
        residual,
        backend,
        "tensor_matmul_bias_add_relu_with_backend",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_matmul_bias_gelu(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
) -> *mut Tensor {
    tensor_matmul_bias_gelu_core(
        lhs,
        rhs,
        bias,
        bias_len,
        MatmulBackend::Auto,
        "tensor_matmul_bias_gelu",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_tensor_matmul_bias_gelu_with_backend(
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    tensor_matmul_bias_gelu_core(
        lhs,
        rhs,
        bias,
        bias_len,
        backend,
        "tensor_matmul_bias_gelu_with_backend",
    )
}

/// Element-wise tensor multiplication. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_hadamard(
    lhs: *const Tensor,
    rhs: *const Tensor,
) -> *mut Tensor {
    tensor_binary_op(lhs, rhs, "tensor_hadamard", Tensor::hadamard)
}

/// Row-wise softmax with an optional backend override. Returns `NULL` on failure.
#[no_mangle]
pub extern "C" fn spiraltorch_tensor_row_softmax(
    handle: *const Tensor,
    backend: SpiraltorchSoftmaxBackend,
) -> *mut Tensor {
    let backend = match map_softmax_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    tensor_unary_op(handle, "tensor_row_softmax", move |tensor| {
        tensor.row_softmax_with_backend(backend)
    })
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_runtime_new(
    worker_threads: usize,
    thread_name: *const c_char,
) -> *mut RuntimeHandle {
    let mut config = GoldenRuntimeConfig::default();
    if worker_threads > 0 {
        config.worker_threads = worker_threads;
    }
    if !thread_name.is_null() {
        match CStr::from_ptr(thread_name).to_str() {
            Ok(name) if name.is_empty() => {
                config.thread_name = None;
            }
            Ok(name) => {
                config.thread_name = Some(name.to_string());
            }
            Err(err) => {
                set_last_error(format!(
                    "runtime_new received invalid thread name utf-8: {err}"
                ));
                return ptr::null_mut();
            }
        }
    }
    match GoldenRuntime::new(config) {
        Ok(runtime) => {
            clear_last_error();
            Box::into_raw(Box::new(RuntimeHandle(runtime)))
        }
        Err(err) => {
            set_last_error(err.to_string());
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_free(handle: *mut RuntimeHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle));
    }
    clear_last_error();
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_worker_count(handle: *const RuntimeHandle) -> usize {
    let runtime = match runtime_from_ptr(handle, "runtime handle") {
        Some(runtime) => runtime,
        None => return 0,
    };
    clear_last_error();
    runtime.worker_count()
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_add(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
) -> *mut Tensor {
    runtime_tensor_binary_op(runtime, lhs, rhs, "runtime_tensor_add", Tensor::add)
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_sub(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
) -> *mut Tensor {
    runtime_tensor_binary_op(runtime, lhs, rhs, "runtime_tensor_sub", Tensor::sub)
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_scale(
    runtime: *const RuntimeHandle,
    handle: *const Tensor,
    value: f32,
) -> *mut Tensor {
    runtime_tensor_unary_op(runtime, handle, "runtime_tensor_scale", move |tensor| {
        tensor.scale(value)
    })
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_transpose(
    runtime: *const RuntimeHandle,
    handle: *const Tensor,
) -> *mut Tensor {
    runtime_tensor_unary_op(runtime, handle, "runtime_tensor_transpose", |tensor| {
        Ok(tensor.transpose())
    })
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_reshape(
    runtime: *const RuntimeHandle,
    handle: *const Tensor,
    rows: usize,
    cols: usize,
) -> *mut Tensor {
    runtime_tensor_unary_op(runtime, handle, "runtime_tensor_reshape", move |tensor| {
        tensor.reshape(rows, cols)
    })
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_matmul(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
) -> *mut Tensor {
    runtime_tensor_binary_op(runtime, lhs, rhs, "runtime_tensor_matmul", Tensor::matmul)
}

unsafe fn runtime_tensor_matmul_bias_relu_core(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: MatmulBackend,
    context: &str,
) -> *mut Tensor {
    let runtime = match runtime_from_ptr(runtime, "runtime handle") {
        Some(runtime) => runtime,
        None => return ptr::null_mut(),
    };
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let (_, cols) = right.shape();
    let bias = match copy_bias_buffer(bias, bias_len, cols, context) {
        Ok(bias) => bias,
        Err(_) => return ptr::null_mut(),
    };
    runtime_spawn(runtime, context, move || {
        left.matmul_bias_relu_with_backend(&right, &bias, backend)
    })
}

unsafe fn runtime_tensor_matmul_bias_add_relu_core(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    residual: *const Tensor,
    backend: MatmulBackend,
    context: &str,
) -> *mut Tensor {
    let runtime = match runtime_from_ptr(runtime, "runtime handle") {
        Some(runtime) => runtime,
        None => return ptr::null_mut(),
    };
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let residual_tensor = match as_tensor(residual, "residual tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let (_, cols) = right.shape();
    let bias = match copy_bias_buffer(bias, bias_len, cols, context) {
        Ok(bias) => bias,
        Err(_) => return ptr::null_mut(),
    };
    runtime_spawn(runtime, context, move || {
        left.matmul_bias_add_relu_with_backend(&right, &bias, &residual_tensor, backend)
    })
}

unsafe fn runtime_tensor_matmul_bias_gelu_core(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: MatmulBackend,
    context: &str,
) -> *mut Tensor {
    let runtime = match runtime_from_ptr(runtime, "runtime handle") {
        Some(runtime) => runtime,
        None => return ptr::null_mut(),
    };
    let left = match as_tensor(lhs, "lhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let right = match as_tensor(rhs, "rhs tensor") {
        Some(tensor) => tensor.clone(),
        None => return ptr::null_mut(),
    };
    let (_, cols) = right.shape();
    let bias = match copy_bias_buffer(bias, bias_len, cols, context) {
        Ok(bias) => bias,
        Err(_) => return ptr::null_mut(),
    };
    runtime_spawn(runtime, context, move || {
        left.matmul_bias_gelu_with_backend(&right, &bias, backend)
    })
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_matmul_with_backend(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    runtime_tensor_binary_op(
        runtime,
        lhs,
        rhs,
        "runtime_tensor_matmul_with_backend",
        move |lhs, rhs| lhs.matmul_with_backend(rhs, backend),
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_runtime_tensor_matmul_bias_relu(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
) -> *mut Tensor {
    runtime_tensor_matmul_bias_relu_core(
        runtime,
        lhs,
        rhs,
        bias,
        bias_len,
        MatmulBackend::Auto,
        "runtime_tensor_matmul_bias_relu",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_runtime_tensor_matmul_bias_relu_with_backend(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    runtime_tensor_matmul_bias_relu_core(
        runtime,
        lhs,
        rhs,
        bias,
        bias_len,
        backend,
        "runtime_tensor_matmul_bias_relu_with_backend",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_runtime_tensor_matmul_bias_add_relu(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    residual: *const Tensor,
) -> *mut Tensor {
    runtime_tensor_matmul_bias_add_relu_core(
        runtime,
        lhs,
        rhs,
        bias,
        bias_len,
        residual,
        MatmulBackend::Auto,
        "runtime_tensor_matmul_bias_add_relu",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_runtime_tensor_matmul_bias_add_relu_with_backend(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    residual: *const Tensor,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    runtime_tensor_matmul_bias_add_relu_core(
        runtime,
        lhs,
        rhs,
        bias,
        bias_len,
        residual,
        backend,
        "runtime_tensor_matmul_bias_add_relu_with_backend",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_runtime_tensor_matmul_bias_gelu(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
) -> *mut Tensor {
    runtime_tensor_matmul_bias_gelu_core(
        runtime,
        lhs,
        rhs,
        bias,
        bias_len,
        MatmulBackend::Auto,
        "runtime_tensor_matmul_bias_gelu",
    )
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_runtime_tensor_matmul_bias_gelu_with_backend(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
    bias: *const f32,
    bias_len: usize,
    backend: SpiraltorchMatmulBackend,
) -> *mut Tensor {
    let backend = match map_matmul_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    runtime_tensor_matmul_bias_gelu_core(
        runtime,
        lhs,
        rhs,
        bias,
        bias_len,
        backend,
        "runtime_tensor_matmul_bias_gelu_with_backend",
    )
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_hadamard(
    runtime: *const RuntimeHandle,
    lhs: *const Tensor,
    rhs: *const Tensor,
) -> *mut Tensor {
    runtime_tensor_binary_op(
        runtime,
        lhs,
        rhs,
        "runtime_tensor_hadamard",
        Tensor::hadamard,
    )
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_random_uniform(
    runtime: *const RuntimeHandle,
    rows: usize,
    cols: usize,
    min: f32,
    max: f32,
    seed: u64,
    has_seed: bool,
) -> *mut Tensor {
    runtime_tensor_generate(runtime, "runtime_tensor_random_uniform", move |runtime| {
        runtime.tensor_random_uniform(rows, cols, min, max, optional_seed(seed, has_seed))
    })
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_random_normal(
    runtime: *const RuntimeHandle,
    rows: usize,
    cols: usize,
    mean: f32,
    std: f32,
    seed: u64,
    has_seed: bool,
) -> *mut Tensor {
    runtime_tensor_generate(runtime, "runtime_tensor_random_normal", move |runtime| {
        runtime.tensor_random_normal(rows, cols, mean, std, optional_seed(seed, has_seed))
    })
}

#[no_mangle]
pub unsafe extern "C" fn spiraltorch_roundtable_classify(
    gradient: *const f32,
    len: usize,
    above_k: usize,
    here_k: usize,
    beneath_k: usize,
    tolerance: f32,
    assignments: *mut u8,
    summary: *mut RoundtableSummary,
) -> bool {
    if gradient.is_null() {
        set_last_error("roundtable_classify received null gradient pointer");
        return false;
    }
    if len == 0 {
        set_last_error("roundtable_classify requires a non-empty gradient");
        return false;
    }

    let gradient = slice::from_raw_parts(gradient, len);
    let assignments_ptr = match require_non_null_mut(assignments, "roundtable_classify assignments")
    {
        Ok(ptr) => ptr,
        Err(_) => return false,
    };
    let summary_ptr = match require_non_null_mut(summary, "roundtable_classify summary") {
        Ok(ptr) => ptr,
        Err(_) => return false,
    };

    let mut indexed: Vec<(usize, f32)> = gradient
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, value.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut bands = vec![1u8; len];

    let top = usize::min(above_k, len);
    for &(idx, _) in indexed.iter().take(top) {
        bands[idx] = 0;
    }

    let bottom = usize::min(beneath_k, len.saturating_sub(top));
    for &(idx, _) in indexed.iter().rev().take(bottom) {
        bands[idx] = 2;
    }

    if here_k > 0 {
        let leftover = len.saturating_sub(top + bottom);
        let here_target = usize::min(here_k, leftover);
        if here_target > 0 {
            let mid_end = len.saturating_sub(bottom);
            let mut assigned = 0usize;
            for &(idx, _) in indexed.iter().skip(top).take(mid_end.saturating_sub(top)) {
                if bands[idx] == 1 {
                    assigned += 1;
                    if assigned >= here_target {
                        break;
                    }
                }
            }
        }
    }

    let tolerance = tolerance.max(0.0);
    for &(idx, magnitude) in &indexed {
        if magnitude <= tolerance {
            bands[idx] = 1;
        }
    }

    unsafe {
        ptr::copy_nonoverlapping(bands.as_ptr(), assignments_ptr, len);
    }

    let mut counts = [0usize; 3];
    let mut energy = [0f32; 3];
    for (idx, value) in gradient.iter().enumerate() {
        let band = bands[idx] as usize;
        if band >= counts.len() {
            continue;
        }
        counts[band] += 1;
        let mag = value.abs();
        let contribution = if mag.is_finite() { mag } else { 0.0 };
        energy[band] += contribution;
    }

    unsafe {
        *summary_ptr = RoundtableSummary {
            above: counts[0],
            here: counts[1],
            beneath: counts[2],
            energy_above: energy[0],
            energy_here: energy[1],
            energy_beneath: energy[2],
        };
    }

    clear_last_error();
    true
}

#[no_mangle]
pub extern "C" fn spiraltorch_runtime_tensor_row_softmax(
    runtime: *const RuntimeHandle,
    handle: *const Tensor,
    backend: SpiraltorchSoftmaxBackend,
) -> *mut Tensor {
    let backend = match map_softmax_backend(backend) {
        Ok(backend) => backend,
        Err(message) => {
            set_last_error(message);
            return ptr::null_mut();
        }
    };
    runtime_tensor_unary_op(
        runtime,
        handle,
        "runtime_tensor_row_softmax",
        move |tensor| tensor.row_softmax_with_backend(backend),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use libc::c_char;
    use std::ffi::{CStr, CString};

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
    fn tensor_random_uniform_respects_bounds_and_seed() {
        let first = spiraltorch_tensor_random_uniform(4, 3, -0.5, 0.75, 7, true);
        let second = spiraltorch_tensor_random_uniform(4, 3, -0.5, 0.75, 7, true);
        assert!(!first.is_null());
        assert!(!second.is_null());

        let mut first_buffer = vec![0.0_f32; 12];
        let mut second_buffer = vec![0.0_f32; 12];
        let first_ok = unsafe {
            spiraltorch_tensor_copy_data(first, first_buffer.as_mut_ptr(), first_buffer.len())
        };
        let second_ok = unsafe {
            spiraltorch_tensor_copy_data(second, second_buffer.as_mut_ptr(), second_buffer.len())
        };
        assert!(first_ok);
        assert!(second_ok);
        assert_eq!(first_buffer, second_buffer);
        assert!(first_buffer
            .iter()
            .all(|value| (-0.5..0.75).contains(value)));

        spiraltorch_tensor_free(second);
        spiraltorch_tensor_free(first);
    }

    #[test]
    fn tensor_random_normal_respects_seed() {
        let first = spiraltorch_tensor_random_normal(2, 5, 0.0, 1.5, 99, true);
        let second = spiraltorch_tensor_random_normal(2, 5, 0.0, 1.5, 99, true);
        assert!(!first.is_null());
        assert!(!second.is_null());

        let mut first_buffer = vec![0.0_f32; 10];
        let mut second_buffer = vec![0.0_f32; 10];
        let first_ok = unsafe {
            spiraltorch_tensor_copy_data(first, first_buffer.as_mut_ptr(), first_buffer.len())
        };
        let second_ok = unsafe {
            spiraltorch_tensor_copy_data(second, second_buffer.as_mut_ptr(), second_buffer.len())
        };
        assert!(first_ok);
        assert!(second_ok);
        assert_eq!(first_buffer, second_buffer);

        spiraltorch_tensor_free(second);
        spiraltorch_tensor_free(first);
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
    fn runtime_accelerates_matmul() {
        let thread_name = CString::new("ffi-golden").expect("thread name");
        let runtime = unsafe { spiraltorch_runtime_new(2, thread_name.as_ptr()) };
        assert!(!runtime.is_null());

        let workers = spiraltorch_runtime_worker_count(runtime);
        assert!(workers >= 1);

        let left_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let right_data = vec![5.0_f32, 6.0, 7.0, 8.0];
        let left =
            unsafe { spiraltorch_tensor_from_dense(2, 2, left_data.as_ptr(), left_data.len()) };
        let right =
            unsafe { spiraltorch_tensor_from_dense(2, 2, right_data.as_ptr(), right_data.len()) };
        assert!(!left.is_null());
        assert!(!right.is_null());

        let product = spiraltorch_runtime_tensor_matmul(runtime, left, right);
        assert!(!product.is_null());

        let mut buffer = vec![0.0_f32; 4];
        let copied =
            unsafe { spiraltorch_tensor_copy_data(product, buffer.as_mut_ptr(), buffer.len()) };
        assert!(copied);
        assert_eq!(buffer, vec![19.0, 22.0, 43.0, 50.0]);

        spiraltorch_tensor_free(product);
        spiraltorch_tensor_free(right);
        spiraltorch_tensor_free(left);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn runtime_random_uniform_matches_direct_seeded_output() {
        let runtime = unsafe { spiraltorch_runtime_new(0, ptr::null()) };
        assert!(!runtime.is_null());

        let direct = spiraltorch_tensor_random_uniform(3, 4, -1.0, 1.0, 17, true);
        let spawned = spiraltorch_runtime_tensor_random_uniform(runtime, 3, 4, -1.0, 1.0, 17, true);

        assert!(!direct.is_null());
        assert!(!spawned.is_null());

        let mut direct_buf = vec![0.0_f32; 12];
        let mut spawned_buf = vec![0.0_f32; 12];
        let direct_ok = unsafe {
            spiraltorch_tensor_copy_data(direct, direct_buf.as_mut_ptr(), direct_buf.len())
        };
        let spawned_ok = unsafe {
            spiraltorch_tensor_copy_data(spawned, spawned_buf.as_mut_ptr(), spawned_buf.len())
        };
        assert!(direct_ok);
        assert!(spawned_ok);
        assert_eq!(direct_buf, spawned_buf);

        spiraltorch_tensor_free(spawned);
        spiraltorch_tensor_free(direct);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn runtime_random_normal_matches_direct_seeded_output() {
        let runtime = unsafe { spiraltorch_runtime_new(1, ptr::null()) };
        assert!(!runtime.is_null());

        let direct = spiraltorch_tensor_random_normal(2, 5, 0.0, 0.5, 99, true);
        let spawned = spiraltorch_runtime_tensor_random_normal(runtime, 2, 5, 0.0, 0.5, 99, true);

        assert!(!direct.is_null());
        assert!(!spawned.is_null());

        let mut direct_buf = vec![0.0_f32; 10];
        let mut spawned_buf = vec![0.0_f32; 10];
        let direct_ok = unsafe {
            spiraltorch_tensor_copy_data(direct, direct_buf.as_mut_ptr(), direct_buf.len())
        };
        let spawned_ok = unsafe {
            spiraltorch_tensor_copy_data(spawned, spawned_buf.as_mut_ptr(), spawned_buf.len())
        };
        assert!(direct_ok);
        assert!(spawned_ok);
        assert_eq!(direct_buf, spawned_buf);

        spiraltorch_tensor_free(spawned);
        spiraltorch_tensor_free(direct);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn tensor_matmul_with_backend_matches_auto() {
        let lhs_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let rhs_data = vec![5.0_f32, 6.0, 7.0, 8.0];
        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let auto = spiraltorch_tensor_matmul(lhs, rhs);
        assert!(!auto.is_null());
        let explicit =
            spiraltorch_tensor_matmul_with_backend(lhs, rhs, SpiraltorchMatmulBackend::CpuNaive);
        assert!(!explicit.is_null());

        let mut auto_buffer = vec![0.0_f32; 4];
        let mut explicit_buffer = vec![0.0_f32; 4];
        let auto_ok = unsafe {
            spiraltorch_tensor_copy_data(auto, auto_buffer.as_mut_ptr(), auto_buffer.len())
        };
        let explicit_ok = unsafe {
            spiraltorch_tensor_copy_data(
                explicit,
                explicit_buffer.as_mut_ptr(),
                explicit_buffer.len(),
            )
        };
        assert!(auto_ok);
        assert!(explicit_ok);
        assert_eq!(auto_buffer, explicit_buffer);

        spiraltorch_tensor_free(explicit);
        spiraltorch_tensor_free(auto);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
    }

    #[test]
    fn tensor_matmul_bias_relu_matches_reference() {
        let lhs_data = vec![1.0_f32, 2.0, -1.0, 0.5, 3.0, -2.0];
        let rhs_data = vec![2.0_f32, -1.0, 0.0, 4.0, -3.0, 1.0, 1.5, 0.5, -2.5];
        let bias = vec![0.25_f32, -0.5, 1.0];

        let lhs_handle =
            unsafe { spiraltorch_tensor_from_dense(2, 3, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs_handle =
            unsafe { spiraltorch_tensor_from_dense(3, 3, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs_handle.is_null());
        assert!(!rhs_handle.is_null());

        let lhs_tensor = Tensor::from_vec(2, 3, lhs_data.clone()).unwrap();
        let rhs_tensor = Tensor::from_vec(3, 3, rhs_data.clone()).unwrap();
        let expected = lhs_tensor.matmul_bias_relu(&rhs_tensor, &bias).unwrap();

        let fused = unsafe {
            spiraltorch_tensor_matmul_bias_relu(lhs_handle, rhs_handle, bias.as_ptr(), bias.len())
        };
        assert!(!fused.is_null());

        let mut buffer = vec![0.0_f32; expected.data().len()];
        let copied =
            unsafe { spiraltorch_tensor_copy_data(fused, buffer.as_mut_ptr(), buffer.len()) };
        assert!(copied);
        assert_eq!(buffer, expected.data());

        spiraltorch_tensor_free(fused);
        spiraltorch_tensor_free(rhs_handle);
        spiraltorch_tensor_free(lhs_handle);
    }

    #[test]
    fn tensor_matmul_bias_relu_with_backend_matches_auto() {
        let lhs_data = vec![0.5_f32, -1.0, 2.0, 3.0];
        let rhs_data = vec![1.0_f32, 0.0, -2.0, 4.0];
        let bias = vec![0.1_f32, -0.2];

        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let auto =
            unsafe { spiraltorch_tensor_matmul_bias_relu(lhs, rhs, bias.as_ptr(), bias.len()) };
        let explicit = unsafe {
            spiraltorch_tensor_matmul_bias_relu_with_backend(
                lhs,
                rhs,
                bias.as_ptr(),
                bias.len(),
                SpiraltorchMatmulBackend::CpuNaive,
            )
        };
        assert!(!auto.is_null());
        assert!(!explicit.is_null());

        let mut auto_buffer = vec![0.0_f32; 4];
        let mut explicit_buffer = vec![0.0_f32; 4];
        let auto_ok = unsafe {
            spiraltorch_tensor_copy_data(auto, auto_buffer.as_mut_ptr(), auto_buffer.len())
        };
        let explicit_ok = unsafe {
            spiraltorch_tensor_copy_data(
                explicit,
                explicit_buffer.as_mut_ptr(),
                explicit_buffer.len(),
            )
        };
        assert!(auto_ok);
        assert!(explicit_ok);
        assert_eq!(auto_buffer, explicit_buffer);

        spiraltorch_tensor_free(explicit);
        spiraltorch_tensor_free(auto);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
    }

    #[test]
    fn tensor_matmul_bias_add_relu_matches_reference() {
        let lhs_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let rhs_data = vec![0.5_f32, -1.0, 1.5, 2.0];
        let bias = vec![0.5_f32, -0.25];
        let residual_data = vec![0.2_f32, -0.1, 0.4, 0.3];

        let lhs_handle =
            unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs_handle =
            unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        let residual_handle = unsafe {
            spiraltorch_tensor_from_dense(2, 2, residual_data.as_ptr(), residual_data.len())
        };
        assert!(!lhs_handle.is_null());
        assert!(!rhs_handle.is_null());
        assert!(!residual_handle.is_null());

        let lhs_tensor = Tensor::from_vec(2, 2, lhs_data.clone()).unwrap();
        let rhs_tensor = Tensor::from_vec(2, 2, rhs_data.clone()).unwrap();
        let residual_tensor = Tensor::from_vec(2, 2, residual_data.clone()).unwrap();
        let expected = lhs_tensor
            .matmul_bias_add_relu(&rhs_tensor, &bias, &residual_tensor)
            .unwrap();

        let fused = unsafe {
            spiraltorch_tensor_matmul_bias_add_relu(
                lhs_handle,
                rhs_handle,
                bias.as_ptr(),
                bias.len(),
                residual_handle,
            )
        };
        assert!(!fused.is_null());

        let mut buffer = vec![0.0_f32; expected.data().len()];
        let copied =
            unsafe { spiraltorch_tensor_copy_data(fused, buffer.as_mut_ptr(), buffer.len()) };
        assert!(copied);
        assert_eq!(buffer, expected.data());

        spiraltorch_tensor_free(fused);
        spiraltorch_tensor_free(residual_handle);
        spiraltorch_tensor_free(rhs_handle);
        spiraltorch_tensor_free(lhs_handle);
    }

    #[test]
    fn tensor_matmul_bias_add_relu_with_backend_matches_auto() {
        let lhs_data = vec![2.0_f32, -1.0, 0.0, 1.5];
        let rhs_data = vec![1.0_f32, 3.0, -0.5, 2.0];
        let bias = vec![0.3_f32, 0.7];
        let residual_data = vec![0.1_f32, 0.2, 0.3, 0.4];

        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        let residual = unsafe {
            spiraltorch_tensor_from_dense(2, 2, residual_data.as_ptr(), residual_data.len())
        };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());
        assert!(!residual.is_null());

        let auto = unsafe {
            spiraltorch_tensor_matmul_bias_add_relu(lhs, rhs, bias.as_ptr(), bias.len(), residual)
        };
        let explicit = unsafe {
            spiraltorch_tensor_matmul_bias_add_relu_with_backend(
                lhs,
                rhs,
                bias.as_ptr(),
                bias.len(),
                residual,
                SpiraltorchMatmulBackend::CpuNaive,
            )
        };
        assert!(!auto.is_null());
        assert!(!explicit.is_null());

        let mut auto_buffer = vec![0.0_f32; 4];
        let mut explicit_buffer = vec![0.0_f32; 4];
        let auto_ok = unsafe {
            spiraltorch_tensor_copy_data(auto, auto_buffer.as_mut_ptr(), auto_buffer.len())
        };
        let explicit_ok = unsafe {
            spiraltorch_tensor_copy_data(
                explicit,
                explicit_buffer.as_mut_ptr(),
                explicit_buffer.len(),
            )
        };
        assert!(auto_ok);
        assert!(explicit_ok);
        assert_eq!(auto_buffer, explicit_buffer);

        spiraltorch_tensor_free(explicit);
        spiraltorch_tensor_free(auto);
        spiraltorch_tensor_free(residual);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
    }

    #[test]
    fn tensor_matmul_bias_gelu_matches_reference() {
        let lhs_data = vec![1.0_f32, -0.5, 2.0, 0.25, -1.5, 3.0];
        let rhs_data = vec![0.75_f32, -1.25, 1.5, 0.5, -0.75, 2.25, 1.0, -0.5, 0.25];
        let bias = vec![0.1_f32, -0.2, 0.3];

        let lhs_handle =
            unsafe { spiraltorch_tensor_from_dense(2, 3, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs_handle =
            unsafe { spiraltorch_tensor_from_dense(3, 3, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs_handle.is_null());
        assert!(!rhs_handle.is_null());

        let lhs_tensor = Tensor::from_vec(2, 3, lhs_data.clone()).unwrap();
        let rhs_tensor = Tensor::from_vec(3, 3, rhs_data.clone()).unwrap();
        let expected = lhs_tensor.matmul_bias_gelu(&rhs_tensor, &bias).unwrap();

        let fused = unsafe {
            spiraltorch_tensor_matmul_bias_gelu(lhs_handle, rhs_handle, bias.as_ptr(), bias.len())
        };
        assert!(!fused.is_null());

        let mut buffer = vec![0.0_f32; expected.data().len()];
        let copied =
            unsafe { spiraltorch_tensor_copy_data(fused, buffer.as_mut_ptr(), buffer.len()) };
        assert!(copied);
        assert_eq!(buffer, expected.data());

        spiraltorch_tensor_free(fused);
        spiraltorch_tensor_free(rhs_handle);
        spiraltorch_tensor_free(lhs_handle);
    }

    #[test]
    fn tensor_matmul_bias_gelu_with_backend_matches_auto() {
        let lhs_data = vec![0.25_f32, -1.0, 1.5, 2.0];
        let rhs_data = vec![1.0_f32, 0.75, -2.0, 0.5];
        let bias = vec![0.05_f32, -0.15];

        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let auto =
            unsafe { spiraltorch_tensor_matmul_bias_gelu(lhs, rhs, bias.as_ptr(), bias.len()) };
        let explicit = unsafe {
            spiraltorch_tensor_matmul_bias_gelu_with_backend(
                lhs,
                rhs,
                bias.as_ptr(),
                bias.len(),
                SpiraltorchMatmulBackend::CpuNaive,
            )
        };
        assert!(!auto.is_null());
        assert!(!explicit.is_null());

        let mut auto_buffer = vec![0.0_f32; 4];
        let mut explicit_buffer = vec![0.0_f32; 4];
        let auto_ok = unsafe {
            spiraltorch_tensor_copy_data(auto, auto_buffer.as_mut_ptr(), auto_buffer.len())
        };
        let explicit_ok = unsafe {
            spiraltorch_tensor_copy_data(
                explicit,
                explicit_buffer.as_mut_ptr(),
                explicit_buffer.len(),
            )
        };
        assert!(auto_ok);
        assert!(explicit_ok);
        assert_eq!(auto_buffer, explicit_buffer);

        spiraltorch_tensor_free(explicit);
        spiraltorch_tensor_free(auto);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
    }

    #[test]
    fn tensor_matmul_bias_relu_validates_bias_length() {
        let lhs_data = vec![1.0_f32, 0.0, 0.0, 1.0];
        let rhs_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bias = vec![0.5_f32];

        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let fused =
            unsafe { spiraltorch_tensor_matmul_bias_relu(lhs, rhs, bias.as_ptr(), bias.len()) };
        assert!(fused.is_null());

        let len = spiraltorch_last_error_length();
        assert!(len > 0);
        let mut buffer = vec![0i8; len + 1];
        let written = spiraltorch_last_error_message(buffer.as_mut_ptr(), buffer.len());
        assert!(written > 0);
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
        assert!(message.contains("expected bias"));

        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
    }

    #[test]
    fn tensor_matmul_bias_gelu_validates_bias_length() {
        let lhs_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let rhs_data = vec![0.5_f32, -1.5, 2.0, 1.0];
        let bias = vec![0.1_f32];

        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let fused =
            unsafe { spiraltorch_tensor_matmul_bias_gelu(lhs, rhs, bias.as_ptr(), bias.len()) };
        assert!(fused.is_null());

        let len = spiraltorch_last_error_length();
        assert!(len > 0);
        let mut buffer = vec![0i8; len + 1];
        let written = spiraltorch_last_error_message(buffer.as_mut_ptr(), buffer.len());
        assert!(written > 0);
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
        assert!(message.contains("expected bias"));

        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
    }

    #[test]
    fn tensor_row_softmax_matches_manual_computation() {
        let data = vec![0.0_f32, 1.0, 2.0, 3.0];
        let tensor = unsafe { spiraltorch_tensor_from_dense(2, 2, data.as_ptr(), data.len()) };
        assert!(!tensor.is_null());

        let softmax = spiraltorch_tensor_row_softmax(tensor, SpiraltorchSoftmaxBackend::Cpu);
        assert!(!softmax.is_null());

        let mut buffer = vec![0.0_f32; 4];
        let copied =
            unsafe { spiraltorch_tensor_copy_data(softmax, buffer.as_mut_ptr(), buffer.len()) };
        assert!(copied);

        let mut expected = Vec::with_capacity(4);
        for row in 0..2 {
            let start = row * 2;
            let slice = &data[start..start + 2];
            let exps: Vec<f32> = slice.iter().map(|v| v.exp()).collect();
            let sum: f32 = exps.iter().sum();
            expected.extend(exps.iter().map(|value| value / sum));
        }

        for (observed, expected) in buffer.iter().zip(expected.iter()) {
            assert!((observed - expected).abs() < 1e-5);
        }

        spiraltorch_tensor_free(softmax);
        spiraltorch_tensor_free(tensor);
    }

    #[test]
    fn tensor_dlpack_roundtrip_preserves_data() {
        let data = vec![1.5_f32, -2.25, 3.75, 4.5];
        let tensor = unsafe { spiraltorch_tensor_from_dense(2, 2, data.as_ptr(), data.len()) };
        assert!(!tensor.is_null());

        let dlpack = spiraltorch_tensor_to_dlpack(tensor);
        assert!(!dlpack.is_null());

        spiraltorch_tensor_free(tensor);

        let roundtrip = unsafe { spiraltorch_tensor_from_dlpack(dlpack) };
        assert!(!roundtrip.is_null());

        let mut buffer = vec![0.0_f32; data.len()];
        let copied =
            unsafe { spiraltorch_tensor_copy_data(roundtrip, buffer.as_mut_ptr(), buffer.len()) };
        assert!(copied);
        assert_eq!(buffer, data);

        spiraltorch_tensor_free(roundtrip);
    }

    #[test]
    fn runtime_matmul_with_backend_matches_auto() {
        let runtime = unsafe { spiraltorch_runtime_new(0, ptr::null()) };
        assert!(!runtime.is_null());

        let lhs_data = vec![1.0_f32, 0.0, 0.0, 1.0];
        let rhs_data = vec![2.0_f32, 3.0, 4.0, 5.0];
        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let auto = spiraltorch_runtime_tensor_matmul(runtime, lhs, rhs);
        let explicit = spiraltorch_runtime_tensor_matmul_with_backend(
            runtime,
            lhs,
            rhs,
            SpiraltorchMatmulBackend::CpuNaive,
        );
        assert!(!auto.is_null());
        assert!(!explicit.is_null());

        let mut auto_buffer = vec![0.0_f32; 4];
        let mut explicit_buffer = vec![0.0_f32; 4];
        let auto_ok = unsafe {
            spiraltorch_tensor_copy_data(auto, auto_buffer.as_mut_ptr(), auto_buffer.len())
        };
        let explicit_ok = unsafe {
            spiraltorch_tensor_copy_data(
                explicit,
                explicit_buffer.as_mut_ptr(),
                explicit_buffer.len(),
            )
        };
        assert!(auto_ok);
        assert!(explicit_ok);
        assert_eq!(auto_buffer, explicit_buffer);

        spiraltorch_tensor_free(explicit);
        spiraltorch_tensor_free(auto);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn runtime_matmul_bias_relu_matches_direct() {
        let runtime = unsafe { spiraltorch_runtime_new(0, ptr::null()) };
        assert!(!runtime.is_null());

        let lhs_data = vec![1.0_f32, -2.0, 0.5, 3.0, -1.0, 4.0];
        let rhs_data = vec![0.5_f32, 1.0, -1.5, 2.0];
        let bias = vec![0.25_f32, -0.75];

        let lhs = unsafe { spiraltorch_tensor_from_dense(3, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let direct =
            unsafe { spiraltorch_tensor_matmul_bias_relu(lhs, rhs, bias.as_ptr(), bias.len()) };
        let spawned = unsafe {
            spiraltorch_runtime_tensor_matmul_bias_relu(
                runtime,
                lhs,
                rhs,
                bias.as_ptr(),
                bias.len(),
            )
        };
        assert!(!direct.is_null());
        assert!(!spawned.is_null());

        let mut direct_buffer = vec![0.0_f32; 6];
        let mut spawned_buffer = vec![0.0_f32; 6];
        let direct_ok = unsafe {
            spiraltorch_tensor_copy_data(direct, direct_buffer.as_mut_ptr(), direct_buffer.len())
        };
        let spawned_ok = unsafe {
            spiraltorch_tensor_copy_data(spawned, spawned_buffer.as_mut_ptr(), spawned_buffer.len())
        };
        assert!(direct_ok);
        assert!(spawned_ok);
        assert_eq!(direct_buffer, spawned_buffer);

        spiraltorch_tensor_free(spawned);
        spiraltorch_tensor_free(direct);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn runtime_matmul_bias_add_relu_matches_direct() {
        let runtime = unsafe { spiraltorch_runtime_new(0, ptr::null()) };
        assert!(!runtime.is_null());

        let lhs_data = vec![1.0_f32, 2.0, -1.0, 0.5];
        let rhs_data = vec![0.5_f32, -0.5, 1.5, 2.5];
        let bias = vec![0.2_f32, -0.3];
        let residual_data = vec![0.1_f32, 0.2, 0.3, 0.4];

        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        let residual = unsafe {
            spiraltorch_tensor_from_dense(2, 2, residual_data.as_ptr(), residual_data.len())
        };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());
        assert!(!residual.is_null());

        let direct = unsafe {
            spiraltorch_tensor_matmul_bias_add_relu(lhs, rhs, bias.as_ptr(), bias.len(), residual)
        };
        let spawned = unsafe {
            spiraltorch_runtime_tensor_matmul_bias_add_relu(
                runtime,
                lhs,
                rhs,
                bias.as_ptr(),
                bias.len(),
                residual,
            )
        };
        assert!(!direct.is_null());
        assert!(!spawned.is_null());

        let mut direct_buffer = vec![0.0_f32; 4];
        let mut spawned_buffer = vec![0.0_f32; 4];
        let direct_ok = unsafe {
            spiraltorch_tensor_copy_data(direct, direct_buffer.as_mut_ptr(), direct_buffer.len())
        };
        let spawned_ok = unsafe {
            spiraltorch_tensor_copy_data(spawned, spawned_buffer.as_mut_ptr(), spawned_buffer.len())
        };
        assert!(direct_ok);
        assert!(spawned_ok);
        assert_eq!(direct_buffer, spawned_buffer);

        spiraltorch_tensor_free(spawned);
        spiraltorch_tensor_free(direct);
        spiraltorch_tensor_free(residual);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn runtime_matmul_bias_gelu_matches_direct() {
        let runtime = unsafe { spiraltorch_runtime_new(0, ptr::null()) };
        assert!(!runtime.is_null());

        let lhs_data = vec![1.0_f32, -0.5, 0.75, 2.0];
        let rhs_data = vec![0.5_f32, 1.25, -1.5, 0.25];
        let bias = vec![0.05_f32, -0.1];

        let lhs = unsafe { spiraltorch_tensor_from_dense(2, 2, lhs_data.as_ptr(), lhs_data.len()) };
        let rhs = unsafe { spiraltorch_tensor_from_dense(2, 2, rhs_data.as_ptr(), rhs_data.len()) };
        assert!(!lhs.is_null());
        assert!(!rhs.is_null());

        let direct =
            unsafe { spiraltorch_tensor_matmul_bias_gelu(lhs, rhs, bias.as_ptr(), bias.len()) };
        let spawned = unsafe {
            spiraltorch_runtime_tensor_matmul_bias_gelu(
                runtime,
                lhs,
                rhs,
                bias.as_ptr(),
                bias.len(),
            )
        };
        assert!(!direct.is_null());
        assert!(!spawned.is_null());

        let mut direct_buffer = vec![0.0_f32; 4];
        let mut spawned_buffer = vec![0.0_f32; 4];
        let direct_ok = unsafe {
            spiraltorch_tensor_copy_data(direct, direct_buffer.as_mut_ptr(), direct_buffer.len())
        };
        let spawned_ok = unsafe {
            spiraltorch_tensor_copy_data(spawned, spawned_buffer.as_mut_ptr(), spawned_buffer.len())
        };
        assert!(direct_ok);
        assert!(spawned_ok);
        assert_eq!(direct_buffer, spawned_buffer);

        spiraltorch_tensor_free(spawned);
        spiraltorch_tensor_free(direct);
        spiraltorch_tensor_free(rhs);
        spiraltorch_tensor_free(lhs);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn runtime_row_softmax_matches_direct_cpu() {
        let runtime = unsafe { spiraltorch_runtime_new(0, ptr::null()) };
        assert!(!runtime.is_null());

        let data = vec![1.0_f32, 2.0, -1.0, -2.0, 0.5, -0.25];
        let tensor = unsafe { spiraltorch_tensor_from_dense(3, 2, data.as_ptr(), data.len()) };
        assert!(!tensor.is_null());

        let direct = spiraltorch_tensor_row_softmax(tensor, SpiraltorchSoftmaxBackend::Cpu);
        let runtime_softmax =
            spiraltorch_runtime_tensor_row_softmax(runtime, tensor, SpiraltorchSoftmaxBackend::Cpu);
        assert!(!direct.is_null());
        assert!(!runtime_softmax.is_null());

        let mut direct_buffer = vec![0.0_f32; data.len()];
        let mut runtime_buffer = vec![0.0_f32; data.len()];
        let direct_ok = unsafe {
            spiraltorch_tensor_copy_data(direct, direct_buffer.as_mut_ptr(), direct_buffer.len())
        };
        let runtime_ok = unsafe {
            spiraltorch_tensor_copy_data(
                runtime_softmax,
                runtime_buffer.as_mut_ptr(),
                runtime_buffer.len(),
            )
        };
        assert!(direct_ok);
        assert!(runtime_ok);
        for (lhs, rhs) in direct_buffer.iter().zip(runtime_buffer.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }

        spiraltorch_tensor_free(runtime_softmax);
        spiraltorch_tensor_free(direct);
        spiraltorch_tensor_free(tensor);
        spiraltorch_runtime_free(runtime);
    }

    #[test]
    fn dlpack_drop_exported_state_is_safe_on_null() {
        unsafe {
            spiraltorch_dlpack_drop_exported_state(ptr::null_mut());
        }
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let tensor = unsafe { spiraltorch_tensor_from_dense(2, 2, data.as_ptr(), data.len()) };
        assert!(!tensor.is_null());
        let dlpack = spiraltorch_tensor_to_dlpack(tensor);
        assert!(!dlpack.is_null());
        unsafe {
            spiraltorch_dlpack_drop_exported_state(dlpack);
        }
        spiraltorch_tensor_free(tensor);
    }
}
