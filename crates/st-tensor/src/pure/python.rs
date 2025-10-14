use super::{AmegaHypergrad, LanguageWaveEncoder, OpenCartesianTopos, Tensor, TensorError};
use core::ptr;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;
use std::sync::{Mutex, OnceLock};

fn error_storage() -> &'static Mutex<Option<CString>> {
    static STORAGE: OnceLock<Mutex<Option<CString>>> = OnceLock::new();
    STORAGE.get_or_init(|| Mutex::new(None))
}

fn store_error(err: TensorError) -> i32 {
    let storage = error_storage();
    let mut slot = storage.lock().expect("last error mutex poisoned");
    let message = CString::new(err.to_string())
        .unwrap_or_else(|_| CString::new("tensor error").expect("literal has no nul"));
    *slot = Some(message);
    -1
}

fn clear_error() {
    let storage = error_storage();
    let mut slot = storage.lock().expect("last error mutex poisoned");
    *slot = None;
}

fn data_length_mismatch(expected: usize, got: usize) -> TensorError {
    TensorError::DataLength { expected, got }
}

#[no_mangle]
pub extern "C" fn st_pure_last_error() -> *const c_char {
    let storage = error_storage();
    let slot = storage.lock().expect("last error mutex poisoned");
    match slot.as_ref() {
        Some(msg) => msg.as_ptr(),
        None => ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn st_pure_clear_last_error() {
    clear_error();
}

#[no_mangle]
pub extern "C" fn st_pure_topos_new(
    curvature: f32,
    tolerance: f32,
    saturation: f32,
    max_depth: usize,
    max_volume: usize,
) -> *mut OpenCartesianTopos {
    match OpenCartesianTopos::new(curvature, tolerance, saturation, max_depth, max_volume) {
        Ok(topos) => Box::into_raw(Box::new(topos)),
        Err(err) => {
            store_error(err);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn st_pure_topos_free(ptr: *mut OpenCartesianTopos) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_new(
    curvature: f32,
    learning_rate: f32,
    rows: usize,
    cols: usize,
) -> *mut AmegaHypergrad {
    match AmegaHypergrad::new(curvature, learning_rate, rows, cols) {
        Ok(hypergrad) => Box::into_raw(Box::new(hypergrad)),
        Err(err) => {
            store_error(err);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_with_topos(
    curvature: f32,
    learning_rate: f32,
    rows: usize,
    cols: usize,
    topos: *const OpenCartesianTopos,
) -> *mut AmegaHypergrad {
    if topos.is_null() {
        store_error(TensorError::EmptyInput("topos handle"));
        return ptr::null_mut();
    }
    let guard = unsafe { &*topos };
    match AmegaHypergrad::with_topos(curvature, learning_rate, rows, cols, guard.clone()) {
        Ok(hypergrad) => Box::into_raw(Box::new(hypergrad)),
        Err(err) => {
            store_error(err);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_free(ptr: *mut AmegaHypergrad) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn st_pure_encoder_new(
    curvature: f32,
    temperature: f32,
) -> *mut LanguageWaveEncoder {
    match LanguageWaveEncoder::new(curvature, temperature) {
        Ok(encoder) => Box::into_raw(Box::new(encoder)),
        Err(err) => {
            store_error(err);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn st_pure_encoder_free(ptr: *mut LanguageWaveEncoder) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_shape(
    hypergrad: *const AmegaHypergrad,
    rows_out: *mut usize,
    cols_out: *mut usize,
) -> i32 {
    if hypergrad.is_null() {
        return store_error(TensorError::EmptyInput("hypergrad handle"));
    }
    if rows_out.is_null() {
        return store_error(TensorError::EmptyInput("rows pointer"));
    }
    if cols_out.is_null() {
        return store_error(TensorError::EmptyInput("cols pointer"));
    }
    let hypergrad = unsafe { &*hypergrad };
    let (rows, cols) = hypergrad.shape();
    unsafe {
        *rows_out = rows;
        *cols_out = cols;
    }
    0
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
    let slice = slice::from_raw_parts(ptr, len);
    let expected = rows.saturating_mul(cols);
    if slice.len() != expected {
        return Err(data_length_mismatch(expected, slice.len()));
    }
    Tensor::from_vec(rows, cols, slice.to_vec())
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_accumulate_pair(
    hypergrad: *mut AmegaHypergrad,
    prediction: *const f32,
    prediction_len: usize,
    target: *const f32,
    target_len: usize,
) -> i32 {
    if hypergrad.is_null() {
        return store_error(TensorError::EmptyInput("hypergrad handle"));
    }
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
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_apply(
    hypergrad: *mut AmegaHypergrad,
    weights: *mut f32,
    weights_len: usize,
) -> i32 {
    if hypergrad.is_null() {
        return store_error(TensorError::EmptyInput("hypergrad handle"));
    }
    if weights.is_null() {
        return store_error(TensorError::EmptyInput("weights"));
    }
    let hypergrad = unsafe { &mut *hypergrad };
    let (rows, cols) = hypergrad.shape();
    let slice = unsafe { slice::from_raw_parts(weights, weights_len) };
    let expected = rows.saturating_mul(cols);
    if slice.len() != expected {
        return store_error(data_length_mismatch(expected, slice.len()));
    }
    let mut tensor = match Tensor::from_vec(rows, cols, slice.to_vec()) {
        Ok(tensor) => tensor,
        Err(err) => return store_error(err),
    };
    match hypergrad.apply(&mut tensor) {
        Ok(_) => {
            let data = tensor.data();
            let dest = unsafe { slice::from_raw_parts_mut(weights, weights_len) };
            dest[..data.len()].copy_from_slice(data);
            0
        }
        Err(err) => store_error(err),
    }
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_gradient(
    hypergrad: *const AmegaHypergrad,
    out: *mut f32,
    out_len: usize,
) -> usize {
    if hypergrad.is_null() {
        store_error(TensorError::EmptyInput("hypergrad handle"));
        return 0;
    }
    let hypergrad = unsafe { &*hypergrad };
    let gradient = hypergrad.gradient();
    if out.is_null() {
        return gradient.len();
    }
    if out_len < gradient.len() {
        store_error(data_length_mismatch(gradient.len(), out_len));
        return 0;
    }
    let dest = unsafe { slice::from_raw_parts_mut(out, out_len) };
    dest[..gradient.len()].copy_from_slice(gradient);
    gradient.len()
}

#[no_mangle]
pub extern "C" fn st_pure_hypergrad_absorb_text(
    hypergrad: *mut AmegaHypergrad,
    encoder: *const LanguageWaveEncoder,
    text: *const c_char,
) -> i32 {
    if hypergrad.is_null() {
        return store_error(TensorError::EmptyInput("hypergrad handle"));
    }
    if encoder.is_null() {
        return store_error(TensorError::EmptyInput("encoder handle"));
    }
    if text.is_null() {
        return store_error(TensorError::EmptyInput("text pointer"));
    }
    let hypergrad = unsafe { &mut *hypergrad };
    let encoder = unsafe { &*encoder };
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    match hypergrad.absorb_text(encoder, &text) {
        Ok(_) => 0,
        Err(err) => store_error(err),
    }
}

#[no_mangle]
pub extern "C" fn st_pure_encoder_encode_z_space(
    encoder: *const LanguageWaveEncoder,
    text: *const c_char,
    out: *mut f32,
    out_len: usize,
) -> usize {
    if encoder.is_null() {
        store_error(TensorError::EmptyInput("encoder handle"));
        return 0;
    }
    if text.is_null() {
        store_error(TensorError::EmptyInput("text pointer"));
        return 0;
    }
    let encoder = unsafe { &*encoder };
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
    if out_len < data.len() {
        store_error(data_length_mismatch(data.len(), out_len));
        return 0;
    }
    let dest = unsafe { slice::from_raw_parts_mut(out, out_len) };
    dest[..data.len()].copy_from_slice(data);
    data.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    unsafe fn last_error_string() -> String {
        let ptr = st_pure_last_error();
        if ptr.is_null() {
            "<no error>".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }

    #[test]
    fn ffi_hypergrad_round_trip() {
        st_pure_clear_last_error();
        let curvature = -1.0;
        let encoder = st_pure_encoder_new(curvature, 0.5);
        if encoder.is_null() {
            panic!("failed to create encoder: {}", unsafe {
                last_error_string()
            });
        }
        let text = CString::new("spiral torch pure").unwrap();
        let required = st_pure_encoder_encode_z_space(encoder, text.as_ptr(), ptr::null_mut(), 0);
        assert!(required > 0);
        let hypergrad = st_pure_hypergrad_new(curvature, 0.1, 1, required);
        if hypergrad.is_null() {
            panic!("failed to create hypergrad: {}", unsafe {
                last_error_string()
            });
        }
        let status = st_pure_hypergrad_absorb_text(hypergrad, encoder, text.as_ptr());
        assert_eq!(status, 0);

        let mut weights = vec![0.0f32; required];
        let apply_status = st_pure_hypergrad_apply(hypergrad, weights.as_mut_ptr(), weights.len());
        assert_eq!(apply_status, 0);

        let mut gradient = vec![0.0f32; required];
        let copied = st_pure_hypergrad_gradient(hypergrad, gradient.as_mut_ptr(), gradient.len());
        assert_eq!(copied, gradient.len());

        st_pure_encoder_free(encoder);
        st_pure_hypergrad_free(hypergrad);
    }

    #[test]
    fn ffi_encoder_len_probe() {
        st_pure_clear_last_error();
        let encoder = st_pure_encoder_new(-1.0, 0.3);
        if encoder.is_null() {
            panic!("failed to create encoder: {}", unsafe {
                last_error_string()
            });
        }
        let text = CString::new("hyperbolic ffi").unwrap();
        let required = st_pure_encoder_encode_z_space(encoder, text.as_ptr(), ptr::null_mut(), 0);
        assert!(required > 0);
        let mut buffer = vec![0.0f32; required];
        let copied = st_pure_encoder_encode_z_space(
            encoder,
            text.as_ptr(),
            buffer.as_mut_ptr(),
            buffer.len(),
        );
        assert_eq!(copied, required);
        st_pure_encoder_free(encoder);
    }
}
