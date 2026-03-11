#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;

use st_tensor::{DifferentialResonance, Tensor, TensorError};
use st_text::TextResonator;

#[repr(C)]
pub struct ResonanceHandle {
    resonator: TextResonator,
}

#[no_mangle]
pub extern "C" fn st_cobol_new_resonator(
    curvature: c_float,
    temperature: c_float,
) -> *mut ResonanceHandle {
    let curvature = if curvature >= 0.0 {
        -curvature.abs().max(1.0e-3)
    } else {
        curvature
    };
    let temperature = if temperature <= 0.0 {
        temperature.abs().max(1.0e-3)
    } else {
        temperature
    };

    match TextResonator::new(curvature, temperature) {
        Ok(resonator) => Box::into_raw(Box::new(ResonanceHandle { resonator })),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn st_cobol_free_resonator(handle: *mut ResonanceHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub extern "C" fn st_cobol_describe(
    handle: *mut ResonanceHandle,
    values: *const c_float,
    len: c_int,
    out_summary: *mut *mut c_char,
) -> c_int {
    if handle.is_null() || values.is_null() || out_summary.is_null() || len < 0 {
        return -1;
    }

    let handle = unsafe { &mut *handle };
    let len = len as usize;
    let coefficients = unsafe { std::slice::from_raw_parts(values, len) };

    let resonance = match resonance_from_coefficients(coefficients) {
        Ok(resonance) => resonance,
        Err(_) => return -2,
    };

    let summary = handle.resonator.describe_resonance(&resonance).summary;
    match CString::new(summary) {
        Ok(c_string) => {
            unsafe {
                *out_summary = c_string.into_raw();
            }
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub extern "C" fn st_cobol_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(ptr));
    }
}

fn resonance_from_coefficients(values: &[f32]) -> Result<DifferentialResonance, TensorError> {
    if values.is_empty() {
        return Err(TensorError::InvalidDimensions { rows: 1, cols: 0 });
    }
    let tensor = Tensor::from_vec(1, values.len(), values.to_vec())?;
    Ok(DifferentialResonance {
        homotopy_flow: tensor.clone(),
        functor_linearisation: tensor.clone(),
        recursive_objective: tensor.clone(),
        infinity_projection: tensor.clone(),
        infinity_energy: tensor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::ptr;

    #[test]
    fn creates_resonator_and_describes_summary() {
        let handle = st_cobol_new_resonator(0.42, 0.58);
        assert!(!handle.is_null());

        let coefficients = [0.1, -0.3, 0.25, -0.4, 0.55];
        let mut summary_ptr: *mut c_char = ptr::null_mut();
        let status = st_cobol_describe(
            handle,
            coefficients.as_ptr(),
            coefficients.len() as c_int,
            &mut summary_ptr,
        );
        assert_eq!(status, 0);
        assert!(!summary_ptr.is_null());

        let summary = unsafe { CStr::from_ptr(summary_ptr) }
            .to_string_lossy()
            .into_owned();
        assert!(!summary.is_empty());
        st_cobol_free_string(summary_ptr);

        st_cobol_free_resonator(handle);
    }

    #[test]
    fn generated_summary_mentions_resonance() {
        let handle = st_cobol_new_resonator(0.5, 0.4);
        assert!(!handle.is_null());
        let coefficients = [0.12, -0.28, 0.44];
        let mut summary_ptr: *mut c_char = ptr::null_mut();
        let status = st_cobol_describe(
            handle,
            coefficients.as_ptr(),
            coefficients.len() as c_int,
            &mut summary_ptr,
        );
        assert_eq!(status, 0);
        let summary = unsafe { CStr::from_ptr(summary_ptr) }
            .to_string_lossy()
            .into_owned();
        assert!(summary.chars().any(|ch| ch.is_alphabetic()));
        st_cobol_free_string(summary_ptr);
        st_cobol_free_resonator(handle);
    }

    #[test]
    fn rejects_invalid_arguments() {
        let mut out: *mut c_char = ptr::null_mut();
        let status = st_cobol_describe(ptr::null_mut(), ptr::null(), 0, &mut out);
        assert_eq!(status, -1);
    }

    #[test]
    fn rejects_empty_coefficients() {
        let handle = st_cobol_new_resonator(0.2, 0.3);
        assert!(!handle.is_null());

        let mut out: *mut c_char = ptr::null_mut();
        let status = st_cobol_describe(handle, ptr::null(), 5, &mut out);
        assert_eq!(status, -1);

        let mut out: *mut c_char = ptr::null_mut();
        let status = st_cobol_describe(handle, ptr::null(), -1, &mut out);
        assert_eq!(status, -1);

        let empty: [f32; 0] = [];
        let mut out: *mut c_char = ptr::null_mut();
        let status = st_cobol_describe(handle, empty.as_ptr(), empty.len() as c_int, &mut out);
        assert_eq!(status, -2);

        st_cobol_free_resonator(handle);
    }
}
