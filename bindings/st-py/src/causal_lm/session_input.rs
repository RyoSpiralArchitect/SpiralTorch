//! Bound immutable Python input before abi3 materializes a full UTF-8 buffer.
use super::error;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PySlice, PyString};
use st_nn::causal_lm::NativeCausalLmTrainingSession;

fn text_bytes<'py>(s: &Bound<'py, PyString>, limit: usize) -> PyResult<Bound<'py, PyBytes>> {
    let py = s.py();
    let ty = py.get_type::<PyString>();
    // Use the built-in descriptors, not arbitrary overrides on a str subclass.
    let n: usize = ty.call_method1("__len__", (s,))?.extract()?;
    let over = || error("session text exceeds byte budget");
    if n > limit {
        return Err(over());
    }
    const CHUNK: usize = 16 * 1024;
    if n <= CHUNK.min(limit / 4) || ty.call_method1("isascii", (s,))?.extract::<bool>()? {
        return s.encode_utf8();
    }
    let chunk = |start: usize| {
        let slice = PySlice::new(py, start as isize, (start + CHUNK).min(n) as isize, 1);
        ty.call_method1("__getitem__", (s, slice))?
            .cast_into::<PyString>()?
            .encode_utf8()
    };
    let mut remaining = limit;
    for start in (0..n).step_by(CHUNK) {
        remaining = remaining
            .checked_sub(chunk(start)?.as_bytes().len())
            .ok_or_else(over)?;
    }
    // A single non-BMP character can make CPython reserve four bytes per code
    // point even in mostly ASCII text. Fill exactly the admitted UTF-8 length.
    PyBytes::new_with(py, limit - remaining, |out| {
        let mut offset = 0;
        for start in (0..n).step_by(CHUNK) {
            let part = chunk(start)?;
            let data = part.as_bytes();
            out[offset..offset + data.len()].copy_from_slice(data);
            offset += data.len();
        }
        Ok(())
    })
}

pub(super) fn payload<'py>(v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    payload_with_limit(v, NativeCausalLmTrainingSession::MAX_PAYLOAD_BYTES)
}

pub(super) fn payload_with_limit<'py>(
    v: &Bound<'py, PyAny>,
    limit: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    if let Ok(b) = v.cast::<PyBytes>() {
        if b.as_bytes().len() > limit {
            return Err(error("session checkpoint exceeds byte budget"));
        }
        return Ok(b.clone());
    }
    text_bytes(v.cast::<PyString>()?, limit)
}

pub(super) fn pin(v: &Bound<'_, PyAny>) -> PyResult<String> {
    let bytes = text_bytes(v.cast::<PyString>()?, "sha256:".len() + 64)?;
    Ok(std::str::from_utf8(bytes.as_bytes())
        .map_err(error)?
        .to_owned())
}
