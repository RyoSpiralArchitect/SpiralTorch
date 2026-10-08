//! ABI3-compatible bulk transport; never retain foreign writable storage.

use pyo3::exceptions::{PyBufferError, PyMemoryError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyBytesMethods, PyMemoryView};

pub(crate) fn read_f32(
    input: &Bound<'_, PyAny>,
    max_values: usize,
    expected_values: Option<usize>,
) -> PyResult<Vec<f32>> {
    let view = PyMemoryView::from(input)?;
    let format: String = view.getattr("format")?.extract()?;
    let native_format = matches!(format.as_str(), "f" | "@f" | "=f")
        || (cfg!(target_endian = "little") && format == "<f")
        || (cfg!(target_endian = "big") && matches!(format.as_str(), ">f" | "!f"));
    if !native_format || view.getattr("itemsize")?.extract::<usize>()? != 4 {
        return Err(PyTypeError::new_err(
            "buffer must contain native-endian float32 values",
        ));
    }
    if !view.getattr("c_contiguous")?.extract::<bool>()? {
        return Err(PyBufferError::new_err(
            "float32 buffer must be C-contiguous",
        ));
    }
    let nbytes: usize = view.getattr("nbytes")?.extract()?;
    if !nbytes.is_multiple_of(4) {
        return Err(PyBufferError::new_err(
            "float32 buffer byte length is invalid",
        ));
    }
    let count = nbytes / 4;
    if count > max_values {
        return Err(PyValueError::new_err(
            "float32 buffer exceeds the value budget",
        ));
    }
    if expected_values.is_some_and(|expected| count != expected) {
        return Err(PyValueError::new_err(
            "float32 buffer direction length does not match the snapshot",
        ));
    }
    // memoryview works with the py38 limited ABI. Copy before releasing the GIL;
    // decoding bytes also avoids assuming the exporter's pointer is f32-aligned.
    let copied = view.call_method0("tobytes")?;
    let bytes = copied.cast::<PyBytes>()?.as_bytes();
    if bytes.len() != nbytes {
        return Err(PyBufferError::new_err(
            "float32 buffer length changed during copying",
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("cannot allocate float32 transport"))?;
    for &value in bytes.as_chunks::<4>().0 {
        values.push(f32::from_ne_bytes(value));
    }
    Ok(values)
}

pub(crate) fn write_f32<'py>(
    py: Python<'py>,
    values: impl ExactSizeIterator<Item = f32>,
) -> PyResult<Bound<'py, PyByteArray>> {
    let bytes = values
        .len()
        .checked_mul(4)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| PyOverflowError::new_err("float32 output byte length overflowed"))?;
    // The writable owner is fresh: modifying a Torch view cannot mutate a snapshot.
    PyByteArray::new_with(py, bytes, |output| {
        for (value, destination) in values.zip(output.as_chunks_mut::<4>().0) {
            *destination = value.to_ne_bytes();
        }
        Ok(())
    })
}
