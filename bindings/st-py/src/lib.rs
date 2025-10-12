use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayDyn, IntoPyArray};

#[inline] fn rowmajor_strides(shape:&[usize]) -> Vec<usize> {
    let nd = shape.len(); let mut st = vec![0usize; nd]; let mut acc = 1usize;
    for d in (0..nd).rev() { st[d] = acc; acc = acc.saturating_mul(shape[d].max(1)); }
    st
}
fn canonicalize_neg_strides(shape:&[usize], strides:&[isize], itemsize:usize, base_ptr:usize) -> (usize, Vec<usize>) {
    let nd = shape.len(); let mut base = base_ptr; let mut st = vec![0usize; nd];
    for d in 0..nd {
        if strides[d] < 0 { if shape[d] > 0 { base = base + ((shape[d]-1) as isize * strides[d].abs()) as usize * itemsize; } st[d] = (strides[d].abs() as usize); }
        else { st[d] = (strides[d] as usize); }
    } (base, st)
}
fn span_bytes(shape:&[usize], strides:&[usize], itemsize:usize) -> usize {
    if shape.is_empty() { return itemsize; } let mut end = 0usize;
    for d in 0..shape.len() { if shape[d] > 0 { end += (shape[d]-1) * strides[d]; } }
    end + itemsize
}

#[pyfunction]
fn topk2d_py<'py>(py: Python<'py>, x: &'py PyArray2<f32>, k: usize, device: Option<&str>) -> PyResult<(&'py PyArray2<f32>, &'py PyArray2<i32>)> {
    let xv = unsafe { x.as_slice()? };
    let shape = x.shape(); let (rows, cols) = (shape[0], shape[1]);
    let want = device.unwrap_or("auto"); let want = if want=="auto" {
        #[cfg(feature="cuda")] { "cuda" }
        #[cfg(all(not(feature="cuda"), feature="mps"))] { "mps" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), feature="wgpu"))] { "wgpu" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), not(feature="wgpu")))] { "cpu" }
    } else { want };
    let (vals, idxs) = match want {
        "cuda" => { #[cfg(feature="cuda")] { st_core::ops::public_api::topk_lastdim_cuda_2d(xv, rows, cols, k) }
                    #[cfg(not(feature="cuda"))] { Err(st_core::error::device("cuda feature not enabled")) } }
        "wgpu" => { #[cfg(feature="wgpu")] { st_core::ops::public_api::topk_lastdim_wgpu_2d_autotuned(xv, rows, cols, k) }
                    #[cfg(not(feature="wgpu"))] { Err(st_core::error::device("wgpu feature not enabled")) } }
        _ => st_core::ops::public_api::topk_lastdim_host_select(xv, rows, cols, k, want),
    }.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
    let v = PyArray2::from_slice(py, &vals).reshape([rows, k]).unwrap();
    let i = PyArray2::from_slice(py, &idxs).reshape([rows, k]).unwrap();
    Ok((v, i))
}

#[pyfunction]
fn where_nd_py<'py>(py: Python<'py>, cond: &'py PyArrayDyn<bool>, x: &'py PyArrayDyn<f32>, y: &'py PyArrayDyn<f32>, device: Option<&str>) -> PyResult<&'py PyArrayDyn<f32>> {
    let want = device.unwrap_or("auto");
    let want = if want=="auto" {
        #[cfg(feature="cuda")] { "cuda" }
        #[cfg(all(not(feature="cuda"), feature="mps"))] { "mps" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), feature="wgpu"))] { "wgpu" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), not(feature="wgpu")))] { "cpu" }
    } else { want };
    let c_shape: Vec<usize> = cond.shape().into();
    let x_shape: Vec<usize> = x.shape().into(); let y_shape: Vec<usize> = y.shape().into();
    if x_shape != y_shape { return Err(pyo3::exceptions::PyValueError::new_err("x and y must match shape")); }
    let out_shape = x_shape.clone();

    let c_ptr = cond.data() as usize; let x_ptr = x.data() as usize; let y_ptr = y.data() as usize;
    let c_strides_el = cond.strides().to_vec(); let x_strides_el = x.strides().to_vec(); let y_strides_el = y.strides().to_vec();

    let (c_base, c_strides_b) = canonicalize_neg_strides(&c_shape, &c_strides_el, 1, c_ptr);
    let (x_base, x_strides_b) = canonicalize_neg_strides(&x_shape, &x_strides_el, 4, x_ptr);
    let (y_base, y_strides_b) = canonicalize_neg_strides(&y_shape, &y_strides_el, 4, y_ptr);

    let c_span = span_bytes(&c_shape, &c_strides_b, 1);
    let x_span = span_bytes(&x_shape, &x_strides_b, 4);
    let y_span = span_bytes(&y_shape, &y_strides_b, 4);

    let c_bytes = unsafe { std::slice::from_raw_parts(c_base as *const u8, c_span) };
    let x_bytes = unsafe { std::slice::from_raw_parts(x_base as *const u8, x_span) };
    let y_bytes = unsafe { std::slice::from_raw_parts(y_base as *const u8, y_span) };

    let cstrides_el_pos: Vec<usize> = c_strides_el.iter().map(|&s| s.unsigned_abs() as usize).collect();
    let xstrides_el_pos: Vec<usize> = x_strides_el.iter().map(|&s| s.unsigned_abs() as usize).collect();
    let ystrides_el_pos: Vec<usize> = y_strides_el.iter().map(|&s| s.unsigned_abs() as usize).collect();

    let res: Vec<f32> = match want {
        "wgpu" => {
            #[cfg(feature="wgpu")]
            { st_core::ops::public_api::where_nd_host_select_strided_bytes_direct_wgpu(
                c_bytes, 0, &c_shape, &cstrides_el_pos,
                x_bytes, 0, &x_shape, &xstrides_el_pos,
                y_bytes, 0, &y_shape, &ystrides_el_pos,
                &out_shape
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))? }
            #[cfg(not(feature="wgpu"))]
            { return Err(pyo3::exceptions::PyRuntimeError::new_err("wgpu feature not enabled")); }
        }
        "mps" => {
            #[cfg(feature="mps")]
            { let out_strides = rowmajor_strides(&out_shape);
              st_core::ops::public_api::where_nd_host_select_strided_bytes_direct_mps(
                c_bytes, 0, &c_shape, &cstrides_el_pos,
                x_bytes, 0, &x_shape, &xstrides_el_pos,
                y_bytes, 0, &y_shape, &ystrides_el_pos,
                &out_shape, &out_strides
              ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))? }
            #[cfg(not(feature="mps"))]
            { return Err(pyo3::exceptions::PyRuntimeError::new_err("mps feature not enabled")); }
        }
        _ => {
            let nelem = out_shape.iter().product::<usize>(); let mut out = vec![0f32; nelem];
            for i in 0..nelem {
                let cb = unsafe { *c_bytes.get_unchecked(i) };
                let xv = f32::from_le_bytes([x_bytes[4*i],x_bytes[4*i+1],x_bytes[4*i+2],x_bytes[4*i+3]]);
                let yv = f32::from_le_bytes([y_bytes[4*i],y_bytes[4*i+1],y_bytes[4*i+2],y_bytes[4*i+3]]);
                out[i] = if cb!=0 { xv } else { yv };
            }
            out
        }
    };
    let ary = PyArrayDyn::<f32>::from_vec(py, res); ary.reshape(out_shape.as_slice()).unwrap(); Ok(ary)
}

#[pymodule]
fn spiraltorch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(topk2d_py, m)?)?;
    m.add_function(wrap_pyfunction!(where_nd_py, m)?)?;
    m.add("__version__", "1.3.72")?;
    Ok(())
}
