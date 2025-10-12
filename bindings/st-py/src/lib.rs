
use pyo3::prelude::*;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyArray2};
use ndarray::Ix2;

#[pyfunction] fn has_wgpu_py() -> PyResult<bool> { Ok(st_core::ops::public_api::wgpu_is_available()) }
#[pyfunction] fn has_cuda_py() -> PyResult<bool> { Ok(st_core::ops::public_api::cuda_is_available()) }
#[pyfunction] fn has_mps_py()  -> PyResult<bool> { Ok(st_core::ops::public_api::mps_is_available()) }

fn bool_like_to_u8<'py>(cond: &'py PyAny) -> PyResult<(Vec<u8>, Vec<usize>, Vec<isize>)> {
    if let Ok(b) = cond.downcast::<PyReadonlyArrayDyn<bool>>() {
        let a = b.as_array();
        let v: Vec<u8> = a.iter().map(|&x| if x {1u8} else {0u8}).collect();
        Ok((v, a.shape().to_vec(), a.strides().iter().map(|&s| s as isize).collect()))
    } else if let Ok(u8a) = cond.downcast::<PyReadonlyArrayDyn<u8>>() {
        let a = u8a.as_array();
        Ok((a.iter().cloned().collect(), a.shape().to_vec(), a.strides().iter().map(|&s| s as isize).collect()))
    } else if let Ok(u32a) = cond.downcast::<PyReadonlyArrayDyn<u32>>() {
        let a = u32a.as_array();
        Ok((a.iter().map(|&x| if x!=0 {1u8} else {0u8}).collect(), a.shape().to_vec(), a.strides().iter().map(|&s| s as isize).collect()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("cond must be bool/uint8/uint32"))
    }
}

fn normalize_neg_strides(shape: &mut [usize], strides: &mut [isize]) -> isize {
    let mut base_shift: isize = 0;
    for d in 0..shape.len() {
        if strides[d] < 0 {
            base_shift += strides[d] * (shape[d] as isize - 1);
            strides[d] = -strides[d];
        }
    }
    base_shift
}

#[pyfunction]
fn where_nd_py(py: Python<'_>, cond: &PyAny, x: PyReadonlyArrayDyn<f32>, y: PyReadonlyArrayDyn<f32>, device: Option<&str>) -> PyResult<Py<PyArrayDyn<f32>>> {
    let (cond_u8, mut cshape, mut cstrides_el) = bool_like_to_u8(cond)?;
    let ax = x.as_array(); let ay = y.as_array();
    let mut xshape = ax.shape().to_vec(); let mut yshape = ay.shape().to_vec();
    let mut xstrides_el: Vec<isize> = ax.strides().iter().map(|&s| s as isize).collect();
    let mut ystrides_el: Vec<isize> = ay.strides().iter().map(|&s| s as isize).collect();

    let nd = cshape.len().max(xshape.len()).max(yshape.len());
    let mut out_shape = vec![1usize; nd];
    for i in 0..nd {
        let da = *cshape.get(cshape.len().wrapping_sub(nd - i)).unwrap_or(&1);
        let db = *xshape.get(xshape.len().wrapping_sub(nd - i)).unwrap_or(&1);
        let dc = *yshape.get(yshape.len().wrapping_sub(nd - i)).unwrap_or(&1);
        let m = da.max(db).max(dc);
        if (da!=1 && da!=m) || (db!=1 && db!=m) || (dc!=1 && dc!=m) {
            return Err(pyo3::exceptions::PyValueError::new_err("broadcast shapes not compatible"));
        }
        out_shape[i]=m;
    }

    let c_base = normalize_neg_strides(&mut cshape, &mut cstrides_el);
    let x_base = normalize_neg_strides(&mut xshape, &mut xstrides_el);
    let y_base = normalize_neg_strides(&mut yshape, &mut ystrides_el);

    let dev = device.unwrap_or("auto").to_string();
    let want = if dev=="auto" {
        if st_core::ops::public_api::cuda_is_available() { "cuda" }
        else if st_core::ops::public_api::mps_is_available() { "mps" }
        else if st_core::ops::public_api::wgpu_is_available() { "wgpu" }
        else { "cpu" }
    } else { dev.as_str() };

    if want=="cuda" or want=="cpu" {
        let x_v: Vec<f32> = ax.iter().cloned().collect();
        let y_v: Vec<f32> = ay.iter().cloned().collect();
        let cstrides = cstrides_el.iter().map(|&s| s as usize).collect::<Vec<_>>();
        let xstrides = xstrides_el.iter().map(|&s| s as usize).collect::<Vec<_>>();
        let ystrides = ystrides_el.iter().map(|&s| s as usize).collect::<Vec<_>>();
        let out = st_core::ops::public_api::where_nd_host_select_strided(
            &cond_u8.iter().map(|&b| b!=0).collect::<Vec<bool>>(), &cshape, &cstrides,
            &x_v, &xshape, &xstrides, &y_v, &yshape, &ystrides, &out_shape, if want=="cuda" {"cuda"} else {"cpu"}
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        let arr = PyArrayDyn::<f32>::from_vec(py, out); arr.reshape(out_shape.as_slice()).unwrap(); return Ok(arr.to_owned());
    }

    // For this artifact, we still use the bytes path for WGPU (single-span). Multi-segment enumerator will be added later.
    unsafe {
        let x_ptr = ax.as_ptr() as *const u8;
        let y_ptr = ay.as_ptr() as *const u8;
        let itemx = std::mem::size_of::<f32>(); let itemy = std::mem::size_of::<f32>(); let itemc = std::mem::size_of::<u8>();
        let mut c_min:isize = c_base * itemc as isize; let mut c_max=isize::from(c_min);
        for d in 0..cshape.len(){ let st = cstrides_el[d]*itemc as isize; if st>=0{ c_max += st*(cshape[d] as isize -1)} else { c_min += st*(cshape[d] as isize -1)} }
        let mut x_min:isize = x_base * itemx as isize; let mut x_max=isize::from(x_min);
        for d in 0..xshape.len(){ let st = xstrides_el[d]*itemx as isize; if st>=0{ x_max += st*(xshape[d] as isize -1)} else { x_min += st*(xshape[d] as isize -1)} }
        let mut y_min:isize = y_base * itemy as isize; let mut y_max=isize::from(y_min);
        for d in 0..yshape.len(){ let st = ystrides_el[d]*itemy as isize; if st>=0{ y_max += st*(yshape[d] as isize -1)} else { y_min += st*(yshape[d] as isize -1)} }
        let c_total = (c_max - c_min) as usize + itemc; let x_total = (x_max - x_min) as usize + itemx; let y_total = (y_max - y_min) as usize + itemy;
        let c_blob = std::slice::from_raw_parts((cond_u8.as_ptr() as *const u8).offset(c_min), c_total);
        let x_blob = std::slice::from_raw_parts(x_ptr.offset(x_min), x_total);
        let y_blob = std::slice::from_raw_parts(y_ptr.offset(y_min), y_total);
        let out = st_core::ops::public_api::where_nd_host_select_strided_bytes(
            c_blob, &cshape, &cstrides_el.iter().map(|&s| s as usize).collect::<Vec<_>>(), (-c_min) as usize,
            x_blob, &xshape, &xstrides_el.iter().map(|&s| s as usize).collect::<Vec<_>>(), (-x_min/4) as usize,
            y_blob, &yshape, &ystrides_el.iter().map(|&s| s as usize).collect::<Vec<_>>(), (-y_min/4) as usize,
            &out_shape, "wgpu"
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        let arr = PyArrayDyn::<f32>::from_vec(py, out); arr.reshape(out_shape.as_slice()).unwrap(); return Ok(arr.to_owned());
    }
}

#[pyfunction]
fn topk2d_py(py: Python<'_>, x: PyReadonlyArrayDyn<f32>, k: usize, device: Option<&str>) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<i32>>)> {
    let a = x.as_array();
    if a.ndim()!=2 { return Err(pyo3::exceptions::PyValueError::new_err("topk2d: x must be 2D [rows, cols]")); }
    let rows = a.shape()[0]; let cols = a.shape()[1];
    let x_v: Vec<f32> = a.iter().cloned().collect();
    let dev = device.unwrap_or("auto");
    let want = if dev=="auto" {
        if st_core::ops::public_api::cuda_is_available() { "cuda" }
        else if st_core::ops::public_api::mps_is_available() { "mps" }
        else if st_core::ops::public_api::wgpu_is_available() { "wgpu" }
        else { "cpu" }
    } else { dev };
    let (vals, idxs) = st_core::ops::public_api::topk_lastdim_host_select(&x_v, rows, cols, k, want)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
    let va = PyArray2::<f32>::from_vec2(py, &vals.chunks(k).map(|c| c.to_vec()).collect::<Vec<_>>()).unwrap();
    let ia = PyArray2::<i32>::from_vec2(py, &idxs.chunks(k).map(|c| c.to_vec()).collect::<Vec<_>>()).unwrap();
    Ok((va.to_owned(), ia.to_owned()))
}

#[pymodule] fn spiraltorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "1.3.52")?;
    m.add_function(wrap_pyfunction!(has_wgpu_py, m)?)?;
    m.add_function(wrap_pyfunction!(has_cuda_py, m)?)?;
    m.add_function(wrap_pyfunction!(has_mps_py, m)?)?;
    m.add_function(wrap_pyfunction!(where_nd_py, m)?)?;
    m.add_function(wrap_pyfunction!(topk2d_py, m)?)?;
    Ok(())
}
