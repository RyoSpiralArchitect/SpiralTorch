
use pyo3::prelude::*;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyArray2};
use ndarray::Ix2;

#[pyfunction]
fn has_wgpu_py() -> PyResult<bool> { Ok(st_core::ops::public_api::wgpu_is_available()) }
#[pyfunction]
fn has_cuda_py() -> PyResult<bool> { Ok(st_core::ops::public_api::cuda_is_available()) }
#[pyfunction]
fn has_mps_py() -> PyResult<bool> { Ok(st_core::ops::public_api::mps_is_available()) }

fn bool_like_to_vec<'py>(cond: &'py PyAny) -> PyResult<(Vec<bool>, Vec<usize>)> {
    if let Ok(b) = cond.downcast::<PyReadonlyArrayDyn<bool>>() {
        let a = b.as_array(); Ok((a.iter().cloned().collect(), a.shape().to_vec()))
    } else if let Ok(u8a) = cond.downcast::<PyReadonlyArrayDyn<u8>>() {
        let a = u8a.as_array(); Ok((a.iter().map(|&x| x!=0).collect(), a.shape().to_vec()))
    } else if let Ok(u32a) = cond.downcast::<PyReadonlyArrayDyn<u32>>() {
        let a = u32a.as_array(); Ok((a.iter().map(|&x| x!=0).collect(), a.shape().to_vec()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("cond must be bool/uint8/uint32"))
    }
}

#[pyfunction]
fn where_nd_py(py: Python<'_>, cond: &PyAny, x: PyReadonlyArrayDyn<f32>, y: PyReadonlyArrayDyn<f32>, device: Option<&str>) -> PyResult<Py<PyArrayDyn<f32>>> {
    let (cond_bool, cshape) = bool_like_to_vec(cond)?;
    let ax = x.as_array(); let ay = y.as_array();
    let (x_v, xshape) = (ax.iter().cloned().collect::<Vec<f32>>(), ax.shape().to_vec());
    let (y_v, yshape) = (ay.iter().cloned().collect::<Vec<f32>>(), ay.shape().to_vec());
    let strides = |shape:&[usize]|{ let mut st=vec![0usize;shape.len()]; let mut acc=1usize; for d in (0..shape.len()).rev(){ st[d]=acc; acc*=shape[d]; } st };
    let cstrides = strides(&cshape);
    let xstrides = strides(&xshape);
    let ystrides = strides(&yshape);

    // broadcast out shape
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

    // device "auto" with capability-aware fallback
    let mut want = device.unwrap_or("auto").to_string();
    if want=="auto" {
        if st_core::ops::public_api::wgpu_is_available() { want="wgpu".into(); }
        else if st_core::ops::public_api::cuda_is_available() && st_core::ops::public_api::supports_where_nd_cuda() { want="cuda".into(); }
        else if st_core::ops::public_api::mps_is_available()  && st_core::ops::public_api::supports_where_nd_mps()  { want="mps".into(); }
        else { want="cpu".into(); }
    }

    let use_wgpu = want=="wgpu";
    let out = st_core::ops::public_api::where_nd_host_use_wgpu_strided(
        &cond_bool, &cshape, &cstrides, &x_v, &xshape, &xstrides, &y_v, &yshape, &ystrides, &out_shape, use_wgpu
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
    let arr = PyArrayDyn::<f32>::from_vec(py, out);
    arr.reshape(out_shape.as_slice()).unwrap();
    Ok(arr.to_owned())
}

#[pyfunction]
fn topk2d_py(py: Python<'_>, x: PyReadonlyArrayDyn<f32>, k: usize, device: Option<&str>) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<i32>>)> {
    let a = x.as_array();
    if a.ndim()!=2 { return Err(pyo3::exceptions::PyValueError::new_err("topk2d: x must be 2D [rows, cols]")); }
    let rows = a.shape()[0]; let cols = a.shape()[1];
    let x_v: Vec<f32> = a.iter().cloned().collect();

    let mut want = device.unwrap_or("auto").to_string();
    if want=="auto" {
        if st_core::ops::public_api::wgpu_is_available() { want="wgpu".into(); }
        else if st_core::ops::public_api::cuda_is_available() && st_core::ops::public_api::supports_topk_cuda() { want="cuda".into(); }
        else if st_core::ops::public_api::mps_is_available()  && st_core::ops::public_api::supports_topk_mps()  { want="mps".into(); }
        else { want="cpu".into(); }
    }
    let use_wgpu = want=="wgpu";

    let (vals, idxs) = st_core::ops::public_api::topk_lastdim_host_use_wgpu(&x_v, rows, cols, k, use_wgpu)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
    let va = PyArray2::<f32>::from_vec2(py, &vals.chunks(k).map(|c| c.to_vec()).collect::<Vec<_>>()).unwrap();
    let ia = PyArray2::<i32>::from_vec2(py, &idxs.chunks(k).map(|c| c.to_vec()).collect::<Vec<_>>()).unwrap();
    Ok((va.to_owned(), ia.to_owned()))
}

#[pymodule]
fn spiraltorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "1.3.30")?;
    m.add_function(wrap_pyfunction!(has_wgpu_py, m)?)?;
    m.add_function(wrap_pyfunction!(has_cuda_py, m)?)?;
    m.add_function(wrap_pyfunction!(has_mps_py, m)?)?;
    m.add_function(wrap_pyfunction!(where_nd_py, m)?)?;
    m.add_function(wrap_pyfunction!(topk2d_py, m)?)?;
    Ok(())
}
