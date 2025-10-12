use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayDyn, IntoPyArray};

#[inline]
fn rowmajor_strides(shape:&[usize]) -> Vec<usize> {
    let nd = shape.len();
    let mut st = vec![0usize; nd];
    let mut acc = 1usize;
    for d in (0..nd).rev() {
        st[d] = acc;
        acc = acc.saturating_mul(shape[d].max(1));
    }
    st
}

/// Canonicalize negative strides: move base to "first element in forward traversal" and make stride positive.
fn canonicalize_neg_strides(shape:&[usize], strides:&[isize], itemsize:usize, base_ptr:usize) -> (usize, Vec<usize>) {
    let nd = shape.len();
    let mut base = base_ptr;
    let mut st = vec![0usize; nd];
    for d in 0..nd {
        if strides[d] < 0 {
            if shape[d] > 0 {
                base = base + ((shape[d]-1) as isize * strides[d].abs()) as usize * itemsize;
            }
            st[d] = (strides[d].abs() as usize);
        } else {
            st[d] = (strides[d] as usize);
        }
    }
    (base, st)
}

/// Enumerate contiguous tail length (largest suffix where stride matches rowmajor)
fn contiguous_tail_len(shape:&[usize], strides:&[usize], itemsize:usize) -> usize {
    let nd = shape.len();
    if nd == 0 { return 0; }
    let mut len = 0usize;
    let mut expect = itemsize;
    for d in (0..nd).rev() {
        if shape[d] == 1 || strides[d] == expect {
            len += 1;
            expect = expect.saturating_mul(shape[d].max(1));
        } else {
            break;
        }
    }
    len
}

/// Generate segments: iterate outer (non-tail) indices, and produce contiguous segments on the tail.
fn enumerate_segments(shape:&[usize], strides:&[usize], itemsize:usize, base:usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, usize) {
    let nd = shape.len();
    let tail = contiguous_tail_len(shape, strides, itemsize);
    let tail_elems = if tail==0 { 1 } else { shape[nd-tail..].iter().product::<usize>().max(1) };
    let tail_bytes = tail_elems * itemsize;
    // number of outer positions
    let outer_dims = if nd>tail { &shape[..nd-tail] } else { &[] };
    let outer_strides = if nd>tail { &strides[..nd-tail] } else { &[] };
    let mut offsets = Vec::new();  // device buffer offsets (destination)
    let mut sizes   = Vec::new();  // bytes per segment
    let mut starts  = Vec::new();  // source host blob offset in bytes
    // iterate outer coords
    let mut idx = vec![0usize; outer_dims.len()];
    let mut total_segments = 0usize;
    loop {
        // compute host offset
        let mut off = base;
        for (d, &len_d) in outer_dims.iter().enumerate() {
            if len_d > 1 {
                off += idx[d] * outer_strides[d];
            }
        }
        // record segment
        offsets.push((total_segments * tail_bytes) as u32);
        sizes.push(tail_bytes as u32);
        starts.push(off as u32);
        total_segments += 1;
        // increment
        if outer_dims.is_empty() { break; }
        let mut p = outer_dims.len();
        while p>0 {
            p -= 1;
            idx[p] += 1;
            if idx[p] < outer_dims[p] { break; }
            idx[p] = 0;
            if p==0 { p = 0; break; }
        }
        if p==0 && idx[0]==0 { break; }
    }
    let total_bytes = total_segments * tail_bytes;
    (offsets, sizes, starts, total_bytes)
}

#[pyfunction]
fn topk2d_py<'py>(py: Python<'py>, x: &'py PyArray2<f32>, k: usize, device: Option<&str>) -> PyResult<(&'py PyArray2<f32>, &'py PyArray2<i32>)> {
    let xv = unsafe { x.as_slice()? };
    let shape = x.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let want = device.unwrap_or("auto");
    let want = if want=="auto" {
        #[cfg(feature="cuda")] { "cuda" }
        #[cfg(all(not(feature="cuda"), feature="mps"))] { "mps" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), feature="wgpu"))] { "wgpu" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), not(feature="wgpu")))] { "cpu" }
    } else { want };
    let (vals, idxs) = match want {
        "cuda" => {
            #[cfg(feature="cuda")]
            { st_core::ops::public_api::topk_lastdim_cuda_2d(xv, rows, cols, k) }
            #[cfg(not(feature="cuda"))]
            { Err(st_core::error::device("cuda feature not enabled")) }
        }
        "wgpu" => {
            #[cfg(feature="wgpu")]
            { st_core::ops::public_api::topk_lastdim_wgpu_2d_autotuned(xv, rows, cols, k) }
            #[cfg(not(feature="wgpu"))]
            { Err(st_core::error::device("wgpu feature not enabled")) }
        }
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
        #[cfg(feature="cuda")] { "cuda" } // (CUDA path for where not implemented yet; falls back internally)
        #[cfg(all(not(feature="cuda"), feature="mps"))] { "mps" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), feature="wgpu"))] { "wgpu" }
        #[cfg(all(not(feature="cuda"), not(feature="mps"), not(feature="wgpu")))] { "cpu" }
    } else { want };

    let c_shape: Vec<usize> = cond.shape().into();
    let x_shape: Vec<usize> = x.shape().into();
    let y_shape: Vec<usize> = y.shape().into();
    if x_shape != y_shape { return Err(pyo3::exceptions::PyValueError::new_err("x and y must match shape")); }
    let out_shape = x_shape.clone();

    // Raw pointers & strides (bytes)
    let c_ptr = cond.data() as usize;
    let x_ptr = x.data() as usize;
    let y_ptr = y.data() as usize;

    let c_strides_el = cond.strides().to_vec();
    let x_strides_el = x.strides().to_vec();
    let y_strides_el = y.strides().to_vec();

    // Canonicalize negative strides & compute base (bytes)
    let (c_base, c_strides_b) = canonicalize_neg_strides(&c_shape, &c_strides_el, 1, c_ptr);
    let (x_base, x_strides_b) = canonicalize_neg_strides(&x_shape, &x_strides_el, 4, x_ptr);
    let (y_base, y_strides_b) = canonicalize_neg_strides(&y_shape, &y_strides_el, 4, y_ptr);

    // Enumerate segments with no repack: cond u8 / x,y f32
    let (coff, csz, cstart, ctotal) = enumerate_segments(&c_shape, &c_strides_b, 1, c_base);
    let (xoff, xsz, xstart, xtotal) = enumerate_segments(&x_shape, &x_strides_b, 4, x_base);
    let (yoff, ysz, ystart, ytotal) = enumerate_segments(&y_shape, &y_strides_b, 4, y_base);

    // Create borrowed slices to original buffers
    let c_bytes = unsafe { std::slice::from_raw_parts(c_ptr as *const u8, ctotal) };
    let x_bytes = unsafe { std::slice::from_raw_parts(x_ptr as *const u8, xtotal) };
    let y_bytes = unsafe { std::slice::from_raw_parts(y_ptr as *const u8, ytotal) };

    let cstrides_el_pos: Vec<usize> = c_strides_el.iter().map(|&s| s.unsigned_abs() as usize).collect();
    let xstrides_el_pos: Vec<usize> = x_strides_el.iter().map(|&s| s.unsigned_abs() as usize).collect();
    let ystrides_el_pos: Vec<usize> = y_strides_el.iter().map(|&s| s.unsigned_abs() as usize).collect();

    let res: Vec<f32> = match want {
        "wgpu" => {
            #[cfg(feature="wgpu")]
            { st_core::ops::public_api::where_nd_host_select_strided_segments_bytes(
                ctotal, &coff, &csz, &cstart, c_bytes,
                &c_shape, &cstrides_el_pos, (c_base - c_ptr), // base elems (bytes/elem=1)
                xtotal, &xoff, &xsz, &xstart, x_bytes,
                &x_shape, &xstrides_el_pos, (x_base - x_ptr)/4,
                ytotal, &yoff, &ysz, &ystart, y_bytes,
                &y_shape, &ystrides_el_pos, (y_base - y_ptr)/4,
                &out_shape, "wgpu"
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))? }
            #[cfg(not(feature="wgpu"))]
            { return Err(pyo3::exceptions::PyRuntimeError::new_err("wgpu feature not enabled")); }
        }
        "mps" => {
            #[cfg(feature="mps")]
            {
                let out_strides = rowmajor_strides(&out_shape);
                st_core::ops::public_api::where_nd_host_select_strided_segments_bytes_mps(
                    ctotal, &coff, &csz, &cstart, c_bytes,
                    xtotal, &xoff, &xsz, &xstart, x_bytes,
                    ytotal, &yoff, &ysz, &ystart, y_bytes,
                    &out_shape, &out_strides,
                    &c_shape, &cstrides_el_pos, (c_base - c_ptr),
                    &x_shape, &xstrides_el_pos, (x_base - x_ptr)/4,
                    &y_shape, &ystrides_el_pos, (y_base - y_ptr)/4
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?
            }
            #[cfg(not(feature="mps"))]
            { return Err(pyo3::exceptions::PyRuntimeError::new_err("mps feature not enabled")); }
        }
        _ => {
            // CPU fallback
            let nelem = out_shape.iter().product::<usize>();
            let mut out = vec![0f32; nelem];
            // naive CPU where for demo
            for i in 0..nelem {
                let cb = unsafe { *c_bytes.get_unchecked((c_base - c_ptr) + i) };
                let xv = f32::from_le_bytes([x_bytes[4*i],x_bytes[4*i+1],x_bytes[4*i+2],x_bytes[4*i+3]]);
                let yv = f32::from_le_bytes([y_bytes[4*i],y_bytes[4*i+1],y_bytes[4*i+2],y_bytes[4*i+3]]);
                out[i] = if cb!=0 { xv } else { yv };
            }
            out
        }
    };
    let ary = PyArrayDyn::<f32>::from_vec(py, res);
    ary.reshape(out_shape.as_slice()).unwrap();
    Ok(ary)
}

#[pymodule]
fn spiraltorch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(topk2d_py, m)?)?;
    m.add_function(wrap_pyfunction!(where_nd_py, m)?)?;
    m.add("__version__", "1.3.56")?;
    Ok(())
}
