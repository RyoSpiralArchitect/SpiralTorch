use ndarray::{ArrayD, IxDyn};

/// Linear interpolation gather along the last dimension.
/// x: [..., N], idx: [..., M] with float indices in [0, N-1].
pub fn gather_real_lastdim(x:&ArrayD<f32>, idx:&ArrayD<f32>) -> ArrayD<f32> {
    let mut out_shape = x.shape().to_vec();
    let m = *idx.shape().last().unwrap();
    *out_shape.last_mut().unwrap() = m;
    let mut y = ArrayD::<f32>::zeros(IxDyn(&out_shape));
    let n = *x.shape().last().unwrap() as isize;
    for (out_ix, yy) in y.indexed_iter_mut() {
        let mut src_ix = out_ix.to_vec();
        let j = *src_ix.last().unwrap() as isize;
        let mut idx_ix = src_ix.clone();
        let _ = idx_ix.pop();
        idx_ix.push(j as usize);
        let tf = *idx.get(IxDyn(&idx_ix)).unwrap();
        let i0 = tf.floor().max(0.0) as isize;
        let i1 = (i0+1).min(n-1);
        let alpha = tf - (i0 as f32);
        src_ix[src_ix.len()-1] = i0 as usize;
        let v0 = *x.get(IxDyn(&src_ix)).unwrap();
        src_ix[src_ix.len()-1] = i1 as usize;
        let v1 = *x.get(IxDyn(&src_ix)).unwrap();
        *yy = (1.0-alpha)*v0 + alpha*v1;
    }
    y
}

/// Backward: returns (gx, gidx)
pub fn gather_real_lastdim_backward(x:&ArrayD<f32>, idx:&ArrayD<f32>, gy:&ArrayD<f32>) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut gx = ArrayD::<f32>::zeros(IxDyn(x.shape()));
    let mut gidx = ArrayD::<f32>::zeros(IxDyn(idx.shape()));
    let n = *x.shape().last().unwrap() as isize;
    let m = *idx.shape().last().unwrap();
    let mut yshape = x.shape().to_vec();
    *yshape.last_mut().unwrap() = m;
    assert_eq!(gy.shape(), &yshape[..]);
    for (out_ix, gyv) in gy.indexed_iter() {
        let mut src_ix = out_ix.to_vec();
        let j = *src_ix.last().unwrap() as isize;
        let mut idx_ix = src_ix.clone();
        let _ = idx_ix.pop();
        idx_ix.push(j as usize);
        let tf = *idx.get(IxDyn(&idx_ix)).unwrap();
        let i0 = tf.floor().max(0.0) as isize;
        let i1 = (i0+1).min(n-1);
        let alpha = tf - (i0 as f32);
        // gx
        src_ix[src_ix.len()-1] = i0 as usize;
        *gx.get_mut(IxDyn(&src_ix)).unwrap() += (1.0 - alpha) * *gyv;
        src_ix[src_ix.len()-1] = i1 as usize;
        *gx.get_mut(IxDyn(&src_ix)).unwrap() += alpha * *gyv;
        // gidx
        let mut xix = out_ix.to_vec();
        xix[xix.len()-1] = i0 as usize;
        let v0 = *x.get(IxDyn(&xix)).unwrap();
        xix[xix.len()-1] = i1 as usize;
        let v1 = *x.get(IxDyn(&xix)).unwrap();
        let mut ixi = out_ix.to_vec();
        let _ = ixi.pop();
        ixi.push(j as usize);
        *gidx.get_mut(IxDyn(&ixi)).unwrap() += *gyv * (v1 - v0);
    }
    (gx, gidx)
}
