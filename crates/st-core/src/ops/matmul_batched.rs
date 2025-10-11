use crate::{Tensor, device::Device, error::Result};

pub fn matmul_batched3d(a:&Tensor, b:&Tensor) -> Result<Tensor> {
    let sa = a.shape(); let sb = b.shape();
    if sa.len()!=3 || sb.len()!=3 { return Err(crate::error::msg("matmul_batched3d expects [B,M,K]@[B,K,N]")); }
    let (ba,m,k) = (sa[0], sa[1], sa[2]); let (bb,k2,n) = (sb[0], sb[1], sb[2]);
    if k!=k2 { return Err(crate::error::msg("inner dim mismatch")); }
    if ba!=bb && ba!=1 && bb!=1 { return Err(crate::error::msg("batch mismatch (broadcast only 1)")); }

    // CPU fallback for now (broadcast)
    let a_np = a.data(); let b_np = b.data();
    let mut out = ndarray::Array3::<f32>::zeros(ndarray::Ix3(ba.max(bb), m, n));
    for i in 0..ba.max(bb) {
        let ai = if ba==1 { a_np.index_axis(ndarray::Axis(0), 0).to_owned() } else { a_np.index_axis(ndarray::Axis(0), i).to_owned() };
        let bi = if bb==1 { b_np.index_axis(ndarray::Axis(0), 0).to_owned() } else { b_np.index_axis(ndarray::Axis(0), i).to_owned() };
        let mut ci = out.index_axis_mut(ndarray::Axis(0), i);
        ndarray::linalg::general_mat_mul(1.0, &ai, &bi, 0.0, &mut ci);
    }
    Ok(Tensor::from_array(out.into_dyn()))
}
