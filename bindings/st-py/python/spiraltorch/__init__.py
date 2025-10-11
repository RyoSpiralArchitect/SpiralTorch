
from .spiraltorch import where_nd_py as where_nd, topk2d_py as _topk2d, has_wgpu_py as has_wgpu, has_cuda_py as has_cuda, has_mps_py as has_mps
import numpy as _np

def topk(x: _np.ndarray, k: int, dim: int = -1, device: str | None = "auto"):
    """
    N-D TopK along axis `dim`. Returns (values, indices) with the same shape as x except dim->k.
    device: "wgpu"|"cuda"|"mps"|"cpu"|"auto"
    """
    x = _np.asarray(x, dtype=_np.float32, order="C")
    if dim < 0: dim = x.ndim + dim
    if dim < 0 or dim >= x.ndim:
        raise ValueError("dim out of range")
    x_m = _np.moveaxis(x, dim, -1)
    rows = int(_np.prod(x_m.shape[:-1])) if x_m.ndim>1 else 1
    cols = x_m.shape[-1]
    x2 = x_m.reshape(rows, cols)
    dev = device if device in ("wgpu","cuda","mps","cpu") else ("wgpu" if has_wgpu() else ("cuda" if has_cuda() else ("mps" if has_mps() else "cpu")))
    vals2, idx2 = _topk2d(x2, k, dev)
    vals2 = _np.asarray(vals2, dtype=_np.float32)
    idx2  = _np.asarray(idx2, dtype=_np.int32)
    out_shape = list(x_m.shape[:-1]) + [k]
    vals = vals2.reshape(out_shape)
    idx  = idx2.reshape(out_shape)
    vals = _np.moveaxis(vals, -1, dim)
    idx  = _np.moveaxis(idx,  -1, dim)
    return vals, idx

__all__ = ["where_nd", "topk", "has_wgpu", "has_cuda", "has_mps"]
