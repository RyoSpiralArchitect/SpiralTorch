
from .spiraltorch import where_nd_py as where_nd, topk2d_py as _topk2d, has_wgpu_py as has_wgpu, has_cuda_py as has_cuda, has_mps_py as has_mps
import numpy as _np, os as _os

_DEVICE_ORDER = None
def _parse_order(s: str):
    order = [t.strip().lower() for t in s.split(",")]
    seen, seq = set(), []
    for t in order:
        if t in ("cuda","mps","wgpu","cpu") and t not in seen:
            seen.add(t); seq.append(t)
    for t in ("cuda","mps","wgpu","cpu"):
        if t not in seen: seq.append(t)
    return seq
def _detect_default_order():
    if has_cuda(): return ["cuda","mps","wgpu","cpu"]
    if has_mps():  return ["mps","wgpu","cpu"]
    if has_wgpu(): return ["wgpu","cpu"]
    return ["cpu"]
def set_device_order(order: str | list[str]):
    global _DEVICE_ORDER
    if isinstance(order, str): _DEVICE_ORDER = _parse_order(order)
    else: _DEVICE_ORDER = _parse_order(",".join(order))
def get_device_order():
    global _DEVICE_ORDER
    if _DEVICE_ORDER is None:
        env = _os.environ.get("SPIRALTORCH_DEVICE_ORDER")
        _DEVICE_ORDER = _parse_order(env) if env else _detect_default_order()
    return list(_DEVICE_ORDER)
def _pick_device(user: str | None):
    if user in ("cuda","mps","wgpu","cpu"): return user
    order = get_device_order()
    for dev in order:
        if dev=="cuda" and has_cuda(): return "cuda"
        if dev=="mps"  and has_mps():  return "mps"
        if dev=="wgpu" and has_wgpu(): return "wgpu"
        if dev=="cpu": return "cpu"
    return "cpu"
def topk(x: _np.ndarray, k: int, dim: int = -1, device: str | None = "auto"):
    x = _np.asarray(x, dtype=_np.float32, order="C")
    if dim < 0: dim = x.ndim + dim
    if dim < 0 or dim >= x.ndim: raise ValueError("dim out of range")
    x_m = _np.moveaxis(x, dim, -1)
    rows = int(_np.prod(x_m.shape[:-1])) if x_m.ndim>1 else 1
    cols = x_m.shape[-1]
    x2 = x_m.reshape(rows, cols)
    dev = _pick_device(device)
    vals2, idx2 = _topk2d(x2, k, dev)
    vals2 = _np.asarray(vals2, dtype=_np.float32)
    idx2  = _np.asarray(idx2, dtype=_np.int32)
    out_shape = list(x_m.shape[:-1]) + [k]
    vals = vals2.reshape(out_shape)
    idx  = idx2.reshape(out_shape)
    vals = _np.moveaxis(vals, -1, dim)
    idx  = _np.moveaxis(idx,  -1, dim)
    return vals, idx
__all__ = ["where_nd", "topk", "has_wgpu", "has_cuda", "has_mps", "set_device_order", "get_device_order"]
