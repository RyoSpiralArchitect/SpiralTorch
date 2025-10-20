from __future__ import annotations
import sys, types as _types
from importlib import import_module
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Rust拡張の本体
try:
    _rs = import_module("spiraltorch.spiraltorch")
except ModuleNotFoundError as exc:
    if exc.name not in {"spiraltorch.spiraltorch", "spiraltorch"}:
        raise
    try:
        _rs = import_module("spiraltorch.spiraltorch_native")
    except ModuleNotFoundError:
        _rs = import_module("spiraltorch_native")

# パッケージ版
try:
    __version__ = _pkg_version("spiraltorch")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

# 追加API（Rust側でエクスポート済みのやつだけ拾う）
_EXTRAS = [
    "golden_ratio","golden_angle","set_global_seed",
    "fibonacci_pacing","pack_nacci_chunks",
    "pack_tribonacci_chunks","pack_tetranacci_chunks",
    "generate_plan_batch_ex",
]
for _n in _EXTRAS:
    if hasattr(_rs, _n):
        globals()[_n] = getattr(_rs, _n)

# 後方互換の別名（存在する方を公開名にバインド）
_COMPAT_ALIAS = {
    "Tensor":   ("Tensor", "PyTensor"),
    "Device":   ("Device", "PyDevice"),
    "Dataset":  ("Dataset", "PyDataset"),
    "Plan":     ("Plan", "PyPlan"),
}
for _pub, _cands in _COMPAT_ALIAS.items():
    for _c in _cands:
        if hasattr(_rs, _c):
            globals()[_pub] = getattr(_rs, _c)
            break

# 空サブモジュール（将来ここに実装をぶら下げる）
def _ensure_submodule(name: str, doc: str = ""):
    fq = f"{__name__}.{name}"
    if fq not in sys.modules:
        m = _types.ModuleType(fq, doc)
        sys.modules[fq] = m
    return sys.modules[fq]

for _name, _doc in [
    ("nn","SpiralTorch neural network primitives"),
    ("frac","Fractal & fractional tools"),
    ("dataset","Datasets & loaders"),
    ("linalg","Linear algebra utilities"),
    ("rl","Reinforcement learning components"),
    ("rec","Reconstruction / signal processing"),
    ("telemetry","Telemetry / dashboards / metrics"),
    ("ecosystem","Integrations & ecosystem glue"),
]:
    _ensure_submodule(_name, _doc)

__all__ = sorted({
    *__EXTRAS,
    *[n for n in _COMPAT_ALIAS if n in globals()],
    "nn","frac","dataset","linalg","rl","rec","telemetry","ecosystem",
    "__version__",
})
