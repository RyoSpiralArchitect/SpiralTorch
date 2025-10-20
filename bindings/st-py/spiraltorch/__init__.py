from __future__ import annotations
import sys, types as _types
from importlib import import_module
from typing import Any as _Any
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
    "capture","share","compat",
    "fibonacci_pacing","pack_nacci_chunks",
    "pack_tribonacci_chunks","pack_tetranacci_chunks",
    "generate_plan_batch_ex","plan","plan_topk",
    "describe_device","hip_probe","z_space_barycenter",
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
def _ensure_submodule(name: str, doc: str = "") -> _types.ModuleType:
    """Return an existing or synthetic child module.

    The stub modules keep ``import spiraltorch.<name>`` from escaping to
    unrelated third-party packages while we gradually expose more bindings
    from the Rust extension.
    """

    fq = f"{__name__}.{name}"
    module = getattr(_rs, name, None)
    if isinstance(module, _types.ModuleType):
        # Bind the real Rust powered submodule directly and make sure the
        # absolute import machinery can still find it.
        if not module.__doc__:
            module.__doc__ = doc
        sys.modules[fq] = module
    else:
        module = sys.modules.get(fq)
        if module is None:
            module = _types.ModuleType(fq, doc)
            sys.modules[fq] = module
    globals()[name] = module
    return module

def _expose_from_rs(name: str) -> None:
    if name in globals():
        return
    if hasattr(_rs, name):
        globals()[name] = getattr(_rs, name)

def _mirror_into_module(name: str, members: list[str]) -> _types.ModuleType:
    module = _ensure_submodule(name)
    exported: set[str] = set(getattr(module, "__all__", ()))
    for member in members:
        _expose_from_rs(member)
        value = globals().get(member)
        if value is None:
            continue
        setattr(module, member, value)
        exported.add(member)
    if exported:
        module.__all__ = sorted(exported)
    return module


for _name, _doc in [
    ("nn","SpiralTorch neural network primitives"),
    ("frac","Fractal & fractional tools"),
    ("dataset","Datasets & loaders"),
    ("linalg","Linear algebra utilities"),
    ("rl","Reinforcement learning components"),
    ("rec","Reconstruction / signal processing"),
    ("telemetry","Telemetry / dashboards / metrics"),
    ("ecosystem","Integrations & ecosystem glue"),
    ("selfsup","Self-supervised objectives"),
    ("export","Model export & compression"),
    ("compat","Interoperability bridges"),
    ("hpo","Hyper-parameter optimization tools"),
    ("inference","Safety inference runtime & auditing"),
]:
    _ensure_submodule(_name, _doc)


_mirror_into_module(
    "inference",
    [
        "SafetyViolation","SafetyVerdict","AuditEvent","AuditLog",
        "InferenceResult","InferenceRuntime",
    ],
)


_CORE_EXPORTS = [
    "Tensor","ComplexTensor","OpenCartesianTopos","LanguageWaveEncoder",
    "GradientSummary","Hypergrad","TensorBiome",
    "BarycenterIntermediate","ZSpaceBarycenter",
    "QueryPlan","RecEpochReport","Recommender",
    "DqnAgent","PpoAgent","SacAgent",
    "DashboardMetric","DashboardEvent","DashboardFrame","DashboardRing",
    "AuditEvent","AuditLog","InferenceResult","InferenceRuntime",
    "SafetyVerdict","SafetyViolation",
]
for _name in _CORE_EXPORTS:
    _expose_from_rs(_name)


def __getattr__(name: str) -> _Any:
    """Defer missing attributes to the Rust extension module.

    This keeps the Python façade lightweight while still exposing the rich
    surface area implemented in Rust.
    """

    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    _expose_from_rs(name)
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    _public = set(__all__)
    _public.update(n for n in dir(_rs) if not n.startswith("_"))
    return sorted(_public)


_EXPORTED = {
    *_EXTRAS,
    *_CORE_EXPORTS,
    *[n for n in _COMPAT_ALIAS if n in globals()],
    "nn","frac","dataset","linalg","rl","rec","telemetry","ecosystem",
    "selfsup","export","compat","hpo","inference",
    "__version__",
}
_EXPORTED.update(
    n for n in getattr(_rs, "__all__", ())
    if isinstance(n, str) and not n.startswith("_")
)
__all__ = sorted(_EXPORTED)
