from __future__ import annotations
import sys, types as _types
from importlib import import_module
from typing import Any as _Any, Iterable as _Iterable, Mapping as _Mapping
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

_FORWARDING_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "nn": {
        "Dataset": ("_NnDataset",),
        "DataLoader": ("_NnDataLoader",),
        "DataLoaderIter": ("_NnDataLoaderIter",),
        "from_samples": ("nn_from_samples", "dataset_from_samples"),
    },
    "compat.torch": {
        "to_torch": ("compat_to_torch", "to_torch"),
        "from_torch": ("compat_from_torch", "from_torch"),
    },
    "compat.jax": {
        "to_jax": ("compat_to_jax", "to_jax"),
        "from_jax": ("compat_from_jax", "from_jax"),
    },
    "compat.tensorflow": {
        "to_tensorflow": ("compat_to_tensorflow", "to_tensorflow"),
        "from_tensorflow": ("compat_from_tensorflow", "from_tensorflow"),
    },
}


class _ForwardingModule(_types.ModuleType):
    """Module stub that forwards attribute lookups to the Rust backend."""

    def __init__(self, name: str, doc: str, key: str) -> None:
        super().__init__(name, doc)
        self.__dict__["_forward_key"] = key

    @property
    def _forward_key(self) -> str:
        return self.__dict__["_forward_key"]

    def __getattr__(self, attr: str) -> _Any:
        if attr.startswith("_"):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{attr}'")

        # Prefer already-exposed globals so top-level mirrors stay consistent.
        if attr in globals():
            value = globals()[attr]
            setattr(self, attr, value)
            _register_module_export(self, attr)
            return value

        hints = _FORWARDING_HINTS.get(self._forward_key, {})
        candidates: list[str] = []
        aliases = hints.get(attr)
        if aliases:
            candidates.extend(aliases)

        namespace_parts = self._forward_key.split(".")
        suffix = namespace_parts[-1]
        flat_suffix = "_".join(namespace_parts)
        candidates.extend(
            [
                attr,
                f"{suffix}_{attr}",
                f"{suffix}_{attr.lower()}",
                f"{flat_suffix}_{attr}",
                f"{flat_suffix}_{attr.lower()}",
            ]
        )

        for candidate in dict.fromkeys(candidates):
            if hasattr(_rs, candidate):
                value = getattr(_rs, candidate)
                setattr(self, attr, value)
                _register_module_export(self, attr)
                return value

        raise AttributeError(f"module '{self.__name__}' has no attribute '{attr}'")

    def __dir__(self) -> list[str]:
        exported = set(getattr(self, "__all__", ()))
        exported.update(super().__dir__())
        hints = _FORWARDING_HINTS.get(self._forward_key, {})
        exported.update(hints.keys())
        suffix = self._forward_key.split(".")[-1] + "_"
        flat_suffix = "_".join(self._forward_key.split(".")) + "_"
        for name in dir(_rs):
            if name.startswith(suffix):
                exported.add(name[len(suffix):])
            elif name.startswith(flat_suffix):
                exported.add(name[len(flat_suffix):])
        return sorted(exported)


def _register_module_export(module: _types.ModuleType, name: str) -> None:
    exported = set(getattr(module, "__all__", ()))
    exported.add(name)
    module.__all__ = sorted(exported)


def _ensure_submodule(name: str, doc: str = "") -> _types.ModuleType:
    """Return an existing or synthetic child module (supports dotted paths)."""

    parts = name.split(".")
    fq = __name__
    parent: _types.ModuleType = sys.modules[__name__]
    for idx, part in enumerate(parts):
        fq = f"{fq}.{part}"
        module = sys.modules.get(fq)
        final = idx == len(parts) - 1
        doc_for_part = doc if final else ""
        if module is None:
            candidate = getattr(parent, part, None)
            if not isinstance(candidate, _types.ModuleType):
                candidate = getattr(_rs, part, None) if idx == 0 else None
            if isinstance(candidate, _types.ModuleType):
                module = candidate
                if doc_for_part and not getattr(module, "__doc__", None):
                    module.__doc__ = doc_for_part
            else:
                key = ".".join(parts[: idx + 1])
                module = _ForwardingModule(fq, doc_for_part, key)
            sys.modules[fq] = module
        elif doc_for_part and not getattr(module, "__doc__", None):
            module.__doc__ = doc_for_part

        setattr(parent, part, module)
        if idx == 0:
            globals()[part] = module
        parent = module
    return parent


def _expose_from_rs(name: str, *aliases: str) -> None:
    if name in globals():
        return
    for candidate in (name, *aliases):
        if hasattr(_rs, candidate):
            globals()[name] = getattr(_rs, candidate)
            return


def _mirror_into_module(
    name: str,
    members: _Iterable[str] | _Mapping[str, _Iterable[str]]
) -> _types.ModuleType:
    module = _ensure_submodule(name)
    exported: set[str] = set(getattr(module, "__all__", ()))
    items: _Iterable[tuple[str, _Iterable[str]]] \
        = members.items() if isinstance(members, _Mapping) else ((m, ()) for m in members)
    for member, aliases in items:
        _expose_from_rs(member, *aliases)
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


_compat_children = {
    "torch": "PyTorch interoperability helpers",
    "jax": "JAX interoperability helpers",
    "tensorflow": "TensorFlow interoperability helpers",
}
for _child, _doc in _compat_children.items():
    _ensure_submodule(f"compat.{_child}", _doc)
_compat_module = globals().get("compat")
if isinstance(_compat_module, _types.ModuleType):
    _compat_exports = set(getattr(_compat_module, "__all__", ()))
    _compat_exports.update(_compat_children.keys())
    _compat_module.__all__ = sorted(_compat_exports)


_mirror_into_module(
    "inference",
    [
        "SafetyViolation","SafetyVerdict","AuditEvent","AuditLog",
        "InferenceResult","InferenceRuntime",
    ],
)


_mirror_into_module(
    "nn",
    {
        "Dataset": ("_NnDataset",),
        "DataLoader": ("_NnDataLoader",),
        "DataLoaderIter": ("_NnDataLoaderIter",),
        "from_samples": ("nn_from_samples", "dataset_from_samples"),
    },
)
_mirror_into_module(
    "frac",
    [
        "gl_coeffs_adaptive",
        "fracdiff_gl_1d",
    ],
)
_mirror_into_module(
    "rl",
    [
        "DqnAgent",
        "PpoAgent",
        "SacAgent",
    ],
)
_mirror_into_module(
    "rec",
    [
        "QueryPlan",
        "RecEpochReport",
        "Recommender",
    ],
)
_mirror_into_module(
    "telemetry",
    [
        "DashboardMetric",
        "DashboardEvent",
        "DashboardFrame",
        "DashboardRing",
    ],
)


for _key, _hint in _FORWARDING_HINTS.items():
    _module = _ensure_submodule(_key)
    if not _hint:
        continue
    _exports = set(getattr(_module, "__all__", ()))
    _exports.update(_hint.keys())
    _module.__all__ = sorted(_exports)


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
