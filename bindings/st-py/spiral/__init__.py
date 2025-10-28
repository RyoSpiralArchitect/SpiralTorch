"""Unified SpiralTorch high-level helpers."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Iterable

from . import cli as cli  # noqa: F401 - re-exported via __all__
from .export import (
    DeploymentTarget,
    ExportConfig,
    ExportPipeline,
    load_benchmark_report,
)

__all__: list[str] = [
    "DeploymentTarget",
    "ExportConfig",
    "ExportPipeline",
    "load_benchmark_report",
    "cli",
]


def _register(name: str, value: object) -> None:
    if name.startswith("_"):
        return
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)


def _merge_public_members(module: ModuleType) -> None:
    exports: Iterable[str] | None = getattr(module, "__all__", None)
    if not exports:
        exports = (name for name in dir(module) if not name.startswith("_"))
    for name in exports:
        value = getattr(module, name, None)
        if value is not None:
            _register(name, value)


def _rebind_module(module: ModuleType, *, relative_name: str) -> None:
    target_name = f"{__name__}.{relative_name}"
    module.__name__ = target_name
    if "." in relative_name:
        module.__package__ = f"{__name__}.{relative_name.rsplit('.', 1)[0]}"
    else:
        module.__package__ = __name__
    sys.modules[target_name] = module
    if "." not in relative_name and not hasattr(sys.modules[__name__], relative_name):
        setattr(sys.modules[__name__], relative_name, module)


def _bridge_pure_python_package() -> None:
    base_dir = pathlib.Path(__file__).resolve().parents[2] / "python" / "spiral"
    init_py = base_dir / "__init__.py"
    if not init_py.exists():
        return

    if importlib.util.find_spec("numpy") is None:
        return

    import spiraltorch as _spiraltorch  # local import to avoid circularity

    if not all(hasattr(_spiraltorch, attr) for attr in ("inference", "hypergrad")):
        return

    bridge_name = "_spiral_py_bridge"
    spec = importlib.util.spec_from_file_location(
        bridge_name,
        init_py,
        submodule_search_locations=[str(base_dir)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    module.__package__ = bridge_name
    module.__path__ = [str(base_dir)]
    sys.modules[bridge_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        sys.modules.pop(bridge_name, None)
        return

    _merge_public_members(module)

    prefix = f"{bridge_name}."
    for name, value in list(sys.modules.items()):
        if name == bridge_name or not name.startswith(prefix):
            continue
        if not isinstance(value, ModuleType):
            continue
        relative_name = name[len(prefix) :]
        _rebind_module(value, relative_name=relative_name)

    for submodule in ("data", "hypergrad", "inference"):
        target = f"{bridge_name}.{submodule}"
        if target in sys.modules:
            continue
        if importlib.util.find_spec(target) is None:
            continue
        loaded = importlib.import_module(target)
        _rebind_module(loaded, relative_name=submodule)

    for submodule in ("data", "hypergrad", "inference"):
        attr = getattr(sys.modules[__name__], submodule, None)
        if isinstance(attr, ModuleType):
            _register(submodule, attr)


_bridge_pure_python_package()

