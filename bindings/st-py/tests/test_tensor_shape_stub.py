from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


def test_tensor_shape_method_available_in_stub(monkeypatch):
    # Ensure we load the pure Python stub bindings rather than the native module.
    project_root = Path(__file__).resolve().parents[3]
    filtered_sys_path = [str(project_root)]
    filtered_sys_path.extend(
        entry
        for entry in sys.path
        if entry != str(project_root) and "bindings/st-py" not in entry
    )
    monkeypatch.setattr(sys, "path", filtered_sys_path)

    for name in list(sys.modules):
        if name == "spiraltorch" or name.startswith("spiraltorch."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    original_find_spec = importlib.machinery.PathFinder.find_spec

    def _raise_for_stub(name, path=None, target=None):
        if name in {"spiraltorch.spiraltorch", "spiraltorch.spiraltorch_native", "spiraltorch_native"}:
            raise ModuleNotFoundError(name=name)
        return original_find_spec(name, path, target)

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", _raise_for_stub)

    original_exec_module = importlib.machinery.SourceFileLoader.exec_module
    native_init_path = str(project_root / "bindings" / "st-py" / "spiraltorch" / "__init__.py")

    def _force_stub(loader, module):
        if getattr(loader, "path", None) == native_init_path:
            original_exec_module(loader, module)
            raise ModuleNotFoundError(name="spiraltorch.spiraltorch")
        return original_exec_module(loader, module)

    monkeypatch.setattr(importlib.machinery.SourceFileLoader, "exec_module", _force_stub)

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.autograd = types.SimpleNamespace(Function=object)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    stub_path = project_root / "spiraltorch" / "__init__.py"
    spec = importlib.util.spec_from_file_location("spiraltorch", stub_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "spiraltorch", module)
    spec.loader.exec_module(module)
    st = module

    # Sanity-check that we are exercising the stub implementation.
    assert hasattr(st, "available_stub_backends"), "stub bindings should expose helper APIs"

    tensor = st.Tensor((2, 3))

    assert tensor.shape() == (2, 3)
    assert tuple(tensor.shape) == (2, 3)
