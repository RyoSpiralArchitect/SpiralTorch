import importlib.util
from pathlib import Path
import sys
import types

import pytest


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


_STUB_SPEC = importlib.util.spec_from_file_location(
    "_spiraltorch_stub_for_tests",
    Path(__file__).resolve().parents[3] / "spiraltorch" / "__init__.py",
)
_STUB_MODULE = importlib.util.module_from_spec(_STUB_SPEC)
assert _STUB_SPEC.loader is not None  # for mypy/static type checking
sys.modules[_STUB_SPEC.name] = _STUB_MODULE
_STUB_SPEC.loader.exec_module(_STUB_MODULE)
_INSTALL_STUB_BINDINGS = getattr(_STUB_MODULE, "_install_stub_bindings")


@pytest.fixture()
def spiraltorch_stub(monkeypatch):
    placeholder = types.ModuleType("spiraltorch")
    for name in list(sys.modules):
        if name == "spiraltorch" or name.startswith("spiraltorch.") or name == "spiral_rl":
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "spiraltorch", placeholder)
    _INSTALL_STUB_BINDINGS(placeholder, ModuleNotFoundError("spiraltorch", name="spiraltorch"))
    return placeholder
