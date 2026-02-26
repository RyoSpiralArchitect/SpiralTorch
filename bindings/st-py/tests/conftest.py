import importlib.util
import pathlib
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture
def stub_spiraltorch(monkeypatch: pytest.MonkeyPatch):
    preexisting = set(sys.modules)
    for name in list(sys.modules):
        if (
            name == "spiraltorch"
            or name.startswith("spiraltorch.")
            or name in {"spiral_rl", "rl"}
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.autograd = types.SimpleNamespace(Function=object)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    spec = importlib.util.spec_from_file_location(
        "spiraltorch", REPO_ROOT / "spiraltorch" / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "spiraltorch", module)
    spec.loader.exec_module(module)
    if hasattr(module, "_install_stub_bindings"):
        module._install_stub_bindings(
            module, ModuleNotFoundError("spiraltorch", name="spiraltorch")
        )
    try:
        yield module
    finally:
        for name in list(sys.modules):
            if name in preexisting:
                continue
            if name == "spiral_rl" or name == "rl" or name.startswith("spiraltorch."):
                sys.modules.pop(name, None)


@pytest.fixture
def spiraltorch_stub(stub_spiraltorch):
    return stub_spiraltorch
