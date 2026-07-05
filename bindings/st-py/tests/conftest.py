import importlib.util
import pathlib
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture
def stub_spiraltorch(monkeypatch: pytest.MonkeyPatch):
    module_names = (
        "spiraltorch",
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    )
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.autograd = types.SimpleNamespace(Function=object)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    stub_path = REPO_ROOT / "spiraltorch" / "__init__.py"
    source = stub_path.read_text()
    prefix, _, _ = source.partition("\n_load_native_package()")

    module = types.ModuleType("spiraltorch")
    module.__file__ = str(stub_path)
    module.__package__ = "spiraltorch"
    module.__path__ = [
        str(stub_path.parent),
        str(REPO_ROOT / "bindings" / "st-py" / "spiraltorch"),
    ]
    module.__spec__ = importlib.util.spec_from_loader("spiraltorch", loader=None)
    monkeypatch.setitem(sys.modules, "spiraltorch", module)
    exec(compile(prefix, str(stub_path), "exec"), module.__dict__)
    if hasattr(module, "_install_stub_bindings"):
        real_find_spec = module.importlib.util.find_spec

        def stub_find_spec(name, *args, **kwargs):
            if name == "numpy":
                return None
            return real_find_spec(name, *args, **kwargs)

        module.importlib.util.find_spec = stub_find_spec
        try:
            module._install_stub_bindings(
                module, ModuleNotFoundError("spiraltorch", name="spiraltorch")
            )
        finally:
            module.importlib.util.find_spec = real_find_spec
        compat = types.ModuleType("spiraltorch.compat")
        module.compat = compat
        monkeypatch.setitem(sys.modules, "spiraltorch.compat", compat)

        def stub_runtime_error(feature: str) -> RuntimeError:
            return RuntimeError(
                "SpiralTorch Python stub: "
                f"{feature} requires the native extension."
            )

        class SpiralSession:
            def __init__(self, *args, **kwargs):
                raise stub_runtime_error("SpiralSession")

        def plan_topk(*args, **kwargs):
            raise stub_runtime_error("plan_topk")

        planner = types.ModuleType("spiraltorch.planner")
        planner.plan_topk = plan_topk
        planner.__doc__ = "SpiralTorch planner stub (native extension unavailable)."

        module.SpiralSession = SpiralSession
        module.plan_topk = plan_topk
        module.planner = planner
        monkeypatch.setitem(sys.modules, "spiraltorch.planner", planner)
    return module


@pytest.fixture
def spiraltorch_stub(stub_spiraltorch):
    return stub_spiraltorch
