import builtins
import contextlib
import importlib.util
import pathlib
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _create_stub_module(*, allow_numpy: bool):
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.autograd = types.SimpleNamespace(Function=object)
        sys.modules["torch"] = torch_stub

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
    module.__spec__ = importlib.util.spec_from_loader("spiraltorch", loader=None, is_package=True)
    sys.modules["spiraltorch"] = module
    exec(compile(prefix, str(stub_path), "exec"), module.__dict__)
    if hasattr(module, "_install_stub_bindings"):
        real_find_spec = module.importlib.util.find_spec
        real_import = builtins.__import__

        def stub_find_spec(name, *args, **kwargs):
            if name == "numpy":
                return None
            return real_find_spec(name, *args, **kwargs)

        def stub_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numpy":
                raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
            if level == 1 and (name == "_blas" or "_blas" in fromlist):
                raise ImportError("SpiralTorch BLAS stub disabled for this fixture")
            return real_import(name, globals, locals, fromlist, level)

        if not allow_numpy:
            module.importlib.util.find_spec = stub_find_spec
            builtins.__import__ = stub_import
        try:
            module._install_stub_bindings(
                module, ModuleNotFoundError("spiraltorch", name="spiraltorch")
            )
        finally:
            module.importlib.util.find_spec = real_find_spec
            builtins.__import__ = real_import
        compat = types.ModuleType("spiraltorch.compat")
        module.compat = compat
        sys.modules["spiraltorch.compat"] = compat

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
        sys.modules["spiraltorch.planner"] = planner
    return module


@contextlib.contextmanager
def _stub_spiraltorch_context(*, allow_numpy: bool):
    def tracked_module(name: str) -> bool:
        return (
            name == "spiraltorch"
            or name.startswith("spiraltorch.")
            or name == "spiraltorch_native"
            or name == "spiral_rl"
            or name.startswith("spiral_rl.")
            or name == "rl"
            or name == "_spiraltorch_stub_py_bridge"
            or name.startswith("_spiraltorch_stub_py_bridge.")
        )

    saved_module_graph = {
        name: module
        for name, module in sys.modules.items()
        if tracked_module(name)
    }
    missing = object()
    torch_saved = sys.modules.get("torch", missing)
    try:
        for name in tuple(sys.modules):
            if tracked_module(name):
                sys.modules.pop(name, None)
        yield _create_stub_module(allow_numpy=allow_numpy)
    finally:
        for name in tuple(sys.modules):
            if tracked_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved_module_graph)
        if torch_saved is missing:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = torch_saved


@pytest.fixture
def stub_spiraltorch_context():
    return _stub_spiraltorch_context


@pytest.fixture
def stub_spiraltorch():
    with _stub_spiraltorch_context(allow_numpy=False) as module:
        yield module


@pytest.fixture
def stub_spiraltorch_with_numpy():
    with _stub_spiraltorch_context(allow_numpy=True) as module:
        yield module


@pytest.fixture
def spiraltorch_stub(stub_spiraltorch):
    return stub_spiraltorch
