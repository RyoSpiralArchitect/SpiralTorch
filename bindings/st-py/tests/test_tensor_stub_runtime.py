from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_tensor_constructor_and_shape_in_stub_environment() -> None:
    module_names = (
        "spiraltorch",
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    torch_saved = sys.modules.get("torch")
    torch_stub_installed = False
    if torch_saved is None:
        torch_stub = types.ModuleType("torch")
        torch_stub.autograd = types.SimpleNamespace(Function=object)
        sys.modules["torch"] = torch_stub
        torch_stub_installed = True
    for name in module_names:
        sys.modules.pop(name, None)

    preexisting = set(sys.modules)
    st_module = None

    try:
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location(
            "spiraltorch", REPO_ROOT / "spiraltorch" / "__init__.py"
        )
        st_module = importlib.util.module_from_spec(spec)
        sys.modules["spiraltorch"] = st_module
        assert spec.loader is not None
        spec.loader.exec_module(st_module)
        if not hasattr(st_module, "Tensor") and hasattr(st_module, "_install_stub_bindings"):
            st_module._install_stub_bindings(
                st_module, ModuleNotFoundError("spiraltorch")
            )
        assert hasattr(st_module, "available_stub_backends")

        tensor = st_module.Tensor((2, 3))
        assert tensor.shape() == (2, 3)
        assert tuple(tensor.shape) == (2, 3)
        assert tensor.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    finally:
        for name in list(sys.modules):
            if name not in preexisting and name.startswith("spiraltorch"):
                sys.modules.pop(name, None)

        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)

        if torch_stub_installed:
            sys.modules.pop("torch", None)
        elif torch_saved is not None:
            sys.modules["torch"] = torch_saved

        if st_module is not None:
            sys.modules.pop("spiraltorch", None)
