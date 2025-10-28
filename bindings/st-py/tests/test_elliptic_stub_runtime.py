from __future__ import annotations

import builtins
import importlib
import pathlib
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_elliptic_autograd_requires_torch_when_unavailable() -> None:
    module_names = (
        "spiraltorch",
        "spiraltorch.elliptic",
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    torch_saved = sys.modules.get("torch")
    preexisting = set(sys.modules)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = fake_import
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.pop("torch", None)

        importlib.invalidate_caches()
        elliptic = importlib.import_module("spiraltorch.elliptic")
        assert hasattr(elliptic, "EllipticWarpFunction")

        with pytest.raises(RuntimeError, match="PyTorch is required"):
            elliptic.elliptic_warp_autograd(None, None)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="PyTorch is required"):
            elliptic.elliptic_warp_partial(None, None)  # type: ignore[arg-type]
    finally:
        builtins.__import__ = real_import

        for name in list(sys.modules):
            if name not in preexisting and name.startswith("spiraltorch"):
                sys.modules.pop(name, None)

        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)

        if torch_saved is not None:
            sys.modules["torch"] = torch_saved
        else:
            sys.modules.pop("torch", None)
