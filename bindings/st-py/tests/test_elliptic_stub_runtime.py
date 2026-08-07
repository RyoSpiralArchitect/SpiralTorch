from __future__ import annotations

import builtins
import importlib.util
import pathlib
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ELLIPTIC_PATH = pathlib.Path(__file__).resolve().parents[1] / "spiraltorch" / "elliptic.py"


def test_elliptic_autograd_requires_torch_when_unavailable() -> None:
    package_name = "_spiraltorch_elliptic_stub_test"
    module_name = f"{package_name}.elliptic"
    torch_saved = sys.modules.get("torch")

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = fake_import
        sys.modules.pop("torch", None)

        package = types.ModuleType(package_name)
        package.__path__ = [str(ELLIPTIC_PATH.parent)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

        spec = importlib.util.spec_from_file_location(module_name, ELLIPTIC_PATH)
        assert spec is not None and spec.loader is not None
        elliptic = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = elliptic
        spec.loader.exec_module(elliptic)
        assert hasattr(elliptic, "EllipticWarpFunction")

        with pytest.raises(RuntimeError, match="PyTorch is required"):
            elliptic.elliptic_warp_autograd(None, None)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="PyTorch is required"):
            elliptic.elliptic_warp_partial(None, None)  # type: ignore[arg-type]
    finally:
        builtins.__import__ = real_import
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)

        if torch_saved is not None:
            sys.modules["torch"] = torch_saved
        else:
            sys.modules.pop("torch", None)
