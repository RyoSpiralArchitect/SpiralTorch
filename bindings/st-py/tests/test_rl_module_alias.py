from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

_PKG_INIT = Path(__file__).resolve().parents[1] / "spiraltorch" / "__init__.py"


@contextmanager
def _isolated_spiraltorch(*, provide_rl: bool, existing_rl: types.ModuleType | None):
    prefix_tracked = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("spiraltorch") or name.startswith("spiral_rl") or name == "rl"
    }
    tracked = set(prefix_tracked)
    tracked.update(
        {
            "spiraltorch.spiraltorch",
            "spiraltorch.spiral_rl",
            "spiraltorch.rl",
            "spiral_rl",
            "rl",
            "spiraltorch.zspace_inference",
            "spiraltorch.elliptic",
        }
    )
    saved_modules = {name: sys.modules.get(name) for name in tracked}
    saved_meta_path = list(sys.meta_path)
    try:
        for name in tracked:
            sys.modules.pop(name, None)

        if existing_rl is not None:
            sys.modules["rl"] = existing_rl

        spec = importlib.util.spec_from_file_location("spiraltorch", _PKG_INIT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["spiraltorch"] = module

        rl_module: types.ModuleType | None = None
        if provide_rl:
            rl_module = types.ModuleType("spiraltorch.spiral_rl")
            rl_module.stAgent = object()
            rl_module.__spiraltorch_placeholder__ = False
            module.spiral_rl = rl_module
            module.rl = rl_module
            sys.modules["spiraltorch.spiral_rl"] = rl_module
            sys.modules["spiraltorch.rl"] = rl_module

        stub_core = types.ModuleType("spiraltorch.spiraltorch")
        if rl_module is not None:
            stub_core.rl = rl_module
            stub_core.spiral_rl = rl_module
        sys.modules["spiraltorch.spiraltorch"] = stub_core

        zspace_stub = types.ModuleType("spiraltorch.zspace_inference")
        for name in [
            "ZMetrics",
            "ZSpaceDecoded",
            "ZSpaceInference",
            "ZSpacePosterior",
            "ZSpacePartialBundle",
            "ZSpaceTelemetryFrame",
            "ZSpaceInferencePipeline",
            "inference_to_mapping",
            "inference_to_zmetrics",
            "prepare_trainer_step_payload",
            "canvas_partial_from_snapshot",
            "canvas_coherence_partial",
            "elliptic_partial_from_telemetry",
            "coherence_partial_from_diagnostics",
            "decode_zspace_embedding",
            "blend_zspace_partials",
            "infer_canvas_snapshot",
            "infer_canvas_transformer",
            "infer_coherence_diagnostics",
            "infer_coherence_from_sequencer",
            "infer_canvas_with_coherence",
            "infer_with_partials",
            "infer_from_partial",
            "weights_partial_from_dlpack",
            "weights_partial_from_compat",
            "infer_weights_from_dlpack",
            "infer_weights_from_compat",
        ]:
            setattr(zspace_stub, name, object())
        sys.modules["spiraltorch.zspace_inference"] = zspace_stub

        elliptic_stub = types.ModuleType("spiraltorch.elliptic")
        for name in [
            "EllipticWarpFunction",
            "elliptic_warp_autograd",
            "elliptic_warp_features",
            "elliptic_warp_partial",
        ]:
            setattr(elliptic_stub, name, object())
        sys.modules["spiraltorch.elliptic"] = elliptic_stub

        spec.loader.exec_module(module)
        yield module, rl_module
    finally:
        sys.meta_path[:] = saved_meta_path
        for name in list(sys.modules):
            if name in tracked or name.startswith("spiraltorch") or name.startswith("spiral_rl"):
                sys.modules.pop(name, None)
        for name, mod in prefix_tracked.items():
            sys.modules[name] = mod
        for name, mod in saved_modules.items():
            if mod is not None and name not in prefix_tracked:
                sys.modules[name] = mod


def test_rl_alias_respects_existing_module() -> None:
    sentinel = types.ModuleType("rl")
    with _isolated_spiraltorch(provide_rl=True, existing_rl=sentinel) as (module, rl_module):
        assert rl_module is not None
        assert sys.modules["rl"] is sentinel
        assert module.rl is rl_module


def test_rl_alias_imports_on_demand() -> None:
    with _isolated_spiraltorch(provide_rl=True, existing_rl=None) as (module, rl_module):
        assert rl_module is not None
        assert "rl" not in sys.modules

        imported = importlib.import_module("rl")
        assert imported is rl_module
        assert sys.modules["rl"] is rl_module
        assert module.rl is rl_module


def test_rl_alias_not_reserved_without_native_target() -> None:
    with _isolated_spiraltorch(provide_rl=False, existing_rl=None) as (module, _):
        assert "rl" not in sys.modules
        assert module._spiraltorch_rl_module(load=False) is None
        finder_cls = module._SpiralTorchRLAliasFinder
        assert all(not isinstance(finder, finder_cls) for finder in sys.meta_path)
