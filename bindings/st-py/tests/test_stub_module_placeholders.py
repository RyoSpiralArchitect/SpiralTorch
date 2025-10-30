from __future__ import annotations

import importlib
import types

import pytest


def _assert_stub_runtime_error(action):
    with pytest.raises(RuntimeError) as excinfo:
        action()
    message = str(excinfo.value).lower()
    assert "stub" in message
    assert "native extension" in message


@pytest.mark.parametrize(
    "submodule_name",
    ["dataset", "linalg", "rec", "telemetry", "ecosystem"],
)
def test_stub_submodules_are_preinstalled(spiraltorch_stub, submodule_name: str):
    parent = importlib.import_module("spiraltorch")
    placeholder = importlib.import_module(f"spiraltorch.{submodule_name}")

    assert hasattr(parent, submodule_name)
    assert isinstance(placeholder, types.ModuleType)
    assert placeholder.__doc__
    assert "native extension" in placeholder.__doc__.lower()

    _assert_stub_runtime_error(lambda: getattr(placeholder, "missing"))


def test_spiral_rl_stub_agents_share_runtime_error(spiraltorch_stub):
    spiral_rl = importlib.import_module("spiraltorch.spiral_rl")

    assert isinstance(spiral_rl, types.ModuleType)
    assert hasattr(spiral_rl, "stAgent")

    stub_cls = spiral_rl.stAgent
    assert stub_cls is spiral_rl.DqnAgent
    assert stub_cls is spiral_rl.PyDqnAgent
    assert stub_cls.__module__ == "spiral_rl"
    assert "stub" in (stub_cls.__doc__ or "").lower()

    _assert_stub_runtime_error(lambda: stub_cls(1, 2, 0.9, 0.01))

    agent = object.__new__(stub_cls)

    _assert_stub_runtime_error(lambda: agent.select_action(0))
    _assert_stub_runtime_error(lambda: agent.set_epsilon(0.5))
    _assert_stub_runtime_error(lambda: agent.state_dict())
