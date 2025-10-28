from __future__ import annotations

import sys
import types

import pytest


def test_spiral_rl_stub_module_is_exposed(spiraltorch_stub):
    from spiraltorch import spiral_rl

    assert isinstance(spiral_rl, types.ModuleType)
    assert spiral_rl.__doc__
    assert "stub" in spiral_rl.__doc__.lower()
    for name in ("stAgent", "DqnAgent", "PyDqnAgent"):
        assert hasattr(spiral_rl, name)


def test_st_agent_stub_signals_missing_extension(spiraltorch_stub):
    from spiraltorch import spiral_rl

    stub_cls = spiral_rl.stAgent

    assert stub_cls is spiral_rl.DqnAgent
    assert stub_cls is spiral_rl.PyDqnAgent
    assert stub_cls.__module__ == "spiral_rl"
    assert "stub" in (stub_cls.__doc__ or "").lower()

    with pytest.raises(RuntimeError) as excinfo:
        stub_cls(1, 2, 0.9, 0.01)
    message = str(excinfo.value)
    assert "stub" in message.lower()
    assert "native extension" in message.lower()

    agent = object.__new__(stub_cls)

    with pytest.raises(RuntimeError):
        agent.select_action(0)
    with pytest.raises(RuntimeError):
        agent.select_actions([0])
    with pytest.raises(RuntimeError):
        agent.update(0, 0, 0.0, 0)
    with pytest.raises(RuntimeError):
        agent.update_batch([0], [0], [0.0], [0])
    with pytest.raises(RuntimeError):
        _ = agent.epsilon
    with pytest.raises(RuntimeError):
        agent.set_epsilon(0.5)
    with pytest.raises(RuntimeError):
        agent.set_exploration(None)
    with pytest.raises(RuntimeError):
        agent.state_dict()
    with pytest.raises(RuntimeError):
        agent.load_state_dict({})
