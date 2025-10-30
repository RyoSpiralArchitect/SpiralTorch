from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


_BINDINGS_DIR = Path(__file__).resolve().parents[1] / "spiraltorch"


def _load_ecosystem(stub_spiraltorch, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "spiraltorch.ecosystem", raising=False)
    package_paths = list(getattr(stub_spiraltorch, "__path__", []))
    bindings_path = str(_BINDINGS_DIR)
    if bindings_path not in package_paths:
        package_paths.append(bindings_path)
        stub_spiraltorch.__path__ = package_paths
    return importlib.import_module("spiraltorch.ecosystem")


def test_tensor_to_torch_and_back_use_compat(stub_spiraltorch, monkeypatch):
    calls: dict[str, tuple[object, dict[str, object]]] = {}

    def to_torch(tensor, **kwargs):
        calls["to_torch"] = (tensor, kwargs)
        return "torch-tensor"

    def from_torch(tensor, **kwargs):
        calls["from_torch"] = (tensor, kwargs)
        return "spiral-tensor"

    torch_namespace = types.SimpleNamespace(to_torch=to_torch, from_torch=from_torch)
    monkeypatch.setattr(stub_spiraltorch.compat, "torch", torch_namespace, raising=False)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    tensor = Tensor(1, 1, [0.0])

    result = ecosystem.tensor_to_torch(
        tensor,
        dtype="float32",
        device="cuda:0",
        requires_grad=True,
        copy=True,
        memory_format="contiguous_format",
    )
    assert result == "torch-tensor"
    assert calls["to_torch"] == (
        tensor,
        {
            "dtype": "float32",
            "device": "cuda:0",
            "requires_grad": True,
            "copy": True,
            "memory_format": "contiguous_format",
        },
    )

    back = ecosystem.torch_to_tensor(
        "torch-tensor",
        dtype="float16",
        device="cpu",
        ensure_cpu=False,
        copy=True,
        require_contiguous=False,
    )
    assert back == "spiral-tensor"
    assert calls["from_torch"] == (
        "torch-tensor",
        {
            "dtype": "float16",
            "device": "cpu",
            "ensure_cpu": False,
            "copy": True,
            "require_contiguous": False,
        },
    )


def test_jax_and_tensorflow_bridge_use_compat(stub_spiraltorch, monkeypatch):
    jax_calls: dict[str, object] = {}
    tf_calls: dict[str, object] = {}

    def to_jax(tensor):
        jax_calls["to"] = tensor
        return "jax-array"

    def from_jax(array):
        jax_calls["from"] = array
        return "jax->spiral"

    def to_tf(tensor):
        tf_calls["to"] = tensor
        return "tf-tensor"

    def from_tf(value):
        tf_calls["from"] = value
        return "tf->spiral"

    monkeypatch.setattr(
        stub_spiraltorch.compat,
        "jax",
        types.SimpleNamespace(to_jax=to_jax, from_jax=from_jax),
        raising=False,
    )
    monkeypatch.setattr(
        stub_spiraltorch.compat,
        "tensorflow",
        types.SimpleNamespace(to_tensorflow=to_tf, from_tensorflow=from_tf),
        raising=False,
    )

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor
    tensor = Tensor(1, 1, [1.0])

    assert ecosystem.tensor_to_jax(tensor) == "jax-array"
    assert jax_calls["to"] is tensor
    assert ecosystem.jax_to_tensor("jax-array") == "jax->spiral"
    assert jax_calls["from"] == "jax-array"

    assert ecosystem.tensor_to_tensorflow(tensor) == "tf-tensor"
    assert tf_calls["to"] is tensor
    assert ecosystem.tensorflow_to_tensor("tf-tensor") == "tf->spiral"
    assert tf_calls["from"] == "tf-tensor"


def test_missing_compat_namespace_raises_helpful_error(stub_spiraltorch, monkeypatch):
    monkeypatch.delattr(stub_spiraltorch.compat, "torch", raising=False)
    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    tensor = stub_spiraltorch.Tensor(1, 1, [0.0])

    with pytest.raises(RuntimeError, match=r"compat\.torch"):
        ecosystem.tensor_to_torch(tensor)


def test_cupy_roundtrip_uses_dlpack(stub_spiraltorch, monkeypatch):
    fake_cupy = types.ModuleType("cupy")

    def from_dlpack(capsule):
        return ("from_dlpack", capsule)

    def to_dlpack(array):
        return ("to_dlpack", array)

    fake_cupy.from_dlpack = from_dlpack
    fake_cupy.toDlpack = to_dlpack
    fake_cupy.to_dlpack = to_dlpack

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    tensor = Tensor(1, 1, [0.5])
    capsule = object()
    monkeypatch.setattr(Tensor, "to_dlpack", lambda self: capsule)
    monkeypatch.setattr(
        Tensor,
        "from_dlpack",
        classmethod(lambda cls, cap: ("from_dlpack", cap)),
    )

    cupy_array = ecosystem.tensor_to_cupy(tensor)
    assert cupy_array == ("from_dlpack", capsule)

    restored = ecosystem.cupy_to_tensor(types.SimpleNamespace(toDlpack=lambda: ("dlpack", 42)))
    assert restored == ("from_dlpack", ("dlpack", 42))


def test_cupy_roundtrip_accepts_stream(stub_spiraltorch, monkeypatch):
    fake_cupy = types.ModuleType("cupy")

    stream_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def from_dlpack(capsule, *, stream=None):
        stream_calls.append(("from", (capsule,), {"stream": stream}))
        return ("from_dlpack", capsule, stream)

    fake_cupy.from_dlpack = from_dlpack

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    tensor = Tensor(1, 1, [0.5])
    capsule = object()
    monkeypatch.setattr(Tensor, "to_dlpack", lambda self: capsule)
    monkeypatch.setattr(
        Tensor,
        "from_dlpack",
        classmethod(lambda cls, cap: ("from_dlpack", cap)),
    )

    stream = object()

    cupy_array = ecosystem.tensor_to_cupy(tensor, stream=stream)
    assert cupy_array == ("from_dlpack", capsule, stream)

    class Array:
        def __init__(self):
            self.calls: list[object | None] = []

        def toDlpack(self, stream=None):
            self.calls.append(stream)
            return ("dlpack", stream)

    array = Array()

    restored = ecosystem.cupy_to_tensor(array, stream=stream)
    assert restored == ("from_dlpack", ("dlpack", stream))

    assert stream_calls[0] == ("from", (capsule,), {"stream": stream})
    assert array.calls == [stream]


def test_cupy_to_tensor_prefers_dunder_dlpack(stub_spiraltorch, monkeypatch):
    fake_cupy = types.ModuleType("cupy")

    def to_dlpack(array, *, stream=None):  # pragma: no cover - fallback guard
        raise AssertionError("module-level exporters should not be used when __dlpack__ exists")

    fake_cupy.toDlpack = to_dlpack
    fake_cupy.to_dlpack = to_dlpack

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)

    Tensor = stub_spiraltorch.Tensor
    capsule = object()
    monkeypatch.setattr(
        Tensor,
        "from_dlpack",
        classmethod(lambda cls, cap: ("from_dlpack", cap)),
    )

    class Array:
        def __dlpack__(self, *, stream=None):
            return ("dlpack", stream)

    restored = ecosystem.cupy_to_tensor(Array(), stream="stream-token")
    assert restored == ("from_dlpack", ("dlpack", "stream-token"))


def test_cupy_to_tensor_uses_module_level_exporter_when_needed(
    stub_spiraltorch, monkeypatch
):
    fake_cupy = types.ModuleType("cupy")

    called: dict[str, tuple[object, object | None]] = {}

    def to_dlpack(array, *, stream=None):
        called["args"] = (array, stream)
        return ("dlpack", stream)

    fake_cupy.to_dlpack = to_dlpack

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    monkeypatch.setattr(
        Tensor,
        "from_dlpack",
        classmethod(lambda cls, cap: ("from_dlpack", cap)),
    )

    class Array:
        pass

    stream = object()
    result = ecosystem.cupy_to_tensor(Array(), stream=stream)
    assert result == ("from_dlpack", ("dlpack", stream))
    assert isinstance(called["args"][0], Array)
    assert called["args"][1] is stream
