from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


_BINDINGS_DIR = Path(__file__).resolve().parents[1] / "spiraltorch"


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_torch = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, payload):
            self.payload = payload

        def to(self, **kwargs):
            return FakeTensor((self.payload, kwargs))

        def clone(self):
            return FakeTensor(("clone", self.payload))

    fake_utils = types.ModuleType("torch.utils")
    fake_dlpack = types.ModuleType("torch.utils.dlpack")

    def from_dlpack(capsule):
        return FakeTensor(("from_dlpack", capsule))

    def to_dlpack(tensor):
        return ("to_dlpack", tensor.payload)

    fake_dlpack.from_dlpack = from_dlpack
    fake_dlpack.to_dlpack = to_dlpack
    fake_utils.dlpack = fake_dlpack

    fake_autograd = types.SimpleNamespace(Function=type("Function", (), {}))

    fake_torch.Tensor = FakeTensor
    fake_torch.utils = fake_utils
    fake_torch.autograd = fake_autograd

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.dlpack", fake_dlpack)

    return fake_torch


def _install_fake_jax(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_jax = types.ModuleType("jax")

    def device_put(array, device=None):
        return ("device_put", array, device)

    fake_jax.device_put = device_put

    fake_dlpack = types.ModuleType("jax.dlpack")

    def from_dlpack(capsule):
        return ("from_dlpack", capsule)

    def to_dlpack(array):
        return ("to_dlpack", array)

    fake_dlpack.from_dlpack = from_dlpack
    fake_dlpack.to_dlpack = to_dlpack

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.dlpack", fake_dlpack)

    return fake_jax


def _install_fake_cupy(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_cupy = types.ModuleType("cupy")

    def from_dlpack(capsule):
        return ("cupy.from_dlpack", capsule)

    def to_dlpack(array):
        return ("cupy.to_dlpack", array)

    fake_cupy.from_dlpack = from_dlpack
    fake_cupy.toDlpack = to_dlpack
    fake_cupy.to_dlpack = to_dlpack

    class FakeArray:
        def __init__(self, payload):
            self.payload = payload

        def toDlpack(self):
            return ("array.toDlpack", self.payload)

    fake_cupy.ndarray = FakeArray

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    return fake_cupy


def _install_fake_tensorflow(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_tf = types.ModuleType("tensorflow")
    fake_experimental = types.SimpleNamespace()
    fake_dlpack = types.SimpleNamespace()

    def from_dlpack(capsule):
        return ("tf.from_dlpack", capsule)

    def to_dlpack(tensor):
        return ("tf.to_dlpack", tensor)

    fake_dlpack.from_dlpack = from_dlpack
    fake_dlpack.to_dlpack = to_dlpack
    fake_experimental.dlpack = fake_dlpack
    fake_tf.experimental = fake_experimental

    class FakeTensor:
        def __init__(self, payload):
            self.payload = payload

    fake_tf.Tensor = FakeTensor

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    return fake_tf


def _load_ecosystem(stub_spiraltorch, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "spiraltorch.ecosystem", raising=False)
    package_paths = list(getattr(stub_spiraltorch, "__path__", []))
    bindings_path = str(_BINDINGS_DIR)
    if bindings_path not in package_paths:
        package_paths.append(bindings_path)
        stub_spiraltorch.__path__ = package_paths
    return importlib.import_module("spiraltorch.ecosystem")


def test_tensor_to_torch_uses_dlpack(
    stub_spiraltorch, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_torch(monkeypatch)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    tensor = Tensor(1, 1, [0.0])

    capsule = object()
    monkeypatch.setattr(Tensor, "to_dlpack", lambda self: capsule)

    torch_tensor = ecosystem.tensor_to_torch(tensor)
    assert torch_tensor.payload == ("from_dlpack", capsule)

    moved = ecosystem.tensor_to_torch(tensor, device="meta", dtype="float32", copy=True)
    assert moved.payload == (
        ("from_dlpack", capsule),
        {"device": "meta", "dtype": "float32", "copy": True},
    )


def test_torch_to_tensor_uses_dlpack(
    stub_spiraltorch, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = _install_fake_torch(monkeypatch)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    captured = {}

    def fake_from_dlpack(cls, capsule):
        captured["capsule"] = capsule
        return ("tensor", capsule)

    monkeypatch.setattr(Tensor, "from_dlpack", classmethod(fake_from_dlpack))

    torch_tensor = fake_torch.Tensor("payload")

    result = ecosystem.torch_to_tensor(torch_tensor, clone=True)
    assert result == ("tensor", ("to_dlpack", ("clone", "payload")))
    assert captured["capsule"] == ("to_dlpack", ("clone", "payload"))

    with pytest.raises(TypeError):
        ecosystem.torch_to_tensor("not_a_tensor")


def test_tensor_to_jax_and_back(
    stub_spiraltorch, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_jax(monkeypatch)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    tensor = Tensor(1, 1, [1.0])
    capsule = object()

    monkeypatch.setattr(Tensor, "to_dlpack", lambda self: capsule)
    monkeypatch.setattr(
        Tensor,
        "from_dlpack",
        classmethod(lambda cls, capsule: ("tensor", capsule)),
    )

    jax_array = ecosystem.tensor_to_jax(tensor, device="TPU")
    assert jax_array == ("device_put", ("from_dlpack", capsule), "TPU")

    restored = ecosystem.jax_to_tensor(("array", 1))
    assert restored == ("tensor", ("to_dlpack", ("array", 1)))


def test_tensor_to_cupy_and_back(
    stub_spiraltorch, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_cupy(monkeypatch)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    tensor = Tensor(1, 1, [0.5])
    capsule = object()

    monkeypatch.setattr(Tensor, "to_dlpack", lambda self: capsule)
    monkeypatch.setattr(
        Tensor,
        "from_dlpack",
        classmethod(lambda cls, capsule: ("tensor", capsule)),
    )

    cupy_array = ecosystem.tensor_to_cupy(tensor)
    assert cupy_array == ("cupy.from_dlpack", capsule)

    restored = ecosystem.cupy_to_tensor(types.SimpleNamespace(toDlpack=lambda: ("dlpack", 42)))
    assert restored == ("tensor", ("dlpack", 42))


def test_tensor_to_tensorflow_and_back(
    stub_spiraltorch, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_tensorflow(monkeypatch)

    ecosystem = _load_ecosystem(stub_spiraltorch, monkeypatch)
    Tensor = stub_spiraltorch.Tensor

    tensor = Tensor(1, 1, [0.25])
    capsule = object()

    monkeypatch.setattr(Tensor, "to_dlpack", lambda self: capsule)
    monkeypatch.setattr(
        Tensor,
        "from_dlpack",
        classmethod(lambda cls, capsule: ("tensor", capsule)),
    )

    tf_tensor = ecosystem.tensor_to_tensorflow(tensor)
    assert tf_tensor == ("tf.from_dlpack", capsule)

    restored = ecosystem.tensorflow_to_tensor("fake_tf_tensor")
    assert restored == ("tensor", ("tf.to_dlpack", "fake_tf_tensor"))
