"""Installed native Linear/LoRA learning without foreign ML imports or stubs."""

import importlib.abc
import math
import sys
import unittest


FOREIGN = {"torch", "transformers", "peft", "accelerate", "numpy", "safetensors"}


class NativeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FOREIGN:
            raise AssertionError("native layer learning must not import " + fullname)


if __name__ == "__main__":
    if any(name.split(".")[0] in FOREIGN for name in sys.modules):
        raise RuntimeError("native learning requires a fresh process without foreign ML modules")
    sys.meta_path.insert(0, NativeOnly())

import spiraltorch as st  # noqa: E402 - install the forbidden-import guard first.


def values(state):
    return {name: tensor.tolist() for name, tensor in state}


def lora():
    layer = st.nn.LoraLinear(3, 4, 2, alpha=4.0, name="adapter")
    layer.load_state_dict([
        ("adapter::lora_a", st.Tensor(3, 2, [0.2, -0.1, 0.3, 0.1, -0.2, 0.25])),
        ("adapter::lora_b", st.Tensor(2, 4, [0.1, -0.2, 0.15, 0.3, 0.2, 0.1, -0.1, 0.25])),
    ])
    return layer


class NativeLayerLearning(unittest.TestCase):
    def check_learning(self, factory, update, frozen_base=False):
        layer = factory()
        initial = values(layer.state_dict())
        base = layer.base_state_dict() if frozen_base else None
        if update == "realgrad":
            layer.attach_realgrad(0.03)
        elif update == "hypergrad":
            layer.attach_hypergrad(-1.0, 0.03)
        inputs = st.Tensor(2, 3, [0.7, -0.4, 0.2, -0.2, 0.9, 0.5])
        target = st.Tensor(2, 4, [0.3, -0.5, 0.2, 0.1, 0.8, 0.2, -0.1, -0.4])
        loss = st.nn.MeanSquaredError()
        before = layer.forward(inputs).tolist()
        for _ in range(2):
            for _ in range(2):
                output = layer.forward(inputs)
                cotangent = loss.backward(output, target)
                self.assertEqual(layer.backward(inputs, cotangent).shape(), inputs.shape())
            layer.apply_step(0.03)
            layer.zero_accumulators()
        self.assertNotEqual(values(layer.state_dict()), initial)
        after = layer.forward(inputs).tolist()
        self.assertNotEqual(after, before)
        self.assertTrue(all(math.isfinite(v) for row in after for v in row))
        restored = factory()
        if frozen_base:
            self.assertEqual(values(layer.base_state_dict()), values(base))
            restored.load_base_state_dict(base)
        restored.load_state_dict(layer.state_dict())
        self.assertEqual(restored.forward(inputs).tolist(), after)

    def test_linear_sgd(self):
        self.check_learning(lambda: st.nn.Linear("head", 3, 4), "sgd")

    def test_linear_realgrad(self):
        self.check_learning(lambda: st.nn.Linear("head", 3, 4), "realgrad")

    def test_linear_hypergrad(self):
        self.check_learning(lambda: st.nn.Linear("head", 3, 4), "hypergrad")

    def test_lora_sgd(self):
        self.check_learning(lora, "sgd", frozen_base=True)

    def test_lora_realgrad(self):
        self.check_learning(lora, "realgrad", frozen_base=True)

    def test_lora_hypergrad(self):
        self.check_learning(lora, "hypergrad", frozen_base=True)


if __name__ == "__main__":
    unittest.main()
