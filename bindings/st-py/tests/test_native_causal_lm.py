"""Installed whole-decoder execution with imports of foreign ML packages forbidden."""
import importlib.abc
import json
import math
import struct
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "accelerate", "safetensors"}:
            raise AssertionError("native causal LM must not import " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402


def fixture():
    config = {"model_type": "gpt2", "vocab_size": 11, "n_positions": 8,
              "n_embd": 4, "n_head": 2, "n_layer": 2}
    header, payload = {}, bytearray()

    def add(name, shape):
        count = math.prod(shape)
        values = [1. if "ln_" in name and name.endswith("weight")
                  else math.sin((i * 7 + len(header) * 3) * .23) * .15 for i in range(count)]
        start = len(payload)
        payload.extend(struct.pack("<" + "f" * count, *values))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}

    add("wte.weight", [11, 4])
    add("wpe.weight", [8, 4])
    for layer in range(2):
        for norm in ("ln_1", "ln_2"):
            for suffix in ("weight", "bias"):
                add(f"h.{layer}.{norm}.{suffix}", [4])
        for name, input_width, output_width in (("attn.c_attn", 4, 12), ("attn.c_proj", 4, 4),
                                                ("mlp.c_fc", 4, 16), ("mlp.c_proj", 16, 4)):
            add(f"h.{layer}.{name}.weight", [input_width, output_width])
            add(f"h.{layer}.{name}.bias", [output_width])
    add("ln_f.weight", [4])
    add("ln_f.bias", [4])
    raw = json.dumps(header).encode()
    return json.dumps(config), st.SafeTensorArchive.from_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def artifact(model, *, path="transformer.h.0.mlp", gate=.5):
    return st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": model.base_model_identity,
        "policy": "full_prefix_f32_no_cache.v1", "training": False, "placements": [{
            "path": path, "training": False,
            "recipe": {"schema": "spiraltorch.channel_affine_adapter.v1", "features": 4, "strength": .5, "max_values": 1024},
            "tensors": [{"name": name, "shape": [4], "values": values, "requires_grad": False, "role": "parameter"}
                        for name, values in [("gate", [gate, 0., -gate, gate]), ("bias", [0.] * 4)]],
        }],
    }))


class NativeCausalLmTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = fixture()
        self.model = st.NativeCausalLm.from_gpt2(self.config, self.weights)

    def test_full_model_shapes_causality_and_batch_independence(self):
        model = self.model
        full = model.forward([1, 2, 3, 4, 5, 6], [2, 3])
        self.assertEqual(full.shape, [2, 3, 11])
        values = full.values()
        for i, row in enumerate(([1, 2, 3], [4, 5, 6])):
            self.assertEqual(values[i * 33:(i + 1) * 33], model.forward(row, [1, 3]).values())
        changed = model.forward([1, 2, 9, 4, 5, 10], [2, 3]).values()
        self.assertEqual(values[:22], changed[:22])
        self.assertEqual(values[33:55], changed[33:55])
        self.assertEqual(model.forward([1, 2, 3, 4, 5, 6], [2, 3], last_only=True).values(),
                         values[22:33] + values[55:66])
        self.assertIn("NativeCausalLm", st.__all__)

    def test_loss_shift_greedy_and_model_identity(self):
        model = self.model
        logits = model.forward([1, 2, 4, 5], [2, 2]).values()
        loss = 0.
        for i, label in enumerate([2, 3, 5, 6]):
            row = logits[i * 11:(i + 1) * 11]
            maximum = max(row)
            loss += maximum - row[label] + math.log(sum(math.exp(x - maximum) for x in row))
        self.assertAlmostEqual(model.next_token_loss([1, 2, 3, 4, 5, 6], [2, 3]), loss / 4, places=6)
        last = model.forward([1, 2, 3], [1, 3], last_only=True).values()
        self.assertEqual(model.next_token_ids([1, 2, 3], [1, 3]), [max(range(11), key=last.__getitem__)])
        self.assertEqual(st.NativeCausalLm.from_gpt2(self.config, self.weights).base_model_identity, model.base_model_identity)
        self.assertTrue(self.weights.content_id.startswith("sha256:"))
        self.assertEqual(json.loads(model.config_json())["vocabulary"], 11)

    def test_geometry_changes_logits_not_the_base_model(self):
        model = self.model
        before = model.forward([1, 2, 3], [1, 3]).values()
        for path in ("transformer.h.0", "transformer.h.1.attn", "transformer.h.0.mlp"):
            identity = model.with_geometry(artifact(model, path=path, gate=0.))
            self.assertEqual(identity.forward([1, 2, 3], [1, 3]).values(), before)
            state = artifact(model, path=path)
            adapted = model.with_geometry(state)
            self.assertEqual(adapted.geometry_artifact_id, state.artifact_id)
            self.assertNotEqual(adapted.forward([1, 2, 3], [1, 3]).values(), before)
        self.assertIsNone(model.geometry_artifact_id)
        self.assertEqual(model.forward([1, 2, 3], [1, 3]).values(), before)
        with self.assertRaisesRegex(ValueError, "unsupported native geometry placement"):
            model.with_geometry(artifact(model, path="transformer.h.99.mlp"))

    def test_rejects_invalid_ids_shapes_and_unsupported_configuration(self):
        for ids, shape in [([11], [1, 1]), ([-1], [1, 1]), ([True], [1, 1]), ([], [1, 1]),
                           ([1], [True, 1]), ([], [1, 9]), ([], [0, 1])]:
            with self.subTest(shape=shape, ids=ids), self.assertRaises((ValueError, TypeError, OverflowError)):
                self.model.forward(ids, shape)
        for key, value in (("model_type", "llama"), ("activation_function", "relu"),
                           ("add_cross_attention", True), ("reorder_and_upcast_attn", True)):
            config = json.loads(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                st.NativeCausalLm.from_gpt2(json.dumps(config), self.weights)
        with self.assertRaises(TypeError):
            st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=True)

    def test_work_budget_precedes_python_token_extraction(self):
        class Poison:
            def __len__(self):
                raise AssertionError("tokens were read before work admission")
        small = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=11)
        with self.assertRaisesRegex(ValueError, "logits"):
            small.forward(Poison(), [1, 2])
        with self.assertRaisesRegex(ValueError, "logits"):
            small.next_token_loss(Poison(), [1, 3])
        self.assertEqual(small.forward([1, 2], [1, 2], last_only=True).shape, [1, 1, 11])
        limited = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_attention_pairs=15)
        with self.assertRaisesRegex(ValueError, "attention pairs"):
            limited.forward(Poison(), [1, 2], last_only=True)


if __name__ == "__main__":
    unittest.main()
