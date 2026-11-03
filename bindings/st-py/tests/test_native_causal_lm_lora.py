"""Native single-projection LoRA, including fresh-interpreter continuation."""
import importlib.abc
import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
            raise AssertionError("native LoRA imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

FIXTURE = str(Path(__file__).with_name("test_native_causal_lm.py"))
fixtures = runpy.run_path(FIXTURE)
WINDOW, SHAPE = [1, 2, 3, 4], [1, 4]


class NativeCausalLmLoraTests(unittest.TestCase):
    def setUp(self):
        config, weights = fixtures["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(config, weights)
        self.initial = self.base.with_lora(0, 2, 4.)

    def step(self, model, window=WINDOW):
        return model.step_lora_sgd(model.lora_loss_gradients(window, SHAPE), .3)

    def test_zero_delta_real_update_and_sealed_gradient_owner(self):
        base_values = self.base.forward(WINDOW, SHAPE).values()
        self.assertEqual(self.initial.forward(WINDOW, SHAPE).values(), base_values)
        gradients = self.initial.lora_loss_gradients(WINDOW, SHAPE)
        self.assertEqual(gradients.loss, self.initial.next_token_loss(WINDOW, SHAPE))
        self.assertEqual(gradients.window_shape, SHAPE)
        self.assertEqual(gradients.target_tokens, 3)
        self.assertEqual(gradients.base_model_identity, self.base.base_model_identity)
        self.assertEqual(gradients.lora_artifact_id, self.initial.lora_artifact_id)
        parameters = gradients.parameter_gradients()
        self.assertEqual(set(parameters), {"a", "b"})
        self.assertTrue(any(v != 0. for v in parameters["b"]))
        updated = self.initial.step_lora_sgd(gradients, .3)
        self.assertNotEqual(updated.lora_artifact_id, self.initial.lora_artifact_id)
        self.assertNotEqual(updated.forward(WINDOW, SHAPE).values(), base_values)
        self.assertEqual(updated.base_model_identity, self.base.base_model_identity)
        self.assertEqual(self.base.forward(WINDOW, SHAPE).values(), base_values)
        self.assertEqual(self.initial.forward(WINDOW, SHAPE).values(), base_values)
        with self.assertRaises(ValueError):
            updated.step_lora_sgd(gradients, .3)
        self.assertIn("NativeCausalLmLoraGradients", st.__all__)
        self.assertIn("NativeCausalLmLoraGradients", st._rs.__all__)

    def test_metadata_generation_and_cached_parity_after_reload(self):
        model = self.step(self.initial)
        restored = self.base.with_lora_json(model.lora_to_json())
        self.assertEqual(restored.lora_artifact_id, model.lora_artifact_id)
        context = st.NativeCausalLmContext(restored, [1, 2])
        self.assertEqual(context.lora_artifact_id, restored.lora_artifact_id)
        context.append(3)
        expected = restored.forward([1, 2, 3], [1, 3], last_only=True).values()
        for actual, wanted in zip(context.logits().values(), expected):
            self.assertAlmostEqual(actual, wanted, places=6)
        evaluation = restored.evaluate_documents([WINDOW], 4)
        self.assertEqual(evaluation.lora_artifact_id, restored.lora_artifact_id)
        self.assertEqual(json.loads(evaluation.to_json())["lora_artifact_id"], restored.lora_artifact_id)
        generator = st.NativeCausalLmGenerator(restored, [1, 2], '{"max_new_tokens": 2}')
        saved = generator.to_json()
        self.assertEqual(json.loads(saved)["lora_artifact_id"], restored.lora_artifact_id)
        with self.assertRaises(ValueError):
            st.NativeCausalLmGenerator.from_json(self.base, saved, generator.checkpoint_id)
        resumed = st.NativeCausalLmGenerator.from_json(restored, saved, generator.checkpoint_id)
        self.assertEqual(generator.step().to_json(), resumed.step().to_json())

    def test_fresh_process_sgd_continuation_and_generation(self):
        script = r'''
import importlib.abc, json, runpy, sys
from pathlib import Path
class NoForeign(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'transformers', 'peft', 'accelerate', 'safetensors'}:
            raise AssertionError('foreign ML import: ' + fullname)
sys.meta_path.insert(0, NoForeign())
import spiraltorch as st
request = json.load(sys.stdin)
if [str(Path(st.__file__).resolve()), str(Path(st._rs.__file__).resolve())] != request['origins']:
    raise ValueError('fresh process loaded a different package or extension')
fixture = runpy.run_path(request['fixture'])['fixture']
config, weights = fixture()
base = st.NativeCausalLm.from_gpt2(config, weights).with_readout_accumulator(request['readout'])
model = base.with_lora_json(request['adapter'])
model = model.step_lora_sgd(model.lora_loss_gradients([4, 3, 2, 1], [1, 4]), .3)
generator = st.NativeCausalLmGenerator(model, [1, 2], '{"max_new_tokens": 2}')
while not generator.finished:
    generator.step()
print(json.dumps({'adapter': model.lora_to_json(), 'tokens': generator.token_ids}))
'''
        for readout in ("f32", "f64"):
            with self.subTest(readout=readout):
                once = self.step(self.initial.with_readout_accumulator(readout))
                twice = self.step(once, [4, 3, 2, 1])
                result = subprocess.run(
                    [sys.executable, "-I", "-B", "-c", script],
                    input=json.dumps({"fixture": FIXTURE, "adapter": once.lora_to_json(), "readout": readout,
                                      "origins":[str(Path(st.__file__).resolve()), str(Path(st._rs.__file__).resolve())]}),
                    text=True, capture_output=True, check=True, timeout=60,
                )
                result = json.loads(result.stdout)
                self.assertEqual(result["adapter"], twice.lora_to_json())
                generator = st.NativeCausalLmGenerator(twice, [1, 2], '{"max_new_tokens": 2}')
                while not generator.finished:
                    generator.step()
                self.assertEqual(result["tokens"], generator.token_ids)

    def test_controls_admission_and_joint_geometry_are_rejected(self):
        for args in ((True, 2, 4.), (0, True, 4.), (0, 2, True), (0, 0, 4.), (2, 2, 4.), (0, 2, float("inf"))):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                self.base.with_lora(*args)
        gradients = self.initial.lora_loss_gradients(WINDOW, SHAPE)
        with self.assertRaises(TypeError):
            self.initial.step_lora_sgd(gradients, True)
        class Poison:
            def __len__(self):
                raise AssertionError("token values accessed before admission")
        with self.assertRaises(ValueError):
            self.initial.lora_loss_gradients(Poison(), [1, 99])
        with self.assertRaises(TypeError):
            self.initial.lora_loss_gradients([1, True, 3, 4], SHAPE)
        colocated = fixtures["artifact"](self.base, path="transformer.h.0.attn", trainable=True)
        with self.assertRaisesRegex(ValueError, "adaptation_loss_gradients"):
            self.initial.with_geometry(colocated).lora_loss_gradients(Poison(), SHAPE)
        geometry = fixtures["artifact"](self.base, trainable=True)
        with self.assertRaises(ValueError):
            self.initial.with_geometry(geometry)
        with self.assertRaises(ValueError):
            self.base.with_geometry(geometry).with_lora(0, 2, 4.)
        with self.assertRaises(ValueError):
            self.base.with_geometry(geometry).with_lora_json(self.initial.lora_to_json())


if __name__ == "__main__":
    unittest.main()
