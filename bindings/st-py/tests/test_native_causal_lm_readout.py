"""Readout arithmetic is Rust-owned and bound to learning/generation continuation."""
import importlib.abc
import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "accelerate", "safetensors"}:
            raise AssertionError("native readout imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

FIXTURES = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))


@unittest.skipIf(__name__ == "__main__", "already running in the isolated child")
class ReadoutIsolationTest(unittest.TestCase):
    def test_suite_with_forbidden_imports_in_fresh_process(self):
        site = str(Path(st.__file__).resolve().parent.parent)
        code = ("import runpy,sys; sys.path.insert(0," + repr(site) + "); "
                "sys.argv=[" + repr(str(Path(__file__).resolve())) + "]; "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class NativeReadoutTests(unittest.TestCase):
    def setUp(self):
        config, weights = FIXTURES["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(config, weights)
        self.model = self.base.with_geometry(FIXTURES["artifact"](self.base, trainable=True))

    def test_policy_is_explicit_immutable_and_rejects_unknown_values(self):
        wide = self.model.with_readout_accumulator("f64")
        self.assertEqual(self.model.readout_accumulator, "f32")
        self.assertEqual(wide.readout_accumulator, "f64")
        self.assertEqual(wide.base_model_identity, self.model.base_model_identity)
        self.assertEqual(wide.geometry_artifact_id, self.model.geometry_artifact_id)
        for invalid in ("F64", "auto", "torch", "", True, 64, None):
            with self.assertRaises((ValueError, TypeError)):
                self.model.with_readout_accumulator(invalid)
        with self.assertRaises(AttributeError):
            wide.readout_accumulator = "f32"

    def test_forward_loss_context_and_gradient_owners_agree(self):
        models = [self.model.with_readout_accumulator(p) for p in ("f32", "f64")]
        windows = [1, 2, 3, 4]
        gradients = [m.geometry_loss_gradients(windows, [1, 4]) for m in models]
        for i, model in enumerate(models):
            with self.subTest(accumulator=model.readout_accumulator):
                own, foreign = gradients[i], gradients[1-i]
                self.assertEqual(own.readout_accumulator, model.readout_accumulator)
                self.assertEqual(own.loss, model.next_token_loss(windows, [1, 4]))
                context = st.NativeCausalLmContext(model, [1, 2])
                self.assertEqual(context.readout_accumulator, model.readout_accumulator)
                context.append(3)
                actual = context.logits().values()
                expected = model.forward([1, 2, 3], [1, 3], last_only=True).values()
                for a, b in zip(actual, expected):
                    self.assertAlmostEqual(a, b, places=6)
                optimizer = st.NativeGeometryAdamW(model.geometry_artifact())
                with self.assertRaises(ValueError):
                    model.step_geometry_sgd(foreign, .01)
                with self.assertRaises(ValueError):
                    model.step_geometry_adamw(foreign, optimizer, .01)
                accumulator = model.geometry_accumulator()
                self.assertEqual(accumulator.readout_accumulator, model.readout_accumulator)
                with self.assertRaises(ValueError):
                    accumulator.add(foreign)
                self.assertEqual(accumulator.microbatches, 0)
                accumulator.add(own)
                with self.assertRaises(ValueError):
                    accumulator.accumulate_loss(models[1-i], windows, [1, 4])
                self.assertEqual(accumulator.microbatches, 1)
                with self.assertRaises(ValueError):
                    accumulator.step_sgd(models[1-i], .01)
                with self.assertRaises(ValueError):
                    accumulator.step_adamw(models[1-i], optimizer, .01)
                updated, _ = accumulator.step_adamw(model, optimizer, .01)
                self.assertEqual(updated.readout_accumulator, model.readout_accumulator)

    def test_generation_resume_binds_arithmetic_in_both_execution_modes(self):
        for policy in ("f32", "f64"):
            model = self.model.with_readout_accumulator(policy)
            other = self.model.with_readout_accumulator("f32" if policy == "f64" else "f64")
            for execution in (None, {"kind": "kv_cache", "max_values": 1024}):
                config = {"max_new_tokens": 4, "seed": 81, "decoding": {
                    "selection": "sample", "top_k": 6, "top_p": .8,
                    "distribution": {"kind": "zspace", "config": {}}}}
                if execution is not None:
                    config["execution"] = execution
                generator = st.NativeCausalLmGenerator(model, [1, 2], json.dumps(config))
                generator.step()
                payload = generator.to_json()
                self.assertEqual("readout_accumulator" in json.loads(payload), policy == "f64")
                with self.assertRaises(ValueError):
                    st.NativeCausalLmGenerator.from_json(other, payload, generator.checkpoint_id)
                restored = st.NativeCausalLmGenerator.from_json(model, payload, generator.checkpoint_id)
                self.assertEqual(restored.to_json(), payload)
                while not generator.finished:
                    self.assertEqual(generator.step().to_json(), restored.step().to_json())
                    self.assertEqual(generator.to_json(), restored.to_json())

    def test_training_resume_and_inference_restore_bind_arithmetic(self):
        docs = [[1, 2, 3, 4, 5, 6, 7]]
        for policy in ("f32", "f64"):
            base = self.base.with_readout_accumulator(policy)
            model = self.model.with_readout_accumulator(policy)
            trainer = st.NativeCausalLmTrainer(model, st.NativeGeometryAdamW(model.geometry_artifact()),
                docs, 4, learning_rate=.001, epochs=2)
            trainer.step()
            payload, pin = trainer.to_json(), trainer.checkpoint_id
            self.assertEqual("readout_accumulator" in json.loads(payload), policy == "f64")
            other = self.base.with_readout_accumulator("f32" if policy == "f64" else "f64")
            with self.assertRaises(ValueError):
                st.NativeCausalLmTrainer.from_json(other, docs, payload, pin)
            with self.assertRaises(ValueError):
                st.NativeCausalLm.from_training_checkpoint(other, payload, pin)
            restored = st.NativeCausalLmTrainer.from_json(base, docs, payload, pin)
            self.assertEqual(restored.to_json(), payload)
            inference = st.NativeCausalLm.from_training_checkpoint(base, payload, pin)
            self.assertEqual(inference.readout_accumulator, policy)
            self.assertEqual(inference.geometry_artifact_id, trainer.model.geometry_artifact_id)
            while not trainer.exhausted:
                self.assertEqual(trainer.step().to_json(), restored.step().to_json())
                self.assertEqual(trainer.to_json(), restored.to_json())


if __name__ == "__main__":
    unittest.main()
