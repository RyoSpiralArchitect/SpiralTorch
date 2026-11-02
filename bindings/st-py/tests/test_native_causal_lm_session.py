"""One Rust-owned training contract, with no optional ML runtime."""
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
            raise AssertionError("training session imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())
import spiraltorch as st  # noqa: E402

FIXTURE = str(Path(__file__).with_name("test_native_causal_lm.py"))
fixtures = runpy.run_path(FIXTURE)
DOCS = [[1, 2, 3, 4, 5, 6], [6, 4, 2, 1]]
WINDOW = [1, 2, 3, 4]


def owners(readout="f32"):
    config, weights = fixtures["fixture"]()
    base = st.NativeCausalLm.from_gpt2(config, weights).with_readout_accumulator(readout)
    geometry = fixtures["artifact"](base, path="transformer.h.0.attn", trainable=True)
    model = base.with_lora(0, 2, 4.).with_geometry(geometry)
    return base, model


def session(model, mode="joint"):
    return st.NativeCausalLmTrainingSession(model, DOCS, 4, learning_rate=.2, mode=mode, epochs=2)


def finish(trainer):
    reports = []
    while (report := trainer.step()) is not None:
        reports.append(report.to_json())
    return reports


class NativeCausalLmSessionTests(unittest.TestCase):
    def test_groups_update_atomically_from_one_snapshot(self):
        base, model = owners()
        initial = model.adaptation_loss_gradients(WINDOW, [1, 4])
        model = model.step_adaptation_sgd(initial, "lora", .2)
        # Freeze a meaningful nonzero LoRA branch, not merely its zero-delta initialization.
        gradients = model.adaptation_loss_gradients(WINDOW, [1, 4])
        self.assertEqual(gradients.loss, model.next_token_loss(WINDOW, [1, 4]))
        self.assertEqual(gradients.readout_accumulator, "f32")
        self.assertEqual(gradients.geometry_artifact_id, model.geometry_artifact_id)
        self.assertEqual(gradients.lora_artifact_id, model.lora_artifact_id)
        self.assertEqual(set(gradients.parameter_gradients()), {"transformer.h.0.attn", "transformer.h.0.attn.c_proj"})
        for mode in ("lora", "geometry", "joint"):
            with self.subTest(mode=mode):
                changed = model.step_adaptation_sgd(gradients, mode, .2)
                self.assertEqual(changed.base_model_identity, base.base_model_identity)
                self.assertEqual(changed.lora_artifact_id == model.lora_artifact_id, mode == "geometry")
                self.assertEqual(changed.geometry_artifact_id == model.geometry_artifact_id, mode == "lora")
                with self.assertRaises(ValueError):
                    changed.step_adaptation_sgd(gradients, mode, .2)
        before = (model.lora_to_json(), model.geometry_artifact().to_json())
        with self.assertRaises(ValueError):
            model.step_adaptation_sgd(gradients, "joint", float("inf"))
        self.assertEqual(before, (model.lora_to_json(), model.geometry_artifact().to_json()))
        with self.assertRaises(ValueError):
            model.lora_loss_gradients(WINDOW, [1, 4])
        with self.assertRaises(ValueError):
            model.geometry_loss_gradients(WINDOW, [1, 4])

    def test_session_modes_checkpoint_and_exhaustion(self):
        for mode in ("lora", "geometry", "joint"):
            for readout in ("f32", "f64"):
                with self.subTest(mode=mode, readout=readout):
                    base, model = owners(readout)
                    whole, split = session(model, mode), session(model, mode)
                    expected = finish(whole)
                    first = split.step().to_json()
                    restored = st.NativeCausalLmTrainingSession.from_json(
                        base.with_readout_accumulator("f32"), DOCS, split.to_json(), split.checkpoint_id)
                    self.assertEqual(restored.model.readout_accumulator, readout)
                    self.assertEqual([first] + finish(restored), expected)
                    self.assertEqual(restored.to_json(), whole.to_json())
                    self.assertEqual(restored.completed_updates, restored.total_updates)
                    self.assertTrue(restored.exhausted)
                    self.assertIsNone(restored.step())
                    self.assertEqual(restored.model.base_model_identity, base.base_model_identity)
                    self.assertEqual(json.loads(restored.config_json())["mode"], mode)
        for name in ("NativeCausalLmAdaptationGradients", "NativeCausalLmTrainingSession", "NativeCausalLmTrainingSessionStep"):
            self.assertIn(name, st.__all__)

    def test_fresh_process_restores_both_groups_and_readout(self):
        script = r'''
import importlib.abc, json, runpy, sys
class NoForeign(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','transformers','accelerate','safetensors'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoForeign())
import spiraltorch as st
request=json.load(sys.stdin)
config, weights=runpy.run_path(request['fixture'])['fixture']()
base=st.NativeCausalLm.from_gpt2(config,weights)
trainer=st.NativeCausalLmTrainingSession.from_json(base,request['docs'],request['checkpoint'],request['id'])
reports=[]
while (report:=trainer.step()) is not None: reports.append(report.to_json())
generator=st.NativeCausalLmGenerator(trainer.model,[1,2],'{"max_new_tokens":2}')
while not generator.finished: generator.step()
print(json.dumps({'reports':reports,'checkpoint':trainer.to_json(),'tokens':generator.token_ids}))
'''
        for readout in ("f32", "f64"):
            base, model = owners(readout)
            trainer = session(model)
            trainer.step()
            payload = {"fixture": FIXTURE, "docs": DOCS, "checkpoint": trainer.to_json(), "id": trainer.checkpoint_id}
            reports = finish(trainer)
            child = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload), text=True,
                                   capture_output=True, timeout=60, cwd=tempfile.gettempdir())
            self.assertEqual(child.returncode, 0, child.stderr)
            result = json.loads(child.stdout)
            self.assertEqual(result["reports"], reports)
            self.assertEqual(result["checkpoint"], trainer.to_json())
            generator = st.NativeCausalLmGenerator(trainer.model, [1, 2], '{"max_new_tokens":2}')
            while not generator.finished:
                generator.step()
            self.assertEqual(result["tokens"], generator.token_ids)

    @unittest.skipUnless(os.environ.get("ST_NATIVE_SESSION_EXAMPLE"),
                         "set ST_NATIVE_SESSION_EXAMPLE to the built Rust native_training_session example")
    def test_python_rust_python_checkpoint_handoff(self):
        for readout in ("f32", "f64"):
            base, model = owners(readout)
            trainer = session(model)
            trainer.step()
            config, weights = fixtures["fixture"](raw_weights=True)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(config)
                (root / "weights.safetensors").write_bytes(weights)
                (root / "docs.json").write_text(json.dumps(DOCS))
                (root / "input.json").write_text(trainer.to_json())
                command = [os.environ["ST_NATIVE_SESSION_EXAMPLE"], str(root / "config.json"),
                           str(root / "weights.safetensors"), str(root / "docs.json"),
                           str(root / "input.json"), trainer.checkpoint_id, str(root / "output.json")]
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
                receipt = json.loads(result.stdout)
                expected = trainer.step()
                expected_report = json.loads(expected.to_json())
                # json! wraps the Rust report in Value, widening f32 to f64 for printing.
                self.assertEqual(receipt["step"].pop("loss"), expected.loss)
                expected_report.pop("loss")
                self.assertEqual(receipt["step"], expected_report)
                checkpoint = (root / "output.json").read_text()
                self.assertEqual(checkpoint, trainer.to_json())
                self.assertEqual(receipt["checkpoint_id"], trainer.checkpoint_id)
                resumed = st.NativeCausalLmTrainingSession.from_json(
                    base.with_readout_accumulator("f32"), DOCS, checkpoint, receipt["checkpoint_id"])
                self.assertEqual(finish(resumed), finish(trainer))
                self.assertEqual(resumed.to_json(), trainer.to_json())

    def test_admission_and_restore_fail_without_mutation(self):
        base, model = owners()
        trainer = session(model)
        trainer.step()
        payload, identity = trainer.to_json(), trainer.checkpoint_id
        with self.assertRaises(ValueError):
            st.NativeCausalLmTrainingSession.from_json(base, [[1, 2, 3]], payload, identity)
        with self.assertRaises(ValueError):
            st.NativeCausalLmTrainingSession.from_json(base, DOCS, payload + " ", identity)
        with self.assertRaises(ValueError):
            st.NativeCausalLmTrainingSession(base, DOCS, 4, learning_rate=.2)
        for kw in ({"mode":"unknown"}, {"epochs": True}, {"learning_rate": True}):
            options={"learning_rate":.2}; options.update(kw)
            with self.assertRaises((ValueError,TypeError)):
                st.NativeCausalLmTrainingSession(model, DOCS, 4, **options)
        class Poison:
            def __len__(self):
                raise AssertionError("token values accessed before shape admission")
        with self.assertRaises(ValueError):
            model.adaptation_loss_gradients(Poison(), [1, 99])
        self.assertEqual(trainer.to_json(), payload)


if __name__ == "__main__":
    unittest.main()
