"""Rust-owned generation in an interpreter that forbids foreign ML imports."""
import hashlib
import importlib.abc
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import struct
import tempfile
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
            raise AssertionError("native generation imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))


def pin(payload):
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class NativeCausalLmGenerationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = fixtures["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(self.config, self.weights)
        self.model = self.base.with_geometry(fixtures["artifact"](self.base, trainable=True))

    def generator(self, prompt=None, **config):
        return st.NativeCausalLmGenerator(self.model, [1, 2] if prompt is None else prompt,
                                          json.dumps({"max_new_tokens": 5, **config}))

    def test_greedy_matches_native_logits_and_owns_prompt(self):
        prompt = [1, 2]
        generator = self.generator(prompt)
        prompt[0] = 8
        self.assertEqual(generator.token_ids, [1, 2])
        for n in range(1, 6):
            expected = self.model.next_token_ids(generator.token_ids, [1, len(generator.token_ids)])[0]
            report = generator.step()
            self.assertEqual((report.step, report.token_id), (n, expected))
            self.assertEqual(json.loads(report.to_json())["token_id"], expected)
            self.assertGreater(report.probability, 0.)
        self.assertTrue(generator.finished)
        self.assertEqual(generator.finish_reason, "length")
        before = generator.to_json()
        self.assertIsNone(generator.step())
        self.assertEqual(before, generator.to_json())
        self.assertEqual(generator.checkpoint_id, pin(before))
        for name in ("NativeCausalLmGenerator", "NativeCausalLmGenerationStep"):
            self.assertIn(name, st.__all__)

    def test_sampled_zspace_exact_resume_and_control_does_not_disappear(self):
        generator = self.generator(seed=132, decoding={"selection": "sample", "top_k": 6, "top_p": .8,
                                  "distribution": {"kind": "zspace", "config": {"curvature": -1.}}})
        for _ in range(5):
            resumed = st.NativeCausalLmGenerator.from_json(self.model, generator.to_json(), generator.checkpoint_id)
            report = generator.step()
            self.assertEqual(report.to_json(), resumed.step().to_json())
            self.assertEqual(generator.to_json(), resumed.to_json())
            payload = json.loads(report.to_json())
            self.assertEqual(payload["zspace_contract"], "spiraltorch.zspace_generation_control.v1")
            self.assertLessEqual(payload["candidates"], 6)
        self.assertEqual(json.loads(generator.config_json())["seed"], 132)

    def test_eos_prompt_is_not_stop_and_generated_eos_is(self):
        expected = self.model.next_token_ids([1, 2], [1, 2])[0]
        generator = self.generator(eos_token_ids=[expected], max_new_tokens=1)
        self.assertFalse(generator.finished)
        self.assertEqual(generator.step().finish_reason, "eos")
        self.assertIsNone(generator.step())
        generator = self.generator([expected], eos_token_ids=[expected])
        self.assertFalse(generator.finished)
        self.assertIsNotNone(generator.step())

    def test_admission_precedes_foreign_token_extraction_and_does_not_iterate(self):
        class TooLong:
            def __len__(self):
                return 100_000
            def __getitem__(self, index):
                raise AssertionError("over-budget token accessed")
        class Indexed:
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return [1, 2][index]
            def __iter__(self):
                raise AssertionError("iterator must not be consumed")
        with self.assertRaises(ValueError):
            self.generator(TooLong())
        self.assertEqual(self.generator(Indexed()).token_ids, [1, 2])
        for prompt in ([], [True], [11], [-1]):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                self.generator(prompt)
        for config in ({"max_new_tokens": True}, {"max_new_tokens": 0}, {"unknown": 1},
                       {"decoding": {"top_k": 0}}, {"decoding": {"top_p": 0.}},
                       {"decoding": {"selection": "torch"}}, {"seed": -1},
                       {"eos_token_ids": [1, 1]}, {"eos_token_ids": [True]}):
            with self.assertRaises(ValueError):
                self.generator(**config)

    def test_pin_geometry_and_resealed_invalid_history_are_rejected(self):
        generator = self.generator()
        original = generator.to_json()
        with self.assertRaisesRegex(ValueError, "pinned"):
            st.NativeCausalLmGenerator.from_json(self.model, original, "wrong")
        with self.assertRaisesRegex(ValueError, "owner"):
            st.NativeCausalLmGenerator.from_json(self.base, original, pin(original))
        changes = [{"rng": "foreign"}, {"schema": "future"}, {"prompt_length": 0},
                   {"tokens": [1, 11]}, {"tokens": [1]}, {"tokens": [1, 2, 3, 4, 5, 6, 7, 8]},
                   {"base_model_identity": "wrong"}]
        for change in changes:
            payload = json.dumps({**json.loads(original), **change})
            with self.assertRaises(ValueError):
                st.NativeCausalLmGenerator.from_json(self.model, payload, pin(payload))
        saved = json.loads(original)
        saved["config"]["eos_token_ids"] = [3]
        saved["tokens"] = [1, 2, 3, 4]
        payload = json.dumps(saved)
        with self.assertRaisesRegex(ValueError, "EOS"):
            st.NativeCausalLmGenerator.from_json(self.model, payload, pin(payload))

    def test_numeric_failure_does_not_advance_state(self):
        generator = self.generator([1, 1], decoding={"selection": "sample",
                                  "distribution": {"kind": "zspace", "config": {"repression_strength": sys.float_info.max}}})
        saved = generator.to_json()
        for _ in range(2):
            with self.assertRaises(ValueError):
                generator.step()
            self.assertEqual(generator.to_json(), saved)

    def test_training_checkpoint_handoff_preserves_learned_geometry(self):
        optimizer = st.NativeGeometryAdamW(self.model.geometry_artifact(), max_gradient_norm=.1)
        trainer = st.NativeCausalLmTrainer(self.model, optimizer, [[1, 2, 3, 4]], 4, learning_rate=.01)
        trainer.step()
        model = st.NativeCausalLm.from_training_checkpoint(self.base, trainer.to_json(), trainer.checkpoint_id)
        self.assertEqual(model.geometry_artifact_id, trainer.model.geometry_artifact_id)
        self.assertNotEqual(model.geometry_artifact_id, self.model.geometry_artifact_id)
        a = st.NativeCausalLmGenerator(model, [1, 2], '{"max_new_tokens": 3}')
        b = st.NativeCausalLmGenerator(trainer.model, [1, 2], '{"max_new_tokens": 3}')
        for _ in range(3):
            self.assertEqual(a.step().to_json(), b.step().to_json())
        with self.assertRaisesRegex(ValueError, "pinned"):
            st.NativeCausalLm.from_training_checkpoint(self.base, trainer.to_json(), "wrong")

    def test_public_example_full_and_resumed_private_outputs_match(self):
        example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/native_geometry_generation.py"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header, payload = {}, bytearray()
            for name in self.weights.names():
                tensor = self.weights.tensor(name)
                values = tensor.values()
                start = len(payload)
                payload.extend(struct.pack("<" + "f" * len(values), *values))
                header[name] = {"dtype": "F32", "shape": tensor.shape, "data_offsets": [start, len(payload)]}
            raw = json.dumps(header).encode()
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
            (root / "config.json").write_text(self.config)
            (root / "prompt.json").write_text("[1, 2]")
            (root / "recipe.json").write_text('{"max_new_tokens": 5, "seed": 123, "decoding": {"selection": "sample"}}')
            fresh = ["--model-dir", str(root), "--prompt-json", str(root / "prompt.json"), "--recipe-json", str(root / "recipe.json")]
            previous = os.umask(0)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    example["main"]([*fresh, "--output", str(root / "full")])
                    example["main"]([*fresh, "--output", str(root / "part"), "--stop-after", "2"])
                    saved = (root / "part/generation.json").read_text()
                    example["main"](["--model-dir", str(root), "--output", str(root / "resumed"),
                                     "--resume", str(root / "part/generation.json"), "--resume-id", pin(saved)])
            finally:
                os.umask(previous)
            self.assertEqual((root / "full/generation.json").read_bytes(), (root / "resumed/generation.json").read_bytes())
            self.assertEqual((root / "full/steps.jsonl").read_bytes(),
                             (root / "part/steps.jsonl").read_bytes() + (root / "resumed/steps.jsonl").read_bytes())
            if os.name == "posix":
                self.assertEqual((root / "full").stat().st_mode & 0o777, 0o700)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                example["main"]([*fresh, "--output", str(root / "full")])

    def test_example_rejects_ambiguous_modes_before_model_io(self):
        example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/native_geometry_generation.py"))
        with tempfile.TemporaryDirectory() as directory:
            common = ["--model-dir", directory + "/missing", "--output", directory + "/run"]
            for extra in ([], ["--resume", "saved"], ["--training-checkpoint", "saved"],
                          ["--resume", "saved", "--resume-id", "pin", "--prompt-json", "prompt"],
                          ["--artifact", "a", "--artifact-id", "pin", "--training-checkpoint", "b", "--training-checkpoint-id", "pin"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    example["main"]([*common, *extra])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
