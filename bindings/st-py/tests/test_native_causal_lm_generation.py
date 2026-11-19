"""Rust-owned generation in an interpreter that forbids foreign ML imports."""
import contextlib
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
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

    def test_id_only_generation_interleaves_with_reports_and_restore_for_every_policy(self):
        for execution in ({"kind":"full_prefix"},{"kind":"kv_cache","max_values":1024}):
            for selection in ("greedy","sample"):
                for dist in ({"kind":"softmax","temperature":.75},{"kind":"zspace","config":{"curvature":-1.}}):
                    with self.subTest(execution=execution,selection=selection,distribution=dist["kind"]):
                        kw=dict(execution=execution,seed=37,decoding=dict(selection=selection,distribution=dist,top_k=6,top_p=.8))
                        full,ids=self.generator(**kw),self.generator(**kw)
                        self.assertEqual(ids.step_token_id(),full.step().token_id)
                        self.assertEqual(ids.checkpoint_id,full.checkpoint_id)
                        ids=st.NativeCausalLmGenerator.from_json(self.model,ids.to_json(),ids.checkpoint_id)
                        self.assertEqual(ids.advance_ids(2),[full.step().token_id for _ in range(2)])
                        self.assertEqual(ids.step().to_json(),full.step().to_json())
                        self.assertEqual(ids.advance_ids(1024),[full.step().token_id])
                        self.assertEqual(ids.generated_ids,full.generated_ids)
                        self.assertEqual(ids.to_json(),full.to_json())
                        self.assertIsNone(ids.step_token_id())
                        self.assertEqual(ids.advance_ids(1),[])

    def test_id_chunk_validation_and_eos_preserve_cursor(self):
        token=self.generator().step().token_id
        ids=self.generator(eos_token_ids=[token])
        before=ids.to_json()
        for count in (0,-1,True,1025):
            with self.assertRaises((ValueError,TypeError,OverflowError)):
                ids.advance_ids(count)
            self.assertEqual(ids.to_json(),before)
        self.assertEqual(ids.advance_ids(4),[token])
        self.assertEqual(ids.finish_reason,"eos")
        self.assertEqual(ids.advance_ids(3),[])

    def test_generation_checkpoint_intervals_match_continuous_sampling(self):
        policy = {"execution": {"kind": "kv_cache", "max_values": 1024}, "seed": 41,
                  "decoding": {"selection": "sample",
                               "distribution": {"kind": "zspace", "config": {"curvature": -1.}}}}
        full, split = self.generator(**policy), self.generator(**policy)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero, partial, final = (root / name for name in ("zero.json", "partial.json", "final.json"))
            empty, initial = split.advance_to_checkpoint_new(0, zero)
            self.assertEqual(empty, [])
            self.assertEqual(split.checkpoint_with_id(), (zero.read_text(encoding="utf-8"), initial))
            with self.assertRaises(FileExistsError):
                split.advance_to_checkpoint_new(2, zero)
            self.assertEqual(split.checkpoint_with_id(), (zero.read_text(encoding="utf-8"), initial))
            for bad_stop in (-1, True, 6):
                with self.assertRaises((TypeError, ValueError, OverflowError)):
                    split.advance_to_checkpoint_new(bad_stop, partial)
                self.assertFalse(partial.exists())

            a, partial_id = split.advance_to_checkpoint_new(2, partial)
            self.assertEqual(split.checkpoint_with_id(), (partial.read_text(encoding="utf-8"), partial_id))
            split = st.NativeCausalLmGenerator.from_json(
                self.model, partial.read_text(encoding="utf-8"), partial_id,
            )
            b, final_id = split.advance_to_checkpoint_new(5, final)
            self.assertEqual([report.to_json() for report in a + b], [full.step().to_json() for _ in range(5)])
            self.assertEqual((final.read_text(encoding="utf-8"), final_id), full.checkpoint_with_id())
            self.assertEqual(split.checkpoint_with_id(), full.checkpoint_with_id())
            if os.name == "posix":
                self.assertEqual(final.stat().st_mode & 0o777, 0o600)

    def test_generation_checkpoint_can_stop_early_at_eos(self):
        token = self.generator().step_token_id()
        generator = self.generator(eos_token_ids=[token])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps, eos_id = generator.advance_to_checkpoint_new(5, root / "eos.json")
            self.assertEqual([(s.token_id, s.finish_reason) for s in steps], [(token, "eos")])
            self.assertTrue(generator.finished)
            self.assertEqual(generator.generated_ids, [token])
            self.assertEqual(generator.advance_to_checkpoint_new(5, root / "copy.json"), ([], eos_id))
            self.assertEqual((root / "eos.json").read_bytes(), (root / "copy.json").read_bytes())

    @unittest.skipUnless(os.name == "posix", "requires a child-local file size limit")
    def test_generation_short_write_preserves_prior_checkpoint_and_warm_cache(self):
        script = r"""
import errno, resource, runpy, signal, sys
from pathlib import Path

test = runpy.run_path(sys.argv[1])["NativeCausalLmGenerationTests"]()
test.setUp()
root = Path(sys.argv[2])
previous, failed = root / "previous.json", root / "failed.json"
recipe = dict(execution={"kind": "kv_cache", "max_values": 1024},
              seed=39, decoding={"selection": "sample"})
generator, reference = test.generator(**recipe), test.generator(**recipe)
assert generator.step().to_json() == reference.step().to_json()
saved = generator.checkpoint_with_id()
assert generator.save_checkpoint_new(previous) == saved[1]
limits = resource.getrlimit(resource.RLIMIT_FSIZE)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (32, limits[1]))
try:
    generator.advance_to_checkpoint_new(4, failed)
except OSError as exc:
    assert exc.errno == errno.EFBIG, exc
else:
    raise AssertionError("32-byte cap did not reject the generation checkpoint")
finally:
    resource.setrlimit(resource.RLIMIT_FSIZE, limits)
assert generator.checkpoint_with_id() == saved
assert previous.read_bytes() == saved[0].encode() and list(root.iterdir()) == [previous]
reports, pin = generator.advance_to_checkpoint_new(4, root / "complete.json")
assert [step.to_json() for step in reports] == [reference.step().to_json() for _ in range(3)]
assert generator.checkpoint_with_id() == reference.checkpoint_with_id()
assert pin == reference.checkpoint_id
"""
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script, str(Path(__file__).resolve()), directory],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(run.returncode, 0, run.stderr)

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
        for config in (b" " * 65_537, "\U0001f300" * 16_385):
            with self.assertRaisesRegex(ValueError, "byte budget"):
                st.NativeCausalLmGenerator(self.model, TooLong(), config)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            st.NativeCausalLmGenerator.from_json(
                self.model, b" " * (1_048_576 + 1), "sha256:" + "0" * 64,
            )
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
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                generator.advance_to_checkpoint_new(2, checkpoint)
            self.assertFalse(checkpoint.exists())
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

    def test_conditional_snapshot_rng_and_generation_ownership(self):
        state, guard, rows, pairs, schedule = fixtures["conditional_session_inputs"](self.base)
        kind = st.NativeCausalLmConditionalAnswerSession
        session = kind(state, guard, rows, schedule, length=.01)
        earlier = kind.serving_from_checkpoint(self.base, *session.checkpoint_with_id())
        repeated = kind(state, guard, rows, schedule, length=.01, epochs=2)
        same_weights = kind.serving_from_checkpoint(self.base, *repeated.checkpoint_with_id())
        self.assertNotEqual(earlier.checkpoint_id, same_weights.checkpoint_id)
        self.assertEqual(earlier.projection_id, same_weights.projection_id)
        empty = earlier.generator([1, 2], '{"max_new_tokens":1}')
        self.assertEqual(same_weights.restore_generator(empty.to_json(), empty.checkpoint_id).to_json(),
                         empty.to_json())
        self.assertTrue(session.step()["report"]["step"]["committed"])
        snapshot = kind.serving_from_checkpoint(self.base, *session.checkpoint_with_id())
        self.assertNotEqual(earlier.projection_id, snapshot.projection_id)
        prompt = [4, 8, 2, 1]
        logits = struct.unpack("<11f", snapshot.prefill(prompt).logits_f32le())
        greedy = snapshot.generator(prompt, '{"max_new_tokens":1,"execution":{"kind":"kv_cache","max_values":1024}}')
        self.assertEqual(greedy.step_token_id(), max(range(len(logits)), key=logits.__getitem__))
        for execution in ({"kind": "full_prefix"}, {"kind": "kv_cache", "max_values": 1024}):
            with self.subTest(execution=execution):
                recipe = json.dumps({
                    "max_new_tokens": 4, "seed": 81, "execution": execution,
                    "decoding": {"selection": "sample", "top_k": 6,
                                 "distribution": {"kind": "zspace", "config": {"curvature": -1.}}},
                })
                full = snapshot.generator(prompt, recipe)
                self.assertEqual(full.projection_id, snapshot.projection_id)
                report = json.loads(full.step().to_json())
                self.assertEqual(report["zspace_contract"], "spiraltorch.zspace_generation_control.v1")
                full.step()
                saved, saved_id = full.to_json(), full.checkpoint_id
                self.assertEqual(json.loads(saved)["projection_id"], snapshot.projection_id)
                with self.assertRaisesRegex(ValueError, "projection owner mismatch"):
                    earlier.restore_generator(saved, saved_id)
                with self.assertRaisesRegex(ValueError, "projection owner mismatch"):
                    st.NativeCausalLmGenerator.from_json(self.base, saved, saved_id)
                with self.assertRaisesRegex(ValueError, "pinned identity"):
                    snapshot.restore_generator(saved + " ", saved_id)
                resumed = snapshot.restore_generator(saved.encode(), saved_id)
                for _ in range(2):
                    self.assertEqual(full.step().to_json(), resumed.step().to_json())
                self.assertEqual((full.checkpoint_id, full.to_json()), (resumed.checkpoint_id, resumed.to_json()))

        class NoValueRead:
            def __len__(self):
                return 100_000

            def __getitem__(self, _):
                raise AssertionError("prompt IDs read before generation budget")

        with self.assertRaisesRegex(ValueError, "generation token count exceeds budget"):
            snapshot.generator(NoValueRead(), b'{"max_new_tokens":4}')

    def test_conditional_generation_example_resumes_without_training_data(self):
        config, weights = fixtures["fixture"](raw_weights=True)
        model = self.base.with_readout_accumulator("f64").with_lora(0, 2, 4.)
        geometry = fixtures["artifact"](model, path="transformer.h.0.attn", trainable=True)
        model = model.with_geometry(geometry)
        state, guard, rows, pairs, schedule = fixtures["conditional_session_inputs"](model)
        kind = st.NativeCausalLmConditionalAnswerSession
        session = kind(state, guard, rows, schedule, length=.01)
        for _ in schedule:
            session.step()
        phase = session.next_phase(rows[::-1], [([0], [pairs[-1]])], length=.02, epochs=2)
        phase.step()
        raw, expected = phase.checkpoint_with_id()
        example = Path(__file__).resolve().parents[1] / "examples/native_geometry_generation.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(config, encoding="utf-8")
            (root / "model.safetensors").write_bytes(weights)
            (root / "phase.ck").write_bytes(raw)
            (root / "lora.json").write_text(model.lora_to_json(), encoding="utf-8")
            (root / "geometry.json").write_text(geometry.to_json(), encoding="utf-8")
            (root / "prompt.json").write_text("[1,2]", encoding="utf-8")
            (root / "recipe.json").write_text(json.dumps({
                "max_new_tokens": 4, "seed": 41, "execution": {"kind": "kv_cache", "max_values": 1024},
                "decoding": {"selection": "sample",
                             "distribution": {"kind": "zspace", "config": {"curvature": -1.}}},
            }), encoding="utf-8")
            base = [sys.executable, "-I", "-B", str(example), "--conditional-checkpoint", "phase.ck",
                    "--conditional-source-lora", "lora.json", "--artifact", "geometry.json",
                    "--artifact-id", geometry.artifact_id, "--readout-accumulator", "f64"]
            prompt = ["--prompt-json", "prompt.json", "--recipe-json", "recipe.json"]

            def run(output, args, *, model_dir=".", conditional_id=expected):
                return subprocess.run(
                    [*base, "--model-dir", model_dir, "--conditional-checkpoint-id", conditional_id,
                     "--output", output, *args], cwd=root, capture_output=True, text=True, timeout=30)

            bad = run("bad", prompt, model_dir="missing", conditional_id="sha256:" + "0" * 64)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("conditional checkpoint differs from pinned identity", bad.stderr)
            self.assertFalse((root / "bad").exists())
            both = run("bad", [*prompt, "--qkv-checkpoint", "missing", "--qkv-checkpoint-id", "pin"])
            self.assertEqual(both.returncode, 2)
            self.assertIn("choose either a QKV projection or a conditional projection", both.stderr)
            self.assertFalse((root / "bad").exists())
            for label, args in (("full", prompt), ("part", [*prompt, "--stop-after", "2"])):
                call = run(label, args)
                self.assertEqual(call.returncode, 0, call.stderr)
            saved = (root / "part/generation.json").read_text(encoding="utf-8")
            continued = run("continued", ["--resume", "part/generation.json", "--resume-id", pin(saved)])
            self.assertEqual(continued.returncode, 0, continued.stderr)
            self.assertEqual((root / "full/generation.json").read_bytes(), (root / "continued/generation.json").read_bytes())
            self.assertEqual((root / "full/steps.jsonl").read_bytes(),
                             (root / "part/steps.jsonl").read_bytes() + (root / "continued/steps.jsonl").read_bytes())
            result = json.loads(continued.stdout)
            self.assertEqual(result["conditional_snapshot"], {
                "checkpoint_id": expected, "projection_id": phase.projection_snapshot().projection_id,
                "position": 1, "total_batches": 2, "training_documents_verified": False,
            })
            self.assertEqual(result["lora_artifact_id"], model.lora_artifact_id)

    def test_public_example_full_and_resumed_private_outputs_match(self):
        for execution in (None, {"kind": "full_prefix"}, {"kind": "kv_cache", "max_values": 100000}):
            with self.subTest(execution=execution):
                self.check_public_example_resume(execution)

    def check_public_example_resume(self, execution):
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
            recipe = {"max_new_tokens": 5, "seed": 123, "decoding": {"selection": "sample"}}
            if execution is not None:
                recipe["execution"] = execution
            (root / "recipe.json").write_text(json.dumps(recipe))
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
            for run in ("full", "part", "resumed"):
                result = json.loads((root / run / "result.json").read_text())
                self.assertEqual(result["execution"], execution or {"kind": "full_prefix"})
                self.assertNotIn("no KV cache", result["scope"])
            from spiraltorch.native_corpus_runs import step_and_save_observation
            base = st.NativeCausalLm.from_gpt2(self.config, st.SafeTensorArchive(root / "model.safetensors"))
            initial = fixtures["artifact"](base, trainable=True)
            trainer = st.NativeCausalLmTrainer(base.with_geometry(initial), st.NativeGeometryAdamW(initial),
                                               [[1,2,3,4]], 4, learning_rate=.01)
            training = root / "training"
            training.mkdir(mode=0o700)
            _, entry = step_and_save_observation(trainer, training)
            boundary = ["--observed-run", str(training), "--observed-update", "1", "--observation-id", entry["observation_id"]]
            with contextlib.redirect_stdout(io.StringIO()):
                example["main"]([*fresh, *boundary, "--output", str(root / "observed")])
                example["main"]([*fresh, "--training-checkpoint", str(training / entry["file"]), "--training-checkpoint-id",
                                 entry["checkpoint_id"], "--output", str(root / "direct")])
                example["main"]([*fresh, *boundary, "--stop-after", "2", "--output", str(root / "observed-part")])
                saved = root / "observed-part/generation.json"
                example["main"](["--model-dir", str(root), *boundary, "--resume", str(saved), "--resume-id", pin(saved.read_text()),
                                 "--output", str(root / "observed-resumed")])
            self.assertEqual((root / "observed/steps.jsonl").read_bytes(), (root / "direct/steps.jsonl").read_bytes())
            self.assertEqual((root / "observed/generation.json").read_bytes(), (root / "direct/generation.json").read_bytes())
            self.assertEqual((root / "observed/generation.json").read_bytes(), (root / "observed-resumed/generation.json").read_bytes())
            self.assertEqual((root / "observed/steps.jsonl").read_bytes(), (root / "observed-part/steps.jsonl").read_bytes()
                             + (root / "observed-resumed/steps.jsonl").read_bytes())
            source = json.loads((root / "observed/result.json").read_bytes())["observed_training_boundary"]
            self.assertEqual(source, {"update":1,"observation_id":entry["observation_id"],"training_documents_verified":False})
            self.assertFalse((training / "result.json").exists())
            if os.name == "posix":
                self.assertEqual((root / "full").stat().st_mode & 0o777, 0o700)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                example["main"]([*fresh, "--output", str(root / "full")])

    def test_example_rejects_ambiguous_modes_before_model_io(self):
        example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/native_geometry_generation.py"))
        with tempfile.TemporaryDirectory() as directory:
            common = ["--model-dir", directory + "/missing", "--output", directory + "/run"]
            for extra in ([], ["--resume", "saved"], ["--training-checkpoint", "saved"],
                          ["--observed-run","run"], ["--observation-id","pin","--observed-update","1"],
                          ["--observed-run","run","--observed-update","0","--observation-id","pin"],
                          ["--observed-run","run","--observed-update","1","--observation-id","pin","--artifact","x","--artifact-id","p"],
                          ["--resume", "saved", "--resume-id", "pin", "--prompt-json", "prompt"],
                          ["--artifact", "a", "--artifact-id", "pin", "--training-checkpoint", "b", "--training-checkpoint-id", "pin"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    example["main"]([*common, *extra])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
