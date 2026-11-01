"""Public document-safe training, exact continuation and no foreign ML imports."""
import hashlib
import importlib.abc
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
            raise AssertionError("native corpus training imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402
from spiraltorch import native_corpus_runs as runs  # noqa: E402

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
DOCS = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 0, 1], [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2]]
POLICY = dict(learning_rate=.001, weight_decay=.01, label_smoothing=.1,
              microbatch_windows=2, accumulation_steps=2, epochs=2)


class NativeCausalLmTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = fixtures["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(self.config, self.weights)
        self.model = self.base.with_geometry(fixtures["artifact"](self.base, trainable=True))
        self.optimizer = st.NativeGeometryAdamW(self.model.geometry_artifact(), max_gradient_norm=.1)

    def trainer(self, *, docs=DOCS, labels=None, **overrides):
        return st.NativeCausalLmTrainer(self.model, self.optimizer, docs, 4,
                                       labels=labels, **{**POLICY, **overrides})

    def test_opt_in_site_observations_share_the_exact_training_trajectory(self):
        plain, measured = self.trainer(), self.trainer()
        self.assertIn("NativeCausalLmObservedTrainingStep", st.__all__)
        self.assertIn("NativeCausalLmObservedTrainingStep", st._rs.__all__)
        for windows in (4, 1, 4, 1):
            expected, observed = plain.step(), measured.step_observed()
            self.assertIsInstance(observed, st.NativeCausalLmObservedTrainingStep)
            self.assertEqual(observed.step.to_json(), expected.to_json())
            payload = json.loads(observed.to_json())
            self.assertEqual(payload["schema"], "spiraltorch.native_lm.observed_training_step.v1")
            self.assertEqual(payload["step"], json.loads(expected.to_json()))
            self.assertEqual(payload["data_id"], plain.data_id)
            self.assertEqual(payload["base_model_identity"], self.base.base_model_identity)
            self.assertEqual(set(payload["sites"]), {"transformer.h.0.mlp"})
            site = payload["sites"]["transformer.h.0.mlp"]
            self.assertEqual(site["elements"], windows * 3 * 4)
            self.assertGreater(site["input_rms"], 0.)
            self.assertGreater(site["relative_residual_rms"], 0.)
            self.assertIsNone(site["pre_log_radius"])
            self.assertIsNone(site["log_radius_gradient"])
            self.assertEqual(measured.to_json(), plain.to_json())
        self.assertIsNone(measured.step_observed())
        bad = self.trainer(learning_rate=3.4e38, weight_decay=3.4e38)
        before = bad.to_json()
        with self.assertRaises(ValueError):
            bad.step_observed()
        self.assertEqual(bad.to_json(), before)

    def test_fixed_order_partial_groups_and_epoch_resume_match_manual_updates(self):
        trainer = self.trainer()
        data = st.HfGeometryDataset(DOCS, 4)
        self.assertEqual(trainer.plan_json(), data.plan.to_json())
        self.assertEqual(trainer.total_updates, 4)
        model, optimizer = self.model, self.optimizer
        for update, (start, stop) in enumerate(((0, 4), (4, 5), (0, 4), (4, 5))):
            group = model.geometry_accumulator(label_smoothing=.1)
            for offset in range(start, stop, 2):
                rows = [data[i]["input_ids"] for i in range(offset, min(offset+2, stop))]
                group.accumulate_loss(model, [v for row in rows for v in row], [len(rows), 4])
            model, optimizer = group.step_adamw(model, optimizer, .001, weight_decay=.01)
            report = trainer.step()
            self.assertEqual((report.update, report.epoch, report.window_start, report.window_end),
                             (update+1, update//2, start, stop))
            self.assertEqual((report.loss, report.target_tokens), (group.loss, group.target_tokens))
            self.assertEqual(trainer.model.geometry_artifact().to_json(), model.geometry_artifact().to_json())
            self.assertEqual(trainer.optimizer.to_json(), optimizer.to_json())
            self.assertEqual(json.loads(report.to_json())["target_tokens"], group.target_tokens)
        saved = trainer.to_json()
        self.assertTrue(trainer.exhausted)
        self.assertIsNone(trainer.step())
        self.assertEqual(trainer.to_json(), saved)
        self.assertEqual((trainer.epoch, trainer.next_window), (2, 5))

    def test_observed_checkpoint_restores_exact_next_steps_without_a_run_summary(self):
        plain, tracked = self.trainer(), self.trainer()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.assertEqual(tracked.step().to_json(), plain.step().to_json())
            report, entry = runs.step_and_save_observation(tracked, directory)
            self.assertEqual(report.to_json(), plain.step().to_json())
            raw = (directory / entry["observation_file"]).read_bytes()
            pin = "sha256:" + hashlib.sha256(raw).hexdigest()
            self.assertEqual(pin, entry["observation_id"])
            receipt = json.loads(raw)
            self.assertEqual(receipt["observation"]["step"], json.loads(report.to_json()))
            self.assertEqual(receipt["checkpoint"]["updates"], 2)
            checkpoint = (directory / entry["file"]).read_bytes()
            self.assertEqual((checkpoint.decode(), entry["checkpoint_id"]), plain.checkpoint_with_id())
            self.assertEqual(entry["checkpoint_id"], "sha256:" + hashlib.sha256(checkpoint).hexdigest())
            self.assertFalse((directory / "result.json").exists())
            inference = runs.load_observed_model(directory, 2, expected_observation_id=pin, base=self.base)
            cfg = json.dumps({"max_new_tokens": 3, "execution": {"kind":"kv_cache","max_values":10000}})
            a = st.NativeCausalLmGenerator(inference, [1,2], cfg)
            b = st.NativeCausalLmGenerator(plain.model, [1,2], cfg)
            while not a.finished:
                self.assertEqual(a.step().to_json(), b.step().to_json())
            self.assertEqual(a.to_json(), b.to_json())
            with self.assertRaises((ValueError, TypeError)):
                st.NativeCausalLm.from_training_checkpoint_at(self.base, checkpoint.decode(), entry["checkpoint_id"],
                    update=True, recorded_data_id=plain.data_id, post_geometry_id=plain.model.geometry_artifact_id)
            def restore(**kw):
                return runs.restore_observed_checkpoint(directory, 2, **{
                    "expected_observation_id": pin, "base": self.base, "token_documents": DOCS, **kw})
            restored = restore()
            while not plain.exhausted:
                self.assertEqual(restored.step().to_json(), plain.step().to_json())
                self.assertEqual(restored.checkpoint_with_id(), plain.checkpoint_with_id())
            self.assertIsNone(runs.step_and_save_observation(restored, directory))
            with self.assertRaisesRegex(ValueError, "pin"):
                restore(expected_observation_id="wrong")
            with self.assertRaises(ValueError):
                restore(token_documents=[list(reversed(d)) for d in DOCS])
            with self.assertRaisesRegex(ValueError, "positive"):
                runs.restore_observed_checkpoint(directory, True, expected_observation_id=pin,
                                                 base=self.base, token_documents=DOCS)
            receipt["observation"]["data_id"] = "different"
            mispaired = json.dumps(receipt).encode()
            (directory / entry["observation_file"]).write_bytes(mispaired)
            with self.assertRaisesRegex(ValueError, "do not agree"):
                restore(expected_observation_id="sha256:" + hashlib.sha256(mispaired).hexdigest())
            with self.assertRaisesRegex(ValueError, "inference boundary"):
                runs.load_observed_model(directory,2,expected_observation_id="sha256:" + hashlib.sha256(mispaired).hexdigest(),
                                         base=self.base)
            (directory / entry["observation_file"]).write_bytes(raw)
            (directory / entry["file"]).write_bytes(checkpoint + b" ")
            with self.assertRaisesRegex(ValueError, "pinned"):
                restore()

    def test_unsealed_observation_is_not_a_resume_point_or_overwritten(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, first = runs.step_and_save_observation(trainer, directory)
            publish = runs._publish
            def fail_on_receipt(path, raw):
                if path.name.startswith("observation-"):
                    raise OSError("simulated storage failure")
                publish(path, raw)
            with mock.patch.object(runs, "_publish", side_effect=fail_on_receipt), self.assertRaises(OSError):
                runs.step_and_save_observation(trainer, directory)
            self.assertEqual(trainer.completed_updates, 2)
            self.assertTrue((directory / "checkpoint-000002.json").is_file())
            self.assertFalse((directory / "observation-000002.json").exists())
            with self.assertRaises(FileNotFoundError):
                runs.restore_observed_checkpoint(directory, 2, expected_observation_id=first["observation_id"],
                                                 base=self.base, token_documents=DOCS)
            prior = runs.restore_observed_checkpoint(directory, 1, expected_observation_id=first["observation_id"],
                                                     base=self.base, token_documents=DOCS)
            with self.assertRaises(FileExistsError):
                runs.step_and_save_observation(prior, directory)
            self.assertEqual(prior.completed_updates, 1)
            plain = self.trainer()
            plain.step()
            self.assertEqual(prior.step().to_json(), plain.step().to_json())
            self.assertEqual(prior.to_json(), trainer.to_json())

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_observed_saves_reject_nonprivate_outputs_before_training(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "shared"
            directory.mkdir()
            directory.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "private"):
                runs.step_and_save_observation(trainer, directory)
            self.assertEqual(trainer.completed_updates, 0)
            self.assertEqual(list(directory.iterdir()), [])

    def test_owned_inputs_snapshots_and_checkpoint_continuation(self):
        docs = [list(d) for d in DOCS]
        trainer = self.trainer(docs=docs)
        snapshot = trainer.model
        docs[1][0] = 10
        first = trainer.step()
        self.assertEqual(snapshot.geometry_artifact_id, first.pre_geometry_artifact_id)
        restored = st.NativeCausalLmTrainer.from_json(self.base, DOCS, trainer.to_json(), trainer.checkpoint_id)
        self.assertEqual(restored.data_id, trainer.data_id)
        self.assertEqual(restored.to_json(), trainer.to_json())
        while not trainer.exhausted:
            self.assertEqual(restored.step().to_json(), trainer.step().to_json())
            self.assertEqual(restored.to_json(), trainer.to_json())
        self.assertEqual(self.model.geometry_artifact_id, snapshot.geometry_artifact_id)
        self.assertEqual(self.optimizer.revision, 0)
        for name in ("NativeCausalLmTrainer", "NativeCausalLmTrainingStep"):
            self.assertIn(name, st.__all__)

    def test_labels_are_native_weighted_and_identity_binds_ignored_tail_tokens(self):
        labels = [list(d) for d in DOCS]
        labels[1][1] = labels[2][1] = -100
        trainer = self.trainer(labels=labels, learning_rate=0.)
        report = trainer.step()
        self.assertEqual(report.target_tokens, 10)
        self.assertEqual(trainer.optimizer.revision, 0)
        self.assertEqual(trainer.model.geometry_artifact_id, self.model.geometry_artifact_id)
        restored = st.NativeCausalLmTrainer.from_json(self.base, DOCS, trainer.to_json(), trainer.checkpoint_id, labels=labels)
        self.assertEqual(restored.to_json(), trainer.to_json())
        for doc, offset in ((0, 0), (1, 8)):
            changed = [list(d) for d in DOCS]
            changed[doc][offset] = 3
            with self.assertRaisesRegex(ValueError, "identity"):
                st.NativeCausalLmTrainer.from_json(self.base, changed, trainer.to_json(), trainer.checkpoint_id, labels=labels)
        labels[1][1] = 5
        with self.assertRaisesRegex(ValueError, "identity"):
            st.NativeCausalLmTrainer.from_json(self.base, DOCS, trainer.to_json(), trainer.checkpoint_id, labels=labels)

    def test_budgets_and_bad_checkpoint_precede_foreign_value_extraction(self):
        class Unreadable:
            def __len__(self):
                raise AssertionError("foreign data touched")
        class TooMany:
            def __len__(self):
                return st.NativeCausalLmTrainer.MAX_DOCUMENTS + 1
            def __getitem__(self, index):
                raise AssertionError("over-budget outer data touched")
        class TooLong:
            def __len__(self):
                return st.NativeCausalLmTrainer.MAX_TOKENS + 1
            def __getitem__(self, index):
                raise AssertionError("over-budget token accessed")
        with self.assertRaisesRegex(ValueError, "document count"):
            self.trainer(docs=TooMany())
        with self.assertRaisesRegex(ValueError, "token count"):
            self.trainer(docs=[TooLong()])
        with self.assertRaises(ValueError):
            self.trainer(docs=Unreadable(), accumulation_steps=0)
        trainer = self.trainer()
        with self.assertRaisesRegex(ValueError, "pinned"):
            st.NativeCausalLmTrainer.from_json(self.base, Unreadable(), trainer.to_json(), "wrong")
        changed = self.model.step_geometry_sgd(self.model.geometry_loss_gradients([1, 2, 3], [1, 3]), .1)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            st.NativeCausalLmTrainer(changed, self.optimizer, Unreadable(), 4, **POLICY)
        class BadLabels:
            def __len__(self):
                raise RuntimeError("label length failed")
        with self.assertRaisesRegex(RuntimeError, "label length failed"):
            self.trainer(labels=BadLabels())
        for kwargs in (dict(epochs=True), dict(microbatch_windows=True), dict(learning_rate=True),
                       dict(label_smoothing=True), dict(weight_decay=-1.)):
            with self.assertRaises((ValueError, TypeError)):
                self.trainer(docs=Unreadable(), **kwargs)

    def test_shapes_labels_cursor_forgery_and_numeric_failure_are_atomic(self):
        trainer = self.trainer(learning_rate=3.4e38, weight_decay=3.4e38)
        before = trainer.to_json()
        with self.assertRaises(ValueError):
            trainer.step()
        self.assertEqual(trainer.to_json(), before)
        for labels in ([[-100]*len(d) for d in DOCS], [[], [], []], [[True]*len(d) for d in DOCS]):
            with self.assertRaises((ValueError, TypeError)):
                self.trainer(labels=labels)
        trainer = self.trainer()
        trainer.step()
        record = json.loads(trainer.to_json())
        for key, value in (("updates", 0), ("updates", 99), ("initial_optimizer_revision", 2**64-1)):
            changed = dict(record, **{key: value})
            payload = json.dumps(changed)
            pin = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
            with self.assertRaises(ValueError):
                st.NativeCausalLmTrainer.from_json(self.base, DOCS, payload, pin)

    def test_conversion_uses_checked_lengths_not_foreign_iterators(self):
        class Indexed:
            def __init__(self, values):
                self.values = values
            def __len__(self):
                return len(self.values)
            def __getitem__(self, index):
                assert 0 <= index < len(self.values)
                return self.values[index]
            def __iter__(self):
                raise AssertionError("unbounded iterator used")
        trainer = self.trainer(docs=Indexed([Indexed(d) for d in DOCS]))
        self.assertEqual(trainer.data_id, self.trainer().data_id)

    def test_resealed_partial_clock_reset_is_not_a_valid_trainer_checkpoint(self):
        record = json.loads(self.model.geometry_artifact().state_json())
        original = record["placements"][0]
        record["placements"] = [dict(original, path=path) for path in
                                ("transformer.h.0.attn", "transformer.h.0.mlp", "transformer.h.1")]
        artifact = st.GeometryAdapterArtifact.create_json(json.dumps(record))
        trainer = st.NativeCausalLmTrainer(self.base.with_geometry(artifact), st.NativeGeometryAdamW(artifact),
                                          DOCS, 4, **POLICY)
        trainer.step()
        record = json.loads(trainer.to_json())
        optimizer = json.loads(record["optimizer_json"])
        optimizer["slots"][0]["steps"] = 0
        for name in ("first", "second"):
            optimizer["slots"][0][name] = [0.] * len(optimizer["slots"][0][name])
        record["optimizer_json"] = json.dumps(optimizer)
        record["optimizer_id"] = "sha256:" + hashlib.sha256(record["optimizer_json"].encode()).hexdigest()
        payload = json.dumps(record)
        pin = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "clocks"):
            st.NativeCausalLmTrainer.from_json(self.base, DOCS, payload, pin)

    def test_example_rejects_ambiguous_modes_and_output_reuse_before_model_io(self):
        example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/native_geometry_corpus_training.py"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            common = ["--model-dir", str(path / "missing-model"), "--data-json", str(path / "missing-data"),
                      "--output", str(path / "new-run")]
            fresh = ["--recipe-json", "recipe", "--artifact", "geometry", "--artifact-id", "pin"]
            for extra in ([], ["--resume", "checkpoint"], ["--resume", "checkpoint", "--checkpoint-id", "pin", *fresh],
                          [*fresh, "--save-every", "0"], [*fresh, "--stop-after", "-1"],
                          [*fresh, "--observe-every", "0"], [*fresh, "--observe-every", "-1"],
                          [*fresh, "--output", str(path)]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    example["main"]([*common, *extra])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(list(path.iterdir()), [])
            with mock.patch.object(st, "NativeCausalLmTrainer", object), contextlib.redirect_stderr(io.StringIO()) as msg:
                with self.assertRaises(SystemExit):
                    example["main"]([*common, *fresh, "--observe-every", "1"])
                self.assertIn("after 0.4.33", msg.getvalue())
                self.assertEqual(list(path.iterdir()), [])
            bounded = path / "bounded.json"
            bounded.write_bytes(b"abc")
            self.assertEqual(example["read"](bounded, 3), b"abc")
            with self.assertRaisesRegex(ValueError, "budget"):
                example["read"](bounded, 2)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_example_run_directory_is_private_even_with_permissive_umask(self):
        example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/native_geometry_corpus_training.py"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent" / "run"
            previous = os.umask(0)
            try:
                example["private_output"](path)
            finally:
                os.umask(previous)
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                example["private_output"](path)


if __name__ == "__main__":
    unittest.main()
