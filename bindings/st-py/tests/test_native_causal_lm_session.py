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
from unittest.mock import patch


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
    def test_checkpoint_file_uses_native_bytes_and_refuses_existing_names(self):
        base, model = owners()
        with tempfile.TemporaryDirectory() as tmp:
            for batch in (1, 2):
                path = Path(tmp) / f"batch-{batch}.json"
                trainer = st.NativeCausalLmTrainingSession(model, DOCS, 4, learning_rate=.2, epochs=2, batch_windows=batch)
                trainer.step()
                payload, pin = trainer.checkpoint_with_id()
                self.assertEqual(trainer.save_checkpoint_new(path), pin)
                self.assertEqual(path.read_bytes(), payload.encode())
                recovered = st.NativeCausalLmTrainingSession.from_json(base, DOCS, path.read_bytes(), pin)
                self.assertEqual(recovered.step().to_json(), trainer.step().to_json())
                cursor = trainer.completed_updates
                with self.assertRaises(FileExistsError):
                    trainer.save_checkpoint_new(path)
                self.assertEqual(trainer.completed_updates, cursor)
                self.assertEqual(path.read_bytes(), payload.encode())

    def test_masked_backward_keeps_the_other_owner_in_the_forward(self):
        _, model = owners()
        full = model.adaptation_loss_gradients(WINDOW, [1, 4])
        all_groups = full.parameter_gradients()
        for mode in ("lora", "geometry", "joint"):
            selected = model.adaptation_loss_gradients(WINDOW, [1, 4], mode=mode)
            self.assertEqual(selected.loss, full.loss)
            self.assertEqual(selected.lora_artifact_id, full.lora_artifact_id)
            self.assertEqual(selected.geometry_artifact_id, full.geometry_artifact_id)
            actual = selected.parameter_gradients()
            self.assertEqual(len(actual), 2 if mode == "joint" else 1)
            self.assertEqual(actual, {k: all_groups[k] for k in actual})
            expected = model.step_adaptation_sgd(full, mode, .2)
            updated = model.step_adaptation_sgd(selected, mode, .2)
            self.assertEqual((updated.lora_artifact_id, updated.geometry_artifact_id),
                             (expected.lora_artifact_id, expected.geometry_artifact_id))
            if mode != "joint":
                with self.assertRaises(ValueError):
                    model.step_adaptation_sgd(selected, "joint", .2)
        with self.assertRaises(ValueError):
            model.adaptation_loss_gradients(WINDOW, [1, 4], mode="both-ish")

    @unittest.skipUnless(sys.implementation.name == "cpython", "CPython allocation guard")
    def test_mixed_unicode_input_and_tiny_file_do_not_allocate_the_worst_case_budget(self):
        probe = Path(__file__).with_name("probe_session_input_allocations.py")
        child = subprocess.run([sys.executable,"-I","-B","-X","tracemalloc=1",str(probe)],
                               capture_output=True,text=True,check=True,timeout=30)
        peaks = json.loads(child.stdout)
        self.assertLess(peaks["unicode_peak"], 2*1024*1024+262144)
        self.assertEqual(len(peaks["file_peaks"]),2)
        self.assertTrue(all(p < 262144 for p in peaks["file_peaks"]))

    @unittest.skipUnless(os.name == "posix", "directory descriptor on POSIX")
    def test_both_example_readers_close_rejected_directory_descriptors(self):
        examples = Path(__file__).parents[1] / "examples"
        with tempfile.TemporaryDirectory() as tmp:
            for filename, helper in (("native_training_session.py","read_limited"),("native_geometry_generation.py","read_bytes")):
                read = runpy.run_path(str(examples/filename))[helper]
                real, opened = os.open, []
                def track(path, flags):
                    fd = real(path, flags)
                    opened.append(fd)
                    return fd
                with patch("os.open", side_effect=track), self.assertRaisesRegex(ValueError,"regular file"):
                    read(Path(tmp),256*1024*1024)
                self.assertEqual(len(opened),1)
                with self.assertRaises(OSError):
                    os.fstat(opened[0])

    def test_batched_joint_steps_do_not_cross_rows_or_drop_epoch_tails(self):
        base, model = owners()
        docs = [[1,2,3,4],[8,7,6,5],[5,4,3,2]]
        def fresh():
            return st.NativeCausalLmTrainingSession(model, docs, 4, learning_rate=.2, epochs=2, batch_windows=2)
        batch = fresh()
        self.assertEqual((batch.batch_windows,batch.updates_per_epoch,batch.total_updates), (2,2,4))
        self.assertEqual(json.loads(batch.config_json())["batch_windows"],2)
        expected = model.adaptation_loss_gradients(docs[0]+docs[1],[2,4])
        update = batch.step()
        self.assertEqual((update.loss,update.target_tokens), (expected.loss,6))
        manual = model.step_adaptation_sgd(expected, "joint", .2)
        self.assertEqual((manual.lora_artifact_id,manual.geometry_artifact_id),
                         (batch.model.lora_artifact_id,batch.model.geometry_artifact_id))
        payload,pin = batch.checkpoint_with_id()
        cut = st.NativeCausalLmTrainingSession.from_json(base, docs, payload, pin)
        self.assertEqual(cut.next_window,2)
        remaining = [json.loads(s) for s in finish(cut)]
        self.assertEqual([(r["epoch"],r["window_start"],r["window_end"],r["target_tokens"]) for r in remaining],
                         [(0,2,3,3),(1,0,2,6),(1,2,3,3)])
        uninterrupted = fresh()
        finish(uninterrupted)
        self.assertEqual(cut.checkpoint_with_id(), uninterrupted.checkpoint_with_id())
        restored = st.NativeCausalLm.from_session_checkpoint_at(base, payload, pin, update=1,
            recorded_data_id=batch.data_id, post_lora_id=manual.lora_artifact_id, post_geometry_id=manual.geometry_artifact_id)
        self.assertEqual(restored.forward(WINDOW,[1,4]).values(), manual.forward(WINDOW,[1,4]).values())

    def test_batch_admission_precedes_token_access(self):
        class Unreadable:
            def __len__(self): raise AssertionError("read a corpus with an invalid policy")
        class Row:
            def __len__(self): return 4
            def __getitem__(self, i): raise AssertionError("read IDs of an inadmissible batch")
        config, weights = fixtures["fixture"]()
        limited = st.NativeCausalLm.from_gpt2(config,weights,max_training_checkpoint_elements=40).with_lora(0,2,4.)
        for value in (0, True, 1_048_577):
            with self.assertRaises((TypeError,ValueError)):
                st.NativeCausalLmTrainingSession(limited, Unreadable(), 4, learning_rate=.1, mode="lora", batch_windows=value)
        with self.assertRaisesRegex(ValueError,"budget"):
            st.NativeCausalLmTrainingSession(limited,[Row(),Row()],4,learning_rate=.1,mode="lora",batch_windows=2)
        within = st.NativeCausalLmTrainingSession(limited,[WINDOW],4,learning_rate=.1,mode="lora",batch_windows=2)
        self.assertEqual(within.step().target_tokens,3)

    def test_joint_session_snapshot_reopens_for_inference_without_documents(self):
        base, model = owners("f64")
        trained = session(model)
        report = trained.step()
        payload, pin = trained.checkpoint_with_id()
        self.assertEqual((payload, pin), (trained.to_json(), trained.checkpoint_id))
        options = dict(update=report.update, recorded_data_id=trained.data_id,
                       post_geometry_id=trained.model.geometry_artifact_id, post_lora_id=trained.model.lora_artifact_id)
        restored = st.NativeCausalLm.from_session_checkpoint_at(base, payload, pin, **options)
        self.assertEqual(restored.readout_accumulator, "f64")
        self.assertEqual(restored.forward(WINDOW, [1,4]).values(), trained.model.forward(WINDOW, [1,4]).values())
        for key, wrong in (("update", 0), ("recorded_data_id", pin), ("post_geometry_id", None), ("post_lora_id", None)):
            with self.assertRaises(ValueError):
                st.NativeCausalLm.from_session_checkpoint_at(base, payload, pin, **{**options, key:wrong})
        with self.assertRaises(TypeError):
            st.NativeCausalLm.from_session_checkpoint_at(base, payload, pin, **{**options,"update":True})
        for key, wrong in (("post_lora_id", model.lora_artifact_id), ("post_geometry_id", model.geometry_artifact_id)):
            self.assertNotEqual(options[key], wrong)
            with self.assertRaises(ValueError):
                st.NativeCausalLm.from_session_checkpoint_at(base, payload.encode(), pin, **{**options,key:wrong})
        self.assertEqual(st.NativeCausalLmTrainingSession.from_json(base, DOCS, payload.encode(), pin).checkpoint_with_id(), (payload,pin))
        class OpaqueStr(str):
            def __len__(self): raise AssertionError("called overridden length")
            def __getitem__(self, i): raise AssertionError("called overridden slice")
            def isascii(self): raise AssertionError("called overridden ASCII query")
            def encode(self, *args): raise AssertionError("called overridden encoding")
        actual = st.NativeCausalLm.from_session_checkpoint_at(base, OpaqueStr(payload), OpaqueStr(pin), **options)
        self.assertEqual(actual.geometry_artifact_id, options["post_geometry_id"])
        with self.assertRaisesRegex(ValueError, "byte budget"):
            st.NativeCausalLm.from_session_checkpoint_at(base, payload, OpaqueStr("\U0001f600"*20), **options)
        with self.assertRaises(UnicodeEncodeError):
            st.NativeCausalLm.from_session_checkpoint_at(base, "\ud800", pin, **options)
        with self.assertRaises(TypeError):
            st.NativeCausalLm.from_session_checkpoint_at(base, bytearray(payload.encode()), pin, **options)

    def test_resume_example_checks_pin_before_loading_model_or_corpus(self):
        cmd = [sys.executable, "-I", "-B", str(Path(__file__).parents[1] / "examples" / "native_training_session.py")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "session.json"
            checkpoint.write_bytes(b"{}")
            cmd.extend(["--config", str(root/"absent.cfg"), "--weights", str(root/"absent.weights"),
                        "--documents", str(root/"absent.docs"), "--resume", str(checkpoint), "--output", str(root/"result")])
            missing = subprocess.run(cmd, capture_output=True,text=True,timeout=30)
            self.assertNotEqual(missing.returncode,0)
            self.assertIn("separately retained --expected-id",missing.stderr)
            wrong = subprocess.run([*cmd,"--expected-id","sha256:"+"0"*64],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(wrong.returncode,0)
            self.assertIn("differs from pinned identity",wrong.stderr)
            self.assertFalse((root/"result").exists())
            pin = "sha256:"+"0"*64
            (root/"boundary").write_text(json.dumps(dict(checkpoint_id=pin,completed_updates=1,
                recorded_data_id=pin,post_lora_id=pin,post_geometry_id=None)))
            wrong = subprocess.run([sys.executable,"-I","-B",str(Path(__file__).parents[1]/"examples"/"native_geometry_generation.py"),
                "--model-dir",str(root/"absent-model"),"--output",str(root/"not-created"),"--session-checkpoint",str(checkpoint),
                "--session-checkpoint-id",pin,"--session-boundary",str(root/"boundary"),"--prompt-json",str(root/"absent-prompt"),
                "--recipe-json",str(root/"absent-recipe")],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(wrong.returncode,0)
            self.assertIn("session checkpoint differs from pinned identity",wrong.stderr)
            self.assertFalse((root/"not-created").exists())

    def test_file_examples_generate_from_a_joint_minibatch_without_the_training_file(self):
        examples = Path(__file__).parents[1] / "examples"
        config, weights = fixtures["fixture"](raw_weights=True)
        base, joint = owners()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"config.json").write_text(config)
            (root/"model.safetensors").write_bytes(weights)
            (root/"geometry.json").write_text(joint.geometry_artifact().to_json())
            docs = root/"docs.json"
            docs.write_text(json.dumps(DOCS))
            ck = root/"checkpoint.json"
            command = [sys.executable,"-I","-B",str(examples/"native_training_session.py"),"--config",str(root/"config.json"),
                       "--weights",str(root/"model.safetensors"),"--documents",str(docs),"--geometry",str(root/"geometry.json"),
                       "--context","4","--mode","joint","--batch-windows","2","--output",str(ck)]
            record = json.loads(subprocess.run(command,check=True,capture_output=True,text=True,timeout=60).stdout)
            docs.unlink()
            boundary = root/"boundary.json"
            boundary.write_text(json.dumps(record))
            (root/"prompt.json").write_text("[1,2]")
            recipe = '{"max_new_tokens":3}'
            (root/"recipe.json").write_text(recipe)
            run = [sys.executable,"-I","-B",str(examples/"native_geometry_generation.py"),"--model-dir",str(root),
                   "--prompt-json",str(root/"prompt.json"),"--recipe-json",str(root/"recipe.json"),
                   "--session-checkpoint",str(ck),"--session-checkpoint-id",record["checkpoint_id"],"--session-boundary",str(boundary)]
            result = json.loads(subprocess.run([*run,"--output",str(root/"gen")],check=True,capture_output=True,text=True,timeout=60).stdout)
            restored = st.NativeCausalLm.from_session_checkpoint_at(base,ck.read_bytes(),record["checkpoint_id"],
                update=record["completed_updates"],**{k:record[k] for k in ("recorded_data_id","post_lora_id","post_geometry_id")})
            gen = st.NativeCausalLmGenerator(restored,[1,2],recipe)
            while not gen.finished:
                gen.step()
            self.assertEqual(result["checkpoint_id"], gen.checkpoint_id)
            self.assertFalse(result["session_boundary"]["training_documents_verified"])
            self.assertEqual(result["lora_artifact_id"], restored.lora_artifact_id)
            record["post_lora_id"] = None
            boundary.write_text(json.dumps(record))
            bad = subprocess.run([*run,"--output",str(root/"wrong")],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(bad.returncode,0)
            self.assertFalse((root/"wrong").exists())

    def test_geometry_checkpoint_does_not_inherit_external_lora(self):
        base, joint = owners()
        geometry = joint.geometry_artifact()
        trainer = st.NativeCausalLmTrainer(
            base.with_geometry(geometry), st.NativeGeometryAdamW(geometry), DOCS, 4,
            learning_rate=.01,
        )
        trainer.step_observed()
        payload, pin = trainer.to_json(), trainer.checkpoint_id
        restored = st.NativeCausalLm.from_training_checkpoint(base, payload, pin)
        self.assertEqual(restored.geometry_artifact_id, trainer.model.geometry_artifact_id)
        lora = base.with_lora(0, 2, 4.)
        changed = lora.step_lora_sgd(lora.lora_loss_gradients(WINDOW, [1, 4]), .2)
        self.assertNotEqual(base.forward(WINDOW, [1, 4]).values(), changed.forward(WINDOW, [1, 4]).values())
        for external in (lora, changed):
            with self.subTest(adapter=external.lora_artifact_id), self.assertRaisesRegex(ValueError, "LoRA"):
                st.NativeCausalLm.from_training_checkpoint(external, payload, pin)

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
from pathlib import Path
class NoForeign(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','transformers','accelerate','safetensors'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoForeign())
import spiraltorch as st
request=json.load(sys.stdin)
assert sys.flags.isolated and sys.dont_write_bytecode
assert [str(Path(m.__file__).resolve()) for m in (st,st._rs)] == request['origins']
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
            payload = {"fixture": FIXTURE, "docs": DOCS, "checkpoint": trainer.to_json(), "id": trainer.checkpoint_id,
                       "origins": [str(Path(m.__file__).resolve()) for m in (st, st._rs)]}
            reports = finish(trainer)
            child = subprocess.run([sys.executable, "-I", "-B", "-c", script], input=json.dumps(payload), text=True,
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
        for readout, batch_windows in (("f32", 1), ("f64", 1), ("f32", 2), ("f64", 2)):
            base, model = owners(readout)
            docs = DOCS if batch_windows == 1 else DOCS + [WINDOW]
            trainer = st.NativeCausalLmTrainingSession(model, docs, 4, learning_rate=.2,
                                                      epochs=2, batch_windows=batch_windows)
            trainer.step()
            config, weights = fixtures["fixture"](raw_weights=True)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(config)
                (root / "weights.safetensors").write_bytes(weights)
                (root / "docs.json").write_text(json.dumps(docs))
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
                    base.with_readout_accumulator("f32"), docs, checkpoint, receipt["checkpoint_id"])
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
