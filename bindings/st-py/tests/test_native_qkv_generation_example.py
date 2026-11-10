"""File-based QKV serving handoff using only small native synthetic fixtures."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import spiraltorch as st
from test_native_causal_lm import artifact, fixture


class NativeQkvGenerationExampleTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "examples/native_geometry_generation.py"
        self.main = runpy.run_path(str(path))["main"]

    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.main([str(x) for x in args])
        return json.loads(output.getvalue())

    def prepare(self, root, *, source="geometry", lag=0, arithmetic="f32"):
        config, raw = fixture(raw_weights=True)
        if lag:
            cfg = json.loads(config)
            cfg.update(model_type="spiraltorch_gpt2_key_lag_v1",
                       spiraltorch_key_lag={"layer": 0, "lag": lag})
            config = json.dumps(cfg)
        (root / "config.json").write_text(config)
        (root / "model.safetensors").write_bytes(raw)
        base = st.NativeCausalLm.from_gpt2(config, st.SafeTensorArchive.from_bytes(raw))
        base = base.with_readout_accumulator(arithmetic)
        if source == "session":
            model = base.with_lora(0, 2, 4.).with_geometry(
                artifact(base, path="transformer.h.0.attn", trainable=True))
            session = st.NativeCausalLmTrainingSession(model, [[1, 2, 3, 4]], 4,
                                                      learning_rate=.02, mode="joint")
            report = session.step()
            payload, pin = session.checkpoint_with_id()
            (root / "session.json").write_text(payload)
            model = session.model
            boundary = dict(checkpoint_id=pin, completed_updates=report.update,
                            recorded_data_id=session.data_id, post_lora_id=model.lora_artifact_id,
                            post_geometry_id=model.geometry_artifact_id)
            (root / "boundary.json").write_text(json.dumps(boundary))
            selected = ["--session-checkpoint", root / "session.json", "--session-checkpoint-id", pin,
                        "--session-boundary", root / "boundary.json"]
        else:
            geometry = artifact(base)
            model = base.with_geometry(geometry)
            (root / "geometry.json").write_text(geometry.to_json())
            selected = ["--artifact", root / "geometry.json", "--artifact-id", model.geometry_artifact_id]
        trainer = st.NativeCausalLmQkvAdamW(model, 0)
        trainer.step_answers([[1, 2], [3, 4, 5]], [3, 6], learning_rate=.03)
        payload, pin = trainer.checkpoint_with_id()
        (root / "qkv.bin").write_bytes(payload)
        (root / "prompt.json").write_text("[1,2,3]")
        args = ["--model-dir", root, "--readout-accumulator", arithmetic, *selected,
                "--qkv-checkpoint", root / "qkv.bin", "--qkv-checkpoint-id", pin]
        return model, trainer, args

    def test_file_roundtrip_matches_direct_api_with_frozen_geometry_or_joint_adapters(self):
        for source in ("geometry", "session"):
            for lag in (0, 1):
                for arithmetic in ("f32", "f32_wide_backward", "f64"):
                    with self.subTest(source=source, lag=lag, arithmetic=arithmetic), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        model, trainer, common = self.prepare(root, source=source, lag=lag, arithmetic=arithmetic)
                        originals = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
                        for execution in ({"kind": "full_prefix"}, {"kind": "kv_cache", "max_values": 96}):
                            recipe = dict(max_new_tokens=4, seed=53, execution=execution, decoding=dict(
                                selection="sample", top_k=6, top_p=.8,
                                distribution=dict(kind="zspace", config=dict(curvature=-1.))))
                            recipe_path = root / (execution["kind"] + "-recipe.json")
                            recipe_path.write_text(json.dumps(recipe))
                            fresh = [*common, "--prompt-json", root / "prompt.json", "--recipe-json", recipe_path]
                            full, part, resumed = [root / (execution["kind"] + suffix) for suffix in ("-full", "-part", "-resumed")]
                            result = self.invoke([*fresh, "--output", full])
                            partial = self.invoke([*fresh, "--output", part, "--stop-after", "2"])
                            self.invoke([*common, "--resume", part / "generation.json", "--resume-id", partial["checkpoint_id"],
                                         "--output", resumed])
                            direct = trainer.generator([1, 2, 3], json.dumps(recipe))
                            reports = []
                            while not direct.finished:
                                reports.append(direct.step().to_json() + "\n")
                            self.assertEqual(result["checkpoint_id"], direct.checkpoint_id)
                            self.assertEqual((full / "steps.jsonl").read_text(), "".join(reports))
                            self.assertEqual((full / "steps.jsonl").read_bytes(),
                                             (part / "steps.jsonl").read_bytes() + (resumed / "steps.jsonl").read_bytes())
                            self.assertEqual((full / "generation.json").read_bytes(), (resumed / "generation.json").read_bytes())
                            self.assertEqual(result["readout_accumulator"], arithmetic)
                            self.assertEqual(result["lora_artifact_id"], model.lora_artifact_id)
                            self.assertEqual(result["qkv_snapshot"]["projection_id"], trainer.serving_id)
                            self.assertEqual(result["qkv_snapshot"]["checkpoint_id"], trainer.checkpoint_with_id()[1])
                            self.assertFalse(result["qkv_snapshot"]["training_documents_verified"])
                            if os.name == "posix":
                                self.assertEqual(full.stat().st_mode & 0o777, 0o700)
                        self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_wrong_projection_readout_or_frozen_adapter_cannot_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trainer, common = self.prepare(root, arithmetic="f64")
            recipe = root / "recipe.json"
            recipe.write_text('{"max_new_tokens":4}')
            fresh = ["--prompt-json", root / "prompt.json", "--recipe-json", recipe]
            initial = self.invoke([*common, *fresh, "--output", root / "part", "--stop-after", "2"])
            bad = root / "refused"
            wrong_readout = ["f32" if value == "f64" else value for value in common]
            without_adapter = common[:4] + common[8:]
            for args in (wrong_readout, without_adapter):
                with self.assertRaisesRegex(ValueError, "owner"):
                    self.invoke([*args, *fresh, "--output", bad])
                self.assertFalse(bad.exists())
            original = (root / "qkv.bin").read_bytes()
            trainer.step_answers([[1, 2]], [3], learning_rate=.03)
            other, pin = trainer.checkpoint_with_id()
            other_path = root / "other.bin"
            other_path.write_bytes(other)
            changed = [*common[:-4], "--qkv-checkpoint", other_path, "--qkv-checkpoint-id", pin]
            resume = ["--resume", root / "part/generation.json", "--resume-id", initial["checkpoint_id"], "--output", bad]
            for args in (changed, common[:-4]):
                with self.assertRaisesRegex(ValueError, "owner"):
                    self.invoke([*args, *resume])
                self.assertFalse(bad.exists())
            self.assertEqual((root / "qkv.bin").read_bytes(), original)

    def test_unpaired_flags_and_bad_pin_stop_before_model_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = ["--model-dir", root / "absent", "--output", root / "out",
                      "--prompt-json", root / "prompt", "--recipe-json", root / "recipe"]
            with patch.object(st, "SafeTensorArchive", side_effect=AssertionError("model IO before admission")):
                for extra in (["--qkv-checkpoint", "missing"], ["--qkv-checkpoint-id", "pin"],
                              ["--qkv-checkpoint", "missing", "--qkv-checkpoint-id", ""],
                              ["--readout-accumulator", "unknown"]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                        self.invoke([*common, *extra])
                    self.assertEqual(exc.exception.code, 2)
                checkpoint = root / "bad.bin"
                checkpoint.write_bytes(b"not a checkpoint")
                with self.assertRaisesRegex(ValueError, "pinned identity"):
                    self.invoke([*common, "--qkv-checkpoint", checkpoint, "--qkv-checkpoint-id", "sha256:" + "0" * 64])
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
